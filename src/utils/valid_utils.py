"""
Multi-GPU validation utilities for diffusion model sampling.

This module provides functions for distributing validation sampling across
multiple GPUs using CUDA streams. The key insight is that diffusion sampling
is iterative (50+ steps), so DataParallel's per-forward-pass distribution
doesn't help - we need each GPU to own its samples throughout all iterations.

Architecture:
    1. Unwrap model from DataParallel
    2. Create persistent model copies on each GPU
    3. Split validation batch across GPUs
    4. Run independent sampling loops in parallel using CUDA streams
    5. Gather results to CPU (not GPU 0) to avoid memory accumulation
    6. Cleanup model copies after validation
"""

import copy
from typing import List, Dict, Any
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm


def get_unwrapped_model(model: nn.Module) -> nn.Module:
    """
    Unwrap a model from DataParallel or DistributedDataParallel.
    
    Parameters
    ----------
    model : nn.Module
        Possibly wrapped model
        
    Returns
    -------
    nn.Module
        Unwrapped model (or original if not wrapped)
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def create_model_copies(
    base_model: nn.Module,
    gpu_ids: List[int]
) -> List[nn.Module]:
    """
    Create independent model copies on each specified GPU.
    
    Parameters
    ----------
    base_model : nn.Module
        The unwrapped model to copy
    gpu_ids : List[int]
        List of GPU device IDs
        
    Returns
    -------
    List[nn.Module]
        List of model copies, one per GPU
        
    Notes
    -----
    Uses deepcopy to ensure complete independence. Each copy has its
    own weights on its own device. The original model is not modified.
    """
    models = []
    for gpu_id in gpu_ids:
        device = f'cuda:{gpu_id}'
        model_copy = copy.deepcopy(base_model).to(device)
        model_copy.eval()
        models.append(model_copy)
    return models


def cleanup_model_copies(models: List[nn.Module]) -> None:
    """
    Delete model copies and free GPU memory.
    
    Parameters
    ----------
    models : List[nn.Module]
        List of model copies to delete
    """
    # Move models to CPU before deletion to ensure GPU memory is freed
    for model in models:
        # Clear all parameters and buffers from GPU
        model.cpu()
        # Delete the model
        del model
    
    # Clear the list
    models.clear()
    
    # Force CUDA cache cleanup
    torch.cuda.empty_cache()


def log_gpu_memory(gpu_ids: List[int], phase: str = "") -> Dict[int, Dict[str, float]]:
    """
    Log and return GPU memory statistics.
    
    Parameters
    ----------
    gpu_ids : List[int]
        GPU device IDs to query
    phase : str, optional
        Label for logging (e.g., "after validation")
        
    Returns
    -------
    Dict[int, Dict[str, float]]
        Memory stats per GPU: {gpu_id: {'allocated_mb': X, 'reserved_mb': Y, 'peak_mb': Z}}
    """
    stats = {}
    phase_str = f" ({phase})" if phase else ""
    
    print(f"\nGPU Memory Usage{phase_str}:")
    for gpu_id in gpu_ids:
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
        peak = torch.cuda.max_memory_allocated(gpu_id) / 1024**2
        
        stats[gpu_id] = {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'peak_mb': peak
        }
        
        print(f"  GPU {gpu_id}: Allocated: {allocated:.1f} MB, "
              f"Reserved: {reserved:.1f} MB, Peak: {peak:.1f} MB")
    
    return stats


def reset_memory_stats(gpu_ids: List[int]) -> None:
    """Reset peak memory statistics for all specified GPUs."""
    for gpu_id in gpu_ids:
        torch.cuda.reset_peak_memory_stats(gpu_id)

def sample_on_device(
    diffusion_params: Dict[str, Any],
    model: nn.Module,
    wrapped_model_class: type,
    sampling_model_wrapper_class: type,
    conditioned_image: torch.Tensor,
    device: str,
    sampling_mode: str,
    ddim_eta: float,
) -> torch.Tensor:
    """
    Run diffusion sampling on a specific device.
    
    This function encapsulates a complete DDIM/DDPM sampling loop on a single GPU.
    It uses the OpenAI improved_diffusion sampling implementation.
    
    Parameters
    ----------
    diffusion_params : Dict[str, Any]
        Parameters to access diffusion object:
        - 'diffusion': The GaussianDiffusion/SpacedDiffusion instance (shared, device-agnostic)
        - 'mask_channels': Number of mask channels
        - 'image_size': Spatial size of outputs
    model : nn.Module
        Model copy on target device
    wrapped_model_class : type
        ConditionalModelWrapper class for wrapping model
    sampling_model_wrapper_class : type
        _SamplingModelWrapper class for handling tuple outputs
    conditioned_image : torch.Tensor
        Conditioning images, already on target device
    device : str
        Target device string (e.g., 'cuda:1')
    sampling_mode : str
        'ddim' or 'ddpm'
    ddim_eta : float
        DDIM eta parameter (0.0 = deterministic)
        
    Returns
    -------
    torch.Tensor
        Sampled masks [B, 1, H, W] in [0, 1] range, on CPU
    """
    from src.models.MedSegDiff.unet_util import unnormalize_to_zero_to_one
    
    batch_size = conditioned_image.shape[0]
    diffusion = diffusion_params['diffusion']
    mask_channels = diffusion_params['mask_channels']
    image_size = diffusion_params['image_size']
    
    shape = (batch_size, mask_channels, image_size, image_size)
    
    # Wrap model for OpenAI compatibility
    wrapped_model = wrapped_model_class(model, condition_key='conditioned_image')
    sampling_model = sampling_model_wrapper_class(wrapped_model)
    
    model_kwargs = {'conditioned_image': conditioned_image}
    
    with torch.no_grad():
        if sampling_mode == 'ddim':
            samples = diffusion.ddim_sample_loop(
                model=sampling_model,
                shape=shape,
                noise=None,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=device,
                progress=False,
                eta=ddim_eta,
            )
        else:
            samples = diffusion.p_sample_loop(
                model=sampling_model,
                shape=shape,
                noise=None,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=device,
                progress=False,
            )
    
    # Denormalize and move to CPU
    samples = unnormalize_to_zero_to_one(samples)
    return samples.cpu()


def sample_parallel(
    diffusion_adapter,
    conditioned_images: torch.Tensor,
    models: List[nn.Module],
    gpu_ids: List[int],
) -> torch.Tensor:
    """
    Sample in parallel across multiple GPUs using CUDA streams and threading.
    
    Parameters
    ----------
    diffusion_adapter : GaussianDiffusionAdapter
        The diffusion adapter (for accessing diffusion params and wrappers)
    conditioned_images : torch.Tensor
        Full batch of conditioning images [B, C, H, W]
    models : List[nn.Module]
        Pre-created model copies, one per GPU
    gpu_ids : List[int]
        GPU device IDs
        
    Returns
    -------
    torch.Tensor
        Sampled masks [B, 1, H, W] on CPU
        
    Notes
    -----
    Each GPU processes its portion of the batch independently using threads.
    Results are gathered to CPU (not GPU 0) to avoid memory accumulation.
    CUDA streams + threading enable true parallel execution across GPUs.
    """
    import threading
    from src.models.wrappers.conditional_wrapper import ConditionalModelWrapper
    from src.diffusion.openai_adapter import _SamplingModelWrapper
    
    num_gpus = len(gpu_ids)
    batch_size = conditioned_images.shape[0]
    
    # Split batch across GPUs
    splits = torch.chunk(conditioned_images, num_gpus, dim=0)
    
    # Prepare diffusion parameters (shared across GPUs)
    diffusion_params = {
        'diffusion': diffusion_adapter.diffusion,
        'mask_channels': diffusion_adapter.mask_channels,
        'image_size': diffusion_adapter.image_size,
    }
    
    # Create CUDA streams
    streams = [torch.cuda.Stream(device=f'cuda:{gpu_id}') for gpu_id in gpu_ids]
    
    # Storage for results and threads
    results = [None] * num_gpus
    threads = []
    
    def sample_worker(idx, gpu_id, stream, model, split):
        """Worker function to run sampling on a specific GPU in a separate thread."""
        if split.shape[0] == 0:
            return
            
        device = f'cuda:{gpu_id}'
        
        with torch.cuda.stream(stream):
            # Move input to target device
            split_device = split.to(device, non_blocking=True)
            
            # Sample on this device
            results[idx] = sample_on_device(
                diffusion_params=diffusion_params,
                model=model,
                wrapped_model_class=ConditionalModelWrapper,
                sampling_model_wrapper_class=_SamplingModelWrapper,
                conditioned_image=split_device,
                device=device,
                sampling_mode=diffusion_adapter.sampling_mode,
                ddim_eta=diffusion_adapter.ddim_eta,
            )
    
    # Launch sampling threads for each GPU
    for i, (gpu_id, stream, model, split) in enumerate(zip(gpu_ids, streams, models, splits)):
        thread = threading.Thread(
            target=sample_worker,
            args=(i, gpu_id, stream, model, split)
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Gather results from CPU tensors
    valid_results = [r for r in results if r is not None]
    return torch.cat(valid_results, dim=0)


def _resolve_validation_progress_metric_keys(cfg) -> List[str]:
    """
    Resolve which metric keys should be shown in validation progress bars.
    """
    configured = OmegaConf.select(cfg, "validation.progress_metrics", default=None)
    if configured is None:
        return ["dice_2d_fg", "f1_2d"]

    keys: List[str] = []
    for key in configured:
        key_str = str(key).strip()
        if key_str:
            keys.append(key_str)
    return keys if keys else ["dice_2d_fg", "f1_2d"]


def _build_validation_progress_postfix(
    metric_values: Dict[str, Any],
    progress_metric_keys: List[str],
) -> Dict[str, str]:
    """
    Build tqdm postfix payload for configured validation progress metrics.
    """
    postfix: Dict[str, str] = {}
    for key in progress_metric_keys:
        if key not in metric_values:
            continue
        value = metric_values[key]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = float(value.detach().float().cpu().item())
        try:
            postfix[key] = f"{float(value):.4f}"
        except (TypeError, ValueError):
            continue
    return postfix


def validate_one_epoch_multigpu(
    diffusion,
    val_dl,
    metrics,
    logger,
    global_step: int,
    cfg,
    gpu_ids: List[int],
) -> Dict[str, float]:
    """
    Multi-GPU validation epoch with parallel sampling.
    
    Parameters
    ----------
    diffusion : GaussianDiffusionAdapter
        Diffusion model adapter
    val_dl : DataLoader
        Validation dataloader
    metrics : List
        List of metric objects
    logger : Logger
        Logger instance
    global_step : int
        Current training step
    cfg : DictConfig
        Hydra configuration
    gpu_ids : List[int]
        GPU device IDs to use
        
    Returns
    -------
    Dict[str, float]
        Computed validation metrics
        
    Notes
    -----
    This function:
    1. Creates model copies on each GPU (once per epoch)
    2. Processes validation batches with parallel sampling
    3. Computes metrics on CPU
    4. Cleans up model copies
    5. Logs memory usage
    """
    diffusion.eval()
    
    num_gpus = len(gpu_ids)
    print(f"\n[Validation] Multi-GPU mode with {num_gpus} GPUs: {gpu_ids}")
    
    # Reset memory stats for tracking
    reset_memory_stats(gpu_ids)
    
    # Unwrap model and create copies
    base_model = get_unwrapped_model(diffusion.model)
    print(f"[Validation] Creating model copies on GPUs...")
    models = create_model_copies(base_model, gpu_ids)
    print(f"[Validation] Model copies created")
    
    # Log memory after model copies
    log_gpu_memory(gpu_ids, "after model copy")
    progress_metric_keys = _resolve_validation_progress_metric_keys(cfg)
    
    pbar = tqdm(val_dl, desc="Validation (Multi-GPU)", leave=True)
    
    try:
        for img, true_mask, _ in pbar:
            # Sample in parallel
            pred_mask = sample_parallel(
                diffusion_adapter=diffusion,
                conditioned_images=img,  # Will be moved to devices inside
                models=models,
                gpu_ids=gpu_ids,
            )
            
            # Compute metrics on CPU (pred_mask is already on CPU)
            batch_size = pred_mask.shape[0]
            for i in range(batch_size):
                pred = pred_mask[i]
                true = true_mask[i]
                for metric in metrics:
                    metric(pred, true)
            
            # Update progress bar with current metrics
            current_results = {}
            for metric in metrics:
                metric_results = metric.compute()
                if isinstance(metric_results, dict):
                    current_results.update(metric_results)
            
            key_metrics = _build_validation_progress_postfix(
                current_results,
                progress_metric_keys,
            )
            if key_metrics:
                pbar.set_postfix(**key_metrics)
    
    finally:
        # Always cleanup, even on error
        print(f"\n[Validation] Cleaning up model copies...")
        cleanup_model_copies(models)
        
        # Synchronize all GPUs to ensure cleanup is complete
        for gpu_id in gpu_ids:
            torch.cuda.synchronize(gpu_id)
        
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
    
    # Log peak memory usage
    memory_stats = log_gpu_memory(gpu_ids, "peak during validation")
    
    # Compute final metrics
    results = {}
    for metric in metrics:
        metric_results = metric.compute()
        if isinstance(metric_results, dict):
            results.update(metric_results)
        else:
            results[metric.__class__.__name__] = metric_results
    
    # Reset metrics for next validation
    for metric in metrics:
        metric.reset()
    
    # Set model back to training mode
    diffusion.train()
    
    # Final memory cleanup before returning to training
    torch.cuda.empty_cache()
    
    return results
