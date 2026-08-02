import sys
print("[DEBUG:trainer.py] Starting imports...", flush=True)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
print("[DEBUG:trainer.py] os, torch, nn done", flush=True)

from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
print("[DEBUG:trainer.py] tqdm, optimizers done", flush=True)

from src.training.optimizer_factory import build_optimizer, build_scheduler
print("[DEBUG:trainer.py] optimizer_factory done", flush=True)

from src.utils.general import device_grad_decorator
print("[DEBUG:trainer.py] general done", flush=True)

from src.utils.logger import Logger
print("[DEBUG:trainer.py] logger done", flush=True)

from src.utils.train_utils import calc_grad_norm, calc_param_norm, _parse_multi_gpu_flag
print("[DEBUG:trainer.py] train_utils done", flush=True)
from src.utils.distribution_utils import (
    barrier_if_needed,
    is_main_process,
    resolve_strategy,
    resolve_train_batch_sizes,
)

from torch import no_grad
from torchvision.utils import make_grid
from src.models.MedSegDiff.unet_util import unnormalize_to_zero_to_one, normalize_to_neg_one_to_one
print("[DEBUG:trainer.py] unet_util done", flush=True)

from src.metrics.metrics import get_metric
print("[DEBUG:trainer.py] metrics done", flush=True)

from src.training.checkpoint_utils import (
    should_save_best_checkpoint,
    save_interval_checkpoint,
    save_best_checkpoint,
    cleanup_interval_checkpoints,
    cleanup_best_checkpoints,
)
print("[DEBUG:trainer.py] checkpoint_utils done", flush=True)

import gc
from contextlib import contextmanager
from omegaconf import OmegaConf

# AMP (Automatic Mixed Precision) support
from torch.amp import autocast, GradScaler

# Ensemble validation support
from src.utils.ensemble import (
    should_ensemble,
    ensemble_predictions,
    should_log_ensembled_image,
)
from src.utils.visualization_utils import (
    prepare_discriminative_sample_panels,
    prepare_discriminative_tensor_panel,
)
print("[DEBUG:trainer.py] All imports complete!", flush=True)


@contextmanager
def _rank0_inference_unwrapped_model(diffusion, strategy):
    """
    Temporarily unwrap DDP model for rank-0-only inference blocks.

    Why:
    During validation/image logging we run heavy sampling only on main process
    and park non-main ranks at a barrier. If the model stays wrapped in DDP,
    forward/sampling may trigger distributed collectives that other ranks never
    enter, leading to NCCL watchdog timeouts.
    """
    original_model = getattr(diffusion, "model", None)
    original_wrapped_model = getattr(diffusion, "wrapped_model", None)
    swapped = False

    if (
        strategy == "ddp"
        and is_main_process()
        and isinstance(original_model, nn.parallel.DistributedDataParallel)
    ):
        base_model = original_model.module
        diffusion.model = base_model
        if original_wrapped_model is not None:
            from src.models.wrappers.conditional_wrapper import ConditionalModelWrapper

            condition_key = getattr(original_wrapped_model, "condition_key", "conditioned_image")
            diffusion.wrapped_model = ConditionalModelWrapper(base_model, condition_key=condition_key)
        swapped = True

    try:
        yield
    finally:
        if swapped:
            diffusion.model = original_model
            if original_wrapped_model is not None:
                diffusion.wrapped_model = original_wrapped_model

def get_optimizer_and_scheduler(cfg, model):
    """
    Build optimizer and scheduler using factory pattern.
    
    Separates concerns:
    - cfg.optimizer: optimizer class + hyperparams
    - cfg.scheduler: LR schedule type + params
    - cfg.training: gradient techniques + checkpointing
    
    Args:
        cfg: Hydra config with optimizer, scheduler, training defined
        model: Model to optimize
    
    Returns:
        (optimizer, scheduler) tuple
    
    Raises:
        KeyError: If required config keys missing
        ValueError: If optimizer/scheduler type unsupported
    """
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    return optimizer, scheduler

def train_one_epoch(diffusion, train_dataloader, optimizer, scheduler, cfg, logger, global_step):
    """
    Train the diffusion model for one epoch.

    Args:
        diffusion: The diffusion model.
        train_dataloader: DataLoader for the training dataset.
        optimizer: Optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        cfg (DictConfig): Hydra configuration object.

    Returns:
        train_mean_loss: Average training loss for the epoch.
    """
    diffusion.train()
    total_loss = 0.0

    with tqdm(train_dataloader, desc="Training", leave=False) as pbar:
        for img, mask, *_ in pbar:
            # Move data to the appropriate device
            img, mask = img.to(cfg.device), mask.to(cfg.device)

            optimizer.zero_grad()
            loss, sample_mses, ts = diffusion.forward(mask, conditioned_image=img)

            loss.backward()
            grad_norm = calc_grad_norm(diffusion.parameters())
            optimizer.step()
            param_norm = calc_param_norm(diffusion.parameters())

            # logging
            batch_size = mask.shape[0]
            global_step += 1
            samples = global_step * batch_size

            logger.logkv_mean('loss', loss.item(), accumulator='train')
            logger.logkv_mean('grad_norm', grad_norm, accumulator='train')
            logger.logkv_mean('param_norm', param_norm, accumulator='train')
            logger.logkv('step', global_step, accumulator='train')
            logger.logkv('samples', samples, accumulator='train')
            logger.logkv('lr', optimizer.param_groups[0]['lr'], accumulator='train')
            logger.logkv_loss_quartiles(diffusion, ts, {'loss': sample_mses}, accumulator='train')

            if global_step % logger.log_interval == 0:
                logger.dumpkvs(global_step, accumulator='train')

            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

    # Compute average loss and update scheduler
    train_mean_loss = total_loss / len(train_dataloader)
    scheduler.step(train_mean_loss)

    return train_mean_loss, global_step

@device_grad_decorator(no_grad=True)
def test_one_epoch(diffusion, test_dataloader, cfg, logger, global_step):
    """
    Evaluate the diffusion model for one epoch.

    Args:
        diffusion: The diffusion model.
        test_dataloader: DataLoader for the testing dataset.
        cfg (DictConfig): Hydra configuration object.

    Returns:
        test_mean_loss: Average testing loss for the epoch.
    """
    diffusion.eval()
    total_loss = 0.0

    with tqdm(test_dataloader, desc="Testing", leave=False) as pbar:
        for img, mask, *_ in pbar:
            # Move data to the appropriate device
            img, mask = img.to(cfg.device), mask.to(cfg.device)

            loss, sample_mses, ts = diffusion.forward(mask, conditioned_image=img)
            batch_size = mask.shape[0]
            global_step += 1
            samples = global_step * batch_size

            logger.logkv_mean('test_loss', loss.item(), accumulator='val')
            logger.logkv('test_step', global_step, accumulator='val')
            logger.logkv('test_samples', samples, accumulator='val')
            logger.logkv_loss_quartiles(diffusion, ts, {'test_loss': sample_mses}, accumulator='val')

            if global_step % logger.log_interval == 0:
                logger.dumpkvs(global_step, accumulator='val')

            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

    test_mean_loss = total_loss / len(test_dataloader)
    return test_mean_loss, global_step


def _resolve_validation_progress_metric_keys(cfg):
    """
    Resolve which metric keys should be shown in validation progress bars.
    """
    configured = OmegaConf.select(cfg, "validation.progress_metrics", default=None)
    if configured is None:
        return ["dice_2d_fg", "f1_2d"]

    keys = []
    for key in configured:
        key_str = str(key).strip()
        if key_str:
            keys.append(key_str)
    return keys if keys else ["dice_2d_fg", "f1_2d"]


def _build_validation_progress_postfix(metric_values, progress_metric_keys):
    """
    Build tqdm postfix payload for configured validation progress metrics.
    """
    postfix = {}
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


def _resolve_validation_batch_label(sample_ids):
    """
    Build concise batch label for validation progress display.
    """
    if isinstance(sample_ids, (list, tuple)):
        labels = []
        for item in sample_ids:
            token = str(item).strip()
            if token:
                labels.append(token)
        if len(labels) == 0:
            return "volume"
        if len(labels) == 1:
            return labels[0]
        return f"{labels[0]} (+{len(labels) - 1})"

    token = str(sample_ids).strip()
    return token if token else "volume"


@device_grad_decorator(no_grad=True)
def validate_one_epoch(diffusion, val_dl, metrics, logger, global_step, cfg):
    """
    Validate the diffusion model for one epoch.
    
    Supports ensemble validation when configured. When ensemble is enabled,
    generates multiple samples per input and combines them using the configured
    method before computing metrics.
    
    Args:
        diffusion: The diffusion model.
        val_dl: DataLoader for validation dataset.
        metrics: List of metric objects.
        logger: Logger instance.
        global_step: Current training step.
        cfg: Hydra configuration object.
    
    Returns:
        dict: Aggregated validation metrics.
    """
    # Check if ensemble is enabled
    use_ensemble = should_ensemble(cfg)
    if use_ensemble:
        num_ensemble_samples = cfg.validation.ensemble.num_samples
        ensemble_method = cfg.validation.ensemble.method
        print(f"  Ensemble validation: {num_ensemble_samples} samples, method='{ensemble_method}'")

    from src.inference.pipeline import build_model_probability_executor
    from src.inference.runtime import (
        AssessmentContext,
        parse_inference_runtime,
        validate_runtime_compatibility,
    )

    infer_batch = build_model_probability_executor(backend=diffusion, cfg=cfg)
    inference_runtime = parse_inference_runtime(
        OmegaConf.select(cfg, "inference_runtime", default={})
    )
    validate_runtime_compatibility(
        inference=infer_batch.policy,
        runtime=inference_runtime,
        assessment=AssessmentContext(requires_ground_truth=True),
    )
    diffusion.eval()
    progress_metric_keys = _resolve_validation_progress_metric_keys(cfg)
    
    pbar = tqdm(val_dl, desc="Validation", leave=True)
    for img, true_mask, sample_ids in pbar:
        img = img.to(cfg.device)
        true_mask = true_mask.to(cfg.device)
        batch_label = _resolve_validation_batch_label(sample_ids)
        
        # Generate predictions with optional ensemble
        if use_ensemble:
            # Generate multiple samples and combine
            samples = []
            for ensemble_idx in range(num_ensemble_samples):
                sample = infer_batch(
                    img,
                    progress_label=batch_label,
                    show_window_progress=(ensemble_idx == 0),
                )
                samples.append(sample)
            # Stack: [N, B, C, *spatial]
            samples = torch.stack(samples, dim=0)
            # Combine using configured method
            pred_mask = ensemble_predictions(samples, cfg.validation.ensemble)
        else:
            # Single sample (original behavior)
            pred_mask = infer_batch(
                img,
                progress_label=batch_label,
                show_window_progress=True,
            )
        
        # Process each sample in batch (since metrics are slice-wise)
        batch_size = img.shape[0]
        for i in range(batch_size):
            pred = pred_mask[i]
            true = true_mask[i]
            for metric in metrics:
                metric(pred, true)  # Call forward
        
        # Compute current running metrics and update postfix
        current_results = {}
        for metric in metrics:
            metric_results = metric.compute()
            if isinstance(metric_results, dict):
                current_results.update(metric_results)
            else:
                current_results[metric.__class__.__name__] = metric_results
        
        key_metrics = _build_validation_progress_postfix(
            current_results,
            progress_metric_keys,
        )
        if key_metrics:
            pbar.set_postfix(**key_metrics)

    # Compute aggregated results
    results = {}
    for metric in metrics:
        metric_results = metric.compute()
        if isinstance(metric_results, dict):  # For aggregators
            results.update(metric_results)
        else:
            results[metric.__class__.__name__] = metric_results

    # Update tqdm postfix with key metrics
    key_metrics = _build_validation_progress_postfix(results, progress_metric_keys)
    if key_metrics:
        pbar.set_postfix(**key_metrics)

    # Reset metrics
    for metric in metrics:
        metric.reset()

    # Validation temporarily switches model to eval mode; restore train mode
    # so subsequent training/logging paths keep expected behavior.
    diffusion.train()

    return results


def _resolve_context_window_cfg(cfg):
    loader_mode = OmegaConf.select(cfg, "data_mode.loader_mode", default=None)
    per_side_context_slices = int(
        OmegaConf.select(cfg, "data_mode.per_side_context_slices", default=0) or 0
    )
    channel_layout = str(
        OmegaConf.select(cfg, "data_mode.channel_layout", default="slice_major") or "slice_major"
    )
    return loader_mode, per_side_context_slices, channel_layout


def _resolve_modality_names_for_channels(cfg, num_channels):
    modality_names = (
        list(cfg.dataset.modalities)
        if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "modalities")
        else []
    )
    if len(modality_names) == num_channels:
        return modality_names
    return [f"Ch{i}" for i in range(num_channels)]


def _should_use_context_modalities_panel(cfg, num_channels):
    loader_mode, per_side_context_slices, _ = _resolve_context_window_cfg(cfg)
    if loader_mode != "nnunet_slices_2d" or per_side_context_slices <= 0:
        return False

    num_modalities = len(
        list(cfg.dataset.modalities)
        if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "modalities")
        else []
    )
    if num_modalities <= 0:
        return False
    num_effective_slices = (2 * per_side_context_slices) + 1
    return num_channels == (num_modalities * num_effective_slices)


def _context_grid_channel_index(
    modality_idx, slice_idx, num_modalities, num_effective_slices, channel_layout
):
    if channel_layout == "slice_major":
        return (slice_idx * num_modalities) + modality_idx
    return (modality_idx * num_effective_slices) + slice_idx


def _build_context_modalities_panel(sample_img_chw, cfg):
    """Create one compact context panel honoring channel_layout semantics."""
    if sample_img_chw.dim() != 3:
        raise ValueError(
            f"Expected sample image tensor with shape [C,H,W], got {tuple(sample_img_chw.shape)}."
        )

    channel_count, height, width = sample_img_chw.shape
    modality_names = list(cfg.dataset.modalities)
    num_modalities = len(modality_names)
    _, per_side_context_slices, channel_layout = _resolve_context_window_cfg(cfg)
    num_effective_slices = (2 * per_side_context_slices) + 1
    expected_channels = num_modalities * num_effective_slices
    if channel_count != expected_channels:
        raise ValueError(
            "Cannot build context modalities panel: channel count mismatch. "
            f"Expected {expected_channels} from data contract, got {channel_count}."
        )

    cells = []
    if channel_layout == "slice_major":
        # Display as [num_effective_slices rows, num_modalities cols].
        for slice_idx in range(num_effective_slices):
            for modality_idx in range(num_modalities):
                channel_idx = _context_grid_channel_index(
                    modality_idx=modality_idx,
                    slice_idx=slice_idx,
                    num_modalities=num_modalities,
                    num_effective_slices=num_effective_slices,
                    channel_layout=channel_layout,
                )
                channel = normalize_to_neg_one_to_one(sample_img_chw[channel_idx:channel_idx + 1])
                cells.append(channel)
        nrow = num_modalities
    else:
        # Display as [num_modalities rows, num_effective_slices cols].
        for modality_idx in range(num_modalities):
            for slice_idx in range(num_effective_slices):
                channel_idx = _context_grid_channel_index(
                    modality_idx=modality_idx,
                    slice_idx=slice_idx,
                    num_modalities=num_modalities,
                    num_effective_slices=num_effective_slices,
                    channel_layout=channel_layout,
                )
                channel = normalize_to_neg_one_to_one(sample_img_chw[channel_idx:channel_idx + 1])
                cells.append(channel)
        nrow = num_effective_slices

    mini_grid = make_grid(cells, nrow=nrow, padding=0, normalize=False)
    if mini_grid.dim() != 3:
        raise RuntimeError(f"Unexpected mini-grid shape: {tuple(mini_grid.shape)}")

    if mini_grid.shape[0] > 1:
        mini_grid = mini_grid.mean(dim=0, keepdim=True)

    # Preserve legacy left-block footprint: height=H, width=num_modalities*W.
    target_width = num_modalities * width
    mini_grid = F.interpolate(
        mini_grid.unsqueeze(0),
        size=(height, target_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return mini_grid


def _context_panel_label(cfg):
    _, per_side_context_slices, channel_layout = _resolve_context_window_cfg(cfg)
    return f"Input Context Grid k={per_side_context_slices} ({channel_layout})"


def _concat_row_panels(panels):
    """Concatenate 1-channel panels into a single row image [1, H, W_total]."""
    normalized = []
    for panel in panels:
        if panel.dim() == 4 and panel.shape[0] == 1:
            panel = panel.squeeze(0)
        if panel.dim() == 2:
            panel = panel.unsqueeze(0)
        if panel.dim() != 3:
            raise ValueError(f"Expected panel with shape [C,H,W], got {tuple(panel.shape)}.")
        if panel.shape[0] > 1:
            panel = panel.mean(dim=0, keepdim=True)
        normalized.append(panel)
    return torch.cat(normalized, dim=-1)


@device_grad_decorator(no_grad=True)
def log_ensembled_segmentation(diffusion, val_dl, logger, global_step, cfg):
    """
    Log ensembled segmentation grid to TensorBoard.
    
    Generates an ensemble of predictions for a few validation samples and logs
    a grid visualization showing modalities, ground truth, and the ensembled
    prediction.
    
    Grid layout:
        Rows = ensembled_image.num_samples slices
        Cols = [modalities...] [ground_truth] [ensembled_pred]
    
    Args:
        diffusion: The diffusion model.
        val_dl: Validation dataloader.
        logger: Logger instance with TensorBoard writer.
        global_step: Current training step.
        cfg: Hydra configuration object.
    """
    diffusion.eval()
    
    ensembled_image_cfg = cfg.validation.ensembled_image
    num_vis_samples = ensembled_image_cfg.num_samples
    
    # Require ensemble config - don't use defaults
    if not hasattr(cfg.validation, 'ensemble'):
        raise ValueError(
            "log_ensembled_segmentation requires 'validation.ensemble' config section. "
            "Please use 'validation: ensemble' in your config or add the ensemble section manually."
        )
    
    ensemble_cfg = cfg.validation.ensemble
    num_ensemble_samples = ensemble_cfg.num_samples
    
    print(f"Logging ensembled segmentation at step {global_step} "
          f"({num_vis_samples} slices, {num_ensemble_samples} ensemble samples)")
    
    # Collect samples from validation dataloader
    collected_imgs = []
    collected_masks = []
    dl_iter = iter(val_dl)
    
    while len(collected_imgs) < num_vis_samples:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(val_dl)
            batch = next(dl_iter)
        
        b_img, b_mask = batch[0], batch[1]
        for j in range(b_img.shape[0]):
            if len(collected_imgs) >= num_vis_samples:
                break
            # Prefer non-empty masks for visualization
            if b_mask[j].sum() > 0:
                collected_imgs.append(b_img[j:j+1])
                collected_masks.append(b_mask[j:j+1])
    
    if len(collected_imgs) < num_vis_samples:
        print(f"Warning: Only found {len(collected_imgs)} non-empty validation samples")
        if len(collected_imgs) == 0:
            return  # Nothing to log
    
    # Stack collected samples
    sample_imgs = torch.cat(collected_imgs, dim=0).to(cfg.device)
    sample_masks = torch.cat(collected_masks, dim=0).to(cfg.device)
    num_samples = sample_imgs.shape[0]
    num_input_channels = sample_imgs.shape[1]
    use_context_panel = _should_use_context_modalities_panel(cfg, num_input_channels)
    modality_names = _resolve_modality_names_for_channels(cfg, num_input_channels)
    num_modalities = len(modality_names) if not use_context_panel else len(cfg.dataset.modalities)
    
    # Build grid
    all_images = []
    labels = []
    
    for i in range(num_samples):
        # Generate ensemble prediction for this slice
        img_slice = sample_imgs[i:i+1]  # [1, C, H, W]
        
        # Collect multiple samples
        samples = []
        for _ in range(num_ensemble_samples):
            sample = diffusion.sample(img_slice, disable_tqdm=True)
            samples.append(sample)
        samples = torch.stack(samples, dim=0)  # [N, 1, 1, H, W]
        
        # Ensemble the predictions
        ensembled_pred = ensemble_predictions(samples, ensemble_cfg)  # [1, 1, H, W]
        
        # Add modality input section (legacy strip or compact context panel)
        if use_context_panel:
            context_panel = _build_context_modalities_panel(sample_imgs[i], cfg)
            gt_norm = normalize_to_neg_one_to_one(sample_masks[i])
            pred_norm = normalize_to_neg_one_to_one(ensembled_pred[0])
            row_image = _concat_row_panels([context_panel, gt_norm, pred_norm])
            all_images.append(row_image)
            labels.append("Context Grid | Ground Truth | Ensemble")
        else:
            for c in range(num_modalities):
                mod_channel = normalize_to_neg_one_to_one(sample_imgs[i, c:c+1])
                all_images.append(mod_channel)
                labels.append(f"Modality: {modality_names[c]}")
            # Add ground truth
            gt_norm = normalize_to_neg_one_to_one(sample_masks[i])
            all_images.append(gt_norm)
            labels.append("Ground Truth")
            # Add ensembled prediction
            pred_norm = normalize_to_neg_one_to_one(ensembled_pred[0])
            all_images.append(pred_norm)
            labels.append(f"Ensemble ({num_ensemble_samples} avg)")
    
    # Calculate grid dimensions
    per_sample_ncol = 1 if use_context_panel else (num_modalities + 2)
    
    # Log the grid
    logger.log_image_grid(
        f"validation/ensembled_segmentation",
        all_images,
        global_step,
        metrics=None,
        grid_layout='horizontal',
        labels=labels,
        per_sample_ncol=per_sample_ncol,
    )


def save_model(diffusion, old_best_epoch, new_best_epoch, model_save_path_template):
    """
    Save the current best model, removing the previous best model file.

    Args:
        diffusion: The diffusion model.
        old_best_epoch: The epoch number of the previous best model.
        new_best_epoch: The epoch number of the current best model.
        model_save_path_template: String template for model save paths.
    """
    # Remove old best model if it exists
    old_model_path = model_save_path_template.format(old_best_epoch)
    if os.path.exists(old_model_path):
        os.remove(old_model_path)

    # Save the new best model
    new_model_path = model_save_path_template.format(new_best_epoch)
    torch.save(diffusion.state_dict(), new_model_path)

def train_and_evaluate(
    cfg,
    diffusion,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    logger,
    model_save_path_template=None,
):
    """
    Train and evaluate the diffusion model.

    Args:
        cfg (DictConfig): Hydra configuration object.
        diffusion: The diffusion model.
        train_dataloader: DataLoader for the training dataset.
        test_dataloader: DataLoader for the test dataset.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        writer: TensorBoard SummaryWriter (optional).
        model_save_path_template: String template for model save paths.

    Returns:
        train_losses: List of average training losses for each epoch.
        test_losses: List of average test losses for each epoch.
        best_epoch: The epoch with the best test loss.
    """
    best_test_loss = float('inf')
    best_test_loss_epoch = -1
    train_losses = []
    test_losses = []

    global_step = 0

    for epoch in range(1, cfg.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.training.num_epochs}")
        print("-" * 30)

        # Train for one epoch
        train_loss, global_step = train_one_epoch(diffusion, train_dataloader, optimizer, scheduler, cfg, logger, global_step)
        train_losses.append(train_loss)

        # Evaluate for one epoch
        test_loss, global_step = test_one_epoch(diffusion, test_dataloader, cfg, logger, global_step)
        test_losses.append(test_loss)

        if logger.writer is not None:
            logger.writer.add_scalar('Loss/train', train_loss, epoch)
            logger.writer.add_scalar('Loss/test', test_loss, epoch)
            logger.writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # Save model if it's the best test loss
        if test_loss < best_test_loss:
            save_model(diffusion, best_test_loss_epoch, epoch, model_save_path_template)
            best_test_loss = test_loss
            best_test_loss_epoch = epoch

        # Log progress
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Best Test Loss: {best_test_loss:.4f} (Epoch {best_test_loss_epoch})")

        # Visualization
        if epoch % cfg.training.visualization_sample_interval == 0 or epoch == cfg.training.num_epochs:
            '''
            sample_and_visualize(
                diffusion,
                test_dataloader.dataset,
                num_samples=cfg.training.visualization_num_samples,
                device=cfg.device,
            )
            '''
            pass

    print(f"\nTraining complete. Best Test Loss: {best_test_loss:.4f} at Epoch {best_test_loss_epoch}.")
    if logger.writer is not None:
        logger.writer.close()
    return train_losses, test_losses, best_test_loss_epoch

def step_based_train(cfg, diffusion, dataloaders, optimizer, scheduler, logger, run_dir=None, resume_state=None):
    """
    Step-based training loop with optional resume support.
    
    Args:
        cfg: Hydra config
        diffusion: Diffusion model
        dataloaders: Dict with 'train', 'val', 'sample' dataloaders
        optimizer: Optimizer
        scheduler: Learning rate scheduler (or None)
        logger: Logger instance
        run_dir: Path to run output directory (with trailing slash)
        resume_state: Optional dict from load_checkpoint() for resuming training.
                      Contains: model_state_dict, optimizer_state_dict, training_state, global_step
    """
    # Access dataloaders
    train_dataloader = dataloaders['train']
    sample_dataloader = dataloaders['sample']  # Formerly test_dataloader
    val_dataloader = dataloaders['val']  # For future validation
    main_process = is_main_process()

    # Startup validation
    if not cfg.training.checkpoint_interval.enabled and not cfg.training.checkpoint_best.enabled:
        print("[WARN] Both checkpoint systems are disabled. No checkpoints will be saved!")
    
    if cfg.training.checkpoint_best.enabled and not cfg.validation.get('validation_interval'):
        print("[WARN] checkpoint_best enabled but validation_interval not set!")

    # Use cfg values directly
    max_steps = cfg.training.max_steps
    log_interval = cfg.logging.interval
    ema_rates = [float(r) for r in cfg.training.ema_rate.split(',') if r.strip()]
    
    diffusion.train()
    train_iterator = iter(train_dataloader)
    train_sampler = getattr(train_dataloader, "sampler", None)
    sampler_epoch = 0
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(sampler_epoch)
    
    # Gradient accumulation setup
    accumulation_steps = cfg.training.gradient.accumulation_steps
    if accumulation_steps is None:
        accumulation_steps = 1
    accumulation_steps = int(accumulation_steps)
    loader_mode = str(OmegaConf.select(cfg, "data_mode.loader_mode", default="") or "")
    if loader_mode == "random_patches_3d":
        return_patches_per_case = int(
            OmegaConf.select(
                cfg,
                "dataset.preprocessing_configs.random_patches_3d.return_patches_per_case.train",
                default=1,
            )
            or 1
        )
    else:
        return_patches_per_case = 1
    if return_patches_per_case <= 0:
        raise ValueError(
            "dataset.preprocessing_configs.random_patches_3d.return_patches_per_case.train "
            f"must be > 0, got {return_patches_per_case}."
        )
    strategy = resolve_strategy(cfg)
    global_physical_batch_size, local_physical_batch_size = resolve_train_batch_sizes(
        int(cfg.data_runtime.train_batch_size),
        strategy=strategy,
    )
    global_effective_batch_size = global_physical_batch_size * accumulation_steps
    global_effective_patch_batch_size = (
        global_physical_batch_size * return_patches_per_case * accumulation_steps
    )
    world_size = max(1, global_physical_batch_size // local_physical_batch_size)
    accumulation_counter = 0
    
    print(f"\nGradient Accumulation Configuration:")
    print(f"  Global batch size: {global_physical_batch_size}")
    print(f"  Per-rank batch size: {local_physical_batch_size} (world_size={world_size})")
    print(f"  Return patches per case: {return_patches_per_case}")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Global effective batch size (cases): {global_effective_batch_size}")
    print(
        "  Global effective patch batch size: "
        f"{global_physical_batch_size}*{return_patches_per_case}*{accumulation_steps}"
        f"={global_effective_patch_batch_size}"
    )
    
    # ==========================================================================
    # AMP (Automatic Mixed Precision) setup
    # ==========================================================================
    amp_cfg = cfg.training.get('amp', {})
    amp_enabled = amp_cfg.get('enabled', False)
    amp_dtype_str = amp_cfg.get('dtype', 'float32')
    
    # Map string to torch dtype
    amp_dtype_map = {
        'float32': None,  # No autocast (explicit FP32)
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    amp_dtype = amp_dtype_map.get(amp_dtype_str, None)
    
    # GradScaler is required for float16, but NOT for bfloat16
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = GradScaler('cuda') if use_scaler else None
    
    print(f"\nAutomatic Mixed Precision (AMP) Configuration:")
    print(f"  Enabled: {amp_enabled}")
    print(f"  Dtype: {amp_dtype_str}")
    print(f"  GradScaler: {'enabled' if use_scaler else 'disabled'}")
    
    # ==========================================================================
    # Initialize training state (fresh or from resume_state)
    # ==========================================================================
    if resume_state is not None:
        # Resuming from checkpoint
        training_state = resume_state.get('training_state', {})
        global_step = resume_state['global_step']
        
        # Restore EMA params
        saved_ema_params = training_state.get('ema_params', [])
        saved_ema_rates = training_state.get('ema_rates', [])
        
        if saved_ema_params and saved_ema_rates == ema_rates:
            # Restore EMA params from checkpoint (move to same device as model)
            device = next(diffusion.parameters()).device
            ema_params = [
                [p.to(device) for p in rate_params]
                for rate_params in saved_ema_params
            ]
            print(f"  ├─ EMA params restored for rates: {ema_rates}")
        else:
            # EMA rates changed or not saved - reinitialize from current model
            ema_params = [copy.deepcopy(list(diffusion.parameters())) for _ in ema_rates] if ema_rates else []
            if saved_ema_rates and saved_ema_rates != ema_rates:
                print(f"  ├─ EMA rates changed ({saved_ema_rates} → {ema_rates}), reinitializing EMA")
            else:
                print(f"  ├─ EMA params initialized fresh")
        
        # Restore best metric tracking
        best_metric_value = training_state.get('best_metric_value')
        best_metric_step = training_state.get('best_metric_step')
        
        # If best_metric_value is None, initialize based on metric mode
        if best_metric_value is None and cfg.training.checkpoint_best.enabled:
            best_metric_value = (
                float('-inf') if cfg.training.checkpoint_best.metric_mode == 'max'
                else float('inf')
            )
        
        # Restore scaler state if using AMP with FP16
        if scaler is not None:
            scaler_state = training_state.get('scaler_state_dict')
            if scaler_state is not None:
                scaler.load_state_dict(scaler_state)
                print(f"  ├─ GradScaler state restored (scale: {scaler.get_scale():.0f})")
            else:
                print(f"  ├─ GradScaler state not found in checkpoint, using defaults")
        
        print(f"\n✓ Resumed training state:")
        print(f"  ├─ Global step: {global_step}")
        print(f"  ├─ Best metric: {best_metric_value} (step {best_metric_step})")
        print(f"  └─ Continuing to max_steps: {max_steps}")
        
    else:
        # Fresh training run
        global_step = 0
        
        # EMA setup (inspired by MedSegDiff)
        ema_params = [copy.deepcopy(list(diffusion.parameters())) for _ in ema_rates] if ema_rates else []
        
        # Checkpoint tracking - initialize based on metric mode
    best_metric_value = (
        float('-inf') if cfg.training.checkpoint_best.enabled and cfg.training.checkpoint_best.metric_mode == 'max'
        else float('inf') if cfg.training.checkpoint_best.enabled and cfg.training.checkpoint_best.metric_mode == 'min'
        else None
    )
    best_metric_step = None
    
    # Checkpoint tracking lists (always start fresh - we don't track old files)
    interval_checkpoints = []  # List of (step, [file_paths])
    best_checkpoints = []  # List of (metric_value, step, [file_paths])
    
    # Scheduling refinements
    interval_loss = 0.0
    interval_count = 0
    
    # Initialize gradients once before training loop
    optimizer.zero_grad()
    
    while max_steps is None or global_step < max_steps:
        try:
            img, mask, *_ = next(train_iterator)
        except StopIteration:
            # Reinitialize iterator if dataset is exhausted
            sampler_epoch += 1
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(sampler_epoch)
            train_iterator = iter(train_dataloader)
            img, mask, *_ = next(train_iterator)
        
        img, mask = img.to(cfg.device), mask.to(cfg.device)
        batch_size = mask.shape[0]  # Define here, after data load
        
        # Forward pass with optional AMP autocast
        with autocast(device_type='cuda', enabled=amp_enabled, dtype=amp_dtype):
            loss, sample_mses, ts, loss_components = diffusion.forward(mask, conditioned_image=img, global_step=global_step)
        
        # NaN/Inf detection (Layer 1: Catch immediately after forward)
        if not torch.isfinite(loss):
            print(f"\n[ERROR] NaN/Inf detected in loss at step {global_step}!")
            print(f"  Loss value: {loss.item()}")
            if loss_components is not None:
                print(f"  Loss components: {loss_components}")
            # Skip this batch but continue training
            print(f"  Skipping batch and resetting gradients...")
            optimizer.zero_grad()
            
            # Aggressive cleanup to prevent cascading failures
            print(f"  Clearing CUDA cache to prevent memory fragmentation...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Reset accumulation counter to avoid partial updates
            accumulation_counter = 0
            
            print(f"  Recovery complete. Continuing training from next batch...")
            continue
        
        # Scale loss for gradient accumulation (normalized to integer, always safe to divide)
        scaled_loss = loss / accumulation_steps
        
        # Backward pass with optional gradient scaling (for FP16 stability)
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Log micro-batch metrics (accumulated across micro-batches)
        # Log unscaled loss (primary metric for model performance)
        if main_process:
            logger.logkv_mean('loss', loss.item(), accumulator='train')
            # Log scaled loss (for verifying accumulation math)
            logger.logkv_mean('loss/scaled', scaled_loss.item(), accumulator='train')
            logger.logkv_loss_quartiles(diffusion, ts, {'loss': sample_mses}, accumulator='train')
        
        # Log loss components if multi-task
        if main_process and loss_components is not None:
            for component_name, component_value in loss_components.items():
                # Log both scaled and unscaled for each component
                logger.logkv_mean(f'loss/{component_name}', component_value, accumulator='train')
                logger.logkv_mean(f'loss/{component_name}_scaled', component_value / accumulation_steps, accumulator='train')
        
        # Increment accumulation counter
        accumulation_counter += 1
        
        # Accumulate loss for interval-based scheduler (if used)
        interval_loss += loss.item()
        interval_count += 1
        
        # Early exit if not yet time for optimizer update (skip macro-batch operations)
        if accumulation_counter < accumulation_steps:
            continue  # Skip to next micro-batch
        
        # ===== Macro-batch operations (only after N micro-batches) =====
        accumulation_counter = 0  # Reset counter
        
        # Unscale gradients BEFORE clipping (required for correct gradient magnitudes)
        # This converts scaled FP16 gradients back to FP32 for proper clipping thresholds
        if scaler is not None:
            scaler.unscale_(optimizer)
        
        # NaN/Inf detection (Layer 2: Check gradients before clipping)
        grad_norm_pre_clip = calc_grad_norm(diffusion.parameters())
        if not torch.isfinite(torch.tensor(grad_norm_pre_clip)):
            print(f"\n[ERROR] NaN/Inf detected in gradients at step {global_step}!")
            print(f"  Gradient norm: {grad_norm_pre_clip}")
            print(f"  Skipping optimizer step and resetting gradients...")
            optimizer.zero_grad()
            accumulation_counter = 0  # Reset accumulation
            if scaler is not None:
                scaler.update()  # Update scaler to potentially reduce scale
            continue
        
        # Gradient clipping (applied to accumulated gradients - correct behavior)
        # Note: clip_norm threshold applies to accumulated gradient, matching large-batch training
        # Do NOT scale clip_norm by accumulation_steps - that would defeat the purpose of clipping
        grad_cfg = cfg.training.gradient
        if grad_cfg.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                diffusion.parameters(),
                max_norm=grad_cfg.clip_norm
            )
        elif grad_cfg.clip_value is not None:
            torch.nn.utils.clip_grad_value_(
                diffusion.parameters(),
                clip_value=grad_cfg.clip_value
            )
        
        # Calculate gradient norm AFTER clipping for logging
        grad_norm = calc_grad_norm(diffusion.parameters())
        
        # Optimizer step (scaler handles inf/nan checking for FP16)
        if scaler is not None:
            scaler.step(optimizer)  # Skips step if gradients contain inf/nan
            scaler.update()         # Adjust scale factor for next iteration
        else:
            optimizer.step()
        
        optimizer.zero_grad()  # Zero gradients for next accumulation cycle
        param_norm = calc_param_norm(diffusion.parameters())
        
        # Step scheduler (per-step for warmup schedulers)
        if scheduler is not None:
            sched_cfg = cfg.scheduler
            if sched_cfg.step_frequency == 'per_step':
                if sched_cfg.scheduler_type != 'reduce_lr':
                    scheduler.step()
        
        # Update EMAs (after optimizer step)
        for rate, params in zip(ema_rates, ema_params):
            for p, ema_p in zip(diffusion.parameters(), params):
                ema_p.data = (1.0 - rate) * p.data + rate * ema_p.data
        
        # Increment global_step here, after training but before logging
        global_step += 1
        samples = global_step * global_effective_patch_batch_size
        samples_cases = global_step * global_effective_batch_size
        
        # Log macro-batch metrics (once per optimizer update)
        if main_process:
            logger.logkv('step', global_step, accumulator='train')
            logger.logkv('samples', samples, accumulator='train')
            logger.logkv('samples_cases', samples_cases, accumulator='train')
            logger.logkv('lr', optimizer.param_groups[0]['lr'], accumulator='train')
            logger.logkv_mean('grad_norm', grad_norm, accumulator='train')
            logger.logkv_mean('param_norm', param_norm, accumulator='train')
        
        # Log clipping diagnostics (only when clipping occurs)
        if main_process and grad_cfg.clip_norm is not None and grad_norm_pre_clip > grad_cfg.clip_norm:
            clip_ratio = grad_norm_pre_clip / grad_cfg.clip_norm
            logger.logkv_mean('grad_norm_pre_clip', grad_norm_pre_clip, accumulator='train')
            logger.logkv_mean('grad_clip_ratio', clip_ratio, accumulator='train')
        
        # Verbose spike logging for debugging gradient explosions
        # Enable with +debug=true or when grad_norm exceeds thresholds
        debug_mode = cfg.get('debug', False)
        if grad_norm_pre_clip > 1e6:
            print(f"\n{'='*60}")
            print(f"[SPIKE ALERT] EXTREME gradient spike at step {global_step}!")
            print(f"  grad_norm_pre_clip: {grad_norm_pre_clip:.2e}")
            print(f"  grad_norm_post_clip: {grad_norm:.4f}")
            if hasattr(diffusion, 'last_loss_components') and diffusion.last_loss_components:
                print(f"  loss_components: {diffusion.last_loss_components}")
            print(f"{'='*60}\n")
        elif grad_norm_pre_clip > 1000 and debug_mode:
            comps = getattr(diffusion, 'last_loss_components', {})
            print(f"[SPIKE] step={global_step} grad_norm_pre_clip={grad_norm_pre_clip:.2e} {comps}")
        elif grad_norm_pre_clip > 100 and debug_mode:
            print(f"[HIGH] step={global_step} grad_norm_pre_clip={grad_norm_pre_clip:.2f}")
        elif grad_norm_pre_clip > 10 and debug_mode:
            print(f"[ELEVATED] step={global_step} grad_norm_pre_clip={grad_norm_pre_clip:.2f}")
        
        # Validation check
        if global_step % cfg.validation.validation_interval == 0 and global_step > 0:
            barrier_if_needed()
            if main_process:
                # Free memory before validation
                gc.collect()
                torch.cuda.empty_cache()
                
                metrics = [get_metric(m['name'], m.get('params', {})) for m in cfg.validation.metrics]
                
                # Check for multi-GPU validation
                gpu_ids = _parse_multi_gpu_flag(cfg.environment.training.multi_gpu)
                use_multi_gpu_val = (
                    cfg.validation.get('multi_gpu_validation', False) 
                    and gpu_ids is not None 
                    and len(gpu_ids) > 1
                )
                
                with _rank0_inference_unwrapped_model(diffusion, strategy):
                    if use_multi_gpu_val:
                        from src.utils.valid_utils import validate_one_epoch_multigpu
                        val_results = validate_one_epoch_multigpu(
                            diffusion, val_dataloader, metrics, logger, global_step, cfg, gpu_ids
                        )
                    else:
                        val_results = validate_one_epoch(
                            diffusion, val_dataloader, metrics, logger, global_step, cfg
                        )
                
                logger.log_metrics_dict("val", val_results, global_step, accumulator='val')
                logger.dumpkvs(global_step, accumulator='val')
                logger.clear_accumulators(accumulator='val')
                
                # Free memory after validation before resuming training
                gc.collect()
                torch.cuda.empty_cache()
                
                # Best checkpoint decision and saving
                if cfg.training.checkpoint_best.enabled:
                    try:
                        should_save, reason, new_best = should_save_best_checkpoint(
                            val_results,
                            best_metric_value,
                            cfg.training.checkpoint_best
                        )
                        
                        if should_save:
                            print(f"✓ {reason}")
                            saved_files = save_best_checkpoint(
                                diffusion, ema_params, ema_rates, global_step,
                                cfg, run_dir, val_results
                            )
                            best_checkpoints.append((new_best, global_step, saved_files))
                            best_metric_value = new_best
                            best_metric_step = global_step
                            
                            # Cleanup old checkpoints if needed
                            keep_n = cfg.training.checkpoint_best.keep_last_n
                            if keep_n is not None and keep_n > 0:
                                best_checkpoints = cleanup_best_checkpoints(
                                    best_checkpoints,
                                    keep_n,
                                    cfg.training.checkpoint_best.metric_mode
                                )
                        else:
                            print(f"✗ {reason}")
                    
                    except KeyError as e:
                        # Metric not found - crash with clear message
                        print(f"[ERROR] {e}")
                        print(f"   Validation returned metrics: {list(val_results.keys())}")
                        print(f"   Check your checkpoint_best.metric_name config!")
                        raise
                
                # Free memory after validation
                gc.collect()
                torch.cuda.empty_cache()
            barrier_if_needed()
        
        # Ensembled segmentation image logging
        if should_log_ensembled_image(cfg, global_step):
            if main_process:
                with _rank0_inference_unwrapped_model(diffusion, strategy):
                    log_ensembled_segmentation(
                        diffusion, val_dataloader, logger, global_step, cfg
                    )
                gc.collect()
                torch.cuda.empty_cache()
            barrier_if_needed()
        
        # New: Image-based logging
        if cfg.logging.enable_image_logging and global_step % cfg.logging.image_log_interval == 0 and global_step >= cfg.logging.min_log_step:
            if main_process:
                print(f"Starting forward image logging at step {global_step}")
                # Log forward process (accumulate fresh batches from dataloader)
                with _rank0_inference_unwrapped_model(diffusion, strategy), torch.no_grad():
                    collected_masks = []
                    collected_imgs = []
                    dl_iter = iter(train_dataloader)  # Use iterator
                    while len(collected_imgs) < cfg.logging.num_log_samples:
                        try:
                            batch = next(dl_iter)
                        except StopIteration:
                            dl_iter = iter(train_dataloader)  # Restart if exhausted
                            batch = next(dl_iter)
                        b_img, b_mask = batch[0], batch[1]  # Full batch
                        for j in range(b_img.shape[0]):
                            if len(collected_imgs) >= cfg.logging.num_log_samples:
                                break
                            if b_mask[j].sum() > 0:  # Only add non-empty
                                collected_imgs.append(b_img[j:j+1])
                                collected_masks.append(b_mask[j:j+1])
                    full_mask = torch.cat(collected_masks, dim=0).to(cfg.device)
                    full_img = torch.cat(collected_imgs, dim=0).to(cfg.device)
                    num_samples = len(full_mask)
                    
                    if num_samples < cfg.logging.num_log_samples:
                        print(f"Warning: Accumulated {num_samples} < num_log_samples {cfg.logging.num_log_samples} for forward.")
                    
                    loss, sample_mses, t, intermediates = diffusion.forward(
                        full_mask, full_img, return_intermediates=True, global_step=global_step
                    )
                    
                    all_images = []
                    labels = []
                    metrics = {}
                    num_input_channels = intermediates['img'].shape[1]
                    modality_names = _resolve_modality_names_for_channels(cfg, num_input_channels)
                    num_modalities = len(modality_names)
                    use_context_panel = _should_use_context_modalities_panel(
                        cfg, num_input_channels
                    )
                    
                    # Check if discriminative mode (no diffusion intermediates)
                    is_discriminative = cfg.diffusion.type == "Discriminative"
                    
                    if is_discriminative:
                        # Discriminative: simpler grid (modalities, prediction, target)
                        per_sample_ncol = 1 if use_context_panel else None
                        for i in range(num_samples):
                            if use_context_panel:
                                context_panel = _build_context_modalities_panel(intermediates['img'][i], cfg)
                                pred_panel = prepare_discriminative_tensor_panel(
                                    tensor=intermediates['pred'][i],
                                    logging_cfg=cfg.logging,
                                )
                                mask_panel = prepare_discriminative_tensor_panel(
                                    tensor=intermediates['mask'][i],
                                    logging_cfg=cfg.logging,
                                )
                                row_image = _concat_row_panels(
                                    [context_panel, pred_panel, mask_panel]
                                )
                                all_images.append(row_image)
                                labels.append("Context Grid | Prediction | Target")
                            else:
                                sample_panels, sample_labels = prepare_discriminative_sample_panels(
                                    sample_img=intermediates['img'][i],
                                    sample_pred=intermediates['pred'][i],
                                    sample_mask=intermediates['mask'][i],
                                    modality_names=modality_names,
                                    logging_cfg=cfg.logging,
                                )
                                all_images.extend(sample_panels)
                                labels.extend(sample_labels)
                                if per_sample_ncol is None:
                                    per_sample_ncol = len(sample_labels)

                        if per_sample_ncol is None:
                            raise RuntimeError(
                                "Unable to resolve per_sample_ncol for discriminative logging."
                            )
                    else:
                        # Diffusion: full grid with x_t, noise, noise_hat, pred_x0
                        for i in range(num_samples):
                            if use_context_panel:
                                input_panel = _build_context_modalities_panel(intermediates['img'][i], cfg)
                                row_image = _concat_row_panels(
                                    [
                                        input_panel,
                                        intermediates['mask'][i],
                                        intermediates['x_t'][i],
                                        intermediates['noise'][i],
                                        intermediates['noise_hat'][i],
                                        intermediates['pred_x0'][i],
                                    ]
                                )
                                sample_images = [row_image]
                                noisy_label = f"Noise Mask t={t[i].item()}"
                                base_labels = [f"Context Grid | Target | {noisy_label} | True Noise | Pred Noise | Recon"]
                            else:
                                sample_images = [
                                    intermediates['img'][i],
                                    intermediates['mask'][i],
                                    intermediates['x_t'][i],
                                    intermediates['noise'][i],
                                    intermediates['noise_hat'][i],
                                    intermediates['pred_x0'][i],
                                ]
                                modality_labels = [f"Modality: {name}" for name in modality_names]
                                noisy_label = f"Noise Mask t={t[i].item()}"
                                base_labels = modality_labels + [
                                    "Target Mask",
                                    noisy_label,
                                    "True Noise",
                                    "Predicted Noise",
                                    "Reconstructed Mask",
                                ]

                            all_images.extend(sample_images)
                            labels.extend(base_labels)
                            metrics[i] = sample_mses[i].item()
                            per_sample_ncol = len(base_labels)
                    
                    logger.log_image_grid(f'Training/Forward_Step_{global_step}', all_images, global_step, metrics, 'horizontal', labels=labels, per_sample_ncol=per_sample_ncol)
            barrier_if_needed()
            
        # Sampling logging (outside image logging block)
        if cfg.logging.enable_sampling_snapshots and global_step % cfg.logging.sampling_log_interval == 0 and global_step >= cfg.logging.min_log_step:
            if main_process:
                print(f"Starting sampling logging at step {global_step}")
                with _rank0_inference_unwrapped_model(diffusion, strategy):
                    # Select dataloader
                    if sample_dataloader is None and cfg.logging.log_test_samples:
                        print("Warning: sample_dataloader is None but log_test_samples=true; falling back to train_dataloader.")
                    dl = sample_dataloader if cfg.logging.log_test_samples and sample_dataloader is not None else train_dataloader
                    if dl is None:
                        print("Warning: No dataloader available for sampling – skipping snapshot logging.")
                    else:
                        # Accumulate samples (images + masks) until we have num_log_samples
                        collected_imgs, collected_masks = [], []
                        remaining = cfg.logging.num_log_samples
                        dl_iter = iter(dl)  # Use iterator to avoid recreating loader each time
                        while len(collected_imgs) < cfg.logging.num_log_samples:
                            try:
                                b_img, b_mask, *_ = next(dl_iter)
                            except StopIteration:
                                dl_iter = iter(dl)  # Restart if exhausted
                                b_img, b_mask, *_ = next(dl_iter)
                            for j in range(b_img.shape[0]):
                                if len(collected_imgs) >= cfg.logging.num_log_samples:
                                    break
                                if b_mask[j].sum() > 0:  # Only add non-empty
                                    collected_imgs.append(b_img[j:j+1])
                                    collected_masks.append(b_mask[j:j+1])
                        sample_imgs = torch.cat(collected_imgs, dim=0).to(cfg.device)
                        sample_masks = torch.cat(collected_masks, dim=0).to(cfg.device)
                        num_samples = sample_imgs.shape[0]
                        
                        # Build composite grid
                        all_images = []
                        labels = []
                        num_input_channels = sample_imgs.shape[1]
                        modality_names = _resolve_modality_names_for_channels(cfg, num_input_channels)
                        num_modalities = len(modality_names)
                        use_context_panel = _should_use_context_modalities_panel(
                            cfg, num_input_channels
                        )
                        # Determine timesteps that sample_with_snapshots will yield
                        snapshots_per_sample = []
                        snapshot_timesteps = []  # Track actual timesteps for labels
                        for i in range(num_samples):
                            snaps = list(diffusion.sample_with_snapshots(sample_imgs[i:i+1], cfg.logging.snapshot_step_interval))
                            # snaps is list of (t, mask); keep order as yielded (descending t) then final 0
                            if i == 0:  # Extract timesteps from first sample (same for all)
                                snapshot_timesteps = [int(t) for t, _ in snaps]
                            snapshots_per_sample.append([normalize_to_neg_one_to_one(m[1]) for m in snaps])  # Normalize snapshots to -1/1
                        num_snapshots = len(snapshots_per_sample[0])
                        
                        # Labels per category
                        modality_labels = [f"Modality: {name}" for name in modality_names]
                        # Use actual timesteps from sample_with_snapshots (works correctly for DDIM respacing)
                        snapshot_labels = [f"t={t}" for t in snapshot_timesteps]
                        row_len = (1 if use_context_panel else num_modalities) + num_snapshots + 1
                        
                        for i in range(num_samples):
                            if use_context_panel:
                                context_panel = _build_context_modalities_panel(sample_imgs[i], cfg)
                                row_panels = [context_panel]
                                row_panels.extend(snapshots_per_sample[i])
                                target_norm = normalize_to_neg_one_to_one(sample_masks[i])
                                row_panels.append(target_norm)
                                all_images.append(_concat_row_panels(row_panels))
                                labels.append("Context Grid | Snapshots | Target")
                            else:
                                # append each modality channel individually
                                for c in range(num_modalities):
                                    mod_channel = normalize_to_neg_one_to_one(sample_imgs[i, c:c+1])  # Normalize to -1/1 for black empty
                                    all_images.append(mod_channel)
                                    labels.append(modality_labels[c])
                                # snapshots
                                for snap in snapshots_per_sample[i]:
                                    all_images.append(snap)
                                labels.extend(snapshot_labels)
                                # target mask (normalize to match forward)
                                target_norm = normalize_to_neg_one_to_one(sample_masks[i])
                                all_images.append(target_norm)
                                labels.append("Target")
                        
                        logger.log_image_grid(
                            f"Sampling/Snapshots_Step_{global_step}",
                            all_images,
                            global_step,
                            metrics=None,
                            grid_layout='horizontal',
                            labels=labels,
                            per_sample_ncol=1 if use_context_panel else row_len,
                        )
            barrier_if_needed()
        
        # Training-metric-based checkpoint (if check_interval is set and not using validation)
        # Must check BEFORE logger clears accumulators
        if main_process and cfg.training.checkpoint_best.enabled:
            check_interval = cfg.training.checkpoint_best.get('check_interval', None)
            if check_interval is not None and global_step % check_interval == 0 and global_step > 0:
                # Get current training metrics from logger (before clearing)
                train_metrics = {k: float(v) for k, v in logger.accumulators['train']['name2val'].items()}
                
                if train_metrics:  # Only proceed if we have metrics
                    try:
                        should_save, reason, new_best = should_save_best_checkpoint(
                            train_metrics,
                            best_metric_value,
                            cfg.training.checkpoint_best
                        )
                        
                        if should_save:
                            print(f"✓ {reason}")
                            saved_files = save_best_checkpoint(
                                diffusion, ema_params, ema_rates, global_step,
                                cfg, run_dir, train_metrics
                            )
                            best_checkpoints.append((new_best, global_step, saved_files))
                            best_metric_value = new_best
                            best_metric_step = global_step
                            
                            # Cleanup old checkpoints if needed
                            keep_n = cfg.training.checkpoint_best.keep_last_n
                            if keep_n is not None and keep_n > 0:
                                best_checkpoints = cleanup_best_checkpoints(
                                    best_checkpoints,
                                    keep_n,
                                    cfg.training.checkpoint_best.metric_mode
                                )
                        else:
                            print(f"✗ {reason}")
                    
                    except KeyError as e:
                        # Metric not found - crash with clear message
                        print(f"[ERROR] {e}")
                        print(f"   Training metrics available: {list(train_metrics.keys())}")
                        print(f"   Check your checkpoint_best.metric_name config!")
                        raise
        
        if main_process and global_step % log_interval == 0:
            logger.dumpkvs(global_step, accumulator='train')
            logger.clear_accumulators(accumulator='train')
        
        # Accumulate for scheduling
        interval_loss += loss.item()
        interval_count += 1
        
        # Interval-based scheduler stepping (only for ReduceLROnPlateau)
        if scheduler is not None:
            sched_cfg = cfg.scheduler
            if sched_cfg.step_frequency == 'per_interval':
                if global_step % sched_cfg.step_interval == 0 and interval_count > 0:
                    mean_loss = interval_loss / interval_count
                    prev_lr = optimizer.param_groups[0]['lr']
                    
                    # Only ReduceLROnPlateau uses this path
                    if sched_cfg.scheduler_type == 'reduce_lr':
                        prev_best = scheduler.best
                        prev_bad_epochs = scheduler.num_bad_epochs
                        prev_cooldown = scheduler.cooldown_counter
                        
                        scheduler.step(mean_loss)
                        
                        new_lr = optimizer.param_groups[0]['lr']
                        reduced = new_lr < prev_lr
                        
                        print(f"Scheduler Update at step {global_step}:")
                        print(f"  Configured Threshold: {scheduler.threshold:.6e}")
                        print(f"  Configured Cooldown: {scheduler.cooldown}")
                        print(f"  Mean Loss: {mean_loss:.6f}")
                        print(f"  Previous Best: {prev_best:.6f}")
                        print(f"  New Best: {scheduler.best:.6f}")
                        print(f"  Bad Epochs: {prev_bad_epochs} → {scheduler.num_bad_epochs}")
                        print(f"  Cooldown: {prev_cooldown} → {scheduler.cooldown_counter}")
                        print(f"  LR: {prev_lr:.6e} → {new_lr:.6e} (Reduced: {reduced})")
                    
                    interval_loss = 0.0
                    interval_count = 0
        
        # Interval checkpoint saving
        if main_process and cfg.training.checkpoint_interval.enabled:
            if global_step % cfg.training.checkpoint_interval.save_interval == 0 and global_step > 0:
                print(f"✓ Saving interval checkpoint at step {global_step}")
                saved_files = save_interval_checkpoint(
                    diffusion, optimizer, global_step, cfg, run_dir,
                    ema_params=ema_params,
                    ema_rates=ema_rates,
                    scheduler=scheduler,
                    best_metric_value=best_metric_value,
                    best_metric_step=best_metric_step,
                    scaler=scaler,
                )
                interval_checkpoints.append((global_step, saved_files))
                
                # Cleanup old checkpoints if needed
                keep_n = cfg.training.checkpoint_interval.keep_last_n
                if keep_n is not None and keep_n > 0:
                    interval_checkpoints = cleanup_interval_checkpoints(
                        interval_checkpoints,
                        keep_n
                    )
        
        # Removed old visualization block
        
    # Final checkpoint save
    if main_process and cfg.training.checkpoint_interval.enabled:
        print(f"✓ Saving final interval checkpoint at step {global_step}")
        saved_files = save_interval_checkpoint(
            diffusion, optimizer, global_step, cfg, run_dir,
            ema_params=ema_params,
            ema_rates=ema_rates,
            scheduler=scheduler,
            best_metric_value=best_metric_value,
            best_metric_step=best_metric_step,
            scaler=scaler,
        )
        # No cleanup on final save
    
    if main_process:
        logger.dumpkvs(global_step, accumulator='train')
        logger.clear_accumulators(accumulator='train')
    
    print(f"Step-based training complete at step {global_step}.")
    if main_process and best_metric_step is not None:
        print(f"Best model saved at step {best_metric_step} with {cfg.training.checkpoint_best.metric_name} = {best_metric_value:.4f}")
