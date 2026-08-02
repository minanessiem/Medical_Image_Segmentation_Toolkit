"""
Training utilities shared between start_training.py and resume_training.py.

This module contains setup functions that are common to both fresh and resumed
training runs, promoting code reuse and consistency.
"""

import math
import os
import random
import copy

import numpy as np
import torch
import torch.distributed as dist
from typing import Iterable, List, Tuple, Optional
from omegaconf import OmegaConf, DictConfig, ListConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from src.utils.distribution_utils import (
    is_main_process,
    get_distribution_state,
    resolve_process_device,
    resolve_strategy,
)
from src.models.model_config import (
    sync_model_image_channels_with_data_contract,
    validate_model_channel_contract,
)


# =============================================================================
# Gradient and Parameter Norm Utilities (existing functions)
# =============================================================================

def calc_grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm of gradients (only params with .grad)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def calc_param_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm of parameters."""
    total = sum(p.data.norm(2).item() ** 2 for p in params)
    return math.sqrt(total)


# =============================================================================
# Multi-GPU Utilities
# =============================================================================

def _parse_multi_gpu_flag(flag):
    """
    Parse multi-GPU configuration flag.
    
    Args:
        flag: Can be None, False, "false", "none", "", a list of GPU ids,
              a tuple of GPU ids, ListConfig, or comma-separated string.
    
    Returns:
        List of GPU ids or None if multi-GPU is disabled.
    
    Examples:
        >>> _parse_multi_gpu_flag(None)
        None
        >>> _parse_multi_gpu_flag([0, 1])
        [0, 1]
        >>> _parse_multi_gpu_flag("0,1,2")
        [0, 1, 2]
    """
    if not flag or str(flag).lower() in {"false", "none", ""}:
        return None
    # Handle OmegaConf ListConfig, plain list, or tuple
    if isinstance(flag, (list, tuple, ListConfig)):
        return [int(x) for x in flag]
    return [int(x.strip()) for x in str(flag).split(',') if x.strip()]


def _resolve_ddp_wrapper_kwargs(cfg: DictConfig) -> dict:
    """Resolve DDP wrapper options from ``cfg.distribution``."""
    distribution_cfg = getattr(cfg, "distribution", None)
    return {
        "find_unused_parameters": bool(
            getattr(distribution_cfg, "find_unused_parameters", False)
        ),
        "broadcast_buffers": bool(getattr(distribution_cfg, "broadcast_buffers", False)),
        "gradient_as_bucket_view": bool(
            getattr(distribution_cfg, "gradient_as_bucket_view", False)
        ),
        "static_graph": bool(getattr(distribution_cfg, "static_graph", False)),
    }


def _wrap_trainable_model_for_strategy(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
    strategy: str,
) -> Tuple[torch.nn.Module, str]:
    """
    Wrap the trainable model exactly once according to distribution strategy.

    Returns:
        Tuple of (wrapped_model, wrapper_name) where wrapper_name is one of
        {"DistributedDataParallel", "DataParallel", "none"}.
    """
    gpu_ids = _parse_multi_gpu_flag(cfg.environment.training.multi_gpu)

    if strategy == "ddp":
        # `environment.training.multi_gpu` is a DP-only knob.
        if gpu_ids:
            raise ValueError(
                "Invalid config: distribution.strategy=ddp does not use "
                "environment.training.multi_gpu. Remove this DP-only setting "
                "or switch strategy to dp."
            )

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "DDP strategy requires an initialized torch.distributed process group "
                "before model wrapping."
            )
        if device.type != "cuda" or device.index is None:
            raise RuntimeError(
                "DDP strategy requires cuda:LOCAL_RANK process device. "
                f"Got device={device}."
            )

        ddp_kwargs = _resolve_ddp_wrapper_kwargs(cfg)
        model = DistributedDataParallel(
            model,
            device_ids=[int(device.index)],
            output_device=int(device.index),
            **ddp_kwargs,
        )
        dist_state = get_distribution_state(strategy)
        print(
            "[Model Wrapping] strategy=ddp wrapper=DistributedDataParallel "
            f"rank={dist_state.rank} local_rank={dist_state.local_rank} "
            f"world_size={dist_state.world_size} ddp_kwargs={ddp_kwargs}"
        )
        return model, "DistributedDataParallel"

    if gpu_ids:
        visible = torch.cuda.device_count()
        if max(gpu_ids) >= visible:
            raise ValueError(f"Requested GPU id {max(gpu_ids)} but only {visible} GPUs visible.")
        print(f"Using GPUs {gpu_ids} with torch.nn.DataParallel")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda(gpu_ids[0])
        print(f"  Wrapped UNet in DataParallel (primary device: cuda:{gpu_ids[0]})")
        print("[Model Wrapping] strategy=dp wrapper=DataParallel")
        return model, "DataParallel"

    print(f"[Model Wrapping] strategy={strategy} wrapper=none")
    return model, "none"


def _assert_ddp_wrapping_invariants(diffusion: torch.nn.Module, strategy: str) -> None:
    """Fail fast if DDP runtime is active without DDP model wrapping."""
    if strategy != "ddp":
        return

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "DDP strategy selected but torch.distributed process group is not initialized."
        )

    trainable_model = getattr(diffusion, "model", None)
    if not isinstance(trainable_model, DistributedDataParallel):
        raise RuntimeError(
            "DDP strategy requires diffusion.model to be wrapped with "
            "torch.nn.parallel.DistributedDataParallel, but got "
            f"{type(trainable_model).__name__}."
        )


# =============================================================================
# Setup Functions
# =============================================================================

def setup_seeds(cfg: DictConfig) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        cfg: Config with random_seed key
    """
    torch.manual_seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)


def setup_device(cfg: DictConfig) -> torch.device:
    """
    Configure and return the training device.
    
    Args:
        cfg: Config with device key ('auto', 'cuda', 'cpu', etc.)
    
    Returns:
        torch.device configured for training
    """
    strategy = resolve_strategy(cfg)
    device = resolve_process_device(cfg.device, strategy=strategy)
    dist_state = get_distribution_state(strategy)
    print(
        f"Using device: {device} "
        f"(strategy={strategy}, rank={dist_state.rank}, "
        f"local_rank={dist_state.local_rank}, world_size={dist_state.world_size})"
    )
    return device


def apply_environment_runtime_context(cfg: DictConfig) -> DictConfig:
    """
    Apply environment-provided runtime context for training bootstrap.
    
    Data contract fields are read directly from dataset/data_mode/data_io/data_runtime.
    This function only maps execution context values that are intentionally sourced
    from the environment group (training paths + device).
    
    Args:
        cfg: Hydra config object
    
    Returns:
        Modified config with runtime context applied
    """
    OmegaConf.set_struct(cfg, False)
    
    # Environment runtime context (training paths)
    cfg.training.output_root = cfg.environment.training.output_root
    cfg.training.model_save_dir = cfg.environment.training.model_save_dir
    cfg.training.multi_gpu = cfg.environment.training.multi_gpu
    
    # Environment runtime context (top-level only)
    cfg.device = cfg.environment.device
    
    OmegaConf.set_struct(cfg, True)
    return cfg


def setup_output_directory(cfg: DictConfig, run_name: str) -> str:
    """
    Create output directory structure for a new run.
    
    Creates:
        - {output_root}/{run_name}/
        - {output_root}/{run_name}/tensorboard/
        - {output_root}/{run_name}/{model_save_dir}/
    
    Args:
        cfg: Config with environment.training settings
        run_name: Name of the run (used as subdirectory)
    
    Returns:
        Path to run output directory (with trailing slash)
    """
    # Normalize output_root to ensure trailing slash (defensive against user input)
    output_root = cfg.environment.training.output_root.rstrip('/') + '/'
    run_output_dir = f"{output_root}{run_name}/"
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(f"{run_output_dir}tensorboard/", exist_ok=True)
    os.makedirs(f"{run_output_dir}{cfg.environment.training.model_save_dir}", exist_ok=True)
    return run_output_dir


def setup_logger(
    cfg: DictConfig, 
    run_dir: str, 
    mode: str = 'train',
    timestamp: Optional[str] = None
) -> Tuple['Logger', SummaryWriter]:
    """
    Initialize tensorboard writer and logger.
    
    Args:
        cfg: Config with logging settings
        run_dir: Path to run directory (with trailing slash)
        mode: Reserved for backwards-compatible callers.
        timestamp: Reserved for backwards-compatible callers.
    
    Returns:
        Tuple of (Logger instance, SummaryWriter instance)
    """
    from src.utils.logger import Logger
    
    main_process = is_main_process()
    _ = (mode, timestamp)
    log_dir = f"{run_dir}tensorboard/"
    writer = SummaryWriter(log_dir=log_dir) if main_process else None
    
    enabled_outputs = list(cfg.logging.outputs) if main_process else []
    logger = Logger(
        log_dir=log_dir,
        enabled_outputs=enabled_outputs,
        log_interval=int(cfg.logging.interval),
        table_format=cfg.logging.table_format,
        writer=writer,
        cfg=cfg.logging,  # Pass logging config
    )
    if main_process:
        logger.print_config(OmegaConf.to_yaml(cfg, resolve=True))
    
    return logger, writer


def build_model_and_diffusion(cfg: DictConfig, device: torch.device):
    """
    Build model and diffusion with strategy-aware wrapping.
    
    - DDP: wraps trainable model with DistributedDataParallel using cfg.distribution
    - DP: wraps trainable model with DataParallel using environment.training.multi_gpu
    - single process: no wrapper

    Wrapping always happens before diffusion construction so adapters see the
    correct trainable module.
    
    Args:
        cfg: Config with model, diffusion, and distribution settings
        device: Target device for model
    
    Returns:
        Diffusion model with distribution-appropriate wrapping
    
    Raises:
        ValueError: If requested GPU id exceeds visible GPUs
    """
    from src.models import build_model
    from src.diffusion.diffusion import Diffusion

    strategy = resolve_strategy(cfg)

    # Keep model channels aligned with the active data contract before model build.
    sync_model_image_channels_with_data_contract(cfg)
    
    # Build model
    unet = build_model(cfg).to(device)

    # Strategy-aware model wrapping (exactly one path)
    unet, wrapper_name = _wrap_trainable_model_for_strategy(
        unet, cfg=cfg, device=device, strategy=strategy
    )
    
    # Build diffusion with potentially wrapped UNet
    diffusion = Diffusion.build_diffusion(unet, cfg, device)
    
    # Ensure diffusion is on correct device
    if wrapper_name == "none":
        diffusion = diffusion.to(device)
    else:
        # Wrapped models are already device-pinned appropriately.
        print(f"  Diffusion built with wrapper={wrapper_name}")

    _assert_ddp_wrapping_invariants(diffusion, strategy=strategy)
    
    return diffusion


def validate_training_runtime_contract(cfg: DictConfig) -> None:
    """
    Fail fast for unsupported training-runtime combinations.

    Current guardrails:
    - 3D + soft-STAPLE validation ensemble for every model family
    - 3D + diffusion (non-discriminative) + sampling snapshot logging
    - 3D + diffusion (non-discriminative) + validation ensemble
    - 3D + diffusion (non-discriminative) + ensembled image validation logging
    """
    dim_token = str(OmegaConf.select(cfg, "data_mode.dim", default="")).strip().lower()
    is_3d = dim_token in {"3", "3d"}
    diffusion_type = str(OmegaConf.select(cfg, "diffusion.type", default="")).strip()
    is_discriminative = diffusion_type.lower() == "discriminative"
    is_diffusion_3d = is_3d and not is_discriminative

    ensemble_enabled = bool(
        OmegaConf.select(cfg, "validation.ensemble.enabled", default=False)
    )
    ensemble_method = str(
        OmegaConf.select(cfg, "validation.ensemble.method", default="unknown")
    ).strip().lower()
    if is_3d and ensemble_enabled and ensemble_method == "soft_staple":
        raise ValueError(
            "Unsupported runtime combination: validation ensemble method "
            "'soft_staple' is 2D-only and cannot be enabled for data_mode.dim='3d'. "
            "Use validation.ensemble.method='mean' or disable validation ensembling."
        )

    if not is_diffusion_3d:
        return

    if bool(OmegaConf.select(cfg, "logging.enable_sampling_snapshots", default=False)):
        raise ValueError(
            "Unsupported runtime combination: data_mode.dim='3d' with "
            f"diffusion.type='{diffusion_type}' and logging.enable_sampling_snapshots=true. "
            "3D diffusion sampling snapshots are disabled for training-time logging."
        )

    if ensemble_enabled:
        raise ValueError(
            "Unsupported runtime combination: data_mode.dim='3d' with "
            f"diffusion.type='{diffusion_type}' and validation.ensemble.enabled=true "
            f"(method='{ensemble_method}'). "
            "3D diffusion ensemble/STAPLE validation in training loops is intentionally disabled."
        )

    if bool(OmegaConf.select(cfg, "validation.ensembled_image.enabled", default=False)):
        raise ValueError(
            "Unsupported runtime combination: data_mode.dim='3d' with "
            f"diffusion.type='{diffusion_type}' and validation.ensembled_image.enabled=true. "
            "3D diffusion ensembled-image validation logging is intentionally disabled."
        )


# =============================================================================
# High-Level Training Entry Points
# =============================================================================

def setup_and_start_training(cfg: DictConfig, run_dir: str):
    """
    Complete setup and training for a fresh run.
    
    Called by start_training.py after Hydra config loading.
    Handles all component initialization and runs training.
    
    Args:
        cfg: Hydra config (runtime context already applied)
        run_dir: Path to run output directory (with trailing slash)
    """
    import sys
    debug = cfg.get("debug", False)
    
    def _debug(msg):
        if debug:
            print(f"[DEBUG:setup_and_start_training] {msg}", flush=True)
            sys.stdout.flush()
    
    _debug("ENTER: setup_and_start_training()")
    
    # Ultra-granular import debugging
    _debug("Step A: About to import src.data.loaders...")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        from src.data.loaders import get_dataloaders
        _debug("Step A: src.data.loaders imported successfully")
    except Exception as e:
        _debug(f"Step A: FAILED to import src.data.loaders: {e}")
        raise
    
    _debug("Step B: About to import src.training.trainer...")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        from src.training.trainer import get_optimizer_and_scheduler, step_based_train
        _debug("Step B: src.training.trainer imported successfully")
    except Exception as e:
        _debug(f"Step B: FAILED to import src.training.trainer: {e}")
        raise
    
    _debug("Setting up device...")
    device = setup_device(cfg)
    _debug(f"Device: {device}")

    # Sync before logger prints resolved config for accurate channel visibility.
    sync_model_image_channels_with_data_contract(cfg)
    validate_training_runtime_contract(cfg)
    
    _debug("Setting up logger...")
    logger, writer = setup_logger(cfg, run_dir, mode='train')
    _debug("Logger done")
    
    _debug("Building model and diffusion...")
    diffusion = build_model_and_diffusion(cfg, device)
    _debug("Model and diffusion built")
    
    _debug("Getting dataloaders...")
    dataloaders = get_dataloaders(cfg)
    _debug(f"Dataloaders ready: {list(dataloaders.keys())}")
    
    # Contract validation: model channels must match active data contract.
    validate_model_channel_contract(cfg)
    
    _debug("Getting optimizer and scheduler...")
    optimizer, scheduler = get_optimizer_and_scheduler(cfg, diffusion)
    _debug("Optimizer and scheduler ready")
    
    # Run step-based training
    _debug("Starting step_based_train()...")
    step_based_train(
        cfg,
        diffusion,
        dataloaders,
        optimizer,
        scheduler,
        logger,
        run_dir=run_dir,
        resume_state=None  # Fresh run
    )
    _debug("step_based_train() complete")
    
    logger.close()
    _debug("EXIT: setup_and_start_training()")


def setup_and_resume_training(cfg: DictConfig, run_dir: str, resume_step: str = 'latest', legacy_mode: bool = False):
    """
    Complete setup and training for a resumed run.
    
    Called by resume_training.py after Compose API config loading.
    Loads checkpoint and restores all training state before continuing.
    
    Args:
        cfg: Hydra config (runtime context already applied)
        run_dir: Path to run directory to resume from (with trailing slash)
        resume_step: Step to resume from (int as string or 'latest')
        legacy_mode: If True, this is a legacy run without .hydra/ or training_state.
                     EMA will be initialized fresh, best metrics reset, scheduler reconstructed.
    """
    from src.data.loaders import get_dataloaders
    from src.training.trainer import get_optimizer_and_scheduler, step_based_train
    from src.training.checkpoint_utils import load_checkpoint, load_model_state_dict_compat
    
    device = setup_device(cfg)
    # Sync before logger prints resolved config for accurate channel visibility.
    sync_model_image_channels_with_data_contract(cfg)
    validate_training_runtime_contract(cfg)
    
    # Load checkpoint first to get step info
    resume_state = load_checkpoint(run_dir, resume_step, cfg, device)
    print(f"✓ Loaded checkpoint from step {resume_state['global_step']}")
    
    logger, writer = setup_logger(cfg, run_dir, mode='train')
    diffusion = build_model_and_diffusion(cfg, device)
    
    # Load model weights
    missing_keys, unexpected_keys = load_model_state_dict_compat(
        diffusion, resume_state['model_state_dict']
    )
    if missing_keys or unexpected_keys:
        print(
            f"  ├─ Model weights loaded with compatibility mode "
            f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
        )
    else:
        print(f"  ├─ Model weights loaded")
    
    # Get dataloaders
    dataloaders = get_dataloaders(cfg)
    
    # Contract validation: model channels must match active data contract.
    validate_model_channel_contract(cfg)
    
    optimizer, scheduler = get_optimizer_and_scheduler(cfg, diffusion)
    
    # Load optimizer state (if available)
    if resume_state.get('optimizer_state_dict'):
        optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        print(f"  ├─ Optimizer state loaded")
    else:
        print(f"  ├─ Optimizer state not found (starting fresh)")
    
    # Handle training state (EMA, best metrics, scheduler)
    training_state = resume_state.get('training_state', {})
    
    if legacy_mode or not training_state:
        # Legacy mode: no training_state file, initialize everything fresh
        print(f"  ├─ [LEGACY] No training state found - initializing fresh:")
        print(f"  │   ├─ EMA: will initialize from current model weights")
        
        # Set best metric based on metric mode
        if cfg.training.checkpoint_best.enabled:
            metric_mode = cfg.training.checkpoint_best.get('metric_mode', 'max')
            if metric_mode == 'max':
                print(f"  │   ├─ Best metric: -inf (mode=max, first validation will be saved)")
            else:
                print(f"  │   ├─ Best metric: inf (mode=min, first validation will be saved)")
        
        # Reconstruct scheduler to resume step
        if scheduler:
            global_step = resume_state['global_step']
            sched_cfg = cfg.scheduler
            if sched_cfg.get('step_frequency', 'per_step') == 'per_step':
                print(f"  │   └─ Scheduler: fast-forwarding {global_step} steps...")
                for _ in range(global_step):
                    scheduler.step()
                print(f"  │       └─ Scheduler reconstructed to step {global_step}")
            else:
                print(f"  │   └─ Scheduler: per-interval mode, will adjust naturally")
        
        # Clear training_state so trainer initializes fresh
        resume_state['training_state'] = {}
    else:
        # Standard mode: load scheduler state if available
        if scheduler and training_state.get('scheduler_state_dict'):
            scheduler.load_state_dict(training_state['scheduler_state_dict'])
            print(f"  ├─ Scheduler state loaded")
    
    print(f"  └─ Resuming training from step {resume_state['global_step']}")
    
    # Run step-based training with resume state
    step_based_train(
        cfg,
        diffusion,
        dataloaders,
        optimizer,
        scheduler,
        logger,
        run_dir=run_dir,
        resume_state=resume_state
    )
    
    logger.close()
