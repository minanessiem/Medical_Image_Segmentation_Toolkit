"""
Ensemble utilities for diffusion model validation.

This module provides methods for combining multiple stochastic samples from
diffusion models into stable, reproducible predictions. Ensemble methods
reduce the variance in validation metrics caused by the inherent stochasticity
of diffusion sampling.

Supported Methods:
    - mean: Simple averaging of probability maps
    - soft_staple: Iterative weighted consensus (soft STAPLE algorithm)
"""

import torch
from torch import Tensor
from omegaconf import DictConfig


def mean_ensemble(samples: Tensor) -> Tensor:
    """
    Combine samples using simple mean averaging.
    
    This is the most straightforward ensemble method. Each sample contributes
    equally to the final prediction. Effective when samples have similar quality.
    
    Args:
        samples: Tensor of shape [N, B, C, *spatial] where N is the number of
                 samples. Values should be probabilities in [0, 1].
    
    Returns:
        Tensor of shape [B, C, *spatial] with averaged predictions.
    
    Example:
        >>> samples = torch.rand(5, 2, 1, 64, 64)  # 5 samples, batch 2
        >>> result = mean_ensemble(samples)
        >>> result.shape
        torch.Size([2, 1, 64, 64])
    """
    # Mean across the sample dimension (dim=0)
    return samples.mean(dim=0)


def soft_staple(
    samples: Tensor,
    max_iters: int = 5,
    tolerance: float = 0.02
) -> Tensor:
    """
    Combine samples using Soft STAPLE (Simultaneous Truth and Performance 
    Level Estimation).
    
    STAPLE is an expectation-maximization algorithm that estimates the "true"
    segmentation by modeling each sample as coming from an expert with unknown
    sensitivity and specificity. Samples that agree with the emerging consensus
    are weighted more heavily.
    
    This implementation uses a soft (probabilistic) variant that works with
    probability maps rather than binary masks, making it differentiable and
    suitable for continuous predictions.
    
    Args:
        samples: Tensor of shape [N, B, C, H, W] where N is number of samples.
                 Values should be probabilities in [0, 1].
        max_iters: Maximum number of EM iterations. Default 5.
        tolerance: Convergence threshold. Stop early if max change in consensus
                   is below this value. Default 0.02.
    
    Returns:
        Tensor of shape [B, C, H, W] with weighted consensus predictions.
    
    References:
        Warfield, S.K., Zou, K.H., & Wells, W.M. (2004). Simultaneous Truth and 
        Performance Level Estimation (STAPLE): An Algorithm for the Validation 
        of Image Segmentation.
    
    Example:
        >>> samples = torch.rand(5, 2, 1, 64, 64)
        >>> result = soft_staple(samples, max_iters=5, tolerance=0.02)
        >>> result.shape
        torch.Size([2, 1, 64, 64])
    """
    n_samples = samples.shape[0]
    eps = 1e-7
    
    # Initialize consensus as simple mean
    consensus = samples.mean(dim=0)  # [B, C, H, W]
    
    # Early exit for edge cases: if all samples nearly agree, return mean
    # This handles cases where all samples are ~0 or ~1
    sample_variance = samples.var(dim=0).max().item()
    if sample_variance < eps:
        # All samples are essentially identical, return the mean
        return consensus
    
    # Clamp consensus to avoid extreme values that cause numerical issues
    consensus = consensus.clamp(eps, 1 - eps)
    
    for _ in range(max_iters):
        prev_consensus = consensus.clone()
        
        # E-step: Estimate expert performance parameters
        # Sensitivity: P(expert says 1 | true = 1)
        # Specificity: P(expert says 0 | true = 0)
        
        # Clamp consensus for stable computation
        consensus_clamped = consensus.clamp(eps, 1 - eps)
        
        # Compute per-expert weights
        weights = []
        for j in range(n_samples):
            sample_j = samples[j].clamp(eps, 1 - eps)  # [B, C, H, W]
            
            # Sensitivity: how well expert j detects positives
            sensitivity_num = (sample_j * consensus_clamped).sum(dim=(-2, -1), keepdim=True)
            sensitivity_den = consensus_clamped.sum(dim=(-2, -1), keepdim=True) + eps
            sensitivity = sensitivity_num / sensitivity_den
            
            # Specificity: how well expert j detects negatives
            specificity_num = ((1 - sample_j) * (1 - consensus_clamped)).sum(dim=(-2, -1), keepdim=True)
            specificity_den = (1 - consensus_clamped).sum(dim=(-2, -1), keepdim=True) + eps
            specificity = specificity_num / specificity_den
            
            # Clamp to valid probability range (avoid log(0) or log(inf))
            sensitivity = sensitivity.clamp(eps, 1 - eps)
            specificity = specificity.clamp(eps, 1 - eps)
            
            # Log-odds contribution from this expert
            # pos_weight = log(P(D=1|T=1) / P(D=1|T=0)) = log(sens / (1-spec))
            # neg_weight = log(P(D=0|T=0) / P(D=0|T=1)) = log(spec / (1-sens))
            pos_weight = torch.log(sensitivity) - torch.log(1 - specificity)
            neg_weight = torch.log(specificity) - torch.log(1 - sensitivity)
            
            # Contribution based on expert's prediction
            expert_logit = sample_j * pos_weight + (1 - sample_j) * neg_weight
            weights.append(expert_logit)
        
        # M-step: Update consensus
        # Stack and sum log-odds, then apply sigmoid
        stacked_weights = torch.stack(weights, dim=0)  # [N, B, C, H, W]
        total_logits = stacked_weights.sum(dim=0)  # [B, C, H, W]
        
        # Convert back to probability
        consensus = torch.sigmoid(total_logits)
        
        # Check convergence
        max_change = (consensus - prev_consensus).abs().max().item()
        if max_change < tolerance:
            break
    
    return consensus


def ensemble_predictions(samples: Tensor, ensemble_cfg: DictConfig) -> Tensor:
    """
    Dispatch to appropriate ensemble method based on configuration.
    
    Args:
        samples: Tensor of shape [N, B, C, H, W] where N is number of samples.
        ensemble_cfg: Configuration dict with 'method' and method-specific params.
                      Expected structure:
                          method: 'mean' | 'soft_staple'
                          soft_staple:
                              max_iters: int
                              tolerance: float
    
    Returns:
        Tensor of shape [B, C, H, W] with ensembled predictions.
    
    Raises:
        ValueError: If method is not recognized.
    
    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({'method': 'mean'})
        >>> samples = torch.rand(5, 2, 1, 64, 64)
        >>> result = ensemble_predictions(samples, cfg)
        >>> result.shape
        torch.Size([2, 1, 64, 64])
    """
    method = ensemble_cfg.method
    
    if method == 'mean':
        return mean_ensemble(samples)
    elif method == 'soft_staple':
        if samples.ndim == 6:
            raise ValueError(
                "validation ensemble method 'soft_staple' is 2D-only and cannot "
                "combine 3D probability tensors [N,B,C,D,H,W]. Use method='mean' "
                "or keep 3D validation ensembling disabled."
            )
        staple_cfg = ensemble_cfg.soft_staple
        return soft_staple(
            samples,
            max_iters=staple_cfg.max_iters,
            tolerance=staple_cfg.tolerance
        )
    else:
        raise ValueError(
            f"Unknown ensemble method: '{method}'. "
            f"Supported methods: 'mean', 'soft_staple'"
        )


def should_ensemble(cfg: DictConfig) -> bool:
    """
    Check if ensemble validation is configured and enabled.
    
    This function uses explicit attribute checks (hasattr) rather than
    .get() with defaults. Missing config sections mean the feature is
    disabled, following the fail-safe design principle.
    
    Args:
        cfg: Full Hydra configuration object.
    
    Returns:
        True if ensemble is configured and enabled, False otherwise.
    
    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({'validation': {'ensemble': {'enabled': True}}})
        >>> should_ensemble(cfg)
        True
        >>> cfg = OmegaConf.create({})
        >>> should_ensemble(cfg)
        False
    """
    # Check if validation section exists
    if not hasattr(cfg, 'validation'):
        return False
    
    validation = cfg.validation
    
    # Check if ensemble section exists
    if not hasattr(validation, 'ensemble'):
        return False
    
    ensemble = validation.ensemble
    
    # Check if enabled flag exists and is True
    if not hasattr(ensemble, 'enabled'):
        return False
    
    return bool(ensemble.enabled)


def should_log_ensembled_image(cfg: DictConfig, global_step: int) -> bool:
    """
    Check if ensembled segmentation image logging is due at this step.
    
    Checks both the configuration and the step interval. Returns False if:
    - validation.ensembled_image section is missing
    - enabled flag is False or missing
    - Current step doesn't match the logging interval
    - Step is 0 (avoid logging before training starts)
    
    Args:
        cfg: Full Hydra configuration object.
        global_step: Current training step.
    
    Returns:
        True if ensembled image should be logged at this step, False otherwise.
    
    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({
        ...     'validation': {
        ...         'ensembled_image': {'enabled': True, 'interval': 5000}
        ...     }
        ... })
        >>> should_log_ensembled_image(cfg, 5000)
        True
        >>> should_log_ensembled_image(cfg, 3000)
        False
    """
    # Never log at step 0
    if global_step == 0:
        return False
    
    # Check if validation section exists
    if not hasattr(cfg, 'validation'):
        return False
    
    validation = cfg.validation
    
    # Check if ensembled_image section exists
    if not hasattr(validation, 'ensembled_image'):
        return False
    
    ensembled_image = validation.ensembled_image
    
    # Check if enabled flag exists and is True
    if not hasattr(ensembled_image, 'enabled'):
        return False
    
    if not ensembled_image.enabled:
        return False
    
    # Check if interval is configured
    if not hasattr(ensembled_image, 'interval'):
        return False
    
    interval = ensembled_image.interval
    
    # Check if current step matches interval
    return global_step % interval == 0
