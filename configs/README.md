# Configuration Guide

This directory contains the Hydra configurations for the repository's
configuration-driven medical-volume segmentation pipeline. The retained
configurations principally support the ISLES'26 DynUNet submission, its
nnU-Net baseline, and the ATLAS v2.1 experiments that informed the final
pipeline.

## Configuration composition

Top-level configurations compose reusable groups from the subdirectories:

- `environment/`: dataset, output, and machine-specific paths
- `dataset/`: split definitions, modalities, and preprocessing
- `data_profile/`: dataset, loading mode, and data-I/O composition
- `data_runtime/`: batch sizes, workers, caching, and memory settings
- `model/`: network architecture
- `augmentation/`: spatial and intensity transformations
- `loss/`: objective functions and supervision policy
- `optimizer/` and `scheduler/`: optimization settings
- `training/`: training duration, precision, and checkpointing
- `validation/` and `inference/`: prediction and evaluation settings
- `nnunet/`: nnU-Net conversion and evaluation settings

Top-level files such as `cluster.yaml` and `local.yaml` provide general
composition bases. More specific top-level configurations override these
defaults for a dataset, model, and experimental setting. Hydra command-line
overrides take precedence over all composed values.

The `diffusion/` group remains part of the shared configuration interface.
ISLES'26 segmentation selects `diffusion/discriminative.yaml`, which
disables diffusion and performs direct segmentation.

## Configure paths and hardware

Before training, create or adapt an environment configuration corresponding
to the intended execution environment.

The principal machine-specific values are:

- `environment.dataset.data_root`
- `environment.dataset.split_file`
- `environment.dataset.nnunet_root`, when required
- `environment.training.output_root`
- `environment.device`

The `data_runtime/` group controls batch sizes, worker counts, caching,
prefetching, and memory-related settings. Some top-level experiment
configurations override these values for the hardware on which the original
runs were performed. Adaptations for other systems should preserve the
experimental configuration while adjusting only the required runtime
parameters.

## ATLAS v2.1 development configurations

The ATLAS v2.1 configurations preserve the staged development work that
preceded the final three-fold training.

Principal top-level configurations include:

- `cluster_isles26_3d_randompatch_dynunet.yaml`
- `local_isles26_3d_randompatch_dynunet.yaml`
- `cluster_isles26_3d_randompatch_swinunetr.yaml`
- `local_isles26_3d_randompatch_swinunetr.yaml`
- `cluster_isles26_3d_randompatch_dynunet_params_26062026.yaml`

The DynUNet and SwinUNETR configurations support the architecture
comparison. The parameter-sweep configuration records the DynUNet capacity
and topology experiments.

The associated configuration groups retain the alternatives evaluated
during staged development, including:

- raw and normalized T1 representations in `data_profile/` and `dataset/`
- spatial and intensity augmentation policies in `augmentation/`
- Dice, BCE, focal, Tversky, generalized-Dice, and Hausdorff-distance
  objective variants in `loss/`
- DynUNet and SwinUNETR architectures in `model/`
- training, optimizer, scheduler, validation, and inference variants

Individual staged experiments are reconstructed by composing the relevant
top-level configuration with the corresponding group selections and
parameter overrides.

The standard ATLAS v2.1 environments are:

- `environment/isles26_cluster.yaml`
- `environment/isles26_local.yaml`

These configurations refer to the ATLAS v2.1 development split and must be
adapted to the local dataset and output paths.

## ATLAS v3.0 configurations

The initial ATLAS v3.0 configuration extends the ATLAS v2.1 pipeline to the
updated training data:

- `cluster_isles26_atlas30_3d_randompatch_dynunet.yaml`

Its associated configuration components include:

- `environment/isles26_atlas30_cluster.yaml`
- `environment/isles26_atlas30_local.yaml`
- `dataset/isles26_atlas30_modalities_t1raw.yaml`
- `data_profile/isles26_atlas30_3d_fullvol_t1raw.yaml`

These files represent the ATLAS v3.0 setup before construction of the final
SOOP-inclusive three-fold split.

## ATLAS v3.0 and SOOP three-fold configurations

The final ISLES'26 training setup uses the combined ATLAS v3.0 and SOOP
cases with a site-aware, metadata-stratified three-fold split.

The supplied split manifest is:

`isles26_split/isles26_soopincluded_3fold_2026_08_09.json`

The environment configurations are:

- `environment/isles26_atlas30_soopincluded_3fold_cluster.yaml`
- `environment/isles26_atlas30_soopincluded_3fold_local.yaml`

The fold-specific dataset configurations are:

- `dataset/isles26_atlas30_soopincluded_3fold_fold1_t1raw.yaml`
- `dataset/isles26_atlas30_soopincluded_3fold_fold2_t1raw.yaml`
- `dataset/isles26_atlas30_soopincluded_3fold_fold3_t1raw.yaml`

Human-readable folds 1, 2, and 3 correspond to manifest fold values 0, 1,
and 2, respectively. Each fold configuration assigns its selected fold to
validation and the remaining folds to training.

Local and cluster top-level training configurations are supplied for every
fold:

- `cluster_isles26_atlas30_soopincluded_3fold_fold[1-3]_3d_randompatch_dynunet.yaml`
- `local_isles26_atlas30_soopincluded_3fold_fold[1-3]_3d_randompatch_dynunet.yaml`

These configurations establish the fold, data representation, preprocessing,
model family, training schedule, validation protocol, and machine profile.
The final submitted DynUNet runs pair them with spatial-only augmentation,
Dice–focal final-head supervision, and feature widths of
`[32, 64, 128, 256]`.

Although the DynUNet exposes auxiliary deep-supervision outputs, the selected
Dice–focal configuration supervises only the final prediction head.

## nnU-Net baseline configurations

The `nnunet/convert/` directory contains local and cluster conversion
configurations for the ATLAS v2.1, ATLAS v3.0, and SOOP-inclusive
three-fold data arrangements.

The final fold-specific conversion configurations follow the naming pattern:

- `isles26_atlas30_soopincluded_3fold_fold[1-3]_cluster_3d_t1raw.yaml`
- `isles26_atlas30_soopincluded_3fold_fold[1-3]_local_3d_t1raw.yaml`

They compose the same raw T1 inputs and fold assignments as the corresponding
DynUNet configurations while supplying the paths and metadata required by
nnU-Net.

## Configuration scope

A small number of legacy defaults remain because `cluster.yaml` and
`local.yaml` still reference them. They support the general configuration
base but do not define the ISLES'26 experiments. For reproduction, select
one of the dataset-specific ATLAS v2.1 or ATLAS v3.0 top-level
configurations rather than either base configuration alone.
