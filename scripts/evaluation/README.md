# Evaluation Package

Unified evaluation package for:
- slice-level 2D metric aggregation,
- volume-level 3D metric aggregation,
- optional reconstructed volume NIfTI export for QA.

The runtime is streaming-first: slices are consumed in order, and volume metrics
are updated when a volume boundary is reached (no full-dataset volume buffering
required for metric computation).

This package is the supported evaluation path for campaign metrics. Training
entrypoints such as `main.py` and `start_training.py` do not provide an
evaluation mode.

## Entry points

- Repository-trained model, config-driven live inference:
  - `python3 -m scripts.evaluation.evaluate_model ...`
- Legacy 2D diffusion/discriminative model:
  - `python3 -m scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions ...`
- nnU-Net post-threshold predictions:
  - `python3 -m scripts.evaluation.compute_segmentation_metrics_for_nnunet_2d_predictions ...`

`evaluate_model` is the current local entrypoint for repository checkpoints. It
loads `<RUN_DIR>/.hydra/config.yaml`, merges an evaluation preset from
`configs/evaluation/`, applies CLI `key=value` overrides, discovers normalized
validation records without constructing training DataLoaders, preprocesses
each record through one reusable typed case producer, and writes JSON/CSV/text
artifacts.

The canonical `evaluate_model` path supports geometry-aware 3D discriminative
volume evaluation. Every prediction/reference pair must declare the same
`model_preprocessed` or `native_input` space and matching shape, affine,
spacing, and orientation before metrics run. One `LabeledPreprocessedCase`
loop handles both spaces. It pairs model-space results with the jointly
transformed label and native-space results with the untouched native label; the
shared inference pipeline alone performs any spatial restoration.

Geometry-aware 3D evaluation requires `validation.val_batch_size=1` in both
spaces because it processes one complete case and its geometry at a time. This
does not limit a job to one case: the full selected split is processed
sequentially. `inference.sliding_window.sw_batch_size` is independent and
controls how many windows from the current case are predicted together.

Repository and nnU-Net 2D slice paths do not currently carry enough typed
parent-volume geometry and reconstruction information to make a truthful 3D
spatial claim. They therefore fail when a volume would be assembled. The
deferred 2D task must define slice placement and invertible preprocessing before
those paths can re-enter geometry-aware volume evaluation. Current 3D
non-discriminative diffusion is likewise rejected with a capability error;
future support must register a compatible `ProbabilityPredictor`.

## Package layout

The package root intentionally contains only user-facing entrypoints and package
metadata. Implementation modules live in focused subpackages:

```text
scripts/evaluation/
├── evaluate_model.py
├── compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py
├── compute_segmentation_metrics_for_nnunet_2d_predictions.py
├── core/
│   ├── contracts.py
│   ├── evaluation_pipeline.py
│   ├── model_config.py
│   └── model_loader.py
├── io/
│   ├── model_slices.py
│   ├── model_volumes.py
│   ├── nnunet.py
│   ├── mask_builder.py
│   ├── provenance.py
│   ├── volume_assembler.py
│   └── volume_exporter.py
├── metrics/
│   ├── engine.py
│   ├── registry_2d.py
│   └── registry_3d.py
├── reporting/
│   ├── reports.py
│   ├── threshold_protocol.py
│   ├── threshold_records.py
│   └── threshold_selection.py
└── slurm_runners/
    ├── run_evaluate_model.py
    ├── run_compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py
    └── run_compute_segmentation_metrics_for_nnunet_2d_predictions.py
```

## Key flags

### Shared volume-export flags

- `--export-reconstructed-volumes`
  - Writes reconstructed 3D NIfTI per case (`pred` and `gt`).
- `--max-export-volumes-per-case`
  - Optional cap for exported reconstructed volumes.

### Config-driven repository-model flags

- `--evaluation-config-name default|fixed_threshold|threshold_sweep|threshold_sweep_with_oracle`
- Required overrides for normal use:
  - `evaluation.run_dir=<RUN_DIR>`
  - `evaluation.model_name=<MODEL_NAME>`
- Common overrides:
  - `dataset.active_subsets.val=val_fast|val_full`
  - `validation=sliding_window_3d_metrics_subset|sliding_window_3d_metrics_full` for 3D volume analysis
  - `inference=sliding_window_model_space` to select a complete shared prediction policy
  - `inference_runtime=native` for ordinary Python execution outside a container
  - `evaluation.levels=[volume]`
  - `evaluation.threshold_protocol.mode=fixed|sweep|oracle_per_case|sweep_with_oracle`
  - `evaluation.threshold_protocol.primary.metric=DiceNativeCoefficient|...`
  - `evaluation.output_dir=<OUTPUT_DIR>`
  - `evaluation.device=cpu|cuda|cuda:0`

### Config-driven repository-model SLURM runner flags

`run_evaluate_model.py` follows the same SLURM submission pattern as the older
evaluation runners and `scripts/slurm/single_job_runner.py`: it merges the
plain `BASE_CONFIG` SLURM dictionary with CLI overrides on the submission side,
then submits a `python3 -m scripts.evaluation.evaluate_model ...` command. The
Hydra/OmegaConf evaluation config is composed inside the submitted container job.

- Convenience evaluation flags:
  - `--run-dir <RUN_DIR>`
  - `--model-name <CHECKPOINT_NAME>`
  - `--evaluation-config-name threshold_sweep_with_oracle`
  - `--validation-config sliding_window_3d_metrics_subset|sliding_window_3d_metrics_full`
  - `--val-subset val_fast|val_full`
  - `--val-batch-size 1`
  - `--override key=value ...` for additional forwarded evaluation overrides
- SLURM/resource flags:
  - `--gpus`, `--partition`, `--qos`, `--cpus-per-task`, `--mem`, `--time`
  - `--container-image`, `--host-outputs-dir`, and other `BASE_CONFIG` fields
    exposed by `scripts.slurm.utils.commandline_utils.add_config_arguments`

### Legacy 2D diffusion/custom flags

- Threshold protocol:
  - `--thresholds 0.05:0.95:0.05` (sweep mode)
  - `--fixed-threshold 0.5` (fixed mode)
  - `--optimize-metric dice`
- Ensemble:
  - `--ensemble-samples 1` or `--ensemble-samples 1,3,5`
  - `--ensemble-method single|mean|soft_staple|both`
  - `--staple-max-iters`
  - `--staple-tolerance`
- Quick test:
  - `--test --test-max-slices 10`

### nnU-Net flags

- `--fixed-threshold 0.5`
- `--allow-shape-mismatch`

## Outputs

For config-driven repository-model evaluation, outputs include:

- `canonical_results.json`
- `evaluation_summary.txt`
- `resolved_evaluation_config.yaml`
- `volume_metrics_per_threshold.csv` for volume-level analysis
- `per_case_threshold_metrics.csv`
- `oracle_per_case_thresholds.csv` when oracle mode is enabled

`canonical_results.json` also records producer type, selected checkpoint,
runtime profile, winning inference-policy source, threshold protocol, and each
case's verified prediction/reference space and geometry. The resolved
evaluation config contains the complete top-level `inference` and
`inference_runtime` policies. `evaluation.input_source` selects the producer; it
does not select the runtime profile.

For an explicitly geometry-declared 3D `VolumeSample`, NIfTI export writes:
- `reconstructed_volumes/<analysis_case_key>/<volume_id>__pred.nii.gz`
- `reconstructed_volumes/<analysis_case_key>/<volume_id>__gt.nii.gz`

## Example commands

### Repository-trained model, config-driven (local)

Evaluate the saved run exactly as configured, at the default fixed threshold:

```bash
python3 -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=/mnt/outputs/<run_dir> \
  evaluation.model_name=<checkpoint_name_without_pth>
```

Run a 3D validation sweep with per-case oracle analysis:

```bash
python3 -m scripts.evaluation.evaluate_model \
  --evaluation-config-name threshold_sweep_with_oracle \
  evaluation.run_dir=/mnt/outputs/<run_dir> \
  evaluation.model_name=<checkpoint_name_without_pth> \
  dataset.active_subsets.val=val_full \
  validation=sliding_window_3d_metrics_full \
  validation.val_batch_size=1
```

The former 2D `evaluate_model` example is intentionally omitted: 2D inputs now
raise the deferred reconstruction-contract error before geometry-aware
assessment.

### Repository-trained model via SLURM runner

```bash
python3 -m scripts.evaluation.slurm_runners.run_evaluate_model \
  --run-dir /mnt/outputs/<run_dir> \
  --model-name <checkpoint_name_without_pth> \
  --evaluation-config-name threshold_sweep_with_oracle \
  --validation-config sliding_window_3d_metrics_full \
  --val-subset val_full \
  --val-batch-size 1 \
  --gpus 1 \
  --cpus-per-task 32 \
  --mem 96G \
  --time 06:00:00
```

For smoke runs, use `--validation-config sliding_window_3d_metrics_subset` and
`--val-subset val_fast`. Add final evaluation overrides with `--override`, for
example `--override evaluation.threshold_protocol.thresholds=0.05:0.90:0.05`.

### Legacy 2D diffusion/custom (local)

```bash
python3 -m scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions \
  --run-dir /mnt/outputs/<run_dir> \
  --model-name <checkpoint_name_without_pth> \
  --thresholds 0.05:0.95:0.05 \
  --optimize-metric dice \
  --ensemble-samples 3 \
  --ensemble-method both \
  --export-reconstructed-volumes \
  --max-export-volumes-per-case 10
```

### nnU-Net (local)

```bash
python3 -m scripts.evaluation.compute_segmentation_metrics_for_nnunet_2d_predictions \
  --pred-dir /mnt/outputs/nnunet/preds \
  --gt-dir /mnt/outputs/nnunet/labels \
  --fixed-threshold 0.5 \
  --export-reconstructed-volumes
```

### Diffusion/custom via SLURM runner

```bash
python3 -m scripts.evaluation.slurm_runners.run_compute_segmentation_metrics_for_diffusionmodel_2d_predictions \
  --run-dir /mnt/outputs/<run_dir> \
  --model-name <checkpoint_name_without_pth> \
  --thresholds 0.05:0.95:0.05 \
  --optimize-metric dice \
  --ensemble-samples 3 \
  --ensemble-method both \
  --export-reconstructed-volumes \
  --max-export-volumes-per-case 10
```

### nnU-Net via SLURM runner

```bash
python3 -m scripts.evaluation.slurm_runners.run_compute_segmentation_metrics_for_nnunet_2d_predictions \
  --pred-dir /mnt/outputs/nnunet/preds \
  --gt-dir /mnt/outputs/nnunet/labels \
  --fixed-threshold 0.5 \
  --export-reconstructed-volumes
```
