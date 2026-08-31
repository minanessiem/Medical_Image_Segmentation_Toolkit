# Medical Image Segmentation Toolkit

Configuration-driven training, evaluation, inference, and Grand Challenge
packaging for 3D medical-image segmentation.

> This README documents the `isles26_submission_branch`, which preserves the
> code, configurations, split manifest, and packaging workflow for our
> ISLES'26 submission. Reproduction work should start from this branch rather
> than `main`.

This repository was developed around our ISLES'26 chronic stroke lesion
segmentation submission. It also provides reusable components for supervised
segmentation of volumetric medical images, including dataset adapters, MONAI
architectures, modular objectives, geometry-aware evaluation, native-space
inference, and independently packaged Grand Challenge model and container
artifacts.

## Clone and install

Clone the ISLES'26 submission branch directly:

```bash
git clone \
  --branch isles26_submission_branch \
  --single-branch \
  https://github.com/minanessiem/Medical_Image_Segmentation_Toolkit.git

cd Medical_Image_Segmentation_Toolkit
```

Create and activate a suitable Python environment, then install the project
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Training requires a compatible CUDA-enabled PyTorch installation. The nnU-Net
baseline additionally requires the nnU-Net v2 command-line tools. Grand
Challenge image construction and lifecycle testing require Docker, the NVIDIA
Container Toolkit, Linux or WSL2, and an available NVIDIA GPU.

## ISLES'26 approach

Our submitted pipeline segments chronic ischaemic stroke lesions from
single-channel T1-weighted MRI volumes.

The final DynUNet configuration comprises:

- raw-intensity T1 input;
- reorientation to RAS and resampling to 1 mm isotropic spacing;
- balanced sampling of $128^3$-voxel patches;
- random 3D flips and rotations without intensity augmentation;
- a four-stage MONAI DynUNet with feature widths `[32, 64, 128, 256]`;
- instance normalization, residual blocks, and leaky-ReLU activations;
- Dice–focal optimization on the final prediction head;
- AdamW with a learning rate of `2e-4`, 10% linear warmup, and cosine annealing;
- 100,000 optimization steps with bfloat16 mixed precision;
- whole-volume sliding-window inference with 50% overlap and Gaussian blending.

The competition model is an equal-weight probability ensemble of three
DynUNets trained on complementary folds of the ATLAS v3.0/SOOP split.
Predictions are restored to the native input geometry and thresholded at
`0.5`. The submitted configuration applies neither test-time augmentation nor
connected-component filtering.

Although DynUNet exposes auxiliary deep-supervision outputs during training,
the selected Dice–focal objective supervises only the final prediction head.

## Experimental development

The repository preserves the staged experiments that informed the final
pipeline.

### ATLAS v2.1

The ATLAS v2.1 development experiments examine:

- DynUNet and SwinUNETR architectures;
- raw and normalized T1 representations;
- spatial and intensity augmentation policies;
- Dice-, BCE-, focal-, Tversky-, generalized-Dice-, and Hausdorff-based objectives;
- patch size and foreground-sampling ratios;
- DynUNet depth, width, and kernel configurations;
- output-threshold selection.

The principal ATLAS v2.1 entry points are:

```text
configs/cluster_isles26_3d_randompatch_dynunet.yaml
configs/cluster_isles26_3d_randompatch_swinunetr.yaml
configs/cluster_isles26_3d_randompatch_dynunet_params_26062026.yaml
```

Individual experiments are reconstructed through Hydra group selections and
parameter overrides. See [`configs/README.md`](configs/README.md) for the
retained experimental configuration map.

### ATLAS v3.0 and SOOP

The final development stage applies the selected pipeline to a site-aware,
metadata-stratified three-fold split containing 1,453 cases from 55 sites.

The split manifest is provided at:

```text
isles26_split/isles26_soopincluded_3fold_2026_08_09.json
```

The repository does not redistribute medical images, masks, or source
metadata. Access to the underlying data remains subject to the ATLAS and ISLES
challenge terms.

Human-readable folds 1, 2, and 3 correspond to manifest fold values 0, 1, and
2, respectively.

## Validation results

The staged ATLAS v2.1 experiments comprise single training runs and should be
interpreted as empirical model-selection results.

| Dataset and protocol | Model | Dice | Surface Dice | HD95 (mm) |
|---|---|---:|---:|---:|
| ATLAS v2.1 development split | nnU-Net | 0.601 | 0.543 | 26.58 |
| ATLAS v2.1 development split | Best DynUNet run | 0.633 | 0.595 | 23.31 |
| ATLAS v3.0/SOOP three-fold mean | nnU-Net | 0.628 | 0.574 | 33.40 |
| ATLAS v3.0/SOOP three-fold mean | DynUNet | 0.610 | 0.561 | 33.02 |

The three DynUNet fold models form the competition ensemble. Out-of-sample
ensemble performance is determined by the hidden ISLES'26 test set and is not
inferred from the fold-wise validation scores.

## Repository structure

```text
configs/                       Hydra experiment and runtime configurations
isles26_split/                 Reproducible ATLAS v3.0/SOOP fold manifest
src/
  data/                        Dataset adapters and preprocessing
  inference/                   Sliding-window and native-space inference
  losses/                      Configurable segmentation objectives
  metrics/                     Segmentation metrics
  models/                      DynUNet, SwinUNETR, and supporting models
  training/                    Training and checkpoint infrastructure
scripts/
  evaluation/                  Geometry-aware model evaluation
  nnunet/                      nnU-Net conversion and evaluation
  gc_submission_builder/      Grand Challenge model and image packaging
  slurm/                       Optional cluster job wrappers
start_training.py              Fresh training entry point
resume_training.py             Checkpoint-resumption entry point
```

## Environment and data configuration

Before training, adapt or create an environment configuration under
`configs/environment/`. At minimum, configure:

```yaml
dataset:
  data_root: /absolute/path/to/the/dataset/root
  split_file: /absolute/path/to/isles26_soopincluded_3fold_2026_08_09.json
  nnunet_root: /absolute/path/to/nnUNet_raw

training:
  output_root: /absolute/path/to/training/outputs

device: cuda
```

Runtime settings such as batch size, worker counts, caching, and prefetching
are defined under `configs/data_runtime/` and may also be overridden through
Hydra.

The supplied split manifest contains relative image, mask, and metadata paths.
`environment.dataset.data_root` must therefore identify the directory against
which those paths resolve.

## Reproduce the ATLAS v3.0/SOOP DynUNet runs

Run all commands from the repository root after configuring dataset and output
paths.

### Fold 1

```bash
python3 start_training.py \
  --config-name cluster_isles26_atlas30_soopincluded_3fold_fold1_3d_randompatch_dynunet \
  loss=discriminative_dicefocal_deepsupervision \
  augmentation=spatial_only_3d \
  'model.filters=[32,64,128,256]'
```

### Fold 2

```bash
python3 start_training.py \
  --config-name cluster_isles26_atlas30_soopincluded_3fold_fold2_3d_randompatch_dynunet \
  loss=discriminative_dicefocal_deepsupervision \
  augmentation=spatial_only_3d \
  'model.filters=[32,64,128,256]'
```

### Fold 3

```bash
python3 start_training.py \
  --config-name cluster_isles26_atlas30_soopincluded_3fold_fold3_3d_randompatch_dynunet \
  loss=discriminative_dicefocal_deepsupervision \
  augmentation=spatial_only_3d \
  'model.filters=[32,64,128,256]'
```

These are direct Python training commands. The SLURM wrappers under
`scripts/slurm/` are optional and are not required to instantiate the runs.

The `local_...` counterparts of the three configurations provide
local-development profiles. Hardware-dependent settings may be overridden
without changing the dataset, model, loss, or fold selection.

## Evaluate a trained DynUNet

Each run stores its resolved Hydra configuration under `.hydra/config.yaml`.
The evaluation entry point reconstructs the model and preprocessing contract
from that saved configuration.

```bash
python3 -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=/absolute/path/to/training-run \
  evaluation.model_name=<checkpoint-name-without-.pth> \
  dataset.active_subsets.val=val_full \
  validation=sliding_window_3d_metrics_full \
  validation.val_batch_size=1
```

Evaluation reports subject-wise 3D metrics and verifies that predictions and
references occupy the same declared spatial geometry. See
[`scripts/evaluation/README.md`](scripts/evaluation/README.md) for threshold
studies, reconstructed-volume export, and additional evaluation modes.

## Reproduce the nnU-Net baseline

The matched nnU-Net v2 baseline uses one exported dataset per held-out fold.
Each dataset contains the two training folds in `imagesTr`/`labelsTr` and the
held-out fold in `imagesTs`/`labelsTs`.

| Human fold | Manifest fold | nnU-Net dataset ID | Dataset folder |
|---:|---:|---:|---|
| 1 | 0 | 266 | `Dataset266_isles26_atlas30_soopincluded_3fold_fold1_t1_raw` |
| 2 | 1 | 267 | `Dataset267_isles26_atlas30_soopincluded_3fold_fold2_t1_raw` |
| 3 | 2 | 268 | `Dataset268_isles26_atlas30_soopincluded_3fold_fold3_t1_raw` |

Configure the standard nnU-Net v2 paths before conversion, planning, training,
or prediction:

```bash
export nnUNet_raw=/absolute/path/to/nnUNet_raw
export nnUNet_preprocessed=/absolute/path/to/nnUNet_preprocessed
export nnUNet_results=/absolute/path/to/nnUNet_results
export nnUNet_compile=False
```

The same locations must be reflected in the selected repository environment
and nnU-Net configuration files.

### 1. Export all three datasets

```bash
python3 -m scripts.nnunet.convert_to_nnunet \
  --config-name=nnunet/convert/isles26_atlas30_soopincluded_3fold_fold1_cluster_3d_t1raw

python3 -m scripts.nnunet.convert_to_nnunet \
  --config-name=nnunet/convert/isles26_atlas30_soopincluded_3fold_fold2_cluster_3d_t1raw

python3 -m scripts.nnunet.convert_to_nnunet \
  --config-name=nnunet/convert/isles26_atlas30_soopincluded_3fold_fold3_cluster_3d_t1raw
```

The cluster presets perform full exports. Equivalent `..._local_3d_t1raw`
presets are available for local paths.

### 2. Plan and preprocess

```bash
nnUNetv2_plan_and_preprocess \
  -d 266 267 268 \
  -c 3d_fullres \
  --verify_dataset_integrity
```

If the installed nnU-Net version does not accept multiple dataset IDs in one
invocation, run the same command separately for `266`, `267`, and `268`.

### 3. Train the three baselines

```bash
nnUNetv2_train 266 3d_fullres all
nnUNetv2_train 267 3d_fullres all
nnUNetv2_train 268 3d_fullres all
```

Here, `all` is the literal nnU-Net fold identifier. Each exported dataset
already excludes its held-out human fold from `imagesTr`, so nnU-Net trains on
all cases present in that dataset. The three runs are independent and may be
scheduled in parallel.

### 4. Predict each held-out fold

```bash
export NNUNET_PRED_ROOT=/absolute/path/to/nnunet_predictions

nnUNetv2_predict \
  -i "$nnUNet_raw/Dataset266_isles26_atlas30_soopincluded_3fold_fold1_t1_raw/imagesTs" \
  -o "$NNUNET_PRED_ROOT/fold1" \
  -d 266 -c 3d_fullres -f all \
  -chk checkpoint_best.pth

nnUNetv2_predict \
  -i "$nnUNet_raw/Dataset267_isles26_atlas30_soopincluded_3fold_fold2_t1_raw/imagesTs" \
  -o "$NNUNET_PRED_ROOT/fold2" \
  -d 267 -c 3d_fullres -f all \
  -chk checkpoint_best.pth

nnUNetv2_predict \
  -i "$nnUNet_raw/Dataset268_isles26_atlas30_soopincluded_3fold_fold3_t1_raw/imagesTs" \
  -o "$NNUNET_PRED_ROOT/fold3" \
  -d 268 -c 3d_fullres -f all \
  -chk checkpoint_best.pth
```

The reported baseline uses `checkpoint_best.pth`, not
`checkpoint_final.pth`.

### 5. Evaluate the held-out predictions

```bash
python3 -m scripts.nnunet.evaluate_nnunet_results \
  --convert-config-name nnunet/convert/isles26_atlas30_soopincluded_3fold_fold1_cluster_3d_t1raw \
  --eval-config-name nnunet/eval/volumes_3d \
  --pred-dir "$NNUNET_PRED_ROOT/fold1" \
  --gt-dir "$nnUNet_raw/Dataset266_isles26_atlas30_soopincluded_3fold_fold1_t1_raw/labelsTs" \
  --output-dir "$NNUNET_PRED_ROOT/fold1-evaluation" \
  --fixed-threshold 0.5

python3 -m scripts.nnunet.evaluate_nnunet_results \
  --convert-config-name nnunet/convert/isles26_atlas30_soopincluded_3fold_fold2_cluster_3d_t1raw \
  --eval-config-name nnunet/eval/volumes_3d \
  --pred-dir "$NNUNET_PRED_ROOT/fold2" \
  --gt-dir "$nnUNet_raw/Dataset267_isles26_atlas30_soopincluded_3fold_fold2_t1_raw/labelsTs" \
  --output-dir "$NNUNET_PRED_ROOT/fold2-evaluation" \
  --fixed-threshold 0.5

python3 -m scripts.nnunet.evaluate_nnunet_results \
  --convert-config-name nnunet/convert/isles26_atlas30_soopincluded_3fold_fold3_cluster_3d_t1raw \
  --eval-config-name nnunet/eval/volumes_3d \
  --pred-dir "$NNUNET_PRED_ROOT/fold3" \
  --gt-dir "$nnUNet_raw/Dataset268_isles26_atlas30_soopincluded_3fold_fold3_t1_raw/labelsTs" \
  --output-dir "$NNUNET_PRED_ROOT/fold3-evaluation" \
  --fixed-threshold 0.5
```

These held-out `imagesTs`/`labelsTs` evaluations produce the nnU-Net results
reported above. They are distinct from nnU-Net's internal training-time
validation summaries.

Optional SLURM wrappers for planning, training, prediction, and evaluation are
available under `scripts/nnunet/slurm_runners/`; the raw commands above do not
depend on that cluster setup.

## Grand Challenge submission artifacts

The Grand Challenge release consists of two independently replaceable
artifacts:

1. a model archive containing the three fold configurations and checkpoints;
2. a Linux/AMD64 container image containing the inference service but no model
   weights.

The final ISLES'26 artifact applies equal-weight probability averaging with:

```text
configs/inference/sliding_window_native_ensemble.yaml
```

It does not select the TTA inference policy.

Run all packaging commands from the repository root in Linux or WSL2.

### Compose the ensemble model archive

Create a release-local builder configuration:

```yaml
members:
  - id: fold1
    run_dir: /absolute/path/to/fold1-run
    checkpoint: models/best/exact-fold1-checkpoint.pth

  - id: fold2
    run_dir: /absolute/path/to/fold2-run
    checkpoint: models/best/exact-fold2-checkpoint.pth

  - id: fold3
    run_dir: /absolute/path/to/fold3-run
    checkpoint: models/best/exact-fold3-checkpoint.pth

inference_policy: /absolute/path/to/Medical_Image_Segmentation_Toolkit/configs/inference/sliding_window_native_ensemble.yaml
output_dir: /absolute/path/to/release/model
archive_name: algorithmmodel.tar.gz
validation_device: cpu  # Build-time artifact validation only; runtime inference requires CUDA.
```

Build and strictly validate the model archive:

```bash
python3 -m scripts.gc_submission_builder.cli build-model \
  --config /absolute/path/to/isles26-ensemble-builder.yaml
```

The generated archive has the root-relative layout expected when Grand
Challenge extracts it beneath `/opt/ml/model/`.

### Build the model-independent container image

```bash
export GC_RELEASE_ROOT=/absolute/path/to/new-release
export GC_IMAGE_OUTPUT="$GC_RELEASE_ROOT/image"
export GC_IMAGE_NAME=medical-image-segmentation-toolkit-gc
export GC_IMAGE_TAG=isles26

python3 -m scripts.gc_submission_builder.cli build-image \
  --image-name "$GC_IMAGE_NAME" \
  --image-tag "$GC_IMAGE_TAG" \
  --interface-manifest scripts/gc_submission_builder/configs/interfaces/isles26.yaml \
  --output-dir "$GC_IMAGE_OUTPUT"
```

The image is built for `linux/amd64`, runs as a non-root user, and contains no
model checkpoint payloads.

### Test the image and model together

Prepare a platform-shaped input fixture:

```text
test-input/
  inputs.json
  stroke-metadata.json
  images/
    t1-brain-mri/
      case.mha
```

Then execute the external Grand Challenge HTTP lifecycle:

```bash
python3 -m scripts.gc_submission_builder.cli test \
  --image-name "$GC_IMAGE_NAME" \
  --image-tag "$GC_IMAGE_TAG" \
  --interface-manifest scripts/gc_submission_builder/configs/interfaces/isles26.yaml \
  --output-dir "$GC_IMAGE_OUTPUT" \
  --model-dir "$GC_RELEASE_ROOT/model/algorithmmodel" \
  --input-dir /absolute/path/to/test-input \
  --test-output-dir "$GC_RELEASE_ROOT/lifecycle-output" \
  --readiness-timeout-seconds 300
```

A passing lifecycle verifies the HTTP service, output types, value ranges,
native input geometry, and required runtime events.

### Save the tested image

```bash
python3 -m scripts.gc_submission_builder.cli save \
  --image-name "$GC_IMAGE_NAME" \
  --image-tag "$GC_IMAGE_TAG" \
  --interface-manifest scripts/gc_submission_builder/configs/interfaces/isles26.yaml \
  --output-dir "$GC_IMAGE_OUTPUT"
```

Before upload, verify and record both artifacts:

```bash
gzip -t "$GC_RELEASE_ROOT"/model/*.tar.gz
gzip -t "$GC_IMAGE_OUTPUT"/*.tar.gz
sha256sum "$GC_RELEASE_ROOT"/model/*.tar.gz "$GC_IMAGE_OUTPUT"/*.tar.gz
tar -tzf "$GC_RELEASE_ROOT"/model/*.tar.gz
```

See
[`scripts/gc_submission_builder/README.md`](scripts/gc_submission_builder/README.md)
for the complete release, validation, and provenance procedure.

## License and acknowledgements

This repository is distributed under the included MIT license. It contains
code derived from the original MedSegDiff implementation and selected
components from OpenAI's `improved-diffusion`; see [`LICENSE`](LICENSE) and
[`LICENSE_OPENAI`](LICENSE_OPENAI).

The ISLES'26 work builds on MONAI, nnU-Net, ATLAS, and the ISLES/SWITCH
challenge infrastructure. Dataset access and redistribution remain governed
by their respective terms.
