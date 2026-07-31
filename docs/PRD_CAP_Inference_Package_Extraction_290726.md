# PRD and CAP: Shared Inference Package Extraction and ISLES26 Grand Challenge Submission Builder

**Document version:** 1.2
**Original date:** 2026-07-29
**Last revised:** 2026-07-31
**Status:** Draft for implementation
**Primary competition:** ISLES26 3D lesion segmentation
**Primary model scope:** Existing 3D discriminative MONAI DynUNet checkpoints
**Primary implementation areas:** `src/inference/`, selected existing `src/` and `scripts/` modules, and `scripts/gc_submission_builder/`
**Reference experiment configuration:** `configs/cluster_isles26_atlas30_3d_randompatch_dynunet.yaml`

---

## 1. Executive decision

This plan introduces one shared, production-oriented inference implementation under `src/inference/`. That implementation will become the common path by which a trained repository model produces a segmentation probability map. Existing training-time validation, offline evaluation, ordinary repository inference, container-based diagnostic validation, and the Grand Challenge submission runtime will all become consumers of that implementation.

The same consumers will compose a shared top-level `inference` config family. Sliding-window execution, numerical precision, result space, TTA, probability ensembling, fixed thresholding, and postprocessing will no longer be defined independently for training validation and Grand Challenge inference. Validation/evaluation config will remain separate because it owns labels, metrics, threshold sweeps, checkpoint selection, and training cadence. A third `inference_runtime` config family will declare and enforce the capabilities and constraints of native Python, diagnostic-container, and production Grand Challenge execution.

The work is intentionally limited to the existing 3D discriminative model family required for the ISLES26 submission, initially DynUNet. It does not add 3D diffusion training or 3D diffusion inference. The shared contract will be narrow enough that a future diffusion predictor can implement it without requiring the spatial, container, or output-writing layers to be redesigned.

The target system has three separate but interlocking release components:

1. **Container infrastructure and code snapshot**: the Linux/amd64 Docker image, pinned Python/CUDA/PyTorch/MONAI runtime, Grand Challenge HTTP lifecycle, and repository inference code at a recorded commit.
2. **Shared inference code**: model construction, preprocessing, probability prediction, sliding-window execution, spatial inversion, TTA/ensembling, thresholding, postprocessing, and output validation.
3. **Model artifact bundle**: one or more immutable complete saved training configs and exact checkpoints, plus the selected shared inference config and provenance manifest, packaged so Grand Challenge expands it under `/opt/ml/model/`. Historical training-validation settings remain present for provenance but are not active when an explicit inference config is selected.

The first certified release must prioritize correctness, spatial fidelity, reproducibility, T4 memory safety, and the ten-minute per-case runtime limit over optional competition enhancements.

---

## 2. Why this work exists

The repository has become a configurable training and evaluation platform for 2D and 3D medical image segmentation. Its central training implementation lives under `src/`, with supporting dataset setup, evaluation, nnU-Net baseline, reporting, analysis, and SLURM tooling under `scripts/`.

The ISLES26 submission introduces a different execution environment from research training:

- one unseen 3D case is processed per container job;
- the model runs on an AWS `g4dn.2xlarge`-class instance with one NVIDIA T4 GPU and 16 GB VRAM, 32 GB host RAM, and 8 vCPUs;
- each case must finish within ten minutes;
- the runtime has no network access;
- model weights are mounted or expanded separately under `/opt/ml/model/`;
- the output segmentation must align with the original input volume, not merely with an internally reoriented or resampled tensor;
- the container must obey the input/output interface and HTTP lifecycle defined by Grand Challenge.

The present repository can already train the intended model and can evaluate it on full volumes, but its inference behavior is distributed across training validation, evaluation, older analysis code, model adapters, and data loaders. It does not yet provide a label-free, metadata-preserving preprocessing and inverse-spatial pipeline suitable for an external inference service.

This project therefore is not simply a Docker packaging exercise. The container should wrap a shared inference capability that is independently testable in the repository. If container inference were implemented separately under `scripts/`, research evaluation and submitted inference could silently diverge in preprocessing, checkpoint loading, sliding-window behavior, activation handling, or geometry.

---

## 3. Terminology

### 3.1 Training configuration

The resolved Hydra configuration saved with a training run, normally at:

```text
<run_dir>/.hydra/config.yaml
```

It is authoritative for model architecture, input channels, dimensionality, preprocessing parameters, diffusion/discriminative adapter selection, and other trained-system semantics. Deployment code must not reconstruct or overwrite these attributes from a second container configuration.

### 3.2 Inference policy

A shared, separately versioned top-level Hydra/OmegaConf configuration family, represented as `cfg.inference`, that controls how a model and preprocessed image become a probability or segmentation result. It is composed by training validation, offline evaluation, local inference, container diagnostics, and Grand Challenge deployment. Examples include:

- result/output space (`model_preprocessed` or `native_input`);
- runtime precision;
- sliding-window batch size, overlap, blend mode, and padding mode;
- TTA operations;
- model ensemble combination;
- probability threshold;
- connected-component or small-lesion filtering.

Inference policy must not redefine model topology, input modality count, trained intensity preprocessing, spacing, or orientation. Its fixed vocabulary is expressed as ordinary validated config strings; a Python `Enum` class is not required.

### 3.3 Result/output space

The model always performs its neural prediction in the preprocessed model tensor space. The configurable choice is the space in which the shared pipeline returns and consumers evaluate/materialize the result:

- `model_preprocessed`: retain the result on the reoriented/resampled model grid;
- `native_input`: invert the floating-point probability map to the original input grid before thresholding and postprocessing.

The config key is therefore `inference.output_space`, not `prediction_space`. Optional retention of both representations is a diagnostic-artifact setting, not a third output-space value.

### 3.4 Validation/evaluation policy

Configuration that requires labels or controls assessment rather than prediction. It owns metric selection, threshold sweep/oracle protocols, analysis level, validation cadence, progress metrics, and best-checkpoint selection. A fixed threshold used to emit a mask belongs to `inference`; evaluating many thresholds and selecting one from labels belongs to `evaluation`.

### 3.5 Inference runtime profile

A separately composed `cfg.inference_runtime` config describing the execution capabilities and hard constraints of a location/mode, initially:

- `native` (native Python execution outside a container);
- `gc_container_test`;
- `gc_submission`.

Runtime profiles enforce allowed output spaces, case batch size, worker policy, timeout, CUDA requirements, label availability, threshold-sweep permission, and diagnostic-artifact permission. They constrain the shared inference policy and fail on incompatible requests rather than silently rewriting them.

### 3.6 Model bundle

The contents of the tarball expanded beneath `/opt/ml/model/`. A bundle identifies the complete saved config and checkpoint for each ensemble member, the selected inference policy, content hashes, compatibility version, and provenance. The runtime projects model-owned fields from each saved config and does not activate its historical `validation.inference` settings when an explicit shared inference policy is supplied.

### 3.7 Predictor

The backend-specific component that maps a preprocessed tensor to a probability tensor. The first implementation is a 3D discriminative predictor. A future 3D diffusion predictor may implement the same high-level contract.

### 3.8 Spatial trace

The metadata required to map a prediction from model/preprocessed space back to the exact input grid, including original shape, affine, orientation, spacing, and the invertible transform history.

### 3.9 Grand Challenge interface, socket, and socket slug

An interface is the input/output combination assigned to a challenge phase. Each input or output is a socket. A socket slug is Grand Challenge's identifier for that socket and appears in `/input/inputs.json`; it is not a model attribute and is not invented by this repository. The phase or generated starter pack supplies the actual socket slugs and relative paths.

Socket handling belongs only to the Grand Challenge transport adapter. No core inference module should know an ISLES26 socket slug.

### 3.10 Cut

An independently implementable CAP unit. Every cut below includes its own context, dependencies, affected files, required changes, tests, and acceptance criteria so it can be assigned to an agent with minimal additional conversation history.

---

## 4. Confirmed requirements and external release gates

### 4.1 Project-confirmed requirements

- The submitted model is an existing 3D discriminative segmentation model, initially the DynUNet represented by `configs/cluster_isles26_atlas30_3d_randompatch_dynunet.yaml`.
- The clinical image modality is one 3D T1-weighted MRI volume.
- The project-standard input and mask format is `.nii.gz`.
- Case batch size is exactly one.
- The target hardware is one T4 with 16 GB VRAM, 32 GB RAM, and 8 vCPUs.
- The maximum execution time is ten minutes per case.
- The model is loaded before per-case invocation whenever the platform lifecycle allows it.
- Model artifacts are separate from the image and reside at `/opt/ml/model/`.
- Model bundle creation accepts a training run directory and a specific checkpoint/snapshot, matching the existing evaluation mental model.
- Existing BF16-trained checkpoints remain eligible. Autocast precision used during training does not encode BF16-only weights into an ordinary PyTorch state dict. Deployment precision is independently selected and tested, initially FP16 on T4 with FP32 fallback for diagnosis.
- Initial sliding-window batch size is one even if the training validation config used a larger window batch.
- Training validation, offline evaluation, local/container diagnostics, and production deployment share the same `inference` configuration schema.
- Repository and diagnostic-container execution may select `model_preprocessed` or `native_input` result space when prediction and reference labels are paired in the same space.
- Production `gc_submission` execution requires `native_input` output, case batch size one, and a fixed deployment threshold. It rejects label-dependent threshold sweeps.
- The output is a binary segmentation with values `{0, 1}` and an integer voxel type, preferably `uint8`.
- The output must match the original input grid and world-space geometry.
- Generative inference, metadata-conditioned inference, and newly trained model families are outside the first submission scope.

### 4.2 Current Grand Challenge platform requirements

Current Grand Challenge documentation describes the following lifecycle:

1. The container starts and loads the model.
2. Grand Challenge polls `GET /health` until HTTP 200.
3. Inputs are made available beneath `/input/`.
4. Grand Challenge calls `POST /invoke`.
5. The container writes outputs beneath `/output/` and returns HTTP 201.
6. The container is stopped after the case.

The image must contain the label:

```dockerfile
LABEL org.grand-challenge.api-method="invoke"
```

The runtime has no network access, should execute as a non-root user, treats `/input` as read-only, and provides `/output` and `/tmp` as writable locations. `/tmp` is transient scratch space.

The implementation should be derived from the current Grand Challenge algorithm template and testing lifecycle while also consulting the historical ISLES Docker template supplied by Ezequiel de la Rosa. Historical template behavior must not override current platform documentation.

Primary external references:

- <https://grand-challenge.org/documentation/building-and-testing-the-container/>
- <https://grand-challenge.org/documentation/runtime-environment/>
- <https://grand-challenge.org/documentation/choose-input-and-output-interfaces/>
- <https://grand-challenge.org/documentation/faq-challenges-algorithm-test-data-to-container/>
- <https://github.com/ezequieldlrosa/isles22-docker-template>

### 4.3 ISLES26 interface manifest release gate

The exact ISLES26 phase interface remains an external acceptance dependency. Before a container is considered upload-ready, capture from the competition phase or generated starter pack:

- the complete set of input socket slugs;
- the complete set of output socket slugs;
- each socket's relative path;
- whether technical-parameter JSON is actually present and whether it is required;
- the platform-declared image/file socket type;
- accepted transport file extensions;
- required segmentation overlay values;
- the official phase timeout and GPU assignment as displayed by the platform.

This is not a request to infer model semantics from socket names. It is an I/O compatibility gate.

There is a specific transport discrepancy to resolve: the project requirement supplied for ISLES26 is NIfTI (`.nii.gz`) input and output, while current general Grand Challenge documentation states that image socket payloads are accepted as MHA or TIFF. The implementation must proceed with NIfTI as the canonical internal and expected challenge format, but the final phase manifest must establish whether ISLES26 uses a file/custom socket for NIfTI or an image socket requiring another transport format.

If boundary conversion is necessary, it must be lossless with respect to voxel values and physical geometry and must live only in the Grand Challenge I/O adapter. The shared inference package remains NIfTI/metadata aware and must not change trained preprocessing because of the transport wrapper.

No fictional socket slug or path may be hardcoded while this manifest is unavailable. Tests may use clearly named fixture slugs.

---

## 5. Goals

### 5.1 Functional goals

1. Provide a reusable `src/inference/` package that loads a trained repository model and returns calibrated probability maps for 3D discriminative segmentation.
2. Make training-time validation and offline evaluation consume the same low-level prediction implementation.
3. Provide label-free preprocessing derived from the existing ISLES26 preprocessing contract.
4. Preserve sufficient spatial metadata to invert prediction probabilities to the original input grid.
5. Support direct and MONAI sliding-window inference with deployment-safe defaults.
6. Introduce one shared top-level inference config family for training validation, offline evaluation, repository inference, container diagnostics, and GC deployment.
7. Introduce separate evaluation and runtime-profile configs so shared inference options can be reused without allowing label-dependent or platform-invalid combinations.
8. Support optional probability-space TTA, multi-checkpoint/model ensembling, thresholding, and conservative postprocessing through the shared inference policy.
9. Build and test a Grand Challenge-compatible Docker image containing the runtime and repository code snapshot.
10. Build a replaceable model tarball from one or more run-directory/checkpoint specifications.
11. Build the image and model bundle independently or together.
12. Produce machine-readable provenance and resource measurements for every release candidate.

### 5.2 Quality goals

- Prediction parity between the pre-extraction and post-extraction evaluation paths within defined numerical tolerance.
- Exact native-output shape and verified affine/world-space consistency with the input; explicit transformed-grid metadata for model-space results.
- Strict, fail-fast model bundle validation.
- Deterministic behavior when TTA and stochastic behavior are disabled.
- No runtime network dependency.
- No data-loader worker subprocess requirement inside the single-case container.
- Peak GPU and host memory safely below platform limits.
- End-to-end invocation below ten minutes on representative worst-case inputs.
- Clear logs without protected-data leakage.

---

## 6. Non-goals

The following are explicitly outside this CAP:

- 3D diffusion training or inference.
- Training a new model merely to make it deployable.
- Metadata-conditioned prediction.
- Changing the scientific semantics of the saved training configuration.
- Replacing Hydra across the repository.
- Rewriting metric implementations.
- Unifying repository-model inference with the external nnU-Net runtime.
- Generic multi-GPU inference in the Grand Challenge container.
- Case batching greater than one.
- Serving multiple simultaneous requests.
- DICOM ingestion.
- Lesion-volume-conditioned postprocessing research.
- Automatic selection of a competition threshold from hidden test data.
- Enabling an enhancement such as TTA, ensembling, or component filtering without validation evidence.

---

## 7. Current repository state and integration evidence

### 7.1 Intended training configuration

`configs/cluster_isles26_atlas30_3d_randompatch_dynunet.yaml` composes:

- `model: dynunet_base`;
- `data_profile: isles26_3d_randompatch_t1raw`;
- `diffusion: discriminative`;
- 3D sliding-window validation;
- BF16 autocast training;
- deep-supervised loss;
- validation case batch size one.

The resolved config, rather than this top-level preset alone, is the authoritative artifact because CLI overrides and Hydra composition may change the final values.

### 7.2 Existing preprocessing

`src/data/loader_stack/isles26_loader.py` currently builds its common transform chain with MONAI `LoadImaged`, `Orientationd`, `Spacingd`, modality processing, channel merging, and `EnsureTyped`. The chain includes the label and the full-volume dataset currently returns tensors in a form optimized for training/evaluation rather than external spatial inversion.

The transform definition is valuable and must be reused. The problems to solve are:

- make label keys optional;
- preserve `MetaTensor` or equivalent transform history;
- expose original image metadata;
- separate deterministic preprocessing from training-only random augmentation;
- provide a tested inverse operation for probability maps.

### 7.3 Existing prediction execution

`src/utils/valid_utils.py::build_validation_inferer()` currently owns direct versus sliding-window inference and calls `diffusion.sample()` for a full input or each window.

`src/training/trainer.py::validate_one_epoch()` calls that inferer and optionally ensembles repeated samples before metric computation.

`scripts/evaluation/io/model_volumes.py` also calls the same validation inferer for 3D live-model evaluation. It correctly rejects current 3D non-discriminative diffusion because the active diffusion samplers are 2D-shaped.

This is the core extraction opportunity: move the reusable prediction behavior into `src/inference/`, then retain compatibility wrappers where useful.

### 7.4 Existing model loading

`scripts/evaluation/core/model_loader.py` provides checkpoint discovery and repository model construction for evaluation. `src/utils/train_utils.py::build_model_and_diffusion()` performs related construction with training-specific DP/DDP wrapping and data-contract channel synchronization.

Inference needs a canonical single-device builder that preserves channel synchronization and the existing model/diffusion factory while excluding distributed training concerns. Deployment loading must be strict. Legacy compatibility loading and training resume must remain permissive where currently required.

### 7.5 Existing probability and threshold analysis

The current evaluation package under `scripts/evaluation/` already provides the canonical metrics, threshold protocols, per-case/global/oracle analysis, reports, and model-volume producers. These capabilities remain outside the core inference package but will consume its probability outputs.

`scripts/analysis/threshold_analysis.py` contains an older duplicate config/checkpoint/inference path. It should not become a third production consumer. After parity is established, it should delegate to the current evaluation package or be formally deprecated.

### 7.6 Existing ensemble behavior

`src/utils/ensemble.py::mean_ensemble()` averages over the ensemble dimension and is spatial-dimension agnostic in implementation.

The existing soft-STAPLE code reduces over only the final two axes. It is therefore not valid as written for `[N, B, C, D, H, W]` volumes. Soft STAPLE must remain disabled for 3D until it is generalized and tested across all spatial dimensions.

### 7.7 Existing ancillary packages

- `scripts/test_validation_memory.py` directly samples models. It should later delegate to the shared predictor or be superseded by the release resource benchmark.
- `scripts/nnunet/` remains a separate external model/runtime pipeline. Its affine and round-trip tests provide useful testing patterns but it should not be forced through `src/inference`.
- `scripts/reporting/` consumes result artifacts and requires no inference dependency.
- `scripts/dataset_setup/` creates/explores dataset definitions and requires no inference dependency.
- generic `scripts/slurm/` remains training-oriented. The new builder may reuse its environment facts and command-construction lessons without coupling core inference to SLURM.

---

## 8. Target architecture

```text
Training run directory
  .hydra/config.yaml
  models/.../<checkpoint>.pth
            |
            v
scripts/gc_submission_builder model build
            |
            v
/opt/ml/model/
  bundle_manifest.json
  inference_policy.yaml
  members/
    model_000/
      config.yaml
      weights.pth

                     Shared config composition
       +-------------------+-----------------------+
       |                   |                       |
 cfg.inference     cfg.inference_runtime    validation/evaluation
 prediction/result   capability limits      labels/metrics/sweeps
       |                   |                       |
       +-------------------+-----------------------+
                           |
Input transport            |          Repository validation data
(/input, inputs.json)       |          (existing dataloaders + labels)
         |                  |                       |
         v                  v                       v
Grand Challenge adapter      src/inference     Evaluation/training wrappers
         |                  shared pipeline               |
         +------------------+-----------------------------+
                            |
            +---------------+----------------+
            |                                |
            v                                v
   model_preprocessed result          native_input result
   for parity/diagnostics             for native validation or GC output
```

### 8.1 Separation of responsibility

| Concern | Owner | Reason |
|---|---|---|
| Model architecture and trained preprocessing | Saved run config | Prevent deployment drift |
| Exact weights | Model bundle member | Independently replaceable artifact |
| Output space, precision, sliding-window policy, TTA, ensemble, fixed threshold, postprocessing | Shared `inference` config | One prediction policy across all consumers |
| Metrics, threshold sweeps/oracles, validation cadence, checkpoint selection | `validation` / `evaluation` config | Label-dependent assessment is not inference |
| Case batch, workers, timeout, device and allowed capabilities | `inference_runtime` profile | Location/mode constraints must not duplicate scientific policy |
| Probability prediction mechanics | `src/inference/` | Shared scientific behavior |
| Pairing result and reference-label space | Evaluation/training consumers with shared contracts | Metrics are valid only when grids match |
| `/input`, `/output`, `inputs.json`, socket slugs, HTTP statuses | GC adapter | Platform transport only |
| Docker base, system libraries, Python packages, non-root user | GC builder/container | Runtime infrastructure only |
| SLURM submission | Existing or new runner wrapper | Cluster orchestration only |

### 8.2 Proposed package layout

The exact module split may be adjusted during implementation, but responsibility boundaries must remain:

```text
src/inference/
  __init__.py
  contracts.py              # typed requests/results/traces/capability errors
  policy.py                 # shared inference config parsing and validation
  runtime.py                # execution-profile capabilities and cross-validation
  bundle.py                 # manifest/config/checkpoint discovery and validation
  model_loader.py           # strict single-device repository model construction
  preprocessing.py          # label-free/shared deterministic preprocessing
  predictors.py             # predictor protocol and discriminative implementation
  sliding_window.py         # MONAI sliding-window orchestration
  augmentation.py           # invertible TTA definitions and de-augmentation
  ensemble.py               # probability-space model/TTA aggregation
  spatial.py                # inverse transforms and native-grid validation
  postprocessing.py         # threshold and optional binary morphology/filtering
  pipeline.py               # end-to-end orchestration independent of transport

scripts/gc_submission_builder/
  __init__.py
  README.md
  cli.py                    # build-image, build-model, build-all, test, save
  build_config.py           # builder-only config validation
  model_artifact.py         # deterministic bundle staging/tar creation
  release_manifest.py       # code/runtime/model provenance and hashes
  runtime/
    app.py                  # /health and /invoke server integration
    inference.py            # init_model() and per-case run()
    interfaces.py           # inputs.json/socket/path dispatch
    image_io.py             # transport format read/write and validation
  container/
    Dockerfile
    requirements.lock
    do_build.*
    do_test_run.*
    do_save.*
  configs/
    default.yaml
    interface.example.yaml
  tests_smoke/
    fixtures/...

configs/inference/
  sliding_window_model_space.yaml
  sliding_window_native.yaml
  sliding_window_native_fp16.yaml

configs/inference_runtime/
  native.yaml
  gc_container_test.yaml
  gc_submission.yaml
```

The inference presets are dimension-agnostic and do not declare ROI. During
policy resolution, existing `model.spatial_dims` / `data_mode.dim` selects
`dataset.preprocessing_configs.roi.slice_2d` or `volume_3d`; those dataset ROI
values are themselves derived from `model.image_size`. The `native` child
overrides only output space, and `native_fp16` overrides only precision. Thus
inference policy neither duplicates 2D/3D model structure nor requires changes
to established data-mode config families. The 2D-compatible config contract
does not add a diffusion predictor or expand the initial ISLES26 3D backend
scope.

Where practical, use the current official Grand Challenge template files rather than creating a novel server implementation. The repository should own and test the inference handler and interface adapter even if `app.py` remains close to the template.

### 8.3 Public inference contracts

The first implementation should expose a small public surface rather than every internal helper. Conceptually:

```python
bundle = load_inference_bundle(
    model_root=Path("/opt/ml/model"),
    device=torch.device("cuda:0"),
)

result = predict_case(
    bundle=bundle,
    image_path=input_path,
    inference_cfg=cfg.inference,
    runtime_cfg=cfg.inference_runtime,
)
```

The returned result should distinguish at least:

- the configured primary result in `model_preprocessed` or `native_input` space;
- optional diagnostic intermediate probabilities when the runtime profile permits them;
- final binary mask in the configured result space;
- spatial trace and geometry checks;
- model and policy provenance;
- timing and memory measurements when instrumentation is enabled.

The predictor-level contract should operate on tensors and return probabilities in `[0, 1]`, with explicit `[B, C, *spatial]` shape semantics. It should not read files, calculate metrics, or write NIfTI outputs.

### 8.4 Model bundle layout and manifest

The model tarball should expand directly into:

```text
/opt/ml/model/
  bundle_manifest.json
  inference_policy.yaml
  members/
    model_000/
      config.yaml
      weights.pth
    model_001/               # optional later ensemble member
      config.yaml
      weights.pth
```

The manifest should contain generated facts, not duplicate model configuration:

```json
{
  "bundle_schema_version": 1,
  "inference_api_version": 1,
  "created_at_utc": "...",
  "code_commit": "...",
  "members": [
    {
      "id": "model_000",
      "source_run": "...",
      "source_checkpoint": "...",
      "config_path": "members/model_000/config.yaml",
      "weights_path": "members/model_000/weights.pth",
      "config_sha256": "...",
      "weights_sha256": "..."
    }
  ],
  "inference_policy_path": "inference_policy.yaml"
}
```

Absolute source paths may be recorded in a local build report but should be optional or redacted from the distributable manifest if they reveal infrastructure details.

Each member's `config.yaml` is the complete resolved training config, including its historical validation section. Keeping it provides reproducibility and avoids prematurely inventing a reduced model-config schema. At runtime, model construction projects only the model-owned and trained-preprocessing fields. When the bundle supplies `inference_policy.yaml`, that explicit shared policy is active; historical `config.yaml::validation.inference` values are retained but are not merged into or allowed to override it.

The builder command should accept one or more pairs of:

- training run directory;
- checkpoint/snapshot name or exact checkpoint path within that run.

Checkpoint discovery may reuse evaluation conventions, but the resolved output must identify one exact file. Ambiguous matches are errors.

The final archive is created with root-relative contents, equivalent to:

```bash
tar -czvf algorithmmodel.tar.gz -C /path/to/algorithmmodel .
```

The trailing `.` is required so extraction populates `/opt/ml/model/` directly rather than adding an unintended enclosing directory.

### 8.5 Shared inference policy schema

An initial schema should be conservative:

```yaml
inference:
  output_space: native_input  # model_preprocessed | native_input
  precision: fp16          # allowed: fp16, fp32; bf16 only if hardware-certified

  sliding_window:
    enabled: true
    sw_batch_size: 1
    overlap: 0.5
    blend_mode: gaussian
    padding_mode: constant

  tta:
    enabled: false

  ensemble:
    enabled: false

  decision:
    threshold: 0.5

  postprocessing:
    enabled: false

  artifacts:
    enabled: false
```

The loader must use an allowlist. `output_space` is an ordinary config string selected from a fixed validated set, not a required Python `Enum` class. Unknown keys, invalid combinations, or unsupported methods fail before inference begins.

`roi_size` is mandatory in the resolved typed policy but is deliberately absent
from new inference YAML because it is model-owned rather than an inference
choice. The resolver maps existing 2D/3D dimensionality to the saved dataset
preprocessing ROI and injects that concrete non-null tuple. A new inference
policy containing `roi_size` is rejected as an unknown model override.
Historical `validation.inference.sliding_window.roi_size: null` resolves to the
same model-owned ROI; a historical concrete value is accepted only when it
matches. Sliding window defaults to enabled. No direct-inference preset is
shipped initially, although an explicit or historical `enabled: false` remains
representable for legacy compatibility.

Until their implementation cuts land, `tta`, `ensemble`, `postprocessing`, and `artifacts` are disabled capability stubs. Each accepts only `enabled: false`; backend-specific parameters must not appear prematurely in production configs. The established fixed threshold remains `0.5`. Output file dtype and foreground/background encoding belong to the later materialization/transport implementation and are not exposed as inactive Cut 1 policy variables.

### 8.6 Runtime profile schema and capability validation

Native Python execution:

```yaml
inference_runtime:
  profile: native
  case_batch_size: 1
  num_workers: 0
  require_cuda: false
  timeout_seconds: null
  constraints:
    allowed_output_spaces: [model_preprocessed, native_input]
    allow_ground_truth: true
    allow_threshold_sweep: true
    allow_intermediate_artifacts: true
```

Production Grand Challenge execution:

```yaml
inference_runtime:
  profile: gc_submission
  case_batch_size: 1
  num_workers: 0
  require_cuda: true
  timeout_seconds: 600
  constraints:
    allowed_output_spaces: [native_input]
    allow_ground_truth: false
    allow_threshold_sweep: false
    allow_intermediate_artifacts: false
```

`gc_container_test` uses the same built runtime but may allow both output spaces, mounted labels, and diagnostic artifacts. It is for controlled validation of the exact container environment, not the public `/invoke` contract.

Cross-config validation applies runtime constraints to the requested inference and evaluation configs. It must reject incompatible combinations rather than silently coerce them. Examples include transformed-space output under `gc_submission`, case batch size greater than one, or a label-dependent threshold sweep during `/invoke`.

### 8.7 Validation/evaluation composition

Training validation and offline evaluation compose the same top-level `inference` group with their label-dependent policy:

```yaml
defaults:
  - /inference: sliding_window_model_space
  - /inference_runtime: native
  - _self_

validation:
  val_batch_size: 1
  validation_interval: 5000
  metrics:
    - dice_3d
    - surface_dice_monai_3d
    - hd95_3d
  checkpoint_best:
    metric_name: dice_3d
```

Offline threshold analysis adds an `evaluation.threshold_protocol`. It does not replace `inference.decision.threshold`; it evaluates alternatives using labels so a fixed deployment value can later be selected and written into an inference config.

For `model_preprocessed` evaluation, prediction is paired with the jointly transformed label. For `native_input` evaluation, prediction is paired with the original native-grid label. The evaluator must validate shape and physical geometry before computing metrics.

### 8.8 Backward compatibility for existing runs

Historical saved runs contain prediction settings under `cfg.validation.inference`, including the current representative DynUNet run. They remain valid inputs.

The migration resolver uses this precedence:

1. explicit top-level `cfg.inference` supplied for the current job;
2. legacy `cfg.validation.inference` translated into the shared schema;
3. a sliding-window policy derived from the saved model/data ROI contract when
   neither policy block exists.

When an explicit top-level inference config exists, values are not merged field-by-field with the historical validation inference block. This is essential to prevent settings such as training-time `sw_batch_size: 4` from leaking into a T4 policy that explicitly selects one.

Legacy translation emits provenance and, eventually, a deprecation warning. It does not rewrite the saved run config on disk.

### 8.9 End-to-end prediction order

The canonical operation order is:

1. Validate the input transport and identify exactly one image.
2. Read image data and native geometry.
3. Apply deterministic trained preprocessing while retaining an invertible trace.
4. Execute each allowed TTA view.
5. Predict patch probabilities using direct or sliding-window inference.
6. Invert each TTA transform in model space.
7. Mean-combine TTA and/or model-member probabilities.
8. If `output_space=native_input`, invert the combined floating-point probability map to the original input grid and validate shape and physical geometry.
9. If `output_space=model_preprocessed`, retain the combined probability on the model grid and record its model-space geometry.
10. Threshold in the configured output space.
11. Apply optional connected-component filtering in that same space; physical-volume filters require valid spacing metadata.
12. Convert to `uint8` values `{0, 1}` when a binary output is requested.
13. Return the configured result to the consumer.
14. For GC transport, write the native result and re-open/validate it before returning success.

Whenever spatial inversion occurs, probability interpolation must happen before thresholding. Interpolating an already binary mask can introduce avoidable geometric and topological artifacts.

---

## 9. Configuration ownership rules

### 9.1 Saved model config owns

- architecture (`DynUNet` initially);
- spatial dimensionality;
- input and output channel counts;
- deep-supervision topology;
- trained modality selection;
- trained intensity preprocessing;
- trained orientation and spacing transforms;
- default/model-compatible ROI size;
- discriminative versus diffusion adapter type.

The complete saved config remains packaged for provenance and construction compatibility. Its historical validation/inference settings are inactive when a current explicit `cfg.inference` is selected.

### 9.2 Inference policy owns

- configured result/output space;
- numerical precision;
- sliding-window execution parameters that do not change model topology;
- TTA;
- ensemble aggregation;
- fixed decision threshold;
- postprocessing;
- optional intermediate artifact retention, subject to runtime permission.

### 9.3 Validation/evaluation config owns

- reference-label access and pairing;
- metric selection and aggregation;
- threshold sweep/oracle protocols;
- analysis levels and reports;
- training validation interval and progress metrics;
- best-checkpoint selection.

It does not define a second sliding-window, TTA, ensemble, output-space, or fixed deployment-threshold schema.

### 9.4 Inference runtime profile owns

- execution profile name (`native`, `gc_container_test`, or `gc_submission`);
- case batch and worker limits;
- timeout and device requirements;
- allowed output spaces;
- permission to access ground truth or execute threshold sweeps;
- permission to retain diagnostic/intermediate artifacts;
- hard capability checks for a particular execution context.

The runtime profile constrains requested behavior. It must not change model architecture, trained preprocessing, or silently rewrite inference choices.

### 9.5 Builder config owns

- image tag and output archive names;
- Docker build context;
- selected runtime lockfile/base image;
- code commit/worktree provenance;
- model run-directory/checkpoint specifications;
- path to inference policy;
- local test fixture paths;
- phase interface manifest path.

Builder config must not contain DynUNet channels, strides, model image size, input modality count, or trained preprocessing values.

### 9.6 Interface manifest owns

- socket slugs;
- input and output relative paths;
- transport file types;
- optional JSON input presence;
- platform interface dispatch key.

It must not contain model architecture or inference-policy settings.

### 9.7 Why this is shared inference composition, not one monolithic validation class

The overlap identified between existing validation configs and proposed GC inference is real: both need the same sliding-window, precision, TTA, ensemble, output-space, fixed-threshold, and postprocessing semantics. Duplicating those fields would recreate the drift this refactor is intended to remove.

The whole job is not nevertheless called validation because production Grand Challenge invocation has no reference label and performs no assessment. Training validation also owns cadence and checkpoint selection, while offline evaluation owns threshold sweeps and reports. Those operations would be meaningless or unsafe inside production `/invoke`.

The agreed composition therefore captures the synergy at the correct boundary:

```text
saved model contract
        +
shared inference policy
        +
consumer assessment/interface policy
        +
runtime capability profile
```

This permits ordinary validation and container diagnostics in either valid result space without allowing evaluation-only behavior to leak into the submission endpoint.

---

## 10. Required changes to existing components

| Existing component | Required relationship to new package | Scope of change |
|---|---|---|
| `configs/validation/*.yaml` | Split prediction from assessment policy | Move new prediction settings to `configs/inference/`; retain metrics/cadence/checkpoint concerns under validation |
| new `configs/inference/*.yaml` | Shared prediction policy | Compose from training validation, evaluation, local inference, container diagnostics, and GC deployment |
| new `configs/inference_runtime/*.yaml` | Execution capability profiles | Enforce native-Python/container/submission constraints without duplicating inference settings |
| `src/data/loader_stack/isles26_loader.py` | Share deterministic preprocessing builder | Extract reusable image-only/image-label transforms; retain training augmentation behavior |
| `src/utils/valid_utils.py` | Compatibility façade and legacy config translator | Delegate prediction/sliding-window construction to `src/inference`; translate historical `validation.inference` when no explicit top-level policy exists |
| `src/training/trainer.py::validate_one_epoch` | Shared predictor/config consumer | Compose top-level inference plus native runtime; retain labels, metrics, progress, cadence, and logging |
| `src/utils/train_utils.py` | Reuse construction primitives only | Keep DP/DDP training builder; expose/reuse channel synchronization without coupling deployment to distributed setup |
| `src/training/checkpoint_utils.py` | Shared prefix normalization where safe | Add strict deployment mode or strict caller; preserve resume compatibility behavior |
| `src/diffusion/discriminative_adapter.py` | First predictor backend dependency | Reuse final-head/probability behavior; validate output-domain contract |
| `src/utils/ensemble.py` | Selective reuse/migration | Mean is eligible; soft STAPLE remains disabled for 3D until generalized |
| `scripts/evaluation/core/model_loader.py` | Delegate common bundle/model loading | Preserve evaluation checkpoint CLI behavior while removing duplicated construction |
| `scripts/evaluation/io/model_volumes.py` | Shared predictor/config consumer | Retain volume sample identity/metadata; route model/native result space and matching label space explicitly |
| `scripts/evaluation/core/evaluation_pipeline.py` | Assessment consumer | Continue metrics, threshold protocols, reports, and provenance; validate prediction/reference geometry |
| `scripts/analysis/threshold_analysis.py` | Legacy migration/deprecation | Do not extend independently; later delegate or retire after parity |
| `scripts/test_validation_memory.py` | Replace or delegate | Convert to shared predictor resource smoke test or supersede with GC benchmark |
| `tests/test_evaluation_*` and loader/validation tests | Regression safety | Update mocks/import paths only where needed; preserve behavior assertions |
| `scripts/nnunet/` | Intentionally separate | Reuse only generic contracts/tests when safe; no forced predictor integration |
| `scripts/reporting/` | Artifact consumer only | No direct dependency required |
| `scripts/dataset_setup/` | No dependency | No change required |
| `scripts/slurm/` | Optional orchestration reuse | Do not import into core inference; add a thin GC runner only if cluster build/test is needed |

---

## 11. Cross-cutting acceptance invariants

Every implementation cut must preserve these invariants where applicable:

1. **Single source of model truth:** saved run config plus exact checkpoint.
2. **One inference schema:** training validation, evaluation, diagnostics, and GC deployment compose the same top-level `cfg.inference` contract.
3. **Separate assessment policy:** metrics and threshold sweeps remain in validation/evaluation config and never become implicit deployment behavior.
4. **Runtime capability enforcement:** incompatible inference/evaluation/runtime combinations fail rather than being silently changed.
5. **Probability contract:** predictor outputs finite probabilities in `[0, 1]` with `[B, C, *spatial]` shape.
6. **Result-space contract:** every result declares `model_preprocessed` or `native_input`; every metric pairs prediction and reference in the same verified space.
7. **GC spatial contract:** production GC output is always `native_input` and matches input shape and physical geometry.
8. **Case batch invariant:** case batch size is one under `gc_submission`.
9. **Strict release loading:** missing/unexpected state-dict keys fail model initialization.
10. **Explicit policy precedence:** top-level `cfg.inference` replaces rather than field-merges with historical `cfg.validation.inference`.
11. **No hidden fallback:** unsupported TTA, ensemble, precision, model family, output space, or interface fails clearly.
12. **No label dependency:** deployment preprocessing accepts an image without a label.
13. **No network runtime:** all required files and packages are inside the image or model mount.
14. **No multiprocessing dependency:** production case loading runs in-process; no DataLoader worker pool is necessary.
15. **Independent artifacts:** the Docker image can be rebuilt without model weights, and a model bundle can be rebuilt without changing the image.
16. **Reproducible provenance:** code, runtime, saved config, active inference config, runtime profile, and weights are hashable and recorded.
17. **Enhancements are opt-in:** baseline inference remains available when TTA, ensembling, and component filtering are disabled.

---

# Change Action Plan

## 12. Cut 0: Baseline lock, environment audit, and release fixtures

### Context

Inference behavior is currently shared informally between training validation and evaluation. Refactoring without a fixed reference could make a numerically plausible change that invalidates threshold studies or competition results. The deployment environment must also be derived from the existing standardized `.sqsh` training image rather than guessed from the stale root `requirements.txt`.

### Dependencies

None. This cut precedes functional changes.

### Affected files and components

- `scripts/slurm/base_run_config.py`
- `scripts/evaluation/`
- `scripts/test_validation_memory.py`
- `requirements.txt` as an audited input, not necessarily modified
- new tests/fixtures under `tests/fixtures/inference/`
- new baseline document or machine-readable record under `docs/` or `tests/fixtures/inference/`
- new environment-inspection helper under `scripts/gc_submission_builder/`

### Desired changes

1. Select a real ISLES26 DynUNet run and exact checkpoint for parity testing.
2. Record its resolved `.hydra/config.yaml`, checkpoint hash, code commit, input case IDs, and expected validation configuration.
3. Record the effective legacy `validation.inference` values and identify which belong in the new shared inference schema.
4. Capture representative probability tensors and metrics for at least:
   - one small/fast validation case;
   - one case with foreground;
   - one case with no or minimal foreground if available;
   - one large-volume case for resource testing.
5. Record direct/sliding-window settings, result space, reference-label space, and seeds.
6. Add an inspector that runs inside `MedSegDiff_nnUNet_010226.sqsh` or the configured successor image and records:
   - Python version;
   - CUDA runtime/toolkit facts;
   - PyTorch and torchvision versions;
   - MONAI, nibabel, SimpleITK, NumPy, SciPy, cc3d, and other runtime-relevant packages;
   - GPU capability when available.
7. Do not treat root `requirements.txt` as a deployable lockfile. It currently lacks several inference dependencies and pins an old PyTorch stack.
8. Obtain or construct synthetic NIfTI fixtures with nontrivial affines. Fixtures must not contain protected patient data.

### Expected tests and testing components

- A baseline command can reproduce the selected probability output using the current evaluation path.
- Fixture hashes are stable.
- The environment inspector emits valid JSON and exits nonzero when a required fact cannot be obtained.
- Synthetic fixtures cover identity, permuted/flipped, anisotropic, translated, and oblique geometries.
- Existing evaluation tests remain green before extraction begins.

### Acceptance criteria

- The baseline model/checkpoint and comparison cases are unambiguous.
- Numerical and metric parity tolerances are written down before code movement.
- A runtime inventory exists from the actual `.sqsh` image.
- No production dependency lock or CUDA image has yet been chosen by guesswork.

### Rollback

Documentation and additive fixtures only; no runtime rollback required.

---

## 13. Cut 1: Scaffold `src/inference` and define shared config contracts

### Context

The shared package needs stable boundaries before existing consumers are migrated. This includes not only the predictor API but also the division between shared inference policy, label-dependent validation/evaluation policy, and runtime capability constraints. The initial API must be sufficient for discriminative 3D inference without attempting to abstract every possible future model.

### Dependencies

Cut 0 baseline records.

### Affected files and components

- new `src/inference/__init__.py`
- new `src/inference/contracts.py`
- new `src/inference/policy.py`
- new `src/inference/runtime.py`
- new `src/inference/predictors.py`
- new `configs/inference/*.yaml`
- new `configs/inference_runtime/*.yaml`
- new `tests/test_inference_contracts.py`
- new `tests/test_inference_policy.py`
- new `tests/test_inference_runtime.py`

### Desired changes

1. Define typed structures for:
   - model bundle/member descriptors;
   - predictor capabilities;
   - preprocessed case;
   - spatial trace;
   - prediction result;
   - timing/resource records.
   - declared output/result space.
2. Define explicit errors for:
   - unsupported model family/dimensionality;
   - invalid bundle;
   - invalid policy;
   - invalid predictor/result tensors;
   - spatial restoration failure;
   - interface/input failure;
   - resource limit failure.
3. Define a minimal predictor protocol that accepts `[B, C, *spatial]` and returns probabilities of the same spatial rank.
4. Implement top-level `cfg.inference` parsing with sliding-window enabled by default, mandatory model-owned ROI resolution, and a strict unknown-key policy.
5. Support fixed validated values `model_preprocessed` and `native_input` for `inference.output_space` without requiring a Python Enum class.
6. Resolve a dimension-appropriate inference ROI from the existing saved model/data preprocessing contract without separate 2D/3D inference policies or changes to established data-mode configs.
7. Implement `cfg.inference_runtime` profiles for `native`, `gc_container_test`, and `gc_submission`.
8. Implement cross-config capability validation that rejects, rather than rewrites, invalid combinations.
9. Encode `case_batch_size == 1`, `output_space == native_input`, no ground truth, and no threshold sweep as `gc_submission` constraints rather than universal inference restrictions.
10. Keep file I/O and Grand Challenge socket concerns out of the shared contracts.

### Expected tests and testing components

- New inference YAML contains no ROI field; resolution against a complete saved
  model/data contract injects its exact ROI and enables sliding-window
  inference.
- Missing, null, empty, dimensionally incompatible model-owned ROI values fail.
- New policy ROI overrides and mismatched historical concrete ROI values fail.
- Unknown keys and unsupported values fail.
- BF16 policy fails on an uncertified deployment profile rather than silently changing precision.
- Repository and container-test profiles accept both declared output spaces.
- Production GC rejects model-space output, case batch sizes other than one, threshold sweeps, ground truth, and intermediate artifact retention.
- Cross-config validation leaves valid requested values unchanged.
- Predictor output validation rejects logits outside the declared probability contract, NaN/Inf values, wrong ranks, and incorrect channels.
- Capability errors clearly explain that current 3D non-discriminative diffusion is unsupported.

### Acceptance criteria

- The public API can represent the discriminative 3D use case without importing `scripts/`.
- Contracts contain no Grand Challenge socket or filesystem assumptions.
- Shared inference, assessment, and runtime responsibilities are separately represented.
- Future diffusion support can implement the predictor protocol without changing spatial result contracts.

### Rollback

Remove the new package and tests; no consumers have migrated yet.

---

## 14. Cut 2: Strict model bundle and single-device model loading

### Context

Evaluation and training currently construct models through related but distinct paths. Deployment requires exact saved-config reconstruction and strict checkpoint loading, while training resume still needs legacy prefix compatibility.

### Dependencies

Cuts 0-1.

### Affected files and components

- new `src/inference/bundle.py`
- new `src/inference/model_loader.py`
- `scripts/evaluation/core/model_loader.py`
- `src/utils/train_utils.py` only if a small common construction helper must be exposed
- `src/training/checkpoint_utils.py` only if strict loading can be added without changing resume defaults
- `src/models/model_factory.py`
- `src/diffusion/diffusion.py`
- `src/diffusion/discriminative_adapter.py`
- new `tests/test_inference_bundle.py`
- new `tests/test_inference_model_loader.py`
- existing `tests/test_evaluation_model_loader.py`

### Desired changes

1. Load each member's saved resolved config without applying architecture overrides.
2. Preserve the complete resolved config, including historical validation fields, as immutable provenance.
3. Project model-owned/trained-preprocessing fields for construction without activating historical validation policy.
4. Preserve existing `sync_model_image_channels_with_data_contract()` behavior or extract a safe shared helper.
5. Build the model through `src.models.build_model()` and the existing diffusion/discriminative factory.
6. Reject non-3D and non-discriminative bundles in the initial backend with an explicit capability error.
7. Resolve exactly one requested checkpoint per member.
8. Normalize known DP/DDP key prefixes, then require no missing or unexpected model keys.
9. Put the model on one explicit device, call `.eval()`, and disable gradients.
10. Validate ensemble member compatibility for:
   - input channels/modalities;
   - spatial dimensionality;
   - preprocessing contract;
   - output channels;
   - native output semantics.
11. Validate manifest hashes before loading.
12. Allow FP16 autocast at execution time without rewriting stored weights.
13. Refactor evaluation's loader to delegate shared construction while preserving its existing run-dir/checkpoint CLI contract.

### Expected tests and testing components

- Load a representative DynUNet state dict saved with no wrapper, DP prefix, and DDP prefix.
- Missing/unexpected keys fail in deployment mode.
- Existing permissive training resume behavior is unchanged.
- A config/weight hash mismatch fails before model construction.
- An explicit top-level inference policy does not inherit historical `validation.inference.sliding_window.sw_batch_size` or other legacy execution values.
- A mismatched ensemble member fails with a field-specific error.
- BF16-trained checkpoint loads under FP32 and FP16 execution modes.
- Existing evaluation model-loader tests remain green.
- A real selected checkpoint passes a CPU construction smoke test if memory permits and a GPU smoke test in the standardized environment.

### Acceptance criteria

- The same saved config and checkpoint produce the same model parameters through old evaluation and new inference loaders.
- Deployment never proceeds after partial state-dict loading.
- Training DP/DDP behavior and resume behavior are not regressed.

### Rollback

Keep the new loader unused and restore evaluation imports to their previous functions.

---

## 15. Cut 3: Extract deterministic, label-optional preprocessing

### Context

The existing ISLES26 loader contains the trained preprocessing contract but assumes labels in its common transform chain. Copying those transforms into the container would create two sources of truth. The extracted implementation must serve both labeled repository data and unlabeled external input.

### Dependencies

Cuts 0-2.

### Affected files and components

- `src/data/loader_stack/isles26_loader.py`
- possibly a new shared module such as `src/data/loader_stack/isles26_transforms.py`
- new `src/inference/preprocessing.py`
- `src/inference/contracts.py`
- existing ISLES26 loader tests
- new `tests/test_inference_preprocessing.py`

### Desired changes

1. Extract the deterministic spatial and intensity transform construction from the dataset-specific orchestration.
2. Allow callers to specify image-only or image-plus-label keys.
3. Ensure random training augmentation remains outside the deployment preprocessing path.
4. Preserve MONAI `MetaTensor` metadata and invertible transform history, or define an equivalent explicit trace if a MONAI transform is not invertible.
5. Record original:
   - array shape;
   - affine;
   - voxel spacing;
   - orientation codes;
   - qform/sform information where available;
   - source dtype.
6. Preserve the exact trained intensity operations from the resolved config.
7. Validate finite input values and expected single-modality/channel semantics.
8. Ensure the full-volume repository loader can continue returning its existing tuple contract unless metadata return is explicitly requested.
9. Do not use a DataLoader worker pool for the external single-file inference path.
10. Make both the jointly transformed label and original native label/metadata available to evaluation when their corresponding output space is requested, without changing the default training tuple unnecessarily.

### Expected tests and testing components

- Image-plus-label preprocessing remains numerically equivalent to the current loader for fixed deterministic settings.
- Image-only preprocessing returns the same image tensor as image-plus-label preprocessing.
- Model-space output can be paired with the jointly transformed label, while native-space output can be paired with an original-grid label whose geometry is verified.
- No label path is required in deployment mode.
- All spatial fixture variants retain an invertible trace.
- Training random-patch and augmentation tests remain green.
- ISLES26 2D and 3D loader routing tests remain green.
- Input with wrong modality count, corrupt NIfTI, nonfinite-only data, or unsupported rank fails clearly.

### Acceptance criteria

- Existing training/evaluation datasets and external inference share one deterministic transform definition.
- The deployment path retains sufficient information for exact native-grid restoration.
- No training augmentation is accidentally applied during inference.

### Rollback

Keep the extracted builder behind the existing loader API. If parity fails, restore the inlined loader transform builder while retaining failing fixtures for investigation.

---

## 16. Cut 4: Discriminative probability predictor and sliding-window execution

### Context

`src/utils/valid_utils.py` currently combines validation policy resolution with prediction execution. The new package must own direct and sliding-window probability generation while reproducing current behavior.

### Dependencies

Cuts 0-3.

### Affected files and components

- new `src/inference/predictors.py`
- new `src/inference/sliding_window.py`
- new `src/inference/pipeline.py`
- `src/inference/policy.py`
- `src/inference/runtime.py`
- `src/utils/valid_utils.py`
- `src/diffusion/discriminative_adapter.py`
- `configs/validation/sliding_window.yaml` as a legacy input and migration target
- new `tests/test_inference_discriminative_predictor.py`
- new `tests/test_inference_sliding_window.py`
- existing validation inferer tests

### Desired changes

1. Implement direct 3D discriminative prediction returning final-head probabilities.
2. Preserve DynUNet deep-supervision semantics: training may return multiple heads, but inference uses only the correct final output.
3. Resolve direct/sliding-window behavior exclusively from the active shared `cfg.inference` after legacy translation.
4. Implement MONAI sliding-window execution with explicit:
   - ROI resolution;
   - `sw_batch_size`;
   - overlap;
   - blend mode;
   - padding mode;
   - progress callback optionality.
5. Set GC deployment default `sw_batch_size=1` through the explicit GC inference policy; do not inherit a larger historical training validation override.
6. Run under `torch.inference_mode()` and selected autocast precision.
7. Validate probability domain after prediction.
8. Keep progress/UI concerns injectable so the Docker service does not emit per-window progress bars.
9. Turn `src/utils/valid_utils.py::build_validation_inferer()` into a compatibility wrapper around the new implementation.
10. Translate legacy `cfg.validation.inference` only when no explicit `cfg.inference` is supplied and record which source won.

### Expected tests and testing components

- Direct output matches the current discriminative adapter on fixed tensors.
- Sliding-window output matches the current validation inferer within the Cut 0 tolerance.
- Explicit `cfg.inference` takes precedence as a complete policy over historical `cfg.validation.inference`; there is no field-level leakage.
- Odd input sizes and inputs smaller than the ROI are padded/cropped correctly.
- Window batch one works under FP32 and FP16 on GPU.
- Deep-supervision training and evaluation output ranks are handled correctly.
- NaN/Inf or out-of-domain output fails.
- Existing `test_valid_utils` and model-volume evaluation tests remain green.
- Selected real-case probability parity passes.

### Acceptance criteria

- Training and evaluation can call the new predictor through the compatibility façade without metric drift.
- The predictor contains no label, metric, NIfTI-writing, or socket logic.
- The T4-safe window-batch default is enforced by deployment policy.

### Rollback

Restore the previous body of `build_validation_inferer()` and leave the new predictor isolated until parity issues are resolved.

---

## 17. Cut 5: Migrate offline repository-model evaluation

### Context

Offline evaluation is the scientific certification surface for deployment behavior. It must use the shared predictor and shared inference config before the container is trusted, while retaining its current command-line interface, label handling, metrics, threshold analysis, and reports. It must also make model-space parity evaluation and native-space deployment certification explicit rather than conflating them.

### Dependencies

Cuts 0-4.

### Affected files and components

- `scripts/evaluation/core/model_loader.py`
- `scripts/evaluation/io/model_volumes.py`
- `scripts/evaluation/core/evaluation_pipeline.py`
- `scripts/evaluation/evaluate_model.py`
- `scripts/evaluation/README.md`
- `configs/evaluation/*.yaml`
- `configs/inference/*.yaml`
- `configs/inference_runtime/native.yaml`
- existing `tests/test_evaluation_model_loader.py`
- existing `tests/test_evaluation_io_model_volumes.py`
- existing `tests/test_evaluation_pipeline.py`
- existing entrypoint/integration tests

### Desired changes

1. Replace evaluation-owned model construction/prediction mechanics with `src.inference` calls.
2. Continue obtaining images, labels, case IDs, and evaluation metadata through existing dataloaders.
3. Continue returning `VolumeSample` contracts to the evaluation engine.
4. Compose the active top-level `cfg.inference` and `cfg.inference_runtime=native` instead of defining a second evaluation-specific prediction schema.
5. Preserve fixed, sweep, oracle, and sweep-with-oracle threshold protocols under evaluation config.
6. Preserve current report schemas unless an explicit schema version is incremented.
7. Record shared inference API version, output space, runtime profile, policy source, and policy hash in evaluation provenance.
8. Permit `model_preprocessed` probabilities for historical parity and `native_input` probabilities for deployment certification.
9. Pair model-space prediction with transformed labels and native-space prediction with original-grid labels; validate shapes and geometry before metrics.
10. Treat threshold sweeps as evaluation-only consumers of probabilities. Write the selected fixed threshold into a candidate inference policy only through an explicit export/selection step.
11. Keep current 2D legacy evaluator behavior functional; do not force its migration into this cut.
12. Keep the explicit current rejection of 3D non-discriminative diffusion.

### Expected tests and testing components

- Existing evaluation unit/integration tests pass.
- The selected baseline checkpoint produces probability and metric parity at threshold 0.5.
- Threshold-sweep best-global selection remains stable within defined tie tolerance.
- Native-space evaluation can be run against labels in original geometry where fixtures/data permit.
- A deliberately mismatched output/reference space fails before metric computation.
- The same inference config can run in native Python and `gc_container_test` environments when both profiles allow its requested capabilities.
- Provenance includes exact bundle member, checkpoint hash, config hash, policy hash, and code version.
- Failure to load the requested checkpoint remains clear at the CLI.

### Acceptance criteria

- The canonical evaluation entrypoint uses `src.inference` for 3D model prediction.
- Evaluation remains a metrics/reporting wrapper rather than a second inference implementation.
- Threshold recommendations intended for deployment are generated using the same probability path the container will use.
- Evaluation-specific sweep/oracle config never becomes active inside `gc_submission`.

### Rollback

Retain a short-lived switch to the old model-volume producer until parity is certified. Do not maintain both paths indefinitely.

---

## 18. Cut 6: Migrate training-time validation without changing training

### Context

The main training pipeline directly invokes the current validation inferer. It should consume the same predictor so future inference fixes benefit both training validation and deployment. This cut must not broaden into a trainer rewrite.

### Dependencies

Cuts 0-5, with evaluation parity already green.

### Affected files and components

- `src/training/trainer.py::validate_one_epoch`
- `src/utils/valid_utils.py`
- `src/utils/ensemble.py` where mean aggregation is reused
- `configs/validation/*.yaml`
- new `configs/inference/*.yaml`
- new `configs/inference_runtime/native.yaml`
- validation/runtime contract tests
- training smoke tests

### Desired changes

1. Route ordinary validation probability generation through the shared predictor/compatibility façade.
2. Keep the trainer responsible for:
   - moving labels to the training device;
   - metric object lifecycle;
   - progress display;
   - checkpoint selection;
   - validation logging.
3. Move new prediction execution settings out of validation presets and into composed top-level `configs/inference/` presets.
4. Keep validation config responsible for metrics, cadence, progress reporting, and checkpoint selection.
5. Preserve old run compatibility by translating `cfg.validation.inference` when `cfg.inference` is absent.
6. Preserve existing ensemble behavior for supported modes, but do not enable 3D soft STAPLE.
7. Leave `train_one_epoch`, optimizer, scheduler, loss, gradient scaling, EMA, DP/DDP, and checkpoint writing unchanged.
8. Leave diffusion sampling snapshots training-specific.
9. Optionally route ordinary ensembled preview images through the predictor later, but do not make this a release blocker.

### Expected tests and testing components

- One-epoch or one-batch training validation smoke test completes.
- Metric parity against Cut 0 baseline passes.
- Best-checkpoint metric names and update behavior remain unchanged.
- New training presets compose `cfg.inference`; old saved configs containing only `cfg.validation.inference` still validate identically.
- Explicit inference overrides replace the legacy policy rather than partially merging with it.
- DP/DDP construction tests remain green.
- Existing 3D diffusion runtime rejection tests remain green.
- Snapshot logging still uses the appropriate diffusion-only API.

### Acceptance criteria

- The validation prediction path shares implementation with evaluation/deployment.
- No training-forward or optimization behavior changed.
- Existing training configs require no model-architecture edits.
- New validation configs no longer duplicate prediction-policy fields.

### Rollback

Revert only the trainer validation call site to the compatibility façade's prior implementation; training remains otherwise unaffected.

---

## 19. Cut 7: Native-space probability restoration and output correctness

### Context

The previous competition failure mode—returning a mask aligned to the reoriented input rather than the original image—is a primary design risk. Spatial correctness must be independently certified before postprocessing or container work can be called complete.

### Dependencies

Cuts 0-4; may proceed in parallel with consumer migrations once contracts are stable.

### Affected files and components

- new `src/inference/spatial.py`
- `src/inference/preprocessing.py`
- `src/inference/pipeline.py`
- `src/inference/contracts.py`
- optionally reusable ideas from `scripts/evaluation/io/volume_exporter.py`
- optionally reusable fixtures/patterns from nnU-Net affine tests
- new `tests/test_inference_spatial_roundtrip.py`
- new `tests/test_inference_native_output.py`

### Desired changes

1. Accept the validated `inference.output_space` and retain model-space results without inversion when `model_preprocessed` is selected.
2. Invert model-space probabilities through spacing and orientation transforms when `native_input` is selected.
3. Use continuous interpolation suitable for probabilities during inversion.
4. Restore exact original array shape.
5. Reconstruct or copy the correct affine/qform/sform semantics.
6. Validate corner or sampled voxel world coordinates within a defined tolerance.
7. Reject restoration if the transform trace is incomplete or inconsistent.
8. Threshold in the selected output space and only after restoration when native output is requested.
9. Write binary arrays as `uint8` with values exactly `{0, 1}`.
10. Re-open written files and verify:
   - loadability;
   - shape;
   - affine;
   - dtype;
   - allowed voxel values.
11. Return output-space and spatial validation facts in the result provenance.
12. Make `gc_submission` reject `model_preprocessed`, while `native` and `gc_container_test` may select it.

### Expected tests and testing components

- Round-trip tests for identity, axis permutation, axis flip, anisotropic spacing, translation, odd shapes, and oblique affine.
- Synthetic landmark test: known world-space points occupy the corresponding output locations after preprocess/predict/invert.
- A probability ramp survives continuous inversion more accurately than threshold-first inversion.
- Model-space mode returns the expected transformed grid without claiming native geometry.
- Runtime profile tests enforce that only diagnostic-container/native-Python execution can select model-space output.
- Empty and full masks preserve geometry.
- Output dtype and allowed values are exact.
- Deliberately corrupted trace or affine mismatch fails rather than writing a plausible file.
- If both nibabel and MONAI metadata are used, their affine interpretation is cross-checked.

### Acceptance criteria

- Every native-output spatial fixture produces an output on the original input grid; every model-space fixture remains on and declares the expected transformed grid.
- Shape equality alone is not accepted as proof; affine/world-space checks pass.
- There is no code path that writes a model-space mask to the production Grand Challenge output socket.

### Rollback

No deployment fallback to un-inverted output is permitted. A failure blocks release and is corrected in this cut.

---

## 20. Cut 8: Shared inference enhancements: TTA, ensemble, threshold, and postprocessing

### Context

Training validation, offline evaluation, local/container diagnostics, and competition inference may benefit from invertible TTA, multi-checkpoint ensembling, a fixed decision threshold, and component filtering. These operations belong to the shared inference schema rather than a GC-only policy, remain separate from the saved model configuration, and must be evaluated incrementally. Label-dependent threshold calibration remains an evaluation concern. Baseline single-model inference must remain available.

### Dependencies

Cuts 0-7 and a working canonical evaluation path.

### Affected files and components

- new `src/inference/augmentation.py`
- new `src/inference/ensemble.py`
- new `src/inference/postprocessing.py`
- `src/inference/policy.py`
- `src/inference/runtime.py`
- `src/inference/pipeline.py`
- `configs/inference/*.yaml`
- selected reusable behavior from `src/utils/ensemble.py`
- `scripts/evaluation/` provenance and policy reporting
- new `tests/test_inference_tta.py`
- new `tests/test_inference_ensemble.py`
- new `tests/test_inference_postprocessing.py`

### Desired changes

1. Implement only exactly invertible TTA operations initially, such as configured spatial-axis flips.
2. Apply TTA to preprocessed images, invert each probability prediction, then average probabilities.
3. Implement mean probability aggregation across compatible model members.
4. Do not enable current soft STAPLE for 3D. If generalized later, reduce across every spatial dimension and add explicit 3D tests.
5. Apply one fixed, previously selected threshold from `inference.decision.threshold` in the configured output space and after native restoration when requested.
6. Keep threshold sweeps/global-oracle selection under `evaluation.threshold_protocol`; provide an explicit way to export the chosen fixed value into a candidate inference config.
7. Implement optional connected-component operations in the configured output space.
8. Express minimum component size in physical volume (`mm3`) only when valid spacing metadata exists.
9. Keep `keep_largest_only` disabled by default because multifocal lesions are scientifically plausible.
10. Record every enabled operation, output space, and parameter in output provenance.
11. Add evaluation recipes comparing baseline versus each enhancement independently and in combination in both scientifically valid result/reference spaces.

### Expected tests and testing components

- TTA transform followed by inverse returns the original tensor for fixtures.
- TTA probability combination is order-independent.
- Mean model ensemble produces the arithmetic mean and rejects incompatible members.
- Soft STAPLE configuration for 3D fails explicitly.
- Thresholding occurs in the configured output space and, for native output, only after native-space restoration.
- Threshold sweep config is rejected by `gc_submission` while its selected fixed threshold is accepted through the shared inference config.
- Connected-component volume filtering respects anisotropic spacing.
- Empty masks and multiple lesions are handled safely.
- Evaluation reports can compare baseline, threshold-only, TTA, ensemble, and postprocessing policies.
- Runtime impact is recorded for each enhancement.

### Acceptance criteria

- Every enhancement is opt-in and independently disableable.
- No enhancement enters the release policy without validation evidence on held-out data.
- The final selected policy remains within T4 memory and timeout constraints.

### Rollback

Set all enhancement flags to disabled and use the certified single-model baseline policy.

---

## 21. Cut 9: Deterministic model artifact builder

### Context

Grand Challenge permits model resources to be uploaded separately and expanded under `/opt/ml/model/`. The repository needs a reproducible builder that accepts the same run-directory/checkpoint specification used by evaluation, retains the complete saved config for every member, adds one explicitly selected shared inference config, and produces a self-validating tarball.

### Dependencies

Cuts 1-4; Cut 8 if the final policy schema is included.

### Affected files and components

- new `scripts/gc_submission_builder/__init__.py`
- new `scripts/gc_submission_builder/cli.py`
- new `scripts/gc_submission_builder/build_config.py`
- new `scripts/gc_submission_builder/model_artifact.py`
- new `scripts/gc_submission_builder/release_manifest.py`
- new `scripts/gc_submission_builder/configs/default.yaml`
- `configs/inference/*.yaml`
- `configs/inference_runtime/gc_submission.yaml` as a validation profile, normally image-owned
- existing `scripts/evaluation/core/model_loader.py` checkpoint-discovery conventions
- new `tests/test_gc_model_artifact.py`
- new `tests/test_gc_builder_config.py`

### Desired changes

1. Provide a command that accepts one or more run-dir/checkpoint member specifications and an explicit shared inference-policy path.
2. Copy, never mutate or trim, each complete resolved training config and exact checkpoint into a staging directory.
3. Reject missing `.hydra/config.yaml`, ambiguous checkpoints, unsupported model contracts, duplicate member IDs, or incompatible ensemble members.
4. Generate SHA-256 hashes and a versioned manifest.
5. Validate the staged bundle by loading it through `src.inference` before packaging.
6. Create `algorithmmodel.tar.gz` with contents rooted correctly for `/opt/ml/model/` extraction.
7. Support independent `build-model` and combined `build-all` commands.
8. Make archive generation deterministic where practical by normalizing member order and metadata timestamps, or document any unavoidable nondeterminism.
9. Emit a local build report containing source paths, artifact paths, sizes, hashes, and validation result.
10. Refuse to package optimizer state, datasets, logs, or unrelated training artifacts.
11. Record that the bundled explicit inference policy is active and historical member `validation.inference` blocks are provenance-only.
12. Validate the bundle against the `gc_submission` runtime profile before packaging, including native output, case batch one, fixed threshold, and no evaluation sweep.

### Expected tests and testing components

- Tarball extracts to the expected root layout with no extra enclosing directory.
- Manifest hashes match extracted files.
- Exact requested checkpoint is used.
- Ambiguous/missing checkpoints fail.
- Single-member and compatible multi-member bundles validate.
- Unsupported diffusion or 2D bundle fails for the initial release profile.
- Historical `validation.inference.sw_batch_size=4` remains visible in the copied saved config but does not override a bundled GC policy selecting one.
- Bundle loads when mounted at a temporary `/opt/ml/model` equivalent.
- Repeated builds from identical inputs have identical logical manifests and, if deterministic tar metadata is implemented, identical archive hashes.

### Acceptance criteria

- A model bundle can be rebuilt without rebuilding the Docker image.
- The bundle is fully load-tested before being declared successful.
- No model architecture value is duplicated in builder config.
- Complete training provenance is retained without activating its historical validation policy.

### Rollback

The builder is additive. Delete a failed staged artifact; source run files remain unchanged.

---

## 22. Cut 10: Grand Challenge container runtime and image builder

### Context

The container must translate platform I/O into a call to the already certified shared inference package. It should not become another inference implementation. The runtime and dependency set must be constructed from the environment audit and current platform constraints.

### Dependencies

Cuts 0-7 and Cut 9. The final exact interface manifest is needed for upload certification but fixture interfaces can support development.

### Affected files and components

- new `scripts/gc_submission_builder/runtime/app.py`
- new `scripts/gc_submission_builder/runtime/inference.py`
- new `scripts/gc_submission_builder/runtime/interfaces.py`
- new `scripts/gc_submission_builder/runtime/image_io.py`
- `configs/inference_runtime/gc_container_test.yaml`
- `configs/inference_runtime/gc_submission.yaml`
- new `scripts/gc_submission_builder/container/Dockerfile`
- new pinned runtime lockfile
- current Grand Challenge template build/test/save scripts adapted into the package
- new example interface manifest and synthetic test inputs
- new `tests/test_gc_interfaces.py`
- new `tests/test_gc_image_io.py`
- container smoke/integration tests

### Desired changes

1. Build for Linux amd64 with a current T4-compatible CUDA runtime.
2. Select Python/PyTorch/MONAI/nibabel/SimpleITK versions from:
   - `.sqsh` environment evidence;
   - checkpoint compatibility testing;
   - current Grand Challenge CUDA constraints.
3. Do not blindly install root `requirements.txt`; create a minimal pinned inference lock.
4. Include all runtime dependencies at image build time; assume no runtime network.
5. Run as a non-root user with write access only where required.
6. Preserve the required Grand Challenge invoke label.
7. Load and validate the model bundle during `init_model()` before health becomes ready.
8. Compose the bundled shared inference policy with `inference_runtime=gc_submission` for the production server and fail on incompatible capabilities.
9. Read `/input/inputs.json`, calculate the sorted socket-slug interface key, and dispatch through the interface manifest.
10. For the ISLES26 single interface, require exactly one 3D T1 image and only the other values specified by the official phase.
11. If technical-parameter JSON is required, parse and validate it for transport/provenance only. Do not condition the model on it in this scope.
12. Write exactly the required native-space output socket value under `/output/`.
13. Re-open and validate output geometry, dtype, and values before returning HTTP 201.
14. Return non-success on any model, input, spatial, inference, or output-validation failure.
15. Keep protected filenames and image contents out of logs. Log timings, shapes, dtypes, device, memory peaks, artifact identifiers, policy hash, and runtime profile.
16. Use `/tmp` only for transient scratch files and clean per-case state.
17. Provide commands for `build-image`, `test`, `save`, and `build-all`.
18. Provide a separate container diagnostic command/profile that can evaluate or retain `model_preprocessed` results without weakening the production `/invoke` contract.

### Expected tests and testing components

- `/health` remains non-ready until model initialization succeeds.
- `/invoke` returns the expected platform status and writes one valid output.
- Missing/extra inputs, wrong interface key, multiple files in a single-value socket, and corrupt input fail clearly.
- Model-init failure prevents health readiness.
- Production initialization rejects transformed-space output, threshold sweeps, labels, or debug artifact retention.
- The diagnostic profile can exercise both result spaces using mounted synthetic/reference data without writing invalid data to a production output socket.
- Network-isolated local test succeeds.
- Read-only `/input` and writable `/output` behavior is tested.
- Non-root execution is verified.
- Container label and architecture are verified.
- Model directory is mounted at `/opt/ml/model` rather than copied into the image.
- NIfTI fixture path works. If official interface requires MHA, lossless boundary conversion round-trip tests cover affine/spacing/direction and mask values.
- Restarting the container does not depend on prior `/tmp` contents.

### Acceptance criteria

- The image and model bundle pass the current Grand Challenge template's local test lifecycle.
- The container performs no scientific prediction logic outside `src.inference`.
- The image is independently buildable and saveable without embedding replaceable weights.

### Rollback

Use the last certified image tag and model bundle. Container releases must be immutable and content-addressed in the release report.

---

## 23. Cut 11: T4 resource qualification and ten-minute certification

### Context

Successful local correctness does not prove viability on the competition instance. Training occurs on an A100 40 GB, while submission runs on a T4 16 GB. Resource testing must use case batch one and deployment-equivalent settings.

### Dependencies

Cuts 0-10.

### Affected files and components

- new or updated GC benchmark/test command
- `scripts/test_validation_memory.py` migrated or deprecated
- release manifests/reports
- synthetic and permissible representative validation inputs
- optional SLURM runner for T4-equivalent testing

### Desired changes

1. Measure separately:
   - container/server startup;
   - model initialization;
   - input read;
   - preprocessing;
   - prediction;
   - spatial inversion;
   - postprocessing/output write;
   - total `/invoke` time.
2. Record peak CUDA allocated/reserved memory and peak host RSS.
3. Benchmark FP16 and FP32. Use FP16 as the initial release candidate only after parity and stability tests.
4. Keep `sw_batch_size=1` as the guaranteed-safe baseline.
5. Test representative median and worst-case volume shapes.
6. Run each candidate policy repeatedly to expose allocator fragmentation or state leakage.
7. Test with zero DataLoader workers and bounded CPU thread counts.
8. Add timeout margin; a configuration that merely finishes at 9 minutes 59 seconds is not acceptable. Set a project release target with operational headroom, recommended no more than 8 minutes on the qualification host.
9. Evaluate enhancements cumulatively because TTA and multiple model members multiply runtime.
10. Produce a machine-readable qualification report linked to image and model hashes.

### Expected tests and testing components

- Baseline single-model FP16 prediction stays below 16 GB VRAM with meaningful margin.
- Host memory remains below 32 GB with meaningful margin.
- Worst-case total per-case time remains below the project target and platform limit.
- Repeated invocations do not show unbounded memory growth.
- FP16 versus FP32 probability/metric differences are quantified.
- OOM is caught and reported as failure; no silent CPU fallback occurs in a GPU-certified release.
- Worker/IPC/ancdata failure modes are absent because the container path does not spawn data workers.

### Acceptance criteria

- A release report proves resource compliance for the exact image, model bundle, interface fixture, inference policy, output space, and runtime profile.
- Optional enhancements that violate the budget are disabled even if they improve offline metrics.
- The selected release configuration has explicit runtime headroom.

### Rollback

Disable enhancements, retain `sw_batch_size=1`, or select a smaller compatible model bundle. Do not change trained model attributes in deployment config to manufacture compatibility.

---

## 24. Cut 12: Platform dry run, release packaging, and documentation closure

### Context

The final cut validates the exact interface and platform behavior, not merely a local approximation. It also removes temporary duplicate paths only after the new route is certified.

### Dependencies

Cuts 0-11 and the official ISLES26 interface manifest/starter pack.

### Affected files and components

- `scripts/gc_submission_builder/README.md`
- exact interface manifest/config
- final shared inference config and `gc_submission` runtime profile
- `scripts/evaluation/README.md`
- `scripts/analysis/README.md`
- `scripts/analysis/threshold_analysis.py` only for delegation/deprecation
- `scripts/test_validation_memory.py` only for delegation/deprecation
- release artifacts and checksums

### Desired changes

1. Replace fixture interface values with the exact official ISLES26 socket set and paths.
2. Verify transport format and implement boundary conversion only if required.
3. Run the official local template test with no network.
4. Save the Docker image archive and model tarball independently.
5. Verify both archives after reloading/extracting into clean temporary locations.
6. Upload or test through Grand Challenge's “Try Out Algorithm” facility when available.
7. Compare returned output geometry with the uploaded test input.
8. Record image digest, archive hashes, model hashes, policy, code commit, runtime versions, and platform job identifier.
9. Record the active output space and prove that the production runtime rejected incompatible validation/evaluation options.
10. Document build, model replacement, repository validation, container-diagnostic validation, platform test, troubleshooting, and rollback procedures.
11. Document the migration from `validation.inference` to top-level `inference` and its precedence rules.
12. Mark the older threshold-analysis implementation as delegated/deprecated once feature parity is confirmed.
13. Replace or mark the old memory script as legacy once the release benchmark supersedes it.
14. Retain compatibility wrappers for at least one stable release cycle where they protect existing workflows.

### Expected tests and testing components

- Clean-machine or clean-environment image load and invocation.
- Model-only replacement test: same image, new compatible model bundle.
- Image-only replacement test: same model bundle, new compatible code image.
- Deliberate incompatibility test fails during initialization.
- Grand Challenge try-out job completes successfully.
- Downloaded/returned segmentation has exact allowed values and expected geometry.
- Documentation commands are executed as written.

### Acceptance criteria

- Both upload artifacts are reproducible, independently replaceable, and verified.
- The exact phase interface is implemented; no placeholder slug remains.
- At least one platform-hosted end-to-end case completes before the competition submission deadline.
- A fresh agent can rebuild and validate the submission using repository documentation alone.

### Rollback

Re-select the previous certified image/model pair by recorded digest. Keep all submitted release pairs and reports immutable.

---

## 25. Cut dependency and execution order

```text
Cut 0  Baseline/environment lock
  |
Cut 1  Contracts/shared config/runtime profiles
  |
Cut 2  Model loading
  |
Cut 3  Shared preprocessing
  |
Cut 4  Predictor/sliding window
  |\
  | +--> Cut 7  Spatial restoration
  |
Cut 5  Evaluation migration and dual-space validation
  |
Cut 6  Training validation/config migration
  |
Cut 8  TTA/ensemble/postprocessing
  |
Cut 9  Model artifact builder
  |
Cut 10 Container runtime/image
  |
Cut 11 T4 qualification
  |
Cut 12 Platform/release closure
```

Cut 7 may begin after Cut 4 and proceed alongside Cuts 5-6, but container release work must wait for its acceptance criteria. Cut 8 may be implemented incrementally; no optional enhancement blocks a correct baseline container.

Recommended pull-request granularity is one cut per PR, except very small scaffolding cuts may be combined if their tests and rollback boundaries remain clear.

---

## 26. Testing strategy

### 26.1 Unit tests

- policy validation;
- runtime-profile capability validation;
- top-level inference versus legacy validation-inference precedence;
- fixed output-space vocabulary and result metadata;
- bundle manifest/hash validation;
- checkpoint prefix handling and strictness;
- predictor shape/domain checks;
- sliding-window parameter validation;
- TTA inversion;
- mean ensemble aggregation;
- connected-component filtering;
- interface dispatch and file discovery;
- image read/write validation.

### 26.2 Spatial property tests

- preprocessing/inversion across generated shapes and affines;
- world-coordinate preservation;
- continuous probability inversion before thresholding;
- output shape, affine, dtype, and allowed values;
- physical-volume filtering under anisotropic spacing.
- model-space result/reference pairing and native-space result/reference pairing;
- rejection of cross-space metric comparisons.

### 26.3 Repository integration tests

- existing ISLES26 loader tests;
- existing validation-inferer tests;
- evaluation model loader and volume producer tests;
- evaluation pipeline and threshold protocol tests;
- training validation smoke tests;
- old saved-run config translation tests;
- new top-level inference config composition tests;
- DP/DDP and checkpoint resume regressions.

### 26.4 Scientific parity tests

For a fixed model/checkpoint/input/config:

- old and new preprocessed inputs match;
- old and new model-space probability maps match within the predeclared tolerance;
- threshold-0.5 masks and metrics match;
- threshold sweep selection remains stable;
- native-space results are assessed against verified native labels;
- evaluation of the same active inference config in native and container-test runtime profiles is numerically consistent;
- FP16 versus FP32 differences are reported rather than assumed negligible.

Exact tolerance should be established in Cut 0. CPU/GPU and FP16/FP32 comparisons may require different tolerances, but any metric-relevant disagreement requires investigation.

### 26.5 Container contract tests

- HTTP startup/health/invoke lifecycle;
- invoke label;
- non-root execution;
- read-only input and writable output;
- offline network mode;
- model mount at `/opt/ml/model`;
- clean `/tmp` behavior;
- exact input/output interface paths;
- failure exit/status behavior;
- saved-image reload test.

### 26.6 Resource tests

- peak GPU allocated and reserved memory;
- peak host RSS;
- stage timings and total invoke time;
- repeated-case memory stability;
- worst-case volume dimensions;
- each proposed TTA/model ensemble policy.

### 26.7 Suggested regression commands

Exact file names will emerge with implementation, but the final suite should support commands equivalent to:

```bash
python3 -m pytest tests/test_inference_*.py
python3 -m pytest tests/test_evaluation_*.py
python3 -m pytest tests/test_isles26_*.py
python3 -m pytest tests/test_training_runtime_contracts.py
python3 -m scripts.gc_submission_builder.cli test --config <release-config>
```

Container tests must run on Linux/amd64. Windows-host development may orchestrate Docker, but successful native Python tests on Windows do not replace Linux container certification.

---

## 27. Release artifacts

A release candidate should produce:

```text
release/
  algorithm_image_<version>.tar.gz
  algorithmmodel_<version>.tar.gz
  release_manifest.json
  build_report.json
  resource_qualification.json
  checksums.sha256
  resolved_inference_policy.yaml
  resolved_inference_runtime.yaml
  interface_manifest.yaml
  test_summary.txt
```

The release manifest links:

- Docker image digest and archive hash;
- model bundle hash;
- every config and checkpoint hash;
- repository commit and dirty-worktree status;
- runtime versions;
- interface manifest version;
- inference API/schema versions;
- active output space and runtime profile;
- source of the active policy (`cfg.inference` or translated legacy fallback);
- local and platform test results;
- selected threshold and postprocessing policy;
- resource qualification results.

A dirty worktree does not automatically prohibit development builds, but competition release builds must either use a clean recorded commit or include a reproducible patch digest in the release report.

---

## 28. Risks and mitigations

### 28.1 Spatially plausible but incorrect output

**Risk:** Shape looks correct while affine/orientation is wrong.
**Mitigation:** Invert probability transforms, test world coordinates and landmarks, re-open output, and block release on mismatch.

### 28.2 Training/evaluation/deployment drift

**Risk:** Three implementations slowly diverge.
**Mitigation:** One `src/inference` predictor and preprocessing definition; wrappers retain consumer-specific work only.

### 28.3 Container configuration overrides the trained model

**Risk:** Duplicated architecture/preprocessing keys conflict.
**Mitigation:** Saved config is authoritative; builder and policy schemas explicitly reject model attributes.

### 28.4 Historical validation policy leaks into deployment

**Risk:** A saved run's `validation.inference` values, such as `sw_batch_size: 4`, are partially merged into an explicitly selected T4 policy.
**Mitigation:** Explicit top-level `cfg.inference` replaces the legacy block as a complete policy; legacy translation occurs only when the top-level config is absent, and the winning source is recorded.

### 28.5 Partial checkpoint loading

**Risk:** Compatibility loading silently skips weights.
**Mitigation:** Prefix normalization followed by strict missing/unexpected-key failure for release loading.

### 28.6 T4 OOM

**Risk:** A100-tested settings exceed 16 GB.
**Mitigation:** case batch one, window batch one, FP16 certification, incremental enhancement benchmarks, and no silent fallback.

### 28.7 Ten-minute timeout

**Risk:** TTA/ensemble or I/O pushes invocation over the limit.
**Mitigation:** stage timing, worst-case inputs, an internal target below the platform limit, and policy rollback to baseline.

### 28.8 CPU worker/IPC errors

**Risk:** Multi-worker loading fails with shared-memory or ancillary-data errors.
**Mitigation:** direct in-process single-case file loading; bounded CPU threads; no deployment DataLoader worker pool.

### 28.9 Dependency incompatibility

**Risk:** Old root pins, current CUDA requirements, and MONAI APIs conflict.
**Mitigation:** inspect `.sqsh`, build a minimal inference lock, test checkpoint load and parity in the actual Docker image.

### 28.10 Grand Challenge file-type mismatch

**Risk:** NIfTI assumptions conflict with platform socket validation.
**Mitigation:** official interface manifest release gate and a lossless transport adapter, never a change to model preprocessing.

### 28.11 Over-aggressive postprocessing

**Risk:** Removing small or secondary components deletes true lesions.
**Mitigation:** disabled by default, physical-volume parameters, held-out validation evidence, and policy-level rollback.

### 28.12 Invalid prediction/reference space pairing

**Risk:** Metrics compare a model-space prediction with a native-space label, or vice versa, producing errors or misleading results.
**Mitigation:** Every result and reference declares its space and geometry; evaluation validates both before metric computation.

### 28.13 Premature generic abstraction

**Risk:** Designing for every future model makes the critical path harder to validate.
**Mitigation:** minimal probability predictor contract; one certified discriminative backend first.

---

## 29. Definition of done for this CAP

This CAP is complete only when all of the following are true:

- [ ] `src/inference/` is the canonical 3D discriminative prediction implementation.
- [ ] Training validation, offline evaluation, native Python inference, container diagnostics, and GC deployment use one top-level `cfg.inference` schema.
- [ ] `cfg.inference_runtime` profiles enforce native-Python, diagnostic-container, and submission capabilities without silently rewriting requested policy.
- [ ] Existing saved runs containing only `cfg.validation.inference` remain supported through a tested compatibility translator.
- [ ] Explicit top-level inference config replaces rather than field-merges with historical validation inference settings.
- [ ] Existing evaluation uses it and passes parity tests.
- [ ] Training-time validation uses it without changing training behavior.
- [ ] Label-free ISLES26 preprocessing shares the trained transform definition.
- [ ] Native-space restoration passes shape, affine, and world-coordinate tests.
- [ ] A model tarball can be built from run-dir/checkpoint specifications and loads strictly from `/opt/ml/model/`.
- [ ] Docker image and model bundle build independently and together.
- [ ] The container implements the current Grand Challenge HTTP lifecycle and runs non-root/offline.
- [ ] The exact ISLES26 interface manifest is implemented with no placeholder socket slugs.
- [ ] Output is a valid binary integer segmentation on the input grid.
- [ ] Repository validation can explicitly and correctly evaluate `model_preprocessed` or `native_input` output against a reference in the same space.
- [ ] Container diagnostics can exercise either output space, while production `/invoke` accepts only `native_input`.
- [ ] The exact release candidate passes T4 memory and ten-minute qualification with headroom.
- [ ] At least one Grand Challenge platform try-out succeeds before submission.
- [ ] Release provenance links code, image, runtime profile, output space, model, policy source/hash, interface, and tests.
- [ ] Optional TTA/ensemble/postprocessing is enabled only if supported by offline and resource evidence.
- [ ] 3D diffusion remains explicitly rejected rather than partially or silently supported.

---

# Post-CAP: Deferred 3D Diffusion Integration

## 30. Why 3D diffusion is not included

The inference extraction is an appropriate time to define a backend boundary that can later accommodate diffusion. It is not an appropriate time to add 3D diffusion itself to the competition-critical implementation.

The current limitation is not merely the absence of an inference adapter:

- `src/diffusion/ddpm_sampler.py` constructs noise as `[B, C, H, W]` and broadcasts schedule coefficients with 2D-shaped assumptions.
- `src/diffusion/openai_adapter.py` also constructs `[B, C, H, W]` sample shapes.
- `src/models/MedSegDiff/` uses concrete 2D convolutions and 2D tensor rearrangements.
- `src/models/DiffSwinTr/` contains concrete 2D convolutional conditioning code.
- parts of ORGMedSegDiff are dimension-parameterized, but the currently configured end-to-end adapter/sampler contract is still 2D.
- `src/utils/train_utils.py::validate_training_runtime_contract()` explicitly rejects several 3D non-discriminative diffusion validation/logging combinations.
- `scripts/evaluation/io/model_volumes.py` explicitly rejects 3D non-discriminative live-model evaluation.
- current soft-STAPLE aggregation is 2D-specific.

Consequently, adding 3D diffusion would require coordinated changes to model architecture, sampler shapes, coefficient broadcasting, training forward paths, validation, ensemble semantics, memory management, and experiment configuration. It would also require proving that iterative 3D sampling—possibly multiplied by sliding windows, stochastic samples, and TTA—can finish on a 16 GB T4 within ten minutes.

Those are research and systems questions with their own failure modes. Coupling them to the first professional ISLES26 submission would materially increase the probability of spatial, runtime, and schedule regressions.

## 31. What this CAP deliberately prepares for

The following boundaries should remain backend-neutral:

- tensor predictor protocol;
- probability-domain validation;
- sliding-window orchestration where the backend is patch-compatible;
- TTA application and de-augmentation;
- probability aggregation;
- spatial trace and native-space inversion;
- thresholding and postprocessing;
- Grand Challenge transport and output validation;
- bundle manifest versioning and capability declarations.

A future diffusion implementation should therefore add a predictor backend and extend bundle capability validation rather than fork the container or spatial pipeline.

The shared interface must nevertheless remain honest: until a backend passes 5D tests, `data_mode.dim=3d` plus a non-discriminative `diffusion.type` must fail with a clear capability error.

## 32. Recommended starting points for the future 3D diffusion task

### 32.1 Sampler shape generalization

Inspect and refactor:

- `src/diffusion/diffusion.py`
- `src/diffusion/ddpm_sampler.py`
- `src/diffusion/ddim_sampler.py`
- `src/diffusion/openai_adapter.py`
- `src/diffusion/noise_scheduler.py`

Required contract:

- conditioning input `[B, C_image, D, H, W]` or the repository's chosen canonical 3D axis order;
- noisy/output mask `[B, C_mask, D, H, W]`;
- schedule coefficient extraction broadcasts to an arbitrary number of spatial dimensions rather than a fixed four-dimensional tensor;
- `sample()` returns probabilities aligned with its conditioning patch/volume;
- `sample_with_snapshots()` either supports 5D tensors or explicitly remains unavailable.

The project should first standardize one canonical tensor axis order across data loaders, MONAI, models, samplers, and metrics. Historical comments using both `H,W,D` and `D,H,W` must not substitute for executable shape contracts.

### 32.2 Model architecture decision

Evaluate model families separately:

- true 3D MedSegDiff conversion;
- a 3D-capable ORGMedSegDiff configuration;
- a new 3D conditional diffusion architecture;
- latent or patch diffusion rather than full voxel-space diffusion.

Do not assume that replacing `Conv2d` with `Conv3d` is sufficient. Attention memory, positional structure, downsampling behavior, anisotropy, conditioning, and receptive field must be designed and tested.

### 32.3 Training contract

Add and validate:

- 5D noisy-mask and conditioning inputs;
- 3D patch sampling and target alignment;
- loss behavior and timestep sampling;
- AMP policy on A100;
- checkpoint/resume behavior;
- training-time validation that does not accidentally invoke prohibitively expensive full diffusion sampling;
- explicit logging/snapshot capabilities.

The existing runtime guardrails in `src/utils/train_utils.py` should be removed only one capability at a time as tests demonstrate support.

### 32.4 3D inference strategy

Decide experimentally whether diffusion sampling occurs:

- on a full volume;
- per sliding-window patch;
- in a latent representation;
- as a refinement of a discriminative proposal.

Patch-wise stochastic diffusion raises seam, blend, and stochastic-consistency questions beyond ordinary discriminative sliding-window inference. The future task must specify whether all windows share a reproducible noise field or independent noise and how overlapping stochastic predictions are combined.

### 32.5 3D stochastic ensemble

Generalize ensemble utilities to `[N, B, C, *spatial]` and reduce across all spatial dimensions where estimating expert weights. Begin with arithmetic mean. Treat soft STAPLE or other learned/iterative consensus methods as separate scientifically validated options.

Distinguish:

- multiple stochastic samples from one diffusion checkpoint;
- multiple diffusion checkpoints;
- TTA samples;
- heterogeneous discriminative/diffusion ensembles.

Their provenance and runtime costs must be recorded separately.

### 32.6 Evaluation and runtime feasibility

Before container integration, require:

- unit tests for every 5D sampler operation;
- small synthetic 3D overfit experiment;
- full repository evaluation through `src/inference`;
- comparison against the discriminative baseline;
- T4 profiling for sampling step counts and window policies;
- end-to-end completion under the same ten-minute limit with margin;
- calibration/threshold analysis of the returned probability semantics.

If 3D diffusion cannot meet the T4 limit, it may remain a training/research backend or be distilled into a discriminative deployment model. The common inference contract should not force every research model to become a competition backend.

## 33. Suggested future 3D diffusion CAP cuts

1. **D0: Canonical 5D tensor and probability contract.** Document axis order, shapes, output domain, and capability tests.
2. **D1: Dimension-agnostic scheduler and sampler primitives.** Remove fixed `[B,C,H,W]` and coefficient reshape assumptions.
3. **D2: One certified 3D conditional diffusion architecture.** Implement only one family first.
4. **D3: 3D training smoke and overfit test.** Demonstrate forward/backward/sample correctness on synthetic data.
5. **D4: Shared-inference diffusion predictor.** Implement the `src/inference` predictor protocol without changing spatial/container layers.
6. **D5: 3D validation and ensemble generalization.** Remove runtime guards only for tested features.
7. **D6: Full-volume/sliding-window scientific evaluation.** Establish accuracy, seam behavior, calibration, and stochastic variance.
8. **D7: T4 feasibility gate.** Profile step counts, precision, memory, and total invocation time.
9. **D8: Optional Grand Challenge enablement.** Permit diffusion bundles only after every previous cut passes and the bundle advertises the certified capability.

This future work should be its own PRD/CAP. The present document supplies the integration boundary and evidence pointers, but does not authorize implementation of 3D diffusion as part of the ISLES26 discriminative submission milestone.

---

## 34. Final implementation guidance

The safest first vertical slice is:

1. lock a real baseline;
2. load one DynUNet bundle strictly;
3. preprocess one unlabeled T1 volume through the shared transforms;
4. produce one sliding-window probability map;
5. invert it to native space;
6. write and validate one binary NIfTI;
7. compare that probability/mask against canonical offline evaluation;
8. only then place the same call behind `/invoke`.

This vertical slice proves the most important claim of the architecture: the Docker container is a transport and runtime wrapper around the same inference implementation used to evaluate the model before submission.
