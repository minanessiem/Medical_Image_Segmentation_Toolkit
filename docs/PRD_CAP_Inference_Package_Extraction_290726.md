# PRD and CAP: Shared Inference Package Extraction and ISLES26 Grand Challenge Submission Builder

**Document version:** 1.6
**Original date:** 2026-07-29
**Last revised:** 2026-08-04
**Status:** Draft for implementation
**Primary competition:** ISLES26 3D lesion segmentation
**Primary model scope:** Existing 3D discriminative MONAI DynUNet checkpoints
**Primary implementation areas:** `src/inference/`, selected existing `src/` and `scripts/` modules, and `scripts/gc_submission_builder/`
**Reference experiment configuration:** `configs/cluster_isles26_atlas30_3d_randompatch_dynunet.yaml`

---

## 1. Executive decision

This plan introduces one shared, production-oriented inference implementation under `src/inference/`. That implementation will become the common path by which a trained repository model produces a segmentation probability map. Existing training-time validation, offline evaluation, native inference, container-based diagnostic validation, and the Grand Challenge submission runtime will all become consumers of that implementation.

The same consumers will compose a shared top-level `inference` config family. Sliding-window execution, numerical precision, result space, TTA, probability ensembling, fixed thresholding, and postprocessing will no longer be defined independently for training validation and Grand Challenge inference. Validation/evaluation config will remain separate because it owns labels, metrics, threshold sweeps, checkpoint selection, and training cadence. A third `inference_runtime` config family will declare and enforce the capabilities and constraints of native Python, diagnostic-container, and production Grand Challenge execution.

The first certified release is intentionally limited to the existing 3D discriminative model family required for the ISLES26 submission, initially DynUNet. The reusable loading, preprocessing, inference, and container-building infrastructure must nevertheless select dataset behavior from the saved model's registered dataset contract rather than hardcode ISLES26. It does not add 3D diffusion training or 3D diffusion inference. The shared contract will be narrow enough that a future diffusion predictor can implement it without requiring the spatial, container, or output-writing layers to be redesigned.

The target system has three separate but interlocking release components:

1. **Container infrastructure and code snapshot**: the Linux/amd64 Docker image, pinned Python/CUDA/PyTorch/MONAI runtime, Grand Challenge HTTP lifecycle, and inference code captured at a recorded repository commit.
2. **Shared inference code**: preprocessing, probability prediction through an already prepared model, sliding-window execution, spatial inversion, TTA/ensembling, thresholding, postprocessing, and output validation.
3. **Model artifact archive and model lifecycle support**: one complete saved training config and one selected checkpoint, plus the selected shared inference config and provenance, packaged so Grand Challenge expands them under `/opt/ml/model/`. Shared model-domain code reconstructs the model through the existing factory, loads its weights, and moves it to the requested device. The saved `dataset.id`, modality contract, and preprocessing config select a registered dataset preprocessing adapter. Historical training-validation settings remain present for provenance but are not active when an explicit inference config is selected.

The first certified release must prioritize correctness, spatial fidelity, reproducibility, T4 memory safety, and the ten-minute per-case runtime limit over optional competition enhancements.

Medical-image transport is selected by the competition interface manifest and
is not synonymous with the shared inference result. The existing native-space
NIfTI writer remains a supported reusable transport for submissions and native
workflows that require NIfTI. The initial ISLES26 release additionally requires
compressed MHA materialization of both the restored continuous probability map
and its thresholded binary segmentation; this MHA adapter must reuse, not
replace, the shared native result and NIfTI output architecture.

Post-training 3D evaluation requires richer case information than training-time validation: it may need both the jointly transformed model-space label and the untouched native-grid label, their geometries, and the exact preprocessing trace. This plan therefore does not enlarge the established training tuple or make ordinary training DataLoaders carry release-only metadata. A validation-only normalized record source and reusable typed evaluation-case producer will supply `LabeledPreprocessedCase` records directly to post-training evaluation. Both `model_preprocessed` and `native_input` evaluation will consume that same typed boundary and differ only in result/reference selection after prediction.

---

## 2. Why this work exists

The repository has become a configurable training and evaluation platform for 2D and 3D medical image segmentation. Its central training implementation lives under `src/`, with supporting dataset setup, evaluation, nnU-Net baseline, reporting, analysis, and SLURM tooling under `scripts/`.

The ISLES26 submission introduces a different execution environment from research training:

- one unseen 3D case is processed per container job;
- the hosted algorithm job exposes one NVIDIA T4 GPU with 16 GB VRAM, 16 GB
  requested host memory, and 8 vCPUs; the earlier `g4dn.2xlarge` description
  remains infrastructure context, but release qualification uses the lower
  16 GB limit reported by the actual Grand Challenge job;
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

A shared, separately versioned top-level Hydra/OmegaConf configuration family, represented as `cfg.inference`, that controls how a model and preprocessed image become a probability or segmentation result. It is composed by training validation, offline evaluation, native inference, container diagnostics, and Grand Challenge deployment. Examples include:

- result/output space (`model_preprocessed` or `native_input`);
- runtime precision;
- sliding-window enablement, batch size, overlap, blend mode, and padding mode;
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

Throughout this document, **native** describes where inference or validation
runs: ordinary Python outside the Grand Challenge container. **Repository
model** describes what is being evaluated: a live PyTorch model implemented by
this repository, as opposed to the separate nnU-Net baseline. Accordingly,
`repository-model evaluation` remains valid provenance terminology, while
`repository inference` and `repository validation` are called `native
inference` and `native validation`.

### 3.6 Model artifact archive

The contents of the tarball expanded beneath `/opt/ml/model/`. The first submission archive contains one complete saved config, one exact checkpoint, the selected inference policy, and release provenance. It is a packaging artifact, not a runtime abstraction for a collection of models. The initial runtime loads one model. Any later multi-model representation must be introduced by the ensemble cut from demonstrated requirements rather than anticipated here.

### 3.7 Predictor

The backend-specific component that maps a preprocessed tensor to a probability tensor. It owns interpretation of the model's raw return value, including deep-supervision head selection and the activation required to produce probabilities. Shared inference sees only the architecture-neutral `ProbabilityPredictor` boundary and must not branch on model class or architecture name. The first registered implementation is a 3D discriminative predictor. A future 3D diffusion predictor may implement the same high-level contract after its probability and sliding-window semantics are certified.

### 3.8 Spatial trace

The metadata required to map a prediction from model/preprocessed space back to the exact input grid, including original shape, affine, orientation, spacing, and the invertible transform history.

### 3.9 Grand Challenge interface, socket, and socket slug

An interface is the input/output combination assigned to a challenge phase. Each input or output is a socket. A socket slug is Grand Challenge's identifier for that socket and appears in `/input/inputs.json`; it is not a model attribute and is not invented by this repository. The phase or generated starter pack supplies the actual socket slugs and relative paths.

Socket handling belongs only to the Grand Challenge transport adapter. No core inference module should know an ISLES26 or other competition socket slug.

### 3.10 Registered dataset preprocessing adapter

A repository-owned adapter selected by the saved `dataset.id`. It translates canonical raw dataset modality keys and the saved preprocessing configuration into the deterministic transform chain used for both labeled native validation and label-free inference. ISLES24 and ISLES26 are the initial registered adapters because the existing loader stack already supports both. Dataset-agnostic infrastructure means it can dispatch to any registered adapter; it does not mean an unknown dataset works without an implemented and tested adapter.

### 3.11 Interface binding

A user-supplied, competition-specific manifest entry that maps an arbitrary Grand Challenge input socket slug to a canonical raw modality key understood by the selected dataset adapter. For example, a phase slug such as `t1-weighted-mri` may bind to dataset key `T1`. The same manifest owns output slug/path/type and any technical-JSON bindings. These bindings are transport configuration, not model or preprocessing configuration.

### 3.12 Result key

A result key is the transport-independent identifier for a value exposed by
`PredictionResult`. The initial materializable result keys are:

- `probability`: the finite continuous probability in the configured result
  space, before fixed-threshold binarization;
- `mask`: the final thresholded binary result in that same space.

An output socket slug maps explicitly to a result key. A transport adapter must
not infer output semantics from slug text or list position.

### 3.13 Output artifact set

The output artifact set is the complete ordered collection of output bindings
required by one interface. Order provides deterministic materialization and
reporting, while each binding's `result_key` determines its semantics. A
production invocation returns HTTP 201 only after every declared output has
been written, reopened, and validated. A partially written set is a failed
invocation, never a partial success.

### 3.14 Cut

An independently implementable CAP unit. Every cut below includes its own context, dependencies, affected files, required changes, tests, and acceptance criteria so it can be assigned to an agent with minimal additional conversation history.

### 3.15 nnU-Net export space

The semantic space represented by the NIfTI files produced by one complete
3D nnU-Net conversion preset. It is declared as
`nnunet.export_space` using the same fixed vocabulary as shared inference:
`model_preprocessed` or `native_input`. It describes the grid written by the
converter; it does not select or alter preprocessing.

NIfTI format, file naming, or nnU-Net ownership does not establish this
semantic classification. The conversion preset owns the declaration, the
converter verifies it against the geometry it actually writes, and the
nnU-Net volume evaluator consumes it from the composed conversion config.

---

## 4. Confirmed requirements and external release gates

### 4.1 Project-confirmed requirements

- The submitted model is an existing 3D discriminative segmentation model, initially the DynUNet represented by `configs/cluster_isles26_atlas30_3d_randompatch_dynunet.yaml`.
- The clinical image modality is one 3D T1-weighted MRI volume.
- The project-standard dataset/training input and mask format is `.nii.gz`.
  Grand Challenge image-kind sockets are a separate transport boundary and may
  materialize that source image as MHA or TIFF before container invocation.
- Case batch size is exactly one.
- The target hardware is one T4 with 16 GB VRAM, 16 GB requested host memory,
  and 8 vCPUs, matching the first hosted algorithm-result allocation rather
  than the larger nominal memory of the underlying instance family.
- The maximum execution time is ten minutes per case.
- The model is loaded before per-case invocation whenever the platform lifecycle allows it.
- Model artifacts are separate from the image and reside at `/opt/ml/model/`.
- Model artifact creation accepts a training run directory and a specific checkpoint/snapshot, matching the existing evaluation mental model.
- Existing BF16-trained checkpoints remain eligible. Autocast precision used during training does not encode BF16-only weights into an ordinary PyTorch state dict. Deployment precision is independently selected and tested, initially FP16 on T4 with FP32 fallback for diagnosis.
- Initial sliding-window batch size is one even if the training validation config used a larger window batch.
- Training validation, offline evaluation, native/container diagnostics, and production deployment share the same `inference` configuration schema.
- Native and diagnostic-container execution may select `model_preprocessed` or `native_input` result space when prediction and reference labels are paired in the same space.
- Production `gc_submission` execution requires `native_input` output, case batch size one, and a fixed deployment threshold. It rejects label-dependent threshold sweeps.
- The final segmentation result is binary with values `{0, 1}` and an integer
  voxel type, preferably `uint8`.
- Every materialized result must match the selected result space and its
  verified physical geometry. The production ISLES26 outputs match the
  original T1 input grid.
- NIfTI native-output materialization remains supported and tested. A
  competition-specific MHA requirement adds a boundary adapter and does not
  remove, redirect, or weaken the NIfTI writer.
- The official ISLES26 interface requires both the native-space binary lesion
  segmentation and the corresponding restored continuous lesion probability
  map.
- Generative inference, metadata-conditioned inference, and newly trained model families are outside the first submission scope.
- The GC builder and runtime are dataset-agnostic across registered repository dataset adapters. The first certified release remains ISLES26-specific only in its selected model artifact and phase interface manifest.
- Dataset construction exposes `load_labels: bool = True`. `test_flag`, which currently selects a data partition, remains independent and never implicitly enables or disables label loading.
- Native blind inference and GC inference use `load_labels=False`; training, native validation, and repository-model evaluation retain `load_labels=True` unless explicitly performing blind inference.
- Native geometry is captured independently for every case before preprocessing and before raw modality metadata is removed during channel merging. Different cases may have different shapes, spacings, orientations, and affines.
- Modalities within one case are expected to satisfy the dataset's alignment contract. This CAP does not add a new pre-preprocessing cross-modality alignment validator.
- Every supported 3D nnU-Net conversion preset explicitly declares
  `nnunet.export_space`; no dataset-name, file-format, or evaluation-time
  fallback may infer it.
- A 3D nnU-Net converter writes the affine/spacing/orientation of the tensor it
  actually exports. A transformed tensor must never be written with its
  untransformed source affine.
- Geometry-aware evaluation in the current CAP accepts 3D volumes only. Both
  repository 2D slice producers and nnU-Net 2D conversion/evaluation require a
  separate reconstruction contract and fail early at the new spatially aware
  boundary.

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
- <https://github.com/ezequieldlrosa/isles26-docker-template>

### 4.3 Dataset-agnostic interface manifest and resolved ISLES26 contract

On 2026-07-31, the competition organizer published the official ISLES26 Docker
template. This CAP pins the initial reconciliation to template commit:

```text
5e25bfc36b1dc6d9c04c8c364f53fb75c6afad32
```

The published `interf0` input socket set is:

| Socket slug | Relative path | Repository binding |
|---|---|---|
| `t1-brain-mri` | `images/t1-brain-mri` | canonical raw dataset key `T1` |
| `stroke-metadata` | `stroke-metadata.json` | required transport/provenance JSON; not model conditioning in this release |

The published required output socket set is:

| Socket slug | Relative path | Result key | Initial transport |
|---|---|---|---|
| `stroke-lesion-segmentation` | `images/stroke-lesion-segmentation` | `mask` | compressed MHA |
| `lesion-probability-map` | `images/lesion-probability-map` | `probability` | compressed MHA |

The metadata object contains `CENTER`, `CHRONICITY`, and
`DAYS_POST_STROKE`. Published examples establish that values may be null.
The manifest declares their required keys, types, and nullability. The
transport adapter validates this structure but does not log identifying values
or pass them to preprocessing/model inference in the current unconditioned
model scope.

The organizer README instructs local tests to place a `.nii.gz` T1 beneath the
published image path, but the hosted challenge interface declares an image-kind
socket read from `/input/images/t1-brain-mri/<uuid>.mha` or
`/input/images/t1-brain-mri/<uuid>.tif`. Dataset format and platform transport
format are therefore separate contracts. NIfTI remains the canonical internal
input and the existing native output transport for compatible consumers; the GC
adapter must discover the platform image, validate it, and losslessly
canonicalize MHA/TIFF to invocation-local `.nii.gz` scratch before calling the
registered preprocessing adapter. This conversion must preserve voxel values
and physical geometry and must not change trained preprocessing, spatial
restoration, thresholding order, or `PredictionResult` semantics.

The organizer's ISLES26 example writes both official outputs as compressed MHA,
so Cut 10B adds an MHA output adapter alongside, not in place of, NIfTI. MHA is
the selected ISLES26 output even though the platform also permits TIFF; no TIFF
output writer is required for this release.

The official local lifecycle calls the HTTP service from outside the algorithm
container and applies a 300-second `/invoke` timeout. This is a local interface
compatibility bound, not a replacement for the confirmed ten-minute total
platform-job limit. Cut 11 measures both and retains project headroom.

Each input binding still maps an arbitrary phase slug to a canonical raw key
required by the selected registered adapter; slugs are never inferred from
model semantics. Each output binding maps a phase slug to an explicit result
key; output semantics are never inferred from list order or slug text. Missing,
duplicate, extra, unknown, or model-incompatible bindings fail before success.

This remains an I/O compatibility concern rather than an ISLES26 dependency in
shared inference. Other competitions may select NIfTI, MHA, or another
implemented transport and may declare different output sets while reusing the
same model/result pipeline. Fixture manifests retain deliberately opaque slugs
to prove that no official slug is hardcoded in Python.

An early Grand Challenge-hosted try-out was executed on 2026-08-04 as algorithm
result `148ccec2-c194-49fb-a5ce-4fecb46b67c7`. The container initialized CUDA,
the DynUNet model, checkpoint, dataset adapter, and inference policy, then failed
before preprocessing with `InterfaceManifestError`. The submitted Cut 10B image
searched the image-kind directory exclusively for `.nii.gz`; the hosted
interface instead materializes MHA or TIFF. The original log suppressed the
exception detail, so it did not retain whether that case used `.mha` or `.tif`,
but it conclusively exposed the invalid NIfTI-only platform-input assumption.

Cut 10B-2 corrects this hosted transport contract and adds stage-aware,
privacy-safe observability. A later hosted try-out remains required to confirm
successful input canonicalization and acceptance of both official MHA outputs.
Hosted differences are resolved in the GC boundary adapter without altering
core preprocessing or inference semantics.

---

## 5. Goals

### 5.1 Functional goals

1. Provide a reusable `src/inference/` package that accepts a prepared repository model/predictor and returns calibrated probability maps for 3D discriminative segmentation.
2. Provide shared model-domain artifact and loading code that reconstructs a trained repository model without coupling model lifecycle behavior to inference execution or an evaluation script.
3. Make training-time validation and offline evaluation consume the same low-level prediction implementation.
4. Provide deterministic, label-optional preprocessing through dataset adapters selected from the saved model's registered `dataset.id`, initially covering ISLES24 and ISLES26 without duplicating their trained transform definitions.
5. Preserve per-case native spatial metadata before preprocessing and sufficient transform history to invert prediction probabilities to the original input grid.
6. Support direct and MONAI sliding-window inference with deployment-safe defaults.
7. Introduce one shared top-level inference config family for training validation, offline evaluation, native inference, container diagnostics, and GC deployment.
8. Introduce separate evaluation and runtime-profile configs so shared inference options can be reused without allowing label-dependent or platform-invalid combinations.
9. Support optional probability-space TTA, multi-checkpoint/model ensembling, thresholding, and conservative postprocessing through the shared inference policy.
10. Build and test a Grand Challenge-compatible Docker image containing the runtime and repository code snapshot.
11. Build a replaceable model tarball from one run-directory/checkpoint specification.
12. Build the image and model artifact archive independently or together.
13. Produce machine-readable provenance and resource measurements for every release candidate.
14. Keep Grand Challenge socket slugs and file layout at the transport boundary through an explicit user-supplied mapping to canonical dataset modality keys.
15. Give post-training 3D evaluation a validation-only normalized record source and reusable typed case producer without constructing training datasets, samplers, worker pools, or augmentation pipelines.
16. Preserve lightweight training dataset items while giving evaluation explicit access to model-space and native-space labels, geometries, and spatial traces.

### 5.2 Quality goals

- Prediction parity between the pre-extraction and post-extraction evaluation paths within defined numerical tolerance.
- Exact native-output shape and verified affine/world-space consistency with the input; explicit transformed-grid metadata for model-space results.
- Strict, fail-fast validation of the eventual release model artifact.
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
- Unifying live repository PyTorch model execution with the external nnU-Net runtime.
- Generic multi-GPU inference in the Grand Challenge container.
- Case batching greater than one for geometry-aware post-training 3D evaluation or Grand Challenge invocation. Sliding-window window batching remains a separate inference-policy concern.
- Serving multiple simultaneous requests.
- DICOM ingestion.
- Lesion-volume-conditioned postprocessing research.
- Automatic selection of a competition threshold from hidden test data.
- Enabling an enhancement such as TTA, ensembling, or component filtering without validation evidence.
- Automatic support for an unregistered or structurally unknown dataset. Adding a dataset requires an explicit repository adapter and parity tests.
- Adding new proactive cross-modality alignment checks before preprocessing; within-case modality alignment remains a dataset contract.

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

`src/data/loader_stack/isles24_loader.py` and `src/data/loader_stack/isles26_loader.py` already implement closely related MONAI preprocessing through `LoadImaged`, `Orientationd`, `Spacingd`, shared modality processing, channel merging, padding, and `EnsureTyped`. The dataset registry in `src/data/loader_stack/registry.py` already distinguishes both dataset identities and their supported loader modes. These are the repository's current preprocessing sources of truth.

Both common transform builders currently hardcode `label` into the keys loaded and spatially transformed. The ISLES26 datalist normalization also requires `label` for every record. This reflects the older training/validation-only assumption and must be updated at dataset construction rather than worked around in the GC adapter. `test_flag` already controls train-versus-validation/test partition selection and still returns labels; it must not be repurposed as a label-loading switch.

`MergeProcessedChannelsTransform` removes raw modality keys and their associated metadata after producing the merged `image`. Native metadata must therefore be captured per case and per raw modality before preprocessing begins, and at the latest before that merge removes the source metadata. The full-volume datasets currently return tensors in a form optimized for training/evaluation rather than external spatial inversion.

The transform definition is valuable and must be reused. The problems to solve are:

- add an explicit, backward-compatible `load_labels: bool = True` dataset-construction contract;
- make transform keys and datalist validation conditional on that contract, independently of `test_flag`;
- preserve `MetaTensor` or equivalent transform history;
- expose original image metadata separately for each case and raw modality;
- separate deterministic preprocessing from training-only random augmentation;
- route preprocessing by registered `dataset.id` rather than an ISLES26-only branch;
- provide a tested inverse operation for probability maps.

### 7.3 Existing prediction execution

`src/utils/valid_utils.py::build_validation_inferer()` currently owns direct versus sliding-window inference and calls `diffusion.sample()` for a full input or each window.

`src/training/trainer.py::validate_one_epoch()` calls that inferer and optionally ensembles repeated samples before metric computation.

`scripts/evaluation/io/model_volumes.py` also calls the same validation inferer for 3D live-model evaluation. It correctly rejects current 3D non-discriminative diffusion because the active diffusion samplers are 2D-shaped.

Current raw-output ownership is already suitable for an architecture-neutral extraction:

- `src/models/model_factory.py` owns architecture construction/registration for DynUNet, SwinUNETR, and other model types.
- The DynUNet model adapter may expose stacked deep-supervision logits during training and a single logit tensor during evaluation; the SwinUNETR adapter exposes a single logit tensor.
- `src/diffusion/discriminative_adapter.py` handles single, list, or stacked discriminative returns generically, selects the final inference head, and applies sigmoid to produce probabilities. It does not branch on a concrete model class.
- `src/losses/discriminative_deep_supervision.py` owns the corresponding generic training-loss interpretation of single/list/stacked heads.

None of this behavior should move into shared inference. Cut 4 characterizes it and consumes the resulting probability boundary. Architecture-specific construction stays in `src.models`; deep-supervision interpretation and activation stay backend-owned.

This is the core extraction opportunity: characterize and move the reusable prediction behavior into `src/inference/`, then migrate the repository's internal consumers directly. Because no active training jobs require the old inference entrypoint during this transition, Cut 4 removes the inference-specific resolver/builder implementation from `valid_utils.py` rather than retaining two internal paths. Unrelated validation helpers in that module remain in place. Compatibility adapters remain appropriate elsewhere only when an actual historical or external consumer requires them.

### 7.4 Existing model loading

`scripts/evaluation/core/model_loader.py` provides checkpoint discovery and repository model construction for evaluation. `src/utils/train_utils.py::build_model_and_diffusion()` performs related construction with training-specific DP/DDP wrapping and data-contract channel synchronization.

The model domain needs a shared home for the existing evaluation model/diffusion construction and checkpoint-loading behavior while distributed training construction remains training-owned. The first step is a physical ownership transfer into `src/models/` with legacy facades and no behavior change. Release-only strictness, artifact validation, and any additional preparation policy are introduced later at their actual deployment boundary. `src/inference/` consumes an already constructed model through its predictor boundary and does not discover checkpoints or own model artifact hashes.

### 7.5 Existing probability and threshold analysis

The current evaluation package under `scripts/evaluation/` already provides the canonical metrics, threshold protocols, per-case/global/oracle analysis, reports, and model-volume producers. These capabilities remain outside the core inference package but will consume its probability outputs.

`scripts/analysis/threshold_analysis.py` contains an older duplicate config/checkpoint/inference path. It should not become a third production consumer. After parity is established, it should delegate to the current evaluation package or be formally deprecated.

### 7.6 Existing ensemble behavior

`src/utils/ensemble.py::mean_ensemble()` averages over the ensemble dimension and is spatial-dimension agnostic in implementation.

The existing soft-STAPLE code reduces over only the final two axes. It is therefore not valid as written for `[N, B, C, D, H, W]` volumes. Soft STAPLE must remain disabled for 3D until it is generalized and tested across all spatial dimensions.

### 7.7 Existing ancillary packages

- `scripts/test_validation_memory.py` directly samples models. It should later delegate to the shared predictor or be superseded by the release resource benchmark.
- `scripts/nnunet/` remains a separate external model/runtime pipeline and is
  not forced through `src/inference`. Its 3D converter currently exports
  tensors produced by repository dataloaders but reconstructs `export_affine`
  from the raw source image. That is valid for a grid-preserving export but is
  incorrect when the loader has reoriented or resampled the tensor. The
  Cut-5 nnU-Net precursor corrects this and makes export space explicit before
  the common evaluator consumes nnU-Net volumes.
- The current nnU-Net 2D converter emits individually resized slice NIfTIs and
  per-slice provenance, but neither that representation nor the downstream
  slice adapter proves a complete invertible parent-volume geometry. It remains
  outside the geometry-aware 3D evaluation scope alongside repository 2D
  slice producers.
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
  artifact_manifest.json
  config.yaml
  weights.pth
  inference_policy.yaml

                     Shared config composition
       +-------------------+-----------------------+
       |                   |                       |
 cfg.inference     cfg.inference_runtime    validation/evaluation
 prediction/result   capability limits      labels/metrics/sweeps
       |                   |                       |
       +-------------------+-----------------------+
                           |
Input transport            |          Post-training 3D evaluation
(/input, inputs.json)       |          (validation-only case records)
         |                  |                       |
         v                  |                       |
GC interface manifest      |                       |
slug -> raw dataset key     |                       |
         |                  |                       |
         +------------+-----+-----------------------+
                      v
       registered dataset preprocessing adapter
       selected by saved dataset.id; load_labels explicit
                      |
                      v
        typed case producer (pipeline built once)
        PreprocessedCase | LabeledPreprocessedCase
                      |
                      v
                 src/inference
                shared pipeline
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
| Deterministic transform construction for a known dataset | Registered adapter in `src/data/loader_stack/` | Reuse the repository's trained preprocessing source of truth for labeled and label-free paths |
| Validation-only case discovery and subset routing | Normalized record source in `src/data/loader_stack/` | Post-training evaluation must not construct training datasets, samplers, augmentations, workers, or require a non-empty training partition merely to discover validation cases |
| Label loading | Explicit dataset-construction argument `load_labels` | Preserve labeled defaults while supporting blind inference without coupling behavior to `test_flag` |
| Per-case native image geometry | Preprocessing request/result contracts | Native restoration cannot rely on a dataset-wide reference geometry |
| Rich post-training evaluation case | Reusable typed case producer in `src/inference/` | Build deterministic preprocessing once and expose model image, trace, both label spaces, and both geometries without inflating training tuples |
| Exact weights | Model artifact archive | Independently replaceable artifact |
| Model artifact validation, reconstruction, weight loading, and device preparation | `src/models/` | Shared model lifecycle behavior must not depend on inference or evaluation frontends |
| Output space, precision, sliding-window policy, TTA, ensemble, fixed threshold, postprocessing | Shared `inference` config | One prediction policy across all consumers |
| Metrics, threshold sweeps/oracles, validation cadence, checkpoint selection | `validation` / `evaluation` config | Label-dependent assessment is not inference |
| Case batch, workers, timeout, device and allowed capabilities | `inference_runtime` profile | Location/mode constraints must not duplicate scientific policy |
| Probability prediction mechanics | `src/inference/` | Shared scientific behavior |
| Pairing result and reference-label space | Evaluation/training consumers with shared contracts | Metrics are valid only when grids match |
| Full-volume post-training case batching | Evaluation request boundary | Geometry-aware 3D evaluation processes one complete case at a time; `inference.sliding_window.sw_batch_size` independently controls windows within that case |
| Semantic space and written geometry of a 3D nnU-Net dataset | Complete nnU-Net conversion preset plus converter | `nnunet.export_space` declares meaning; the exporter must write and verify the transformed/native grid it actually produced |
| `/input`, `/output`, `inputs.json`, socket slugs, slug-to-canonical-key/result-key bindings, required output sets, transport formats, HTTP statuses | GC adapter and interface manifest | Platform transport only; shared preprocessing sees canonical dataset keys and shared materialization sees result keys, never competition-specific slug semantics |
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
  preprocessing.py          # dataset-adapter dispatch and typed preprocessing requests/results
  case_producer.py          # reusable producer for rich post-training evaluation cases
  predictors.py             # architecture-neutral protocol, validation, and backend registration
  sliding_window.py         # MONAI sliding-window orchestration
  augmentation.py           # invertible TTA definitions and de-augmentation
  ensemble.py               # probability-space model/TTA aggregation
  spatial.py                # inverse transforms and native-grid validation
  postprocessing.py         # threshold and optional binary morphology/filtering
  pipeline.py               # end-to-end orchestration independent of transport

src/models/
  model_factory.py          # existing architecture construction registry
  model_config.py           # model/data channel contract helpers moved from training
  checkpoint_loading.py     # existing state-dict extraction/normalization/loading
  model_loader.py           # existing single-model evaluation construction/loading

src/data/loader_stack/
  registry.py               # registered dataset capabilities and preprocessing-adapter lookup
  record_source.py          # normalized validation-only record discovery without DataLoaders
  preprocessing.py          # shared deterministic builder/contracts if extraction proves useful
  isles24_loader.py          # ISLES24 adapter plus existing dataset orchestration
  isles26_loader.py          # ISLES26 adapter plus existing dataset orchestration

scripts/gc_submission_builder/
  __init__.py
  README.md
  cli.py                    # build-image, build-model, build-all, test, save
  build_config.py           # builder-only config validation
  model_artifact.py         # deterministic artifact staging/tar creation
  release_manifest.py       # code/runtime/model provenance and hashes
  runtime/
    app.py                  # /health and /invoke server integration
    inference.py            # init_model() and per-case run()
    interfaces.py           # inputs.json/socket/path dispatch
    image_io.py             # input canonicalization plus output transport validation
    observability.py        # privacy-safe structured events, stage timing, and resources
  container/
    Dockerfile
    requirements.lock
    do_build.*
    do_test_run.*
    do_save.*
  configs/
    default.yaml
    interfaces/
      fixture_single_nifti.yaml
      isles26.yaml
  tests_smoke/
    fixtures/...

configs/inference/
  direct_model_space.yaml
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
`direct_model_space` inherits the same complete model-space policy and changes
only `sliding_window.enabled=false`; it preserves established direct 2D
training validation without introducing a dimension-specific direct preset.
inference policy neither duplicates 2D/3D model structure nor requires changes
to established data-mode config families. The 2D-compatible config contract
does not add a diffusion predictor or expand the initial ISLES26 3D backend
scope.

Where practical, use the current official Grand Challenge template files rather than creating a novel server implementation. The repository should own and test the inference handler and interface adapter even if `app.py` remains close to the template.

### 8.3 Public inference contracts

The first implementation should expose a small public surface rather than every internal helper. Conceptually:

```python
model, missing_keys, unexpected_keys = load_model(
    cfg=saved_cfg,
    checkpoint_path=Path("/opt/ml/model/weights.pth"),
    device=torch.device("cuda:0"),
)
predictor = build_probability_predictor(model)

case_producer = build_case_producer(
    dataset_id=saved_cfg.dataset.id,
    dataset_cfg=saved_cfg.dataset,
    load_labels=False,
)
preprocessed = case_producer.preprocess(
    {"caseID": case_id, **canonical_raw_inputs}
)

result = predict_case(
    predictor=predictor,
    preprocessed_case=preprocessed,
    inference_cfg=cfg.inference,
    runtime_cfg=cfg.inference_runtime,
)
```

Model construction, checkpoint loading, and device placement are model-domain operations.
The inference API begins from prepared predictors and should not know how their
config and checkpoint paths were discovered. Preprocessing begins from canonical raw dataset keys, not GC socket slugs, and dispatches through the adapter registered for the saved `dataset.id`. Its typed `PreprocessedCase` result contains the processed image, case identifier, native metadata, model-space geometry, and spatial trace. When `load_labels=True`, the required `LabeledPreprocessedCase` extension exposes both the jointly transformed model-space label and the untouched original native label, with independent verified geometry for each. When `load_labels=False`, `PreprocessedCase` contains no dummy, zero, or `None` label field. The producer constructs the deterministic adapter pipeline once and reuses it across records; `preprocess_case()` may remain as a one-case convenience facade. The returned prediction result
should distinguish at least:

- the configured primary result in `model_preprocessed` or `native_input` space;
- optional diagnostic intermediate probabilities when the runtime profile permits them;
- final binary mask in the configured result space;
- spatial trace and geometry checks;
- model and policy provenance;
- timing and memory measurements when instrumentation is enabled.

The predictor-level contract should operate on tensors and return probabilities in `[0, 1]`, with explicit `[B, C, *spatial]` shape semantics. Backend selection may dispatch through an explicit registered backend family/capability, but never through DynUNet, SwinUNETR, or other model-architecture identities. Backend adapters own raw model-output interpretation, deep-supervision head selection, and activation. Unsupported configured backends fail during predictor preparation, before direct/sliding-window execution. The predictor should not read files, calculate metrics, or write medical-image outputs. NIfTI and MHA materialization are transport/output concerns around the shared result, not predictor behavior.

### 8.4 Model artifact archive layout and manifest

The model tarball should expand directly into:

```text
/opt/ml/model/
  artifact_manifest.json
  config.yaml
  weights.pth
  inference_policy.yaml
```

The manifest should contain generated facts, not duplicate model configuration:

```json
{
  "inference_api_version": 1,
  "created_at_utc": "...",
  "code_commit": "...",
  "source_run": "...",
  "source_checkpoint": "...",
  "config_path": "config.yaml",
  "weights_path": "weights.pth",
  "config_sha256": "...",
  "weights_sha256": "...",
  "inference_policy_path": "inference_policy.yaml"
}
```

Absolute source paths may be recorded in a local build report but should be optional or redacted from the distributable manifest if they reveal infrastructure details.

`config.yaml` is the complete resolved training config, including its historical validation section. Keeping it provides reproducibility and avoids prematurely inventing a reduced model-config schema. When the archive supplies `inference_policy.yaml`, that explicit shared policy is active; historical `config.yaml::validation.inference` values are retained but are not merged into or allowed to override it.

The initial builder command accepts one pair of:

- training run directory;
- checkpoint/snapshot name or exact checkpoint path within that run.

Checkpoint discovery may reuse evaluation conventions, but the resolved output must identify one exact file. Ambiguous matches are errors.

By default, the archive filename is the complete selected training-run directory
name with `.tar.gz` appended. The builder does not abbreviate or synthesize a
different model name. An explicit builder override may select another root-level
`.tar.gz` filename.

The final archive is created with root-relative contents, equivalent to:

```bash
tar -czvf <training-run-directory-name>.tar.gz -C /path/to/algorithmmodel .
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

The `gc_submission.timeout_seconds: 600` value represents the confirmed
ten-minute total case/job ceiling. The organizer local compatibility harness
independently limits the HTTP `/invoke` call to 300 seconds. Container testing
must enforce that stricter local call bound without rewriting the runtime
profile's total platform budget.

Cross-config validation applies runtime constraints to the requested inference and evaluation configs. It must reject incompatible combinations rather than silently coerce them. Examples include transformed-space output under `gc_submission`, case batch size greater than one, or a label-dependent threshold sweep during `/invoke`.

### 8.7 Interface manifest schema

The concrete parser names may be refined in Cut 10B, but the manifest must
express input and output separation explicitly. Fixture slugs are intentionally
not canonical modality or result names:

```yaml
interfaces:
  - name: fixture-single-nifti
    inputs:
      - slug: fixture-image-socket
        dataset_key: T1
        relative_path: images/fixture-input
        kind: image
        accepted_formats: [nii_gz]
        canonical_format: nii_gz
        cardinality: one

    technical_inputs: []

    outputs:
      - slug: fixture-segmentation-socket
        result_key: mask
        relative_path: images/fixture-output
        file_type: nifti
```

The first official manifest is:

```yaml
interfaces:
  - name: isles26
    inputs:
      - slug: t1-brain-mri
        dataset_key: T1
        relative_path: images/t1-brain-mri
        kind: image
        accepted_formats: [mha, tif, tiff, nii, nii_gz]
        canonical_format: nii_gz
        cardinality: one

    technical_inputs:
      - slug: stroke-metadata
        relative_path: stroke-metadata.json
        file_type: json
        required: true
        schema:
          CENTER: {type: string, nullable: true}
          CHRONICITY: {type: integer, nullable: true}
          DAYS_POST_STROKE: {type: number, nullable: true}

    outputs:
      - slug: stroke-lesion-segmentation
        result_key: mask
        relative_path: images/stroke-lesion-segmentation
        file_type: mha
      - slug: lesion-probability-map
        result_key: probability
        relative_path: images/lesion-probability-map
        file_type: mha
```

`dataset_key` is selected from the registered adapter's canonical raw keys; it
is not derived from `slug`. `result_key` is selected from the shared prediction
result contract; it is not derived from slug text or output position. The list
order is deterministic but never semantic. Duplicate slugs, dataset keys,
result keys, or relative paths are invalid within an interface.

Input `kind: image` describes the platform socket rather than the dataset's
original storage format. `accepted_formats` is an explicit transport allowlist;
`canonical_format` declares what the registered preprocessing adapter receives.
The ISLES26 adapter accepts the hosted MHA/TIFF forms plus NIfTI forms used by
the organizer's local harness and reusable fixtures, then exposes only canonical
`.nii.gz` paths downstream. Output `file_type` continues to select a concrete
materializer.

Technical schemas are transport validation, not model conditioning. Model
architecture, processed channel names, preprocessing parameters, ROI,
threshold, output foreground/background encoding, and precision are invalid in
this manifest. The transport file type selects an implemented boundary writer.
Adding MHA therefore does not remove or redirect the NIfTI writer. A multimodal
registered dataset supplies one binding per required raw modality; another
competition supplies the exact output set it requires.

### 8.8 Validation/evaluation composition

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

Geometry-aware post-training 3D evaluation processes complete cases one at a time in both spaces. `validation.val_batch_size`/the effective evaluation case batch must therefore be one at the evaluation request boundary; this is not a native-space or Grand Challenge-only limitation. It also does not mean one case per process or job: an evaluation run may iterate an entire split sequentially. `inference.sliding_window.sw_batch_size` remains independent and controls how many windows from the current case are sent through the predictor together. The shared `configs/validation/default.yaml` uses case batch one; any explicitly larger specialized training-validation preset is not accepted by the geometry-aware post-training 3D evaluator unless overridden to one.

### 8.9 Backward compatibility for existing runs

Historical saved runs contain prediction settings under `cfg.validation.inference`, including the current representative DynUNet run. They remain valid inputs.

The migration resolver uses this precedence:

1. explicit top-level `cfg.inference` supplied for the current job;
2. legacy `cfg.validation.inference` translated into the shared schema;
3. a sliding-window policy derived from the saved model/data ROI contract when
   neither policy block exists.

When an explicit top-level inference config exists, values are not merged field-by-field with the historical validation inference block. This is essential to prevent settings such as training-time `sw_batch_size: 4` from leaking into a T4 policy that explicitly selects one.

Legacy translation emits provenance and, eventually, a deprecation warning. It does not rewrite the saved run config on disk.

### 8.10 End-to-end prediction order

The canonical operation order is:

1. Validate the input transport and resolve the active interface manifest.
2. Map each input socket slug to its declared canonical raw dataset key; reject missing, duplicate, extra, or unknown bindings.
3. Select the registered preprocessing adapter from the saved model's `dataset.id` and validate that the bound raw keys can produce every saved processed modality.
4. Read each raw modality and capture that case's native metadata before any preprocessing or channel merge. Modalities within the case are trusted to satisfy the dataset alignment contract; this CAP adds no separate alignment precheck.
5. Apply deterministic trained preprocessing with explicit `load_labels` behavior while retaining an invertible trace.
6. Execute each allowed TTA view.
7. Predict probabilities using direct inference or, for sliding-window inference, make the backend predictor produce a probability for every window before MONAI blends overlapping windows. Probability-before-blending is the parity-preserving baseline; blending logits and applying an activation afterward is a separate future experimental policy, not an implicit refactor.
8. Invert each TTA transform in model space.
9. Mean-combine TTA and/or model-member probabilities.
10. If `output_space=native_input`, invert the combined floating-point probability map to the selected native reference grid and validate shape and physical geometry.
11. If `output_space=model_preprocessed`, retain the combined probability on the model grid and record its model-space geometry.
12. Threshold in the configured output space.
13. Apply optional connected-component filtering in that same space; physical-volume filters require valid spacing metadata.
14. Convert to `uint8` values `{0, 1}` when a binary output is requested.
15. Return the configured result to the consumer.
16. For GC transport, map each declared result key to the shared native result,
    materialize it through the selected NIfTI or MHA boundary writer, and
    reopen/validate the complete output artifact set before returning success.

Whenever spatial inversion occurs, probability interpolation must happen before thresholding. Interpolating an already binary mask can introduce avoidable geometric and topological artifacts.

---

## 9. Configuration ownership rules

### 9.1 Saved model config owns

- architecture (`DynUNet` initially);
- spatial dimensionality;
- input and output channel counts;
- deep-supervision topology;
- trained modality selection;
- dataset identity and canonical raw-to-processed modality contract;
- trained intensity preprocessing;
- trained orientation and spacing transforms;
- default/model-compatible ROI size;
- discriminative versus diffusion adapter type.

The complete saved config remains packaged for provenance and construction compatibility. Its historical validation/inference settings are inactive when a current explicit `cfg.inference` is selected.

### 9.2 Dataset construction and preprocessing adapters own

- whether labels are loaded for a dataset instance, through explicit `load_labels: bool = True`;
- label-conditional datalist requirements and MONAI transform keys;
- deterministic spatial and intensity transform assembly from the saved dataset config;
- dataset-specific raw-to-processed modality production;
- per-case native metadata capture before transformations discard raw keys;
- deterministic selection of the canonical raw modality whose native grid is the output reference when a dataset has multiple aligned inputs;
- label-required training augmentation checks.

The loader stack also owns validation-only normalized record discovery. That source resolves the requested subset and dataset-specific datalist format into canonical case records without constructing datasets, DataLoaders, samplers, augmentation pipelines, or worker pools. It must not require a usable training partition when only validation records were requested.

`test_flag` remains an independent partition-selection input. It must not set `load_labels`, and `load_labels` must not change partition selection. Existing labeled training, validation, and repository-model evaluation retain the default. Blind native inference and GC inference explicitly pass `False`.

`src.inference` owns reusable case production from those normalized records. Label-free production returns `PreprocessedCase`; labeled post-training evaluation returns `LabeledPreprocessedCase` with the model image, spatial trace, native input metadata, jointly transformed model label and geometry, and untouched native label and geometry. This rich result is an evaluation/deployment boundary, not a replacement for lightweight training dataset tuples.

### 9.3 Inference policy owns

- configured result/output space;
- numerical precision;
- sliding-window execution parameters that do not change model topology;
- TTA;
- ensemble aggregation;
- fixed decision threshold;
- postprocessing;
- optional intermediate artifact retention, subject to runtime permission.

### 9.4 Validation/evaluation config owns

- reference-label access and pairing;
- metric selection and aggregation;
- threshold sweep/oracle protocols;
- analysis levels and reports;
- training validation interval and progress metrics;
- best-checkpoint selection.
- complete-case batch size for post-training evaluation, fixed to one for geometry-aware 3D evaluation in this CAP.

It does not define a second sliding-window, TTA, ensemble, output-space, or fixed deployment-threshold schema.
Complete-case batch size must not be confused with `inference.sliding_window.sw_batch_size`, which controls window throughput within one case.

### 9.5 Inference runtime profile owns

- execution profile name (`native`, `gc_container_test`, or `gc_submission`);
- case batch and worker limits;
- timeout and device requirements;
- allowed output spaces;
- permission to access ground truth or execute threshold sweeps;
- permission to retain diagnostic/intermediate artifacts;
- hard capability checks for a particular execution context.

The runtime profile constrains requested behavior. It must not change model architecture, trained preprocessing, or silently rewrite inference choices.

### 9.6 Builder config owns

- image tag and output archive names;
- Docker build context;
- selected runtime lockfile/base image;
- code commit/worktree provenance;
- model run-directory/checkpoint specifications;
- path to inference policy;
- local test fixture paths;
- phase interface manifest path.

Builder config must not contain DynUNet channels, strides, model image size, input modality count, or trained preprocessing values.

### 9.7 Interface manifest owns

- socket slugs;
- explicit input socket-slug to canonical raw dataset-key bindings;
- explicit output socket-slug to shared result-key bindings;
- the complete ordered output artifact set required by each interface;
- input and output relative paths;
- transport file types;
- optional technical-JSON socket bindings, required status, and structural
  schemas/nullability;
- platform interface dispatch key.

It must not contain model architecture, processed modality definitions, trained
preprocessing, threshold choice, or other inference-policy settings. Runtime
validation compares its canonical raw keys with the selected registered adapter
and saved model contract and compares its result keys with values exposed by
the shared result contract. Shared preprocessing receives only the canonical
input mapping; output writers receive result keys plus transport bindings. No
shared scientific component receives or infers behavior from a socket slug.

### 9.8 Why this is shared inference composition, not one monolithic validation class

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

This permits native validation and container diagnostics in either valid result space without allowing evaluation-only behavior to leak into the submission endpoint.

### 9.9 nnU-Net conversion and evaluation ownership

The complete 3D conversion preset owns `nnunet.export_space`. The precursor
updates current presets to declare it at their leaf so environment-local and
cluster conversion commands remain independently auditable. The field is required for
`full_volumes_3d`/`3d` conversion and accepts only `model_preprocessed` or
`native_input`; it has no inferred default.

The converter owns proof that the files it writes agree with that declaration.
It records raw source geometry separately from exported geometry and writes the
affine carried by the transformed image/label grid. For a `native_input`
declaration, exported and selected native-reference geometry must agree. For a
`model_preprocessed` declaration, exported geometry must agree with the
transformed tensor. Image channels and label must occupy the same export grid.

The nnU-Net evaluator does not redeclare or guess this value. Its existing
composition of conversion and evaluation configs carries `nnunet.export_space`
into the 3D volume producer. That producer still reads prediction and reference
NIfTI geometry independently and rejects shape, affine, spacing, or orientation
disagreement before metrics. The declaration supplies semantic meaning;
geometry agreement supplies physical-grid evidence.

This is deliberately narrow provenance: reuse the existing `dataset.json` source
context and `export_provenance.jsonl`. Cut 5 does not add conversion hashes or a
second nnU-Net provenance subsystem.

---

## 10. Required changes to existing components

| Existing component | Required relationship to new package | Scope of change |
|---|---|---|
| `configs/validation/*.yaml` | Split prediction from assessment policy | Move new prediction settings to `configs/inference/`; retain metrics/cadence/checkpoint concerns under validation |
| `configs/inference/*.yaml` | Shared prediction policy | Compose from training validation, evaluation, native inference, container diagnostics, and GC deployment |
| `configs/inference_runtime/*.yaml` | Execution capability profiles | Enforce native-Python/container/submission constraints without duplicating inference settings |
| `src/data/loader_stack/registry.py` and contracts/factory routing | Register preprocessing capabilities | Select an implemented adapter by saved `dataset.id`; reject unknown/unimplemented dataset preprocessing clearly |
| new `src/data/loader_stack/record_source.py` | Validation-only normalized record source | Reuse dataset-specific datalist parsing and subset routing without constructing training datasets, DataLoaders, samplers, augmentations, workers, or requiring a non-empty training partition |
| `src/data/loader_stack/isles24_loader.py` | Share deterministic preprocessing and make labels explicit | Add backward-compatible `load_labels=True`, dynamic transform keys, metadata capture, and parity-preserving ISLES24 adapter behavior |
| `src/data/loader_stack/isles26_loader.py` | Share deterministic preprocessing and make labels explicit | Remove the unconditional label assumption through `load_labels=True`, dynamic datalist/transform keys, metadata capture, and parity-preserving ISLES26 adapter behavior |
| shared modality transforms, including `MergeProcessedChannelsTransform` | Preserve preprocessing semantics and metadata boundary | Capture native metadata before raw keys/meta are removed; do not duplicate modality processing in the container |
| `src/inference/preprocessing.py`, new `src/inference/case_producer.py`, and contracts | Dataset-agnostic preprocessing and reusable case production | Build deterministic preprocessing once, dispatch canonical raw-key inputs through the registered adapter, and return typed label-free `PreprocessedCase` or rich `LabeledPreprocessedCase` results without changing training tuples |
| `src/utils/valid_utils.py` | Source of the legacy inferer implementation, not its permanent façade | In Cut 4, transfer/remove only inference-policy resolution and direct/sliding-window prediction helpers; retain unrelated validation, memory, parallel, and generative utilities. Legacy config translation belongs in `src/inference.policy` |
| `src/training/trainer.py::validate_one_epoch` | Direct shared predictor/config consumer | In Cut 4, migrate probability generation directly to `src.inference`; in Cut 6, finish config composition, preserve the lightweight tuple/model-space-only guard, and certify labels, metrics, progress, cadence, logging, and checkpoint behavior |
| `src/utils/train_utils.py` | Transfer model-owned channel helpers | Import the unchanged helpers from `src/models/model_config.py`; keep the DP/DDP training builder in place |
| `src/training/checkpoint_utils.py` | Compatibility facade for moved loading helpers | Re-export the unchanged checkpoint-to-model helpers from `src/models/checkpoint_loading.py`; preserve resume behavior |
| `src/diffusion/discriminative_adapter.py` | First predictor backend implementation/dependency | Retain backend-owned single/list/stacked-head interpretation, final-head selection, and sigmoid behavior; expose finite `[B, C, *spatial]` probabilities to the shared contract without architecture-name branching in `src.inference` |
| `src/utils/ensemble.py` | Selective reuse/migration | Mean is eligible; soft STAPLE remains disabled for 3D until generalized |
| `scripts/evaluation/core/model_loader.py` | Delegate common model construction/loading to `src/models/` | Preserve evaluation checkpoint CLI behavior and run-directory discovery while removing duplicated model lifecycle behavior |
| `scripts/evaluation/io/model_volumes.py` | Unified typed-case prediction consumer | Replace separate model/native loops and dataloader-internal record reconstruction with one `LabeledPreprocessedCase` loop that selects the matching label/geometry from the declared result space |
| `scripts/evaluation/core/evaluation_pipeline.py` | Assessment consumer and full-volume request boundary | Load normalized validation records, build the reusable case producer, enforce one complete case at a time, then continue metrics, threshold protocols, reports, provenance, and geometry validation |
| `configs/nnunet/convert/*3d*.yaml` | Explicit 3D export-space source of truth | Declare `nnunet.export_space` as `native_input` or `model_preprocessed`; do not infer it in evaluation |
| `scripts/nnunet/core/conversion_core.py` and `exporters.py` | Produce spatially truthful 3D nnU-Net datasets | Validate the declared export space, write transformed tensor geometry rather than a stale source affine, and extend existing dataset/provenance records |
| `scripts/nnunet/core/io_adapters.py` and evaluation orchestration | Compliant external-model 3D producer | Consume conversion-owned space, validate prediction/reference NIfTI geometry, and emit the finalized volume-space contract before Cut 5 |
| `scripts/analysis/threshold_analysis.py` | Legacy migration/deprecation | Do not extend independently; later delegate or retire after parity |
| `scripts/test_validation_memory.py` | Replace or delegate | Convert to shared predictor resource smoke test or supersede with GC benchmark |
| `scripts/gc_submission_builder/runtime/interfaces.py` | Generic interface and complete-output-set contract | Bind image-kind input slugs to canonical dataset keys, discover one allowed platform representation, bind output slugs to explicit result keys, and validate required technical JSON without model conditioning |
| `scripts/gc_submission_builder/runtime/image_io.py` | Input canonicalization and parallel output boundary adapters | Losslessly normalize allowed MHA/TIFF/NIfTI inputs to downstream `.nii.gz`, preserve the existing NIfTI writer, and materialize/validate MHA mask and probability outputs |
| `scripts/gc_submission_builder/runtime/observability.py` | Structured production telemetry | Emit privacy-safe build/runtime identity, stage durations, resource summaries, geometry contracts, and actionable failure codes without logging patient filenames, metadata values, tensors, or image contents |
| `scripts/gc_submission_builder/container_builder.py` | Organizer-style external lifecycle test | Call health/invoke from an external tester sidecar, enforce the local 300-second invoke bound and hosted 16 GB memory limit, require the complete declared output set, and assert required telemetry milestones |
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
6. **Backend-owned output interpretation:** raw model-output parsing, deep-supervision head selection, and activation remain behind the predictor adapter; shared inference contains no architecture-name branching.
7. **Probability-before-blending parity:** the initial sliding-window path predicts a probability for each window and blends those probabilities. Logit blending is not introduced without an explicit, separately validated policy.
8. **Result-space contract:** every result declares `model_preprocessed` or `native_input`; every metric pairs prediction and reference in the same verified space.
9. **GC spatial contract:** production GC output is always `native_input` and matches input shape and physical geometry.
10. **Full-volume case batch invariant:** geometry-aware post-training 3D evaluation and `gc_submission` process one complete case at a time in both result spaces. This is independent of sliding-window window batching and does not restrict how many cases a job may process sequentially.
11. **Strict release loading:** the eventual release gate must reject missing/unexpected state-dict keys. The transition cut does not change the established permissive training/evaluation behavior.
12. **Explicit policy precedence:** top-level `cfg.inference` replaces rather than field-merges with historical `cfg.validation.inference`.
13. **No hidden fallback:** unsupported TTA, ensemble, precision, model family, output space, or interface fails clearly.
14. **No label dependency:** deployment preprocessing accepts an image without a label.
15. **No network runtime:** all required files and packages are inside the image or model mount.
16. **No multiprocessing dependency:** production case loading runs in-process; no DataLoader worker pool is necessary.
17. **Independent artifacts:** the Docker image can be rebuilt without model weights, and the model artifact archive can be rebuilt without changing the image.
18. **Reproducible provenance:** code, runtime, saved config, active inference config, runtime profile, and weights are hashable and recorded.
19. **Enhancements are opt-in:** baseline inference remains available when TTA, ensembling, and component filtering are disabled.
20. **Registered dataset dispatch:** saved `dataset.id` selects preprocessing. An unregistered adapter fails clearly; no container code hardcodes ISLES26 transforms.
21. **Explicit label contract:** `load_labels=True` is backward compatible, `False` is label-free, and `test_flag` never controls label behavior.
22. **No synthetic labels:** blind inference returns a typed label-free case, not a dummy tensor or placeholder label.
23. **Per-case geometry:** native metadata is captured for every case before preprocessing/channel merging; no dataset-wide geometry is assumed.
24. **Within-case alignment contract:** raw modalities within a case are expected aligned. This CAP does not introduce an additional pre-preprocessing alignment check.
25. **Transport isolation:** interface manifests map arbitrary socket slugs to canonical dataset keys; core inference and preprocessing never depend on competition slug names.
26. **Truthful nnU-Net export geometry:** every 3D nnU-Net conversion declares
    its export space and writes the affine, spacing, orientation, and shape of
    the tensor grid actually emitted.
27. **No semantic space guessing:** nnU-Net evaluation consumes
    `nnunet.export_space` from its conversion config and still validates both
    NIfTI geometries independently.
28. **3D-only geometry-aware evaluation:** repository and nnU-Net 2D slice
    paths fail before geometry-aware assessment until their parent-volume and
    reconstruction contracts are implemented and tested.
29. **Lightweight training boundary:** ordinary training and training-validation dataset items retain their established lightweight tuple contract; native labels, source records, and restoration traces are not added to every training batch.
30. **Evaluation record-source isolation:** post-training 3D evaluation obtains normalized validation records without constructing ordinary training datasets, DataLoaders, samplers, augmentation pipelines, or worker pools.
31. **One rich evaluation case contract:** both repository-model result spaces consume `LabeledPreprocessedCase`; the evaluator selects the reference matching `PredictionResult.output_space` instead of maintaining separate preprocessing/prediction loops.
32. **Complete-output invariant:** production success requires every output
    declared by the selected interface; a partial artifact set never returns
    HTTP 201.
33. **Explicit result-binding invariant:** output slugs map to explicit result
    keys. Neither slug spelling nor output order determines whether a value is
    a probability or mask.
34. **Probability/mask invariant:** the probability output is the continuous
    result restored to the configured space; the mask is produced by applying
    the configured threshold in that same space and after native restoration
    when `native_input` is selected.
35. **Output-grid identity:** all ISLES26 output values occupy the same verified
    native T1 physical grid. A transport format may not change result space.
36. **Transport coexistence:** adding an MHA writer does not remove, wrap, or
    silently redirect the certified native NIfTI writer. The manifest selects
    among explicit implemented transports around the same shared result.
37. **Lossless boundary conversion:** NIfTI/MHA representation changes preserve
    numeric meaning and voxel-to-world geometry. Unsupported or
    non-representable geometry fails rather than being approximated.
38. **Metadata isolation:** required technical JSON is structurally validated
    at the transport boundary and does not enter preprocessing or prediction
    unless a future saved model contract explicitly declares conditioning.
39. **External HTTP reachability:** the release lifecycle calls `/health` and
    `/invoke` from outside the algorithm container; a localhost self-probe is
    insufficient certification.
40. **Platform-input canonicalization:** image-kind transport is resolved and
    validated entirely within the GC adapter. Registered preprocessing receives
    canonical `.nii.gz` paths and never branches on MHA, TIFF, socket slugs, or
    platform-generated filenames.
41. **Source-grid preservation:** input representation conversion does not
    reorient, resample, crop, or normalize. Source and canonical shapes, scalar
    voxel meaning, physical geometry, and asymmetric world-coordinate landmarks
    must agree before preprocessing begins.
42. **Actionable production observability:** every startup and invocation emits
    a stable build/runtime identity, stage outcomes, finite nonnegative elapsed
    times, and bounded resource summaries. Failures name the last active stage,
    stable error code, error type, and sanitized detail rather than only an
    exception class.
43. **Telemetry isolation:** logging and timing must not expose patient
    filenames, metadata values, tensors, or image contents; must not emit one
    record per sliding window; and must not change inference behavior or become
    a second orchestration path.

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
   - predictor capabilities;
   - preprocessed case;
   - spatial trace;
   - prediction result;
   - timing/resource records.
   - declared output/result space.
2. Define explicit errors for:
   - unsupported model family/dimensionality;
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
- Capability errors clearly explain that the initial shared backend is discriminative-only; Cut 4 extends the message with the future `ProbabilityPredictor` adapter-registration hook. Existing diffusion model/training code outside this shared predictor boundary is not removed.

### Acceptance criteria

- The public API can represent the discriminative 3D use case without importing `scripts/`.
- Contracts contain no Grand Challenge socket or filesystem assumptions.
- Shared inference, assessment, and runtime responsibilities are separately represented.
- Future diffusion support can implement the predictor protocol without changing spatial result contracts.

### Rollback

Remove the new package and tests; no consumers have migrated yet.

---

## 14. Cut 2: Transfer single-model loading into `src/models`

### Context

Evaluation and training currently own model-related helpers in workflow-oriented locations. Before extracting prediction execution, the existing single-model evaluation loader, checkpoint-to-model helpers, and channel-contract helpers need model-domain homes. This cut is an ownership transition, not a deployment-loader redesign: externally observable loading, partial-load reporting, device mapping, configuration mutation, gradient state, and prefix handling remain unchanged. Strict release enforcement and artifact validation are deferred to the later release-artifact/runtime cuts.

### Dependencies

Cuts 0-1.

### Affected files and components

- new `src/models/checkpoint_loading.py`
- new `src/models/model_config.py`
- new `src/models/model_loader.py`
- `src/inference/contracts.py` and `src/inference/__init__.py` to remove the provisional, unneeded multi-model artifact types from the inference API
- `scripts/evaluation/core/model_loader.py` as the legacy evaluation facade
- `src/utils/train_utils.py` as the legacy training consumer of moved channel helpers
- `src/training/checkpoint_utils.py` as the compatibility facade for moved checkpoint helpers
- `src/models/model_factory.py`
- `src/diffusion/diffusion.py`
- `src/diffusion/discriminative_adapter.py`
- new `tests/test_model_loader.py`
- existing `tests/test_inference_contracts.py`
- existing `tests/test_evaluation_model_loader.py`

### Desired changes

1. Transfer `_extract_checkpoint_state_dict()`, `_normalize_state_dict_keys_for_model()`, and `load_model_state_dict_compat()` from `src/training/checkpoint_utils.py` to `src/models/checkpoint_loading.py` without broadening their accepted layouts or changing their strict-first/permissive-fallback behavior.
2. Re-export those functions from the training module so historical imports and training resume behavior continue to work.
3. Transfer the existing model/data channel-contract helpers from `src/utils/train_utils.py` to `src/models/model_config.py` without changing their config reads, mutation, messages, or validation behavior.
4. Keep the existing training utilities as consumers of those transferred helpers; do not move or redesign distributed training construction.
5. Transfer evaluation's existing single-model construction sequence into `src/models/model_loader.py`: build through `src.models.build_model()`, wrap through `Diffusion.build_diffusion()`, load the checkpoint on the requested device, move to that device, and call `.eval()`.
6. Do not add config projection/copying, read-only provenance objects, CPU-first loading, parameter freezing, strict-only modes, new prefix transforms, model-family restrictions, manifests, hashes, or multi-model APIs.
7. Keep evaluation checkpoint discovery, model-name/EMA selection, diagnostics, and public function signatures in `scripts/evaluation/core/model_loader.py`.
8. Make that evaluation module a thin facade over the moved single-model lifecycle while preserving its observable return values and diagnostic reporting.
9. Remove the provisional model-artifact/member contracts from `src/inference`; inference receives an already constructed model/predictor and does not discover checkpoints or artifacts.

### Expected tests and testing components

- Existing unwrapped and supported leading-wrapper checkpoint layouts load exactly as before.
- Existing partial state-dict loading still reports missing/unexpected keys rather than becoming a new strict failure.
- Tests exercise the moved implementation directly and the legacy training/evaluation import paths.
- The shared loader receives the original config object and requested `map_location`, and preserves the established model build/wrap/load/`eval()` sequence.
- Existing evaluation checkpoint discovery and model-loader tests remain green.
- Existing training checkpoint, resume, data-contract, 2D, and diffusion tests remain green.
- A real selected checkpoint produces identical state tensors through the old characterized sequence and the moved implementation in the standardized desktop environment.

### Acceptance criteria

- The same config and checkpoint produce the same model parameters and preparation state through the legacy evaluation interface and the moved shared loader.
- The evaluation-facing checkpoint-discovery and model-loading API remains compatible.
- Training DP/DDP construction, checkpoint prefix handling, partial-load reporting, and resume behavior are not regressed.
- No bundle, ensemble-member, manifest, hash-validation, or strict-release abstraction is introduced.
- `src/inference/` has no model-artifact discovery or model-loader responsibility.

### Rollback

Restore evaluation and training imports to their previous local implementations; no artifact or inference consumer has migrated yet.

---

## 15. Cut 3: Extract dataset-agnostic deterministic, label-optional preprocessing

### Context

The existing ISLES24 and ISLES26 loaders contain the trained preprocessing contracts, share modality-processing primitives, and are already selected through the loader-stack registry. Both common transform chains nevertheless hardcode a label because they were designed when every training/validation record was labeled. ISLES26 datalist validation likewise unconditionally requires a label. Blind native inference and Grand Challenge invocation make that assumption obsolete.

This cut updates the assumption at dataset construction instead of adding a container-only workaround. It establishes explicit `load_labels` behavior, extracts or exposes each registered dataset's deterministic transform assembly, and lets `src/inference` select the correct adapter from the saved model's `dataset.id`. The first adapters are ISLES24 and ISLES26; the release model remains ISLES26 3D DynUNet, but shared infrastructure is not allowed to branch on the competition name.

Spatial restoration is not implemented here, but its source evidence must survive. Native metadata is case-specific and must be captured before preprocessing begins and before `MergeProcessedChannelsTransform` removes raw modality keys/meta. Different cases may have unrelated shapes, spacings, orientations, and affines. Within a case, the dataset contract says its modalities are aligned; this cut deliberately does not introduce an additional pre-preprocessing alignment validator.

### Dependencies

Cuts 0-2.

### Affected files and components

- `src/data/loader_stack/registry.py`
- `src/data/loader_stack/contracts.py`
- `src/data/loader_stack/factory.py` and the `src/data/loaders.py` compatibility facade where constructor arguments are routed
- `src/data/loader_stack/isles24_loader.py`
- `src/data/loader_stack/isles26_loader.py`
- `src/utils/loader_transforms.py` only if metadata preservation requires a shared transform-boundary change
- optionally one dataset-neutral helper under `src/data/loader_stack/` for deterministic transform assembly, native metadata capture, and/or typed adapter registration
- new `src/inference/preprocessing.py`
- `src/inference/contracts.py`
- existing ISLES24 and ISLES26 full-volume, 2D, random-patch, datalist, facade-routing, and registry tests
- new `tests/test_inference_preprocessing.py`

### Desired changes

1. Add `load_labels: bool = True` at dataset construction and thread it through loader-stack/facade routing. The default preserves every existing labeled caller. Declare `load_labels: true` explicitly in the current ISLES24 and ISLES26 base dataset configs; historical resolved model configs that predate the field continue to use the backward-compatible `True` runtime default.
2. Keep `test_flag` semantically independent. It continues to select the existing train/validation/test partition behavior and must neither derive nor override `load_labels`. `is_training` continues to control augmentation rather than label presence.
3. Make datalist normalization conditional:
   - with `load_labels=True`, a missing label remains an informative error;
   - with `load_labels=False`, a record may omit `label`, and a supplied label is not silently loaded.
4. Build MONAI key lists dynamically for `LoadImaged`, `Orientationd`, `Spacingd`, `SpatialPadd`, and `EnsureTyped`. Use nearest-neighbor label interpolation only when a label is loaded. Image processing must be numerically identical in labeled and label-free modes.
5. Keep label-dependent random-patch operations, including `RandCropByPosNegLabeld`, explicitly label-required. A random-patch dataset requested with `load_labels=False` fails immediately with an informative specialized error rather than constructing a partially valid pipeline.
6. Extract or expose deterministic transform assembly without moving dataset-specific scientific choices into `src/inference`. Shared modality processing remains shared; ISLES24- and ISLES26-specific rules remain in their registered adapters.
7. Extend registry capabilities so the saved `dataset.id` resolves an implemented preprocessing adapter. Unknown or registered-but-unimplemented preprocessing fails clearly. Do not infer the adapter from a path, modality name, or competition slug.
8. Resolve saved processed modalities through the adapter's canonical raw-to-processed modality contract. One raw modality may produce multiple processed channels; the saved `dataset.modalities` and preprocessing config remain authoritative.
9. Capture native metadata before deterministic preprocessing starts, separately for each case and raw modality:
   - canonical raw modality key and safe source provenance;
   - array shape and source dtype;
   - affine and voxel spacing;
   - orientation codes;
   - qform and sform values plus codes where available.
10. Let the registered adapter select a deterministic canonical raw modality as the native output reference, then retain that reference geometry, model-space geometry, MONAI transform history, and any explicit non-MONAI trace required by Cut 7C. Because within-case inputs are contractually aligned, this selection does not add geometry-comparison checks. Do not implement probability inversion in this cut.
11. Trust the established within-case modality-alignment contract. Do not add new shape/affine/spacing/world-coordinate comparisons between raw modalities before preprocessing.
12. Preserve the exact trained orientation, spacing, intensity, modality-processing, channel-order, and padding behavior from the resolved saved config.
13. Preserve the default labeled item/tuple contract, including `(image, label, case_id)` where currently exposed. Do not force existing training or evaluation callers to consume a new object merely to gain blind inference support.
14. Define an explicit typed `PreprocessedCase` for label-free/shared inference with processed image, case ID, native metadata, model-space geometry, and spatial trace. Do not fabricate a zero label or expose a dummy/`None` label field.
15. Establish the data required by the later `LabeledPreprocessedCase` contract: the jointly transformed model-space label and geometry plus the untouched original native label and geometry. Cut 7B turns this evidence into the reusable typed evaluation-case producer. The predictor remains label-free regardless of which preprocessing result was used, and ordinary training dataset tuples remain unchanged.
16. Do not use a DataLoader worker pool for the external single-case inference path.
17. Keep Grand Challenge socket/path parsing out of this cut. `src/inference.preprocessing` receives canonical raw modality keys after the transport adapter has applied its user-supplied interface bindings.

### Expected tests and testing components

- ISLES24 and ISLES26 image-plus-label preprocessing remain numerically equivalent to the pre-cut loaders for fixed deterministic settings.
- For the same record and config, `load_labels=False` produces the same processed image tensor as `load_labels=True`.
- Omitting `load_labels` preserves the labeled default and rejects a missing label with a dataset/case-specific message.
- A missing label succeeds only when `load_labels=False`; a present label is not returned or transformed in that mode.
- `test_flag=True, load_labels=True` remains labeled, and tests prove that changing either flag never implicitly changes the other.
- Image-only transform chains contain no label keys and do not request label interpolation or label typing.
- Label-dependent random-patch construction with `load_labels=False` fails before iteration with an informative error.
- Registry tests route saved `dataset.id=isles24` and `dataset.id=isles26` to their respective preprocessing adapters; unknown/unimplemented adapters fail clearly.
- One raw modality producing multiple configured processed channels retains the saved channel order and numerical behavior.
- Two fixture cases with deliberately different native geometries each retain their own shape, affine, spacing, orientation, qform/sform, and dtype metadata.
- Tests prove native metadata is captured before raw modality keys/meta are removed by channel merging.
- All supported spatial fixtures retain a trace sufficient for later inversion, without claiming that inversion is already implemented.
- Fixtures prove the jointly transformed label and original native label/geometry can be retained for the Cut 7B typed producer without changing the ordinary training tuple.
- Existing ISLES24/ISLES26 2D, 3D, full-volume, random-patch, datalist, and facade-routing tests remain green.
- Input with wrong modality count, corrupt NIfTI, invalid/nonfinite data, unsupported rank, or an unavailable adapter fails clearly.
- No new test asserts proactive within-case cross-modality alignment validation; that behavior is intentionally out of scope.

### Acceptance criteria

- Existing training/evaluation datasets and external inference share the registered dataset's deterministic transform definition.
- Dataset selection comes from the saved `dataset.id`; neither `src/inference` nor the GC runtime hardcodes ISLES26 preprocessing.
- `load_labels=True` preserves legacy callers, `load_labels=False` supports blind inference without placeholders, and `test_flag` remains independent.
- The deployment path retains per-case information sufficient for the Cut 7B typed producer and exact native-grid restoration in Cut 7C.
- The loader interface leaves a clear validation-record and rich-case extraction seam for Cuts 7A-7B without requiring training batches to carry the additional payload.
- No training augmentation is accidentally applied during inference.

### Rollback

Keep extracted builders/adapters behind the existing loader API. If parity fails, restore the inlined deterministic builder while retaining the explicit `load_labels` contract, metadata fixtures, and failing parity cases for investigation. Do not replace the update with a GC-only preprocessing copy.

### Explicit non-goals for this cut

- Prediction, sliding-window execution, or activation handling (Cut 4).
- Probability inversion or native-output writing (Cut 7C).
- Socket-slug discovery, `inputs.json` parsing, or output path dispatch (Cuts
  10A-10B).
- TTA, ensembling, threshold calibration, or postprocessing (Cut 8).
- Automatic support for datasets without a registered, tested adapter.
- New within-case raw-modality alignment checks.

---

## 16. Cut 4: Backend-neutral probability prediction with the initial discriminative backend

### Context

`src/utils/valid_utils.py` currently combines validation-policy resolution with direct/sliding-window prediction execution. `src/training/trainer.py::validate_one_epoch()` and `scripts/evaluation/io/model_volumes.py` consume that implementation. Cut 4 transfers this reusable probability-generation behavior into `src/inference/` and migrates those internal consumers directly; it does not preserve `build_validation_inferer()` as a second façade because there are no active training jobs that require an intermediate migration window.

The shared layer must be backend- and architecture-neutral. Its input is an injected `ProbabilityPredictor` that returns finite `[B, C, *spatial]` probabilities. Interpretation of raw model returns—including single versus list/stacked deep-supervision outputs, final-head selection, and sigmoid or any future backend-specific normalization—remains owned by the backend adapter. Existing `DynUNetAdapter`/`SwinUNETRAdapter` and `DiscriminativeAdapter` behavior is characterized rather than reimplemented inside inference.

Current sliding-window behavior asks the discriminative adapter for a probability on every patch and then lets MONAI blend those probabilities. This is a valid, parity-critical baseline rather than an identified bug. Cut 4 preserves it. A possible logits-first blend would be a separately configured and scientifically evaluated future experiment.

The package boundary is deliberately capable of receiving a future diffusion-backed `ProbabilityPredictor`, but this cut registers only the discriminative backend. Unsupported generative configurations must fail before direct/window execution with an actionable capability error that names the future predictor-adapter extension point.

### Dependencies

Cuts 0-3.

### Affected files and components

- existing `src/inference/predictors.py`, extended with the executable predictor boundary and discriminative registration/construction
- new `src/inference/sliding_window.py`
- `src/inference/pipeline.py` only if a small coordinator is needed to keep direct/sliding-window policy execution out of consumers; do not create an empty pass-through layer
- `src/inference/policy.py`
- `src/inference/runtime.py`
- `src/utils/valid_utils.py`, removing only its inference-specific resolver/builder functions and imports while retaining unrelated validation utilities
- `src/training/trainer.py::validate_one_epoch`
- `scripts/evaluation/io/model_volumes.py`
- `src/diffusion/discriminative_adapter.py`, only for the minimal generic predictor boundary if required; do not move or duplicate its output parser
- existing model adapters and `src/losses/discriminative_deep_supervision.py` as characterized backend-owned behavior, not inference-package migration targets
- `configs/validation/sliding_window.yaml` as the legacy Cut 4 migration input;
  Cut 6 removes this now-empty compatibility group after the 3D metric bundles
  inherit `validation/default` directly
- new or extended `tests/test_inference_discriminative_predictor.py`
- new `tests/test_inference_sliding_window.py`
- existing validation-inferer, trainer-validation, and model-volume evaluation tests, migrated to the new direct boundary

### Desired changes

1. Make direct and sliding-window execution consume the architecture-neutral `ProbabilityPredictor` protocol from Cut 1 rather than a concrete architecture or diffusion object.
2. Register/build the current discriminative probability backend as the first supported implementation. Keep raw-output interpretation, deep-supervision parsing/final-head selection, and activation inside the existing backend/model-adapter layer.
3. Keep `src.inference` free of DynUNet, SwinUNETR, or other model-class/name branches. It validates only the predictor's declared tensor rank, channels, finiteness, and probability domain.
4. Resolve direct/sliding-window behavior exclusively from the active shared `cfg.inference` after legacy translation.
5. Implement MONAI sliding-window execution with only the established repository parameters:
   - ROI resolution;
   - `sw_batch_size`;
   - overlap;
   - blend mode;
   - padding mode;
   - progress callback optionality.
6. Preserve the established aggregation order: invoke the probability predictor separately for each window, then blend overlapping window probabilities using the configured `blend_mode`; preserve the configured `padding_mode` for window-edge/input padding. Do not introduce logits-first blending or additional unestablished MONAI policy fields.
7. Set the GC-safe `sw_batch_size=1` through the explicitly selected inference policy; do not inherit a larger historical training-validation override.
8. Run under `torch.inference_mode()` and the selected precision/autocast policy, and validate the probability contract after direct prediction and after sliding-window aggregation.
9. Keep progress/UI concerns injectable so the Docker service does not emit per-window progress bars.
10. Reject unsupported generative/diffusion configurations at the shared predictor boundary with `UnsupportedModelError` before allocating or executing inference windows. The message must identify implementation/registration of a backend adapter satisfying `src.inference.contracts.ProbabilityPredictor` as the extension point; `src.inference` must not import DDPM/DDIM/OpenAI diffusion sampling logic in this cut. This capability rejection does not remove diffusion model construction, training-forward, checkpoint, or training-specific snapshot code elsewhere in the repository.
11. Remove `_resolve_validation_inference_mode`, `should_use_sliding_window_validation`, `resolve_validation_sliding_window_roi`, `build_validation_inferer`, and inference-only parsing helpers/imports from `src/utils/valid_utils.py`. Preserve its unrelated model-copy, memory, parallel-validation, and generative-validation utilities.
12. Migrate all repository call sites of the removed inferer directly to `src.inference`, including training validation and live-model evaluation. `scripts/test_validation_memory.py` does not call this inferer and remains outside Cut 4; its unrelated multi-GPU/generative validation helpers remain in `valid_utils.py`. No internal compatibility façade remains.
13. Translate legacy `cfg.validation.inference` only when no explicit `cfg.inference` is supplied and record which source won.
14. Return model/preprocessed-space probabilities only in this cut. If the Cut 4 entrypoint receives `output_space=native_input` before Cut 7C is integrated, it must fail with an explicit unsupported-capability error rather than relabel a model-space tensor. Do not threshold predictions, calculate metrics, handle labels, write NIfTI, or add GC socket behavior; those remain later-cut or consumer responsibilities.

### Expected tests and testing components

- A generic mock `ProbabilityPredictor` exercises direct and sliding-window paths without importing or naming a model architecture.
- Direct output matches the current discriminative adapter on fixed tensors.
- Sliding-window output matches the current validation inferer within the Cut 0 tolerance.
- A regression test proves that patch probabilities are produced before overlap blending; no hidden logits-then-activation reorder occurs.
- Explicit `cfg.inference` takes precedence as a complete policy over historical `cfg.validation.inference`; there is no field-level leakage.
- Odd input sizes and inputs smaller than the ROI are padded/cropped correctly.
- Window batch one works under FP32 and FP16 on GPU.
- Existing backend/model-adapter tests continue to cover single, list, and stacked deep-supervision returns and final-head selection; shared inference tests assert only the normalized probability boundary.
- A configured generative backend fails before prediction/window execution with an informative `UnsupportedModelError` naming `ProbabilityPredictor` adapter registration as the future hook.
- NaN/Inf or out-of-domain output fails.
- Trainer-validation, model-volume evaluation, and relevant former `test_valid_utils` behavior tests use the direct shared entrypoint and remain green.
- Repository search confirms that no consumer imports or calls `build_validation_inferer` and that the removed inference symbols no longer exist in `valid_utils.py`.
- Selected real-case probability parity passes.

### Acceptance criteria

- Training validation and live-model evaluation call the same `src.inference` probability path directly without metric drift.
- `src/utils/valid_utils.py` no longer owns or re-exports inference-policy resolution or direct/sliding-window construction; its unrelated functionality remains intact.
- Shared inference contains no architecture-name branching, raw-output/deep-supervision interpretation, or activation choice.
- Sliding-window aggregation preserves probability-before-blending parity with the accepted baseline.
- The initial discriminative backend is supported and configured generative inference fails early with the documented future adapter hook.
- The predictor contains no label, metric, NIfTI-writing, or socket logic.
- The T4-safe window-batch default is enforced by deployment policy.
- Cut 4 results are truthfully identified as model/preprocessed-space probabilities; native restoration remains Cut 7C.

### Rollback

If parity cannot be established, atomically restore the old inference-specific `valid_utils.py` implementation and its direct call sites while keeping the new shared implementation isolated. Do not leave both implementations active or partially route consumers between them.

---

## 16A. Cut-5-nnunet-precursor: Space-aware 3D nnU-Net conversion

### Context

The external nnU-Net runtime remains separate from `src.inference`, but its 3D
datasets and predictions are consumers of the same scientific evaluation
contract. Current 3D conversion loads tensors through repository preprocessing,
then `VolumeExportStrategy` reconstructs `export_affine` from the untransformed
source image. This is correct only for a grid-preserving conversion. ISLES26
conversion enables RAS orientation and 1 mm isotropic spacing, so writing its
transformed tensor with the source affine can create a spatially plausible but
physically false NIfTI.

NIfTI format also cannot tell the evaluator whether a grid is semantically
`native_input` or `model_preprocessed`. That fact is known when the conversion
preset is selected and must travel with the converted dataset. This precursor
makes every supported 3D nnU-Net producer comply before Cut 5 changes the
shared evaluator. Cut 5 must not return to `scripts/nnunet/` to repair producer
contracts.

The current repository contains both kinds of 3D conversion. The ISLES24 local
and cluster baseline presets disable orientation, spacing, and full-volume
padding and are assigned `native_input` by this cut. The ordinary ISLES26
local, cluster, and ATLAS30 T1-raw presets enable RAS orientation and 1 mm
isotropic spacing and are assigned `model_preprocessed`. A separate local
ISLES26 T1-raw native baseline disables orientation and spacing, opts into
native spacing explicitly, retains the existing no-padding full-volume policy,
and is assigned `native_input`. `T1_RAW` describes modality/intensity handling;
it does not by itself establish native spatial geometry.

### Dependencies

Cuts 0-4, including the Cut 3 `SpatialGeometry` contract and per-case metadata
work. This cut is a mandatory dependency of Cut 5.

### Affected files and components

- `configs/nnunet/convert/isles24_cluster_3d_baseline.yaml`
- `configs/nnunet/convert/isles24_local_3d_baseline.yaml`
- `configs/nnunet/convert/isles26_cluster_3d_t1raw.yaml`
- `configs/nnunet/convert/isles26_local_3d_t1raw.yaml`
- `configs/nnunet/convert/isles26_local_3d_t1raw_native.yaml`
- `configs/nnunet/convert/isles26_atlas30_cluster_3d_t1raw.yaml`
- `configs/dataset/isles26_modalities_t1raw_native.yaml`
- `configs/data_profile/isles26_3d_fullvol_t1raw_native.yaml`
- `scripts/nnunet/core/conversion_core.py`
- `scripts/nnunet/core/exporters.py`
- `scripts/nnunet/core/io_adapters.py`
- `scripts/nnunet/core/evaluation_pipeline.py`
- `scripts/nnunet/evaluate_nnunet_results.py`
- `scripts/evaluation/core/contracts.py`, only for the forward spatial seam
  consumed by both nnU-Net and Cut 5
- nnU-Net conversion/evaluation documentation
- existing converter affine, round-trip, config, and volume-adapter tests
- new focused 3D conversion-space and volume-geometry tests

### Desired changes

1. Add required `nnunet.export_space` to each complete 3D conversion preset.
   Use only `model_preprocessed` and `native_input`; do not infer a default from
   dataset identity, preprocessing flags, NIfTI format, or nnU-Net ownership.
2. Set both ISLES24 3D baseline presets to `native_input`. Set the ordinary
   ISLES26 local/cluster T1-raw presets and the ISLES26 ATLAS30 3D preset to
   `model_preprocessed`. Add the local ISLES26 T1-raw native baseline as
   `native_input`, with a distinct nnU-Net dataset identity and explicit
   orientation-disabled, spacing-disabled/allowed-native preprocessing.
3. Require and validate `nnunet.export_space` for
   `loader_mode=full_volumes_3d` with `dim=3d` before output directories are
   cleared or files are written. Existing 2D conversion does not receive an
   invented space declaration in this cut.
4. Capture raw source geometry separately from export geometry. Obtain export
   geometry from the transformed MONAI image/label grid, not by copying the raw
   source affine after preprocessing.
5. Use the jointly transformed label `MetaTensor` as an available authoritative
   export-grid source, cross-check image metadata when retained, and validate
   that every image channel and label has the same spatial shape and geometry.
   Do not reconstruct a transformed affine from requested spacing/orientation
   config values when the actual tensor metadata is available.
6. Write image channels and labels with the verified export affine and
   consistent qform/sform. Preserve the source affine, spacing, orientation,
   and shape as separate provenance rather than relabeling them as output
   geometry.
7. For `native_input`, require exported shape and physical geometry to agree
   with the selected raw reference grid. For `model_preprocessed`, require the
   written shape and geometry to agree with the transformed tensor grid.
8. Add `export_space` and actual export shape/affine/spacing/orientation to the
   existing `dataset.json` source context and `export_provenance.jsonl`; do not
   add hashes or a second provenance artifact.
9. Prepare the finalized 3D `VolumeSample` seam with explicit prediction space,
   reference space, prediction geometry, and reference geometry. These fields
   may be temporarily optional only so this precursor remains independently
   compatible with the not-yet-migrated repository-model producer; the nnU-Net
   3D producer must populate all four immediately, and Cut 5 removes acceptance
   of missing fields from canonical 3D evaluation.
10. Resolve the common 3D nnU-Net space from the composed conversion config.
    The nnU-Net evaluator must not ask for a duplicate evaluation-space value.
11. Load prediction and ground-truth NIfTI geometry independently and reject
    shape, affine, spacing, or orientation disagreement before creating a
    compliant 3D sample. Replace metadata such as `nnunet_native_volumes` with
    the neutral producer identity `nnunet_volumes`.
12. Reject nnU-Net `slices_2d` at the geometry-aware evaluation boundary with
    an actionable error explaining that parent-volume geometry, in-plane
    resize inversion, slice placement, and reconstruction remain uncertified.
    Existing 2D conversion may continue for legacy use, but it is not certified
    for the new evaluator.
13. Do not route nnU-Net model execution through `src.inference`, change nnU-Net
    training/prediction semantics, or implement general 2D spatial compliance.

### Expected tests and testing components

- Compose every current 3D conversion preset and assert its exact declared
  export space.
- Missing or unknown `nnunet.export_space` fails for 3D before filesystem
  mutation; no compatibility/default inference is accepted.
- A synthetic native-grid volume preserves source shape and affine.
- Synthetic reorientation and anisotropic-resampling fixtures write the
  transformed shape, affine, spacing, and orientation rather than the source
  affine.
- Image-channel and label geometry disagreement fails.
- A false `native_input` declaration for a changed grid fails.
- Written NIfTIs are reopened and checked for shape, affine, spacing,
  orientation, qform, and sform; provenance contains distinct source/export
  geometry and `export_space`.
- A matching 3D nnU-Net prediction/reference pair creates a spatially complete
  `VolumeSample` for both allowed semantic spaces.
- Equal-shaped prediction/reference files with different affines fail before
  metrics; spacing and orientation mismatches also fail.
- nnU-Net 2D geometry-aware evaluation fails with the documented deferred-work
  error, while legacy 2D conversion tests remain green.
- On the desktop, export and reopen at least one ISLES24 native-grid case and
  one ISLES26 transformed-grid case when the data is available. Compare the
  written geometry with the actual loader outputs, and run nnU-Net dataset
  integrity verification where practical.

### Acceptance criteria

- Every supported 3D nnU-Net conversion preset declares its semantic export
  space explicitly.
- Every written 3D NIfTI uses the geometry of the tensor actually exported;
  transformed ISLES26 tensors never receive stale raw-source affines.
- A `native_input` declaration is verified against the selected raw reference
  rather than trusted blindly.
- Existing per-case provenance distinguishes source and export geometry and
  records export space without a new hash/manifest subsystem.
- The 3D nnU-Net evaluation producer supplies explicit, independently verified
  prediction/reference space and geometry.
- nnU-Net 2D inputs cannot enter the geometry-aware evaluation path.
- Cut 5 requires no changes beneath `scripts/nnunet/` or `configs/nnunet/`.

Previously generated spatially transformed 3D nnU-Net datasets are not
retroactively certified by a config edit. Existing ISLES26 exports must be
inspected and should be regenerated if they were written with raw source
affines. This cut never deletes or overwrites external datasets automatically.

### Rollback

Revert the precursor as one unit and treat all nnU-Net volume inputs as
non-compliant with the new evaluator. Do not restore evaluation-time space
guessing and do not continue using a transformed dataset known to carry a stale
affine.

---

## 17. Cut 5: Migrate geometry-aware offline evaluation

### Context

Offline evaluation is the scientific certification surface for deployment
behavior. Cut 4 already routed live repository-model, model-space probability
generation in `model_volumes.py` through the shared predictor. The nnU-Net
precursor has separately made 3D external-model volumes declare their space and
geometry. Cut 5 now completes the common evaluator integration: shared config
composition, final `VolumeSample` enforcement, label/reference selection,
metrics, threshold protocols, operational provenance, and reports.

This cut changes the evaluator package, not nnU-Net conversion or nnU-Net
producer code. It retains the current repository-model CLI and makes
model-space parity evaluation and native-space deployment certification
explicit rather than conflating them.

Cut 5 establishes the evaluator's explicit `VolumeSample` space/geometry contract. It does not freeze the temporary repository-model data acquisition path as the final design. Cuts 7A-7D subsequently replace ordinary DataLoader construction and any evaluator-side reconstruction of raw records with the normalized validation-record source, reusable `LabeledPreprocessedCase` producer, and one dual-space model-volume loop. nnU-Net remains an already materialized external-volume producer and is unaffected by that repository-model producer refactor.

### Dependencies

Cuts 0-4 and Cut-5-nnunet-precursor. Model-space evaluation integration can
complete immediately. Native-space repository-model evaluation assertions
depend on the Cut 7C restoration contract and the Cut 7D producer migration,
and may be finalized when those dependent cuts land.

### Affected files and components

- `scripts/evaluation/core/model_loader.py`
- `scripts/evaluation/core/contracts.py`
- `scripts/evaluation/io/model_volumes.py`
- `scripts/evaluation/io/volume_assembler.py`
- `scripts/evaluation/core/evaluation_pipeline.py`
- `scripts/evaluation/evaluate_model.py`
- `scripts/evaluation/reporting/` where spatial facts enter existing reports
- `scripts/evaluation/README.md`
- `configs/evaluation/*.yaml`
- `configs/inference/*.yaml`
- `configs/inference_runtime/native.yaml`
- existing `tests/test_evaluation_model_loader.py`
- existing `tests/test_evaluation_io_model_volumes.py`
- existing `tests/test_evaluation_pipeline.py`
- evaluation contract, reporting, entrypoint, and integration tests

Explicitly unaffected are `scripts/nnunet/`, `configs/nnunet/`, nnU-Net dataset
generation, model architecture implementations, and Cut 7C spatial inversion.

### Desired changes

1. Replace any remaining evaluation-owned model construction with `src.models`
   calls. Retain the direct `src.inference` probability path established in
   Cut 4 and remove remaining evaluation-local prediction normalization or
   interpretation that duplicates the shared probability contract. Do not
   apply another sigmoid merely because a tensor crosses the evaluation
   boundary.
2. For the initial Cut 5 migration, continue obtaining images, labels, case IDs,
   and evaluation metadata through existing dataset infrastructure with
   explicit `load_labels=True`. Treat any evaluator-side record reconstruction
   needed for early native-space work as transitional only; Cut 7D must replace
   it with the Cut 7A normalized record source and Cut 7B
   `LabeledPreprocessedCase` producer.
3. Finalize `VolumeSample` so every accepted 3D sample explicitly carries
   `prediction_space`, `reference_space`, `prediction_geometry`, and
   `reference_geometry`. Remove the precursor's transitional acceptance of
   absent fields from canonical 3D evaluation.
4. Validate before any metric computation that tensor shapes agree with their
   declared `SpatialGeometry`, prediction/reference spaces are identical, and
   shape, affine, spacing, and orientation agree within explicit tolerances.
   Shape equality alone is insufficient.
5. Compose `/inference: sliding_window_model_space` and
   `/inference_runtime: native` from `configs/evaluation/default.yaml`. Existing
   evaluation presets inherit this unless they explicitly select another
   complete inference policy. Evaluation supersedes saved training-validation
   inference settings; an explicit top-level policy is never field-merged with
   them.
6. Preserve `evaluation.input_source` as producer selection (`live_model` versus
   fixed outputs), not as a runtime profile.
7. Preserve fixed, sweep, oracle, and sweep-with-oracle threshold protocols
   under evaluation config. Threshold sweeps consume shared probabilities;
   exporting one selected deployment threshold remains an explicit action.
8. Pair `model_preprocessed` probabilities with the jointly transformed label.
   After Cuts 7C-7D, pair `native_input` probabilities with the original native-grid
   label. Never resample a reference ad hoc inside the metric engine to conceal
   a space mismatch.
9. Record useful operational provenance: producer type, prediction/reference
   spaces and verified geometries, runtime profile, policy source, selected
   checkpoint identity, threshold protocol, and selected threshold. Do not add
   a new SHA-256/config-hash subsystem in this cut.
10. Preserve existing report schemas where possible. If spatial fields require
    a public schema change, make that change explicit and update its consumers
    rather than silently altering meaning.
11. Reject repository-model `data_mode.dim=2d`, legacy `SliceSample` assembly,
    or any 2D sample lacking the deferred typed reconstruction contract before
    geometry-aware assessment. `VolumeAssembler` must not manufacture a parent
    affine or infer space from first-slice metadata.
12. Keep the explicit current rejection of 3D non-discriminative diffusion and
    name `ProbabilityPredictor` registration as the future inference hook.
13. Keep shared evaluation free of DynUNet, SwinUNETR, or other architecture
    branches; raw return/deep-supervision interpretation remains backend-owned.

### Expected tests and testing components

- Existing relevant evaluation unit/integration tests pass after their fixtures
  provide explicit 3D space and geometry.
- A fully declared matching 3D sample passes; missing/unknown space, missing
  geometry, cross-space pairing, tensor/geometry shape disagreement, and
  equal-shape/different-affine pairing all fail before metrics.
- Shared predictor probabilities pass through the repository-model producer
  unchanged; no second sigmoid or probability reinterpretation occurs.
- The selected baseline checkpoint produces model-space probability and metric
  parity at threshold 0.5 within the predeclared environment-appropriate
  tolerance.
- Threshold-sweep best-global and oracle selection remain stable within the
  defined tie tolerance.
- Explicit top-level inference policy replaces saved training-validation
  inference policy as a whole, and `evaluation.input_source` does not alter the
  runtime profile.
- A compliant sample emitted by the nnU-Net precursor enters the same metrics
  engine without any Cut 5 change to nnU-Net code.
- Repository and nnU-Net 2D inputs fail early with the actionable deferred
  geometry/reconstruction message.
- Native-space repository-model evaluation runs against original-grid labels
  after Cuts 7C-7D where fixtures/data permit.
- The same inference policy can run in native Python and
  `gc_container_test` when both runtime profiles permit its capabilities.
- Failure to load the requested checkpoint remains clear at the CLI.

### Acceptance criteria

- The canonical evaluation entrypoint uses `src.inference` for 3D
  repository-model prediction.
- Evaluation remains a geometry-validating metrics/reporting wrapper rather
  than a second inference implementation.
- Repository-model and compliant nnU-Net 3D samples enter one evaluation engine
  through the finalized explicit space/geometry contract.
- No `scripts/nnunet/` or `configs/nnunet/` file changes in Cut 5.
- Threshold recommendations intended for deployment use the same probability
  path as container inference; sweep/oracle config never becomes active inside
  `gc_submission`.
- No 2D slice path is presented as geometry-aware or assigned a guessed output
  space.
- Native-space certification remains explicitly incomplete until Cuts 7C-7E pass.

### Rollback

Revert evaluation-specific config/result-space/provenance integration to the
accepted Cut 4 model-space shared-predictor state, and treat both repository and
nnU-Net inputs as not yet accepted by the new geometry-aware evaluator. Do not
restore the removed `valid_utils.py` inferer, introduce a second model-volume
prediction implementation, or undo the independently correct nnU-Net precursor.

---

## 18. Cut 6: Complete training-validation configuration migration and certify unchanged training

### Context

Cut 4 has already migrated training-time probability generation directly from the removed validation inferer to the shared predictor. Cut 6 completes the configuration and consumer-responsibility migration around that accepted path, then certifies that labels, metrics, progress, logging, checkpoint selection, and optimization behavior remain unchanged. This cut must not repeat the predictor migration or broaden into a trainer rewrite.

### Dependencies

Cuts 0-5, including the mandatory Cut-5-nnunet-precursor, with evaluation
parity already green.

### Affected files and components

- `src/training/trainer.py::validate_one_epoch`
- `src/utils/ensemble.py` where mean aggregation is reused
- `configs/validation/*.yaml`
- existing `configs/inference/*.yaml`
- existing `configs/inference_runtime/native.yaml`
- validation/runtime contract tests
- training smoke tests

### Desired changes

1. Retain and certify the direct shared-predictor probability path established by Cut 4; do not reintroduce a `valid_utils` compatibility façade.
2. Keep the trainer responsible for:
   - moving labels to the training device;
   - metric object lifecycle;
   - progress display;
   - checkpoint selection;
   - validation logging.
3. Move prediction execution settings out of validation presets and into the already established, composed top-level `configs/inference/` presets.
   Compose `direct_model_space` for base 2D training profiles and override it
   with `sliding_window_model_space` for the current 3D discriminative
   profiles. Both presets derive ROI from the model-owned dataset contract.
   Remove the empty legacy `validation/sliding_window` group and make the two
   `sliding_window_3d_metrics_*` assessment bundles inherit `validation/default`
   directly; their names describe the 3D validation use case but do not own
   prediction execution.
4. Keep validation config responsible for metrics, cadence, progress reporting, and checkpoint selection.
5. Preserve old run compatibility by translating `cfg.validation.inference` when `cfg.inference` is absent.
6. Compose `inference_runtime=native` for new training profiles and validate
   that training validation is permitted to access ground truth before model
   execution. Historical configs without a runtime profile retain the native
   compatibility default.
7. Preserve existing ensemble behavior for supported modes, but reject current
   2D-only soft STAPLE for every 3D model family. Mean aggregation remains
   spatial-dimension agnostic.
8. Leave `train_one_epoch`, optimizer, scheduler, loss, gradient scaling, EMA, DP/DDP, and checkpoint writing unchanged.
9. Leave diffusion sampling snapshots training-specific.
10. Optionally route ordinary ensembled preview images through the predictor later, but do not make this a release blocker.
11. Keep during-training validation on `model_preprocessed` output. Its lightweight tuple batches intentionally do not carry the case-specific native trace and untouched native label required for restoration; requesting `native_input` must fail early with an actionable message directing users to post-training repository-model evaluation.
12. Set `configs/validation/default.yaml::validation.val_batch_size` to one. Explicit specialized training-validation presets may retain larger values for their own demonstrated use, but geometry-aware post-training 3D evaluation rejects any effective complete-case batch other than one at its request boundary.

### Expected tests and testing components

- One-epoch or one-batch training validation smoke test completes.
- Metric parity against Cut 0 baseline passes.
- Training validation still imports/calls the direct shared predictor, and no removed `valid_utils` inferer symbol is restored.
- Best-checkpoint metric names and update behavior remain unchanged.
- New training presets compose `cfg.inference`; old saved configs containing only `cfg.validation.inference` still validate identically.
- Explicit inference overrides replace the legacy policy rather than partially merging with it.
- DP/DDP construction tests remain green.
- Existing 3D diffusion runtime rejection tests remain green.
- Snapshot logging still uses the appropriate diffusion-only API.
- During-training `native_input` validation fails before prediction and explains that post-training evaluation is the supported native-space assessment path.
- The default validation config resolves to complete-case batch one, while tests keep complete-case batching distinct from sliding-window `sw_batch_size`.

### Acceptance criteria

- The validation prediction path shares implementation with evaluation/deployment.
- No training-forward or optimization behavior changed.
- Existing training configs require no model-architecture edits.
- New validation configs no longer duplicate prediction-policy fields.
- Training tuples remain lightweight and are not expanded to carry native labels or spatial traces for the sake of post-training evaluation.

### Rollback

Revert only Cut 6's validation-config composition and surrounding assessment wiring to the accepted Cut 4 state; training remains otherwise unaffected. Do not restore the removed legacy inferer or fork prediction mechanics.

---

## 19. Cut 7 umbrella: Typed evaluation cases, native restoration, and dual-space certification

Cut 7 is divided into five dependency-ordered subcuts. This split preserves the already proven spatial work while correcting the temporary evaluator architecture that reconstructed native cases by reaching behind an ordinary DataLoader. Each subcut has its own rollback boundary. Cut 8 and all container release work depend on the complete Cut 7A-7E chain, not merely on the existence of a spatial resampler.

### 19.1 Cut 7A: Validation-only normalized record source

#### Context

Post-training evaluation needs canonical case records and validation-subset routing, but it does not need training datasets, samplers, random augmentation, DataLoader workers, or a valid/non-empty training partition. Constructing the full loader stack for record discovery couples evaluation to training-only assumptions and makes label/native-metadata acquisition unnecessarily indirect.

This cut extracts only the existing dataset-specific record-reading and subset-selection behavior. It does not preprocess tensors or change training dataset construction.

#### Dependencies

Cuts 0-6 and the dataset-loader contracts established in Cut 3.

#### Affected files and components

- new `src/data/loader_stack/record_source.py`
- `src/data/loader_stack/registry.py` and contracts only where reader registration is required
- narrow ISLES24/ISLES26 datalist-reader wrappers around existing parsing behavior
- new focused record-source tests
- existing loader-stack routing and datalist parser tests

#### Desired changes

1. Add a public `load_case_records(cfg, subset_role="val", load_labels=True)`-style boundary that returns normalized canonical records for a requested subset.
2. Dispatch record reading through the existing dataset registry and dataset-specific datalist parsers; do not infer behavior from paths or competition names.
3. Preserve the established partition semantics and `test_flag`/`load_labels` independence without constructing datasets or DataLoaders.
4. Permit validation-only discovery when the training partition is absent or empty.
5. With `load_labels=True`, retain the existing informative missing-label failure. With `False`, return records that contain no fabricated label.
6. Do not instantiate transforms, samplers, augmentation pipelines, worker pools, tensors, or GPU state.
7. Keep ordinary training and validation loader entrypoints unchanged.

#### Expected tests and testing components

- ISLES24 and ISLES26 validation records match the corresponding existing loader-stack record identities and canonical modality paths.
- Train/validation/test subset routing is characterized for both registered datasets.
- `load_labels=True` and `False` preserve the Cut 3 label contract.
- A validation-only config with an empty or absent training partition succeeds.
- A spy/patch proves no Dataset, DataLoader, sampler, transform pipeline, or worker pool is constructed.
- Unknown dataset IDs and malformed records fail with dataset/subset/case context.
- Existing datalist parser and loader-routing regressions remain green.

#### Acceptance criteria

- Post-training consumers can obtain normalized validation records without constructing the ordinary training loader stack.
- Record identity, modality order, partition selection, and label requirements match the registered dataset contract.
- No existing training dataset item or loader signature changes.

#### Rollback

Remove the unused record-source boundary and its registrations. Because no evaluator consumer migrates until Cut 7D, rollback does not affect accepted training or evaluation behavior.

### 19.2 Cut 7B: Reusable typed evaluation-case producer

#### Context

The existing lightweight `(image, label, case_id)` dataset item is appropriate for training and training-time validation, but insufficient for dual-space post-training evaluation. Native evaluation additionally needs the untouched native label, both label geometries, native input metadata, and the exact spatial trace. Adding those fields to every ordinary training batch would increase I/O, collation complexity, and memory pressure for consumers that do not use them.

This cut introduces a separate rich evaluation interchange. It turns deterministic preprocessing into a reusable producer built once per evaluation job rather than rebuilding the adapter pipeline per case.

#### Dependencies

Cut 7A and the deterministic adapter/metadata evidence from Cut 3.

#### Affected files and components

- `src/inference/contracts.py`
- `src/inference/preprocessing.py`
- new `src/inference/case_producer.py`
- `src/inference/__init__.py`
- registered dataset preprocessing adapters/builders
- new typed-case producer tests
- existing Cut 3 preprocessing and loader parity tests

#### Desired changes

1. Define `LabeledPreprocessedCase` as an extension or sibling of `PreprocessedCase` with explicit fields for:
   - processed model input and case ID;
   - model-space geometry and complete spatial trace;
   - case-specific native input reference metadata;
   - jointly transformed model-space label and geometry;
   - untouched native-grid label and independent native label geometry.
2. Keep label-free `PreprocessedCase` free of dummy, zero, or `None` label fields.
3. Build the registered deterministic preprocessing pipeline once per producer and reuse it across normalized records.
4. Accept canonical normalized records from Cut 7A and resolve raw modality paths without Grand Challenge socket knowledge.
5. Capture the native label before joint transforms while also retaining the transformed label produced by the exact image/label preprocessing chain.
6. Validate tensor rank, finite values, required trace fields, and each declared geometry without assuming that different cases share a grid.
7. Preserve `preprocess_case()` as a narrow one-case convenience facade implemented through the same producer/core where practical.
8. Do not change ordinary training dataset tuples, collation, augmentation, or DataLoader behavior.

#### Expected tests and testing components

- One producer instance preprocesses multiple records while constructing the deterministic transform pipeline once.
- Label-free and labeled production yield numerically identical processed images for the same record/config.
- `LabeledPreprocessedCase` exposes both labels and their correct independent geometries.
- The model-space label matches the processed image grid; the native label retains its original grid.
- Cases with distinct shapes, spacings, orientations, translations, and oblique affines retain independent metadata and traces.
- Results match the accepted deterministic loader output at the shared model-tensor boundary.
- Missing labels, incomplete traces, unsupported ranks, and unavailable adapters fail before prediction with case-specific messages.
- Existing training tuple and loader tests prove no payload or signature expansion.

#### Acceptance criteria

- Evaluation can obtain every artifact required for either output space through one typed case contract.
- The producer reuses the repository's deterministic preprocessing source of truth and does not rebuild it for every record.
- Training and training-time validation remain on their lightweight established data path.

#### Rollback

Remove the producer and rich contract while retaining the accepted Cut 3 label-optional preprocessing APIs. No evaluator switches to the new boundary until Cut 7D.

### 19.3 Cut 7C: Native-space probability restoration and output correctness

#### Context

Cut 7C consumes the architecture-neutral model-space probability result established in Cut 4 and the case-specific trace/reference contract finalized in Cut 7B. It does not inspect raw model outputs or apply an activation.

The previous competition failure mode—returning a mask aligned to the reoriented input rather than the original image—is a primary design risk. Spatial correctness must be independently certified before postprocessing or container work can be called complete.

#### Dependencies

Cuts 0-4 and Cut 7B. Spatial fixtures that do not require evaluator integration may be developed earlier, but final acceptance requires the typed case contract.

#### Affected files and components

- new `src/inference/spatial.py`
- new `src/inference/output.py`
- `src/inference/preprocessing.py`
- `src/inference/pipeline.py`
- `src/inference/contracts.py`
- `src/inference/__init__.py`
- optionally reusable ideas from `scripts/evaluation/io/volume_exporter.py`
- optionally reusable fixtures/patterns from nnU-Net affine tests
- new `tests/test_inference_spatial_roundtrip.py`
- new `tests/test_inference_native_output.py`
- related inference pipeline and sliding-window tests

#### Desired changes

1. Accept the validated `inference.output_space` and retain model-space results without inversion when `model_preprocessed` is selected.
2. Invert model-space probabilities through spacing and orientation transforms when `native_input` is selected, using the case-specific native reference geometry retained by the registered dataset adapter rather than a dataset-wide geometry assumption.
3. Use continuous interpolation suitable for probabilities during inversion.
4. Restore exact original array shape.
5. Reconstruct or copy the correct affine/qform/sform semantics.
6. Validate corner or sampled voxel world coordinates within a defined tolerance.
7. Reject restoration if the transform trace is incomplete or inconsistent.
8. Threshold in the selected output space and only after restoration when native output is requested.
9. Keep the shared file writer intentionally native-only. It must refuse a `model_preprocessed` result so diagnostic tensors cannot accidentally be materialized as production segmentations.
10. Write binary arrays as `uint8` with values exactly `{0, 1}`.
11. Re-open written files and verify:
   - loadability;
   - shape;
   - affine;
   - qform/sform values and codes;
   - dtype;
   - allowed voxel values.
12. Return output-space and spatial validation facts in the result provenance.
13. Make `gc_submission` reject `model_preprocessed`, while `native` and `gc_container_test` may select it.

#### Expected tests and testing components

- Round-trip tests for identity, axis permutation, axis flip, anisotropic spacing, translation, odd shapes, and oblique affine.
- Synthetic landmark test: known world-space points occupy the corresponding output locations after preprocess/predict/invert.
- A probability ramp survives continuous inversion more accurately than threshold-first inversion.
- Model-space mode returns the expected transformed grid without claiming native geometry.
- Runtime profile tests enforce that only diagnostic-container/native-Python execution can select model-space output.
- Empty and full masks preserve geometry.
- Output dtype and allowed values are exact.
- Deliberately corrupted trace or affine mismatch fails rather than writing a plausible file.
- If both nibabel and MONAI metadata are used, their affine interpretation is cross-checked.
- Cases with different native grids restore independently to their own recorded reference geometry.
- The native writer rejects model-space results and reopens/validates successful outputs.

#### Acceptance criteria

- Every native-output spatial fixture produces an output on the original input grid; every model-space fixture remains on and declares the expected transformed grid.
- Shape equality alone is not accepted as proof; affine/world-space checks pass.
- There is no code path that writes a model-space mask to the production Grand Challenge output socket.

#### Rollback

No deployment fallback to un-inverted output is permitted. A failure blocks release and is corrected in this subcut.

### 19.4 Cut 7D: Unified full-volume repository-model evaluator migration

#### Context

An early native-space evaluator implementation may prove restoration by reaching through `dataloader.dataset`, re-resolving records and raw paths, rebuilding preprocessing, and maintaining a separate native-only loop. That is evidence, not an acceptable final architecture. Model-space and native-space evaluation run the same predictor on the same preprocessed case; they differ only in where the probability result is returned and which already-carried reference label/geometry is selected.

This cut removes the evaluator's dependence on ordinary DataLoader internals and gives both spaces one typed case-production and prediction loop. It also makes complete-case batch one a generic geometry-aware post-training 3D evaluation contract rather than misleadingly presenting it as native-space or container-specific.

#### Dependencies

Cuts 5-6 and Cuts 7A-7C.

#### Affected files and components

- `scripts/evaluation/core/evaluation_pipeline.py`
- `scripts/evaluation/io/model_volumes.py`
- `scripts/evaluation/evaluate_model.py` and SLURM evaluation argument plumbing where required
- `configs/validation/default.yaml`
- `scripts/evaluation/README.md`
- `tests/test_evaluation_pipeline.py`
- `tests/test_evaluation_io_model_volumes.py`
- evaluation entrypoint/config/SLURM tests
- related training-validation guard tests

#### Desired changes

1. For 3D repository-model `live_model` evaluation, call Cut 7A record discovery and construct one Cut 7B case producer. Do not call the ordinary training `get_dataloaders()` path.
2. Iterate `LabeledPreprocessedCase` values through one repository-model volume loop:
   - move the model image to the requested device;
   - invoke the shared `predict_preprocessed_case`/equivalent pipeline;
   - inspect `PredictionResult.output_space`;
   - pair `model_preprocessed` with the transformed label/geometry or `native_input` with the untouched native label/geometry;
   - construct the common `VolumeSample` and provenance.
3. Remove `_iter_native_model_volume_samples`-style native-only orchestration, dataset/`Subset` unwrapping, evaluator-owned raw-path resolution, evaluator-owned adapter construction, and duplicate prediction loops.
4. Keep space-specific operations inside the shared inference result path. Evaluator code may select the matching reference but must not independently restore probabilities.
5. Enforce complete-case batch size one at the repository-model 3D evaluation request boundary for both output spaces. The error must explain the distinction from sliding-window `sw_batch_size` and that one job may still process many cases sequentially.
6. Set `configs/validation/default.yaml::validation.val_batch_size` to one. Do not delete specialized opt-in training-validation profiles merely because the post-training evaluator rejects their larger case batch.
7. Preserve the accepted during-training guard: training validation consumes lightweight transformed tuples and rejects `native_input`, directing users to post-training repository-model evaluation.
8. Keep the native-only NIfTI writer from Cut 7C unchanged.
9. Preserve the same evaluation entrypoint and composed DictConfig behavior for direct native Python execution and the LRZ SLURM wrapper. Move OmegaConf/dataset config conversion to the producer boundary; `model_volumes.py` receives typed cases, not raw config mappings.
10. Keep nnU-Net's compliant 3D materialized-volume producer unchanged and keep all uncertified 2D paths failing early.

#### Expected tests and testing components

- The same synthetic `LabeledPreprocessedCase` passes through the one model-volume loop in both output spaces and selects the correct reference/geometry.
- A spy proves the repository-model evaluator uses normalized records and does not construct an ordinary DataLoader.
- Validation-only evaluation succeeds with an empty/absent training partition.
- Tests prove there is no native-only record-resolution or prediction loop left in `model_volumes.py`.
- Complete-case batch values other than one fail equally for both result spaces with an actionable case-versus-window batching message.
- `configs/validation/default.yaml` resolves to case batch one; sliding-window `sw_batch_size` remains independently configurable.
- Direct native execution and SLURM-composed DictConfig reach the same typed producer/evaluator boundary.
- Model-space parity, native-space geometry validation, threshold protocols, and result/report schemas remain green.
- During-training native output still fails early; model-space training validation remains green.
- Existing compliant 3D nnU-Net evaluation remains green without entering the repository-model case producer.

#### Acceptance criteria

- Repository-model post-training 3D evaluation constructs no ordinary training DataLoader and does not inspect Dataset/Subset internals.
- One typed loop handles both result spaces; only reference selection differs after the shared prediction result declares its space.
- Complete-case batch one is expressed as a generic full-volume evaluation rule, not as a native-space synonym or a one-case-per-job restriction.
- Direct native and LRZ SLURM evaluation use the same public entrypoint and config contract.
- Training-time validation remains lightweight and model-space only.

#### Rollback

Revert the evaluator consumer to the accepted Cut 5/Cut 7C state while retaining Cuts 7A-7C as independently tested, unused capabilities. Do not restore a second spatial inversion implementation or alter training tuples.

### 19.5 Cut 7E: Live parity, full-split certification, and evidence closure

#### Context

Unit and synthetic spatial tests establish contracts, but release confidence also requires a real saved model, real heterogeneous ISLES26 geometries, both output spaces, label-free output writing, and a completed validation traversal. Prior exploratory native evaluation is useful evidence but does not replace certification of the finalized Cut 7D architecture.

This is primarily a verification/evidence cut. Production-code changes should occur only if a discovered defect requires correction and must then be covered by the appropriate preceding subcut's tests.

#### Dependencies

Cuts 7A-7D and the accepted Cut 0 checkpoint/environment evidence.

#### Affected files and components

- `docs/INFERENCE_EXTRACTION_EVIDENCE_LEDGER.md`
- temporary non-versioned Desktop fixtures, output masks, evaluation reports, and runner scripts
- focused regression commands/documentation only where necessary
- no production source file in the expected success path

#### Desired changes

1. Use the pinned representative DynUNet run/config/checkpoint and record its identity plus the evaluated code state.
2. Prove that model-space and native-space policies consume the same preprocessed model input and predictor path.
3. Re-run model-space parity at the earliest shared representation within the environment-appropriate tolerance.
4. Run label-free native output on at least one nontrivial case and reopen the NIfTI to verify exact input geometry, qform/sform, `uint8`, and `{0,1}` values.
5. Exercise heterogeneous native grids, including orientation/spacing/shape changes and oblique cases supported by the dataset.
6. Confirm known source image/label geometry violations fail closed with the recorded case and reason; do not weaken tolerances to complete a split.
7. Create an evidence-only validation selection that excludes exactly the documented invalid source cases, then run all remaining valid cases as one completed evaluation through the finalized Cut 7D path. Previously explored partial traversals may guide selection but do not substitute for this completed run.
8. Verify unique expected case count, no unaccounted failures, output/report completeness, spatial metadata, timings, and relevant artifact/config hashes.
9. Distinguish Desktop GPU evidence from later T4/Grand Challenge certification; do not claim T4 resource compliance in this cut.
10. Update the evidence ledger with commands, environment, case selection/exclusions, results, hashes, artifacts, limitations, and disposition of any failure.

#### Expected tests and testing components

- Focused Cut 7A-7D unit/integration suites pass in the Desktop MedSegDiff environment.
- Model-space replay matches the accepted predictor/evaluator baseline within the declared tolerance.
- Native label-free output passes reopened-file validation.
- One-case and sequential multi-case live evaluation exercise both spaces through the unified producer.
- The evidence-only full valid split completes once, with the expected unique case count and exactly documented exclusions.
- Known invalid source geometry fails before metrics/output rather than being silently resampled or paired.
- `git diff --check` and artifact/secret review pass for the versioned changes.

#### Acceptance criteria

- The finalized typed producer and unified evaluator, not the superseded exploratory helper, have live evidence across both output spaces.
- Every valid selected case completes with verified result/reference geometry; every exclusion is evidence-backed and explicit.
- Label-free native output is a valid binary NIfTI on the original input grid.
- Evidence clearly separates scientific/spatial correctness on the Desktop from deferred T4/container resource qualification.
- Cut 7A-7E is ready for review as the complete native-restoration and evaluator-boundary milestone.

#### Rollback

Evidence artifacts may be discarded and the run repeated without reverting accepted code. A scientific or spatial failure reopens the responsible subcut; it must not be waived in release documentation.

---

## 20. Cut 8: Shared inference enhancements: TTA, ensemble, threshold, and postprocessing

### Context

Training validation, offline evaluation, native/container diagnostics, and competition inference may benefit from invertible TTA, multi-checkpoint ensembling, a fixed decision threshold, and component filtering. These operations belong to the shared inference schema rather than a GC-only policy, remain separate from the saved model configuration, and must be evaluated incrementally. Label-dependent threshold calibration remains an evaluation concern. Baseline single-model inference must remain available.

### Dependencies

Cuts 0-6 and the complete Cut 7A-7E chain, including the finalized canonical evaluation path.

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
12. Apply enhancements to outputs from prepared `ProbabilityPredictor` instances; TTA and ensemble code must not inspect model architecture, deep-supervision structure, activation, or diffusion sampler internals.

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

Grand Challenge permits model resources to be uploaded separately and expanded under `/opt/ml/model/`. The repository needs a reproducible builder that accepts the same singular run-directory/checkpoint specification used by evaluation, retains the complete saved config, adds one explicitly selected shared inference config, and produces a self-validating tarball.

### Dependencies

Cuts 1-4 and the complete Cut 7A-7E chain; Cut 8 if the final enhancement policy schema is included.

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

1. Provide a command that accepts one run-dir/checkpoint specification and an explicit shared inference-policy path.
2. Copy, never mutate or trim, the complete resolved training config and exact checkpoint into a staging directory.
3. Reject a missing `.hydra/config.yaml`, ambiguous checkpoint, unsupported initial-release model contract, or saved `dataset.id` without an implemented preprocessing adapter.
4. Generate SHA-256 hashes and a versioned manifest.
5. Validate the staged artifact by loading its singular model through the shared model-domain and inference paths before packaging.
6. Create `<training-run-directory-name>.tar.gz` by default, with an explicit
   filename override available and contents rooted correctly for
   `/opt/ml/model/` extraction.
7. Support independent `build-model` and combined `build-all` commands.
8. Make archive generation deterministic where practical by normalizing metadata timestamps and file order, or document any unavoidable nondeterminism.
9. Emit a local build report containing source paths, artifact paths, sizes, hashes, and validation result.
10. Refuse to package optimizer state, datasets, logs, or unrelated training artifacts.
11. Record that the archived explicit inference policy is active and the historical saved `validation.inference` block is provenance-only.
12. Validate the artifact against the `gc_submission` runtime profile before packaging, including native output, case batch one, fixed threshold, and no evaluation sweep.

### Expected tests and testing components

- Tarball extracts to the expected root layout with no extra enclosing directory.
- Manifest hashes match extracted files.
- Exact requested checkpoint is used.
- Ambiguous/missing checkpoints fail.
- A singular model artifact validates and loads strictly at the release boundary.
- Unsupported diffusion or 2D artifact fails for the initial release profile.
- ISLES24 and ISLES26 artifacts resolve their registered preprocessing adapters in generic builder validation; the ISLES26 release profile then applies its narrower 3D discriminative certification constraints.
- Historical `validation.inference.sw_batch_size=4` remains visible in the copied saved config but does not override an archived GC policy selecting one.
- The artifact loads when mounted at a temporary `/opt/ml/model` equivalent.
- Repeated builds from identical inputs have identical logical manifests and, if deterministic tar metadata is implemented, identical archive hashes.

### Acceptance criteria

- A model artifact archive can be rebuilt without rebuilding the Docker image.
- The artifact is fully load-tested before being declared successful.
- No model architecture value is duplicated in builder config.
- Complete training provenance is retained without activating its historical validation policy.

### Rollback

The builder is additive. Delete a failed staged artifact; source run files remain unchanged.

---

## 22. Cut 10A: Generic Grand Challenge container runtime and image builder

### Context

The container must translate platform I/O into a call to the already certified shared inference package. It should not become another inference implementation. The runtime and dependency set must be constructed from the environment audit and current platform constraints.

Cut 10A establishes the model-independent image, HTTP service, fixture
interface dispatch, label-free shared-inference call, native NIfTI output,
build/test/save commands, and archive behavior. It is intentionally a
fixture-based scaffold and is not an ISLES26 upload artifact. Evidence items
E018 and E019 record the initial lifecycle and bounded closure of this
pre-official-interface milestone.

The builder and runtime are not ISLES26 preprocessing implementations. A
user-supplied interface manifest binds arbitrary socket slugs to canonical raw
modality keys. The saved model config then selects a registered dataset adapter
and its trained preprocessing. Consequently, another repository model trained
on a supported dataset can use the same builder/runtime design with its own
compatible artifact and interface manifest; adding an unknown dataset still
requires a repository adapter.

### Dependencies

Cuts 0-6, the complete Cut 7A-7E chain, and Cut 9. Fixture interfaces are
sufficient for this scaffold; Cut 10B owns official-interface reconciliation.

### Affected files and components

- new `scripts/gc_submission_builder/runtime/app.py`
- new `scripts/gc_submission_builder/runtime/inference.py`
- new `scripts/gc_submission_builder/runtime/interfaces.py`
- new `scripts/gc_submission_builder/runtime/image_io.py`
- new `scripts/gc_submission_builder/runtime/diagnostic.py`
- new `scripts/gc_submission_builder/container_config.py`
- new `scripts/gc_submission_builder/container_builder.py`
- `scripts/gc_submission_builder/cli.py`
- `scripts/gc_submission_builder/__init__.py`
- `configs/inference_runtime/gc_container_test.yaml`
- `configs/inference_runtime/gc_submission.yaml`
- new `scripts/gc_submission_builder/container/Dockerfile`
- new pinned runtime lockfile
- current Grand Challenge template build/test/save scripts adapted into the package
- new container config, fixture-only example interface manifest, and synthetic test inputs
- new package README and build-context `.dockerignore`
- `src/utils/general.py` only if needed to keep the runtime import graph free of unused training/plotting dependencies
- new `tests/test_gc_interfaces.py`
- new `tests/test_gc_image_io.py`
- new `tests/test_gc_runtime.py`
- new `tests/test_gc_diagnostic.py`
- new `tests/test_gc_container_builder.py`
- `tests/test_gc_builder_config.py`
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
7. Load and validate the singular model artifact during `init_model()` before health becomes ready, including the saved `dataset.id`, required processed modalities, availability of its registered preprocessing adapter, and successful construction of a registered `ProbabilityPredictor`. Unsupported generative artifacts fail through the same predictor-capability error used by native execution rather than a container-only model branch.
8. Compose the archived shared inference policy with `inference_runtime=gc_submission` for the production server and fail on incompatible capabilities.
9. Require an explicit interface manifest at build configuration time. For each input it declares the socket slug, canonical raw dataset key, relative path/file type, and required cardinality; the Cut 10A fixture also declares one NIfTI output slug/path/type and any technical-JSON bindings.
10. Read `/input/inputs.json`, calculate the sorted socket-slug interface key, and dispatch through that manifest without treating a slug as a modality name.
11. Validate the manifest and invocation against the saved model contract: every required canonical raw key is bound once, no unsupported input is silently consumed, and the selected adapter can derive the saved processed modalities in their configured order.
12. Pass the resulting canonical raw-key-to-path mapping to shared preprocessing with `load_labels=False`. Core preprocessing and prediction receive no socket identifiers.
13. Prove with deliberately opaque fixture slugs that an input socket can bind to canonical key `T1` without Python code depending on the slug or dataset identity.
14. Parse configured technical JSON as transport/provenance only. Do not condition the model on it in this scope; Cut 10B adds the official schema.
15. Write the fixture's native-space mask through the existing NIfTI output binding beneath `/output/`.
16. Re-open and validate output geometry, dtype, and values before returning HTTP 201.
17. Return non-success on any model, dataset-adapter, interface-binding, input, spatial, inference, or output-validation failure.
18. Keep protected filenames and image contents out of logs. Log timings, shapes, dtypes, device, memory peaks, artifact identifiers, policy hash, dataset adapter identity, and runtime profile.
19. Use `/tmp` only for transient scratch files and clean per-case state.
20. Provide commands for `build-image`, `test`, `save`, and `build-all`.
21. Provide a separate container diagnostic command/profile that can evaluate or retain `model_preprocessed` results without weakening the production `/invoke` contract.

### Expected tests and testing components

- `/health` remains non-ready until model initialization succeeds.
- `/invoke` returns the expected platform status and writes one valid output.
- Missing/extra inputs, wrong interface key, multiple files in a single-value socket, and corrupt input fail clearly.
- An arbitrary fixture slug that is explicitly bound to canonical key `T1` produces the same canonical input and preprocessing result as another slug bound to `T1`; no test depends on naming the socket `T1`.
- Missing, duplicate, unknown, or processed-modality-incompatible canonical bindings fail before preprocessing.
- ISLES24 and ISLES26 fixture manifests dispatch to their registered adapters in runtime unit/integration tests; the production certification remains the ISLES26 3D model.
- Runtime calls shared preprocessing with `load_labels=False`; no label path, dummy label, or `test_flag` coupling is present.
- Model-init failure prevents health readiness.
- Production initialization rejects transformed-space output, threshold sweeps, labels, or debug artifact retention.
- The diagnostic profile can exercise both result spaces using mounted synthetic/reference data without writing invalid data to a production output socket.
- Network-isolated local test succeeds.
- Read-only `/input` and writable `/output` behavior is tested.
- Non-root execution is verified.
- Container label and architecture are verified.
- Model directory is mounted at `/opt/ml/model` rather than copied into the image.
- The NIfTI fixture path works and retains the existing native-output writer's shape, affine, qform/sform, dtype, and binary-value checks.
- Restarting the container does not depend on prior `/tmp` contents.

### Acceptance criteria

- The image and model artifact archive pass the fixture-based Cut 10A local lifecycle.
- The container performs no scientific prediction logic outside `src.inference`.
- The GC transport layer owns slugs and paths; saved model config plus the registered data adapter own preprocessing and modality semantics.
- The builder/runtime can be configured for a compatible model from any registered repository dataset without editing ISLES26-specific Python branches.
- The image is independently buildable and saveable without embedding replaceable weights.
- Cut 10A is explicitly not upload-ready until Cut 10B implements and tests the complete official interface.

### Rollback

Use the last certified image tag and model artifact archive. Container releases must be immutable and content-addressed in the release report.

---

## 22A. Cut 10B: Official ISLES26 interface reconciliation

### Context

The official ISLES26 starter template was published after the Cut 10A fixture
runtime was implemented. It resolves the input socket set and introduces two
requirements that affect the generic transport boundary rather than model
inference: one interface requires an ordered set of two result values, and the
organizer example materializes both as compressed MHA. The official local test
also calls HTTP from outside the algorithm container with a 300-second invoke
timeout.

Cut 10B extends the transport layer around the already certified
`PredictionResult`. It does not create another prediction implementation and
does not discard the existing NIfTI input/output path. The NIfTI writer remains
available for native workflows, fixtures, and future challenge manifests that
select it.

### Dependencies

Cuts 0-10A and official ISLES26 template commit:

```text
5e25bfc36b1dc6d9c04c8c364f53fb75c6afad32
```

Cut 10B-2 follows the original Cut 10B implementation after hosted validation
exposed its NIfTI-only input assumption. Cut 11 depends on Cut 10B-2 so resource
qualification measures input canonicalization, production telemetry, and the
actual two-output transport workload.

### Affected files and components

- `scripts/gc_submission_builder/runtime/interfaces.py`
- `scripts/gc_submission_builder/runtime/image_io.py`
- `scripts/gc_submission_builder/runtime/inference.py`
- `scripts/gc_submission_builder/runtime/app.py` only if complete-output
  success/error handling requires adjustment
- `scripts/gc_submission_builder/container_builder.py`
- `scripts/gc_submission_builder/container_config.py`
- `scripts/gc_submission_builder/configs/container.yaml`
- new `scripts/gc_submission_builder/configs/interfaces/isles26.yaml`
- relocated
  `scripts/gc_submission_builder/configs/interfaces/fixture_single_nifti.yaml`
- `scripts/gc_submission_builder/container/Dockerfile` only for the selected
  manifest path; the scientific runtime remains unchanged
- `scripts/gc_submission_builder/README.md`
- `tests/test_gc_interfaces.py`
- `tests/test_gc_image_io.py`
- `tests/test_gc_runtime.py`
- `tests/test_gc_container_builder.py`
- synthetic official-interface inputs and output fixtures outside protected
  data
- `docs/INFERENCE_EXTRACTION_EVIDENCE_LEDGER.md`

### Desired changes

1. Replace the singular interface `output` with an ordered `outputs`
   collection.
2. Add explicit output `result_key` binding. Initially support `mask` and
   `probability` from the shared `PredictionResult` contract.
3. Use list order only for deterministic materialization/reporting; never infer
   semantics from position or slug spelling.
4. Reject duplicate input/output slugs, canonical dataset keys, result keys,
   or relative paths and reject unknown/unavailable result keys before success.
5. Add the exact official ISLES26 manifest from Section 8.7 and select it from
   the production container config.
6. Retain the opaque single-output NIfTI fixture manifest and its complete test
   coverage to prove transport coexistence and dataset/interface generality.
7. Validate the required `stroke-metadata` object at the transport boundary:
   `CENTER` is string-or-null, `CHRONICITY` is integer-or-null, and
   `DAYS_POST_STROKE` is number-or-null. Do not pass these values to the current
   unconditioned model or log their contents.
8. Bind `t1-brain-mri` to canonical raw key `T1` only in manifest
   configuration. Shared preprocessing receives no socket identifier.
9. Bind `stroke-lesion-segmentation` to `PredictionResult.mask` and
   `lesion-probability-map` to `PredictionResult.probability`.
10. Preserve the restored native continuous probability before fixed-threshold
    binarization. Do not threshold, normalize, or otherwise reinterpret the
    probability transport value.
11. Keep the existing NIfTI native writer and validations unchanged. Add
    parallel compressed-MHA materializers selected explicitly by
    `file_type: mha`.
12. Convert array order and RAS/LPS physical conventions explicitly when
    constructing SimpleITK images. Represent spacing, origin, and direction
    without silently approximating unsupported geometry.
13. Materialize the segmentation as binary `uint8` and the probability as
    finite floating point, initially `float32`, constrained to `[0, 1]`.
14. Reopen both MHA files and verify size, spacing, origin, direction,
    world-coordinate landmarks, dtype/domain, and identical output grids.
    MHA has no NIfTI qform/sform codes; validation therefore compares physical
    geometry rather than inventing equivalent form-code semantics.
15. Return HTTP 201 only after every declared output exists and passes reopened
    validation. Any partially written set returns non-success.
16. Replace the Cut 10A localhost `docker exec` HTTP probe with a request from
    an external tester sidecar on a Docker network. Keep the algorithm
    container without runtime internet access.
17. Apply the organizer's 300-second local `/invoke` timeout. Preserve the
    separately enforced ten-minute total platform-job limit for Cut 11.
18. Clear/reprovision output and transient state so neither a repeated invoke
    nor a restarted container can pass with stale artifacts.
19. Run the reconciled lifecycle with the same pinned p5n1 model artifact and
    a non-patient synthetic official-interface case.
20. Record a new evidence item that supersedes only E019's interface/output and
    internal-self-probe evidence. E018 and E019 remain valid as historical Cut
    10A image, model, dependency, and NIfTI-fixture lifecycle evidence for the
    exact states they tested.

### Expected tests and testing components

- The checked-in official manifest contains exactly the published input and
  output socket set, paths, result bindings, and transport types.
- `t1-brain-mri` maps to `T1`; no production Python branch contains an ISLES26
  slug or architecture identity.
- Valid metadata values and an all-null documented value set pass.
- Missing metadata keys, invalid types, arrays/non-object payloads, malformed
  JSON, or undeclared extras according to the selected schema policy fail with
  actionable transport errors.
- Missing/extra input sockets and path disagreement fail before preprocessing.
- Missing, duplicate, unknown, or unavailable output result bindings fail
  before HTTP success.
- The NIfTI fixture still writes and reopens its original single binary NIfTI;
  MHA support does not wrap, replace, or mutate this route.
- The MHA segmentation is binary `uint8`; the MHA probability is floating,
  finite, and within `[0, 1]`.
- With postprocessing disabled, the segmentation equals the configured
  threshold applied to the same native probability result.
- Both MHA outputs occupy the same verified native T1 grid.
- Synthetic MHA round trips cover anisotropic spacing, translation, axis
  flips/permutations, and oblique rotation with world-coordinate landmarks.
- Geometry not losslessly representable by the MHA/SimpleITK contract fails
  rather than being approximated.
- The external tester receives health 200 without redirect and invoke 201.
- HTTP 201 is impossible when either required file is missing, corrupt, or
  spatially invalid.
- The local compatibility invoke is bounded at 300 seconds.
- Read-only model/input, writable output/tmp, non-root execution, offline
  runtime, required label, and Linux/amd64 inspection remain green.
- Fresh-container and reprovisioned-invocation tests cannot reuse stale output
  or `/tmp` contents.
- The dependency-wide inference/evaluation/training/GC regression matrix
  remains green.

### Acceptance criteria

- The exact published ISLES26 input socket set dispatches through manifest
  bindings without hardcoded slug behavior in shared or transport Python.
- Both official output sockets are produced at their exact relative paths as
  compressed MHA.
- The segmentation is binary `uint8`; the probability map is continuous,
  finite floating point in `[0, 1]`.
- Both outputs match one another and the native T1 physical grid after being
  reopened independently.
- HTTP 201 is impossible until the complete output artifact set passes
  validation.
- The service is reachable from outside the algorithm container on port 4743
  and completes the official-style local invoke within 300 seconds for the
  development fixture.
- The existing NIfTI transport remains directly supported and tested.
- The image remains independently buildable/saveable and contains no model
  weights.
- An opaque non-ISLES fixture remains green, demonstrating that the runtime is
  configurable across registered datasets and interface manifests.

### Rollback

Retain the Cut 10A image and model pair strictly as development evidence. A
Cut 10B failure rolls back the MHA/multi-output adapter without removing the
working NIfTI transport. Cut 10A must not be uploaded as ISLES26 because it
lacks the complete official output set.

---

## 22B. Cut 10B-2: Hosted image-input normalization and production observability

### Context

Cut 10B correctly reconciled the published ISLES26 socket slugs, technical JSON,
ordered probability/segmentation output set, compressed-MHA writers, external
HTTP sidecar lifecycle, and complete-output transaction. Its Desktop tests used
the organizer README's local `.nii.gz` fixture convention and therefore retained
the Cut 10A assumption that the platform image directory itself would contain a
NIfTI.

The current challenge interface distinguishes the uploaded dataset format from
the algorithm transport format. `t1-brain-mri` is an image-kind socket read
from:

```text
/input/images/t1-brain-mri/<uuid>.mha
```

or:

```text
/input/images/t1-brain-mri/<uuid>.tif
```

An early hosted try-out of the uncommitted Cut 10B release candidate produced
Grand Challenge result:

```text
148ccec2-c194-49fb-a5ce-4fecb46b67c7
```

The T4 container initialized the CUDA runtime, reconstructed the DynUNet,
strictly loaded the model artifact, selected the ISLES26 adapter, and validated
the deployment policy. `/invoke` then failed with `InterfaceManifestError`
before preprocessing because the resolver accepted only `.nii.gz`. The image
logged only the exception type, so the exact hosted suffix and failed invariant
were not retained. The platform page nevertheless establishes that the
NIfTI-only input assumption is invalid.

Cut 10B-2 closes that hosted contract without changing trained preprocessing or
shared inference. The GC boundary discovers one allowed image representation,
canonicalizes it losslessly to invocation-local `.nii.gz`, validates the source
and canonical physical grids, and passes only canonical dataset-key paths into
the existing `PreprocessedCaseProducer`. It also adds structured, privacy-safe
startup/invocation telemetry so one expensive image upload yields actionable
evidence about stage progress, elapsed time, resource use, and failure cause.

The model archive from the failed try-out remains valid: startup proved that it
verified and loaded successfully. This cut rebuilds the independently replaceable
Docker image only.

### Dependencies

- Cuts 0-10A.
- The complete Cut 10B implementation and Desktop evidence; its closure commit
  is recorded in the evidence ledger during this cut.
- Official ISLES26 template commit
  `5e25bfc36b1dc6d9c04c8c364f53fb75c6afad32`.
- Hosted challenge interface declarations for the four published sockets.
- Hosted failure result `148ccec2-c194-49fb-a5ce-4fecb46b67c7`.

Cut 11 depends on Cut 10B-2, not merely the pre-hosted Cut 10B state, so T4
resource qualification measures input canonicalization, both MHA outputs, and
the production telemetry overhead. Cut 12 repeats the hosted try-out with the
corrected release candidate.

### Affected files and components

- `scripts/gc_submission_builder/runtime/interfaces.py`
- `scripts/gc_submission_builder/runtime/image_io.py`
- `scripts/gc_submission_builder/runtime/inference.py`
- `scripts/gc_submission_builder/runtime/app.py`
- new `scripts/gc_submission_builder/runtime/observability.py`, unless a smaller
  cohesive name is selected during implementation
- `scripts/gc_submission_builder/configs/interfaces/isles26.yaml`
- `scripts/gc_submission_builder/configs/interfaces/fixture_single_nifti.yaml`
- `scripts/gc_submission_builder/container/Dockerfile`
- `scripts/gc_submission_builder/container_builder.py`
- `scripts/gc_submission_builder/container_config.py` and container config only
  where resource/test parameters or build identity require them
- `scripts/gc_submission_builder/README.md`
- `tests/test_gc_interfaces.py`
- `tests/test_gc_image_io.py`
- `tests/test_gc_runtime.py`
- `tests/test_gc_container_builder.py`
- synthetic scalar 3D MHA, TIFF, `.nii`, and `.nii.gz` fixtures with nontrivial
  physical geometry
- `docs/INFERENCE_EXTRACTION_EVIDENCE_LEDGER.md`

No shared predictor, model adapter, trained transform, dataset modality mapping,
spatial restoration algorithm, threshold policy, or model artifact content is
changed by this cut. `src/inference/` remains transport-agnostic. If deeper
prediction substage timing cannot be observed without duplicating or bypassing
the shared pipeline, this cut records the shared prediction pipeline as one
span rather than weakening that boundary.

### Desired changes

1. Replace the input-only `file_type: nifti` assumption with an explicit
   image-kind manifest contract. Each image input declares its accepted source
   formats, canonical downstream format, cardinality, socket slug, relative
   path, and canonical dataset key.
2. Configure the ISLES26 image binding to accept `mha`, `tif`, `tiff`, `nii`,
   and `nii_gz`, with `nii_gz` as the canonical downstream format. MHA/TIFF are
   hosted image-kind representations; NIfTI forms retain organizer-local,
   fixture, and reusable competition support.
3. Preserve the opaque fixture manifest as a NIfTI-only accepted-format case so
   the generic parser proves that accepted source formats are manifest-owned and
   not globally inferred from ISLES26.
4. Discover regular files in the declared image directory without logging or
   depending on platform-generated filenames. Count allowed formats, require
   exactly one allowed image for `cardinality: one`, reject symlink/path escape,
   and fail clearly for zero or ambiguous supported files.
5. Ignore unrelated regular auxiliary files only when they cannot match an
   allowed image format; include privacy-safe counts by extension category in
   telemetry. Do not select the first match according to filesystem order.
6. Represent the resolved platform image with a typed transport value containing
   its slug, canonical dataset key, safe source-format identifier, byte count,
   and internal path. Do not pass this transport object into `src/inference/`.
7. Create one invocation-local scratch directory beneath writable `/tmp` and
   remove it on success or failure. Never write to read-only `/input` or mix
   canonical inputs with `/output` artifacts.
8. For `.nii.gz`, validate and use the original read-only path directly; do not
   rewrite a canonical file that already satisfies the downstream contract.
9. For `.nii`, gzip the single-file NIfTI bytes losslessly into scratch
   `.nii.gz`. A decompression test must reproduce the original `.nii` bytes.
10. For MHA and TIFF, use SimpleITK to read the source as a scalar 3D spatial
    image and write an invocation-local compressed `.nii.gz`. Do not manually
    transpose, flip, reorient, resample, crop, normalize, or construct a new
    model grid during transport canonicalization.
11. Reject two-dimensional, four-dimensional, vector/multicomponent, corrupt,
    nonfinite-geometry, or otherwise unsupported platform images before trained
    preprocessing begins. TIFF is supported only when SimpleITK resolves one
    scalar 3D image; a 2D image or unsupported series representation fails
    explicitly.
12. Inspect the source through its native reader and the canonical `.nii.gz`
    through nibabel/SimpleITK. Verify shape/size, scalar dtype semantics, finite
    positive spacing, finite origin/direction, equivalent RAS/LPS physical
    geometry, and multiple asymmetric voxel-to-world landmarks.
13. Treat this conversion as representation normalization only. It must not
    change the source grid or replace the saved dataset preprocessing. The
    registered adapter continues to own all trained orientation, spacing,
    intensity, crop, padding, and channel transforms.
14. Pass the canonical mapping `{dataset_key: canonical_nii_gz_path}` into the
    existing label-free `PreprocessedCaseProducer`. No shared or dataset code
    branches on MHA, TIFF, Grand Challenge slugs, or platform UUID filenames.
15. Retain a source-geometry inspection alongside the in-memory invocation so
    final output validation can compare both official MHA outputs with the
    originally received platform grid, not only with the intermediate NIfTI.
16. Preserve the existing compressed-MHA probability/mask transaction and its
    numeric validations. MHA remains the selected ISLES26 output; this cut does
    not add a TIFF output writer because the published interface accepts MHA.
17. Introduce one structured logging helper that emits stable single-line JSON
    payloads prefixed by a recognizable marker such as `GC_EVENT`. All fields
    must be JSON-serializable, deterministically named, bounded in size, and
    flushed to container stdout/stderr.
18. Add an image build/source fingerprint to both the exported image report and
    runtime startup log. It must identify the exact copied runtime/manifest/
    dependency state even when the build uses an uncommitted working tree; a Git
    commit alone is insufficient in that case.
19. Emit startup events for build identity, dependency/runtime versions, Python,
    PyTorch/CUDA/cuDNN/MONAI/SimpleITK/nibabel versions, GPU identity and total
    VRAM, visible device count, CPU count, cgroup host-memory limit, writable
    scratch capacity, model-artifact hashes, model parameter/dtype/device
    summary, dataset adapter/raw keys, active inference/runtime policy, interface
    bindings, initialization outcome, and initialization resource summary.
20. Emit invocation events for socket/interface resolution, safe image inventory,
    selected source format/size, metadata keys/types/nullness without values,
    source/canonical geometry, conversion outcome, preprocessing shapes,
    prediction tensor/device/policy summary, restoration/output space, probability
    and mask domain summaries, output geometry/compression checks, and final
    resource summary.
21. Replace exception-type-only startup and invocation logs with stable stage and
    error codes plus sanitized actionable detail. Known contract errors may log
    their safe message. Unexpected errors report exception type and internal
    module/function/line locations without dumping filenames, metadata values,
    tensors, image contents, or uncontrolled third-party exception text.
22. Keep logging bounded. Do not emit per-voxel, per-slice, or per-sliding-window
    events. Logging must not synchronize CUDA more often than required for
    accurate enclosing-stage timing and must not materially alter the ten-minute
    workload.
23. Update the organizer-style smoke runtime from 32 GB to the 16 GB host-memory
    request shown by the hosted job. Preserve 8 CPUs, GPU access, read-only
    input/model mounts, writable output, bounded writable `/tmp`, non-root user,
    offline algorithm networking, and the 300-second local invoke bound.
24. Record the hosted failure and corrected evidence without treating the failed
    model archive as invalid. E020 remains evidence for the exact Desktop Cut 10B
    state but no longer establishes upload readiness.

### Timing and performance telemetry contract

Timing uses a monotonic high-resolution clock such as `time.perf_counter()` and
reports elapsed seconds as finite nonnegative numbers. Wall-clock log timestamps
remain useful for sequence inspection but are not used to calculate durations.
The minimum startup timing fields are:

- `artifact_validation_seconds`;
- `policy_runtime_resolution_seconds`;
- `model_construction_checkpoint_load_seconds`;
- `adapter_manifest_initialization_seconds`;
- `startup_total_seconds`.

The minimum per-case invocation timing fields are:

- `interface_resolution_seconds`;
- `input_discovery_seconds`;
- `input_canonicalization_seconds` (zero or near-zero for validated
  `.nii.gz` pass-through, but still explicitly reported);
- `preprocessing_seconds`;
- `device_transfer_seconds`;
- `prediction_pipeline_seconds` (shared probability execution, sliding-window
  orchestration, native restoration, fixed threshold, and enabled
  postprocessing as one minimum span);
- `output_materialization_validation_seconds`;
- `invoke_total_seconds`.

If the shared pipeline exposes a natural nonduplicative observation seam,
`prediction_pipeline_seconds` may additionally report
`model_probability_execution_seconds`, `spatial_restoration_seconds`, and
`threshold_postprocessing_seconds`. These optional subdivisions must wrap the
canonical shared operations; Cut 10B-2 must not reproduce inference or
restoration logic merely to obtain more granular timing.

Every successful invocation emits one final timing summary. Every failed
startup/invocation emits all completed stage durations, the elapsed duration of
the active failed stage when safely measurable, total elapsed time until failure,
and the last active stage. Stage totals need not sum exactly to the enclosing
total because validation, logging, scheduling, and framework overhead remain,
but no child span may be negative or exceed its enclosing span beyond a declared
clock tolerance.

The same summary records peak CUDA allocated and reserved bytes, peak host RSS or
the closest reliable Linux/cgroup equivalent, configured cgroup memory limit,
and scratch free/used bytes around canonicalization. Resource values are
telemetry, not a fallback mechanism: an OOM or timeout remains a failed
invocation and is never silently retried on CPU.

These measurements cover code executing inside the algorithm container. They do
not claim to measure Grand Challenge instance preparation, image download,
container-image download, model expansion, or result upload outside the process.
Hosted execution-history timestamps remain the authority for those external
phases.

### Expected tests and testing components

- Manifest parsing accepts the explicit image-kind schema and rejects unknown
  kinds, formats, canonical formats, unsafe paths, or unsupported cardinality.
- The ISLES26 manifest carries the exact published slugs/paths and image-kind
  source allowlist; the opaque fixture remains NIfTI-only.
- Exactly one MHA, TIFF, `.nii`, or `.nii.gz` resolves correctly according to
  the selected manifest. Zero or multiple allowed images fail deterministically
  with counts rather than filenames.
- `.nii.gz` pass-through does not rewrite the source. `.nii` gzip round-trip is
  byte-identical after decompression and cleans scratch state.
- Synthetic MHA/TIFF canonicalization preserves scalar voxel values, shape,
  spacing, origin, direction, oblique/flipped/permuted geometry where the source
  format represents it, and asymmetric world-coordinate landmarks.
- Corrupt, 2D, 4D, vector, nonfinite, or nonrepresentable inputs fail before the
  case producer.
- A canonicalized MHA/TIFF record reaches the existing `PreprocessedCaseProducer`
  through canonical dataset key `T1` and a `.nii.gz` path; no shared-inference
  API changes to accept GC formats.
- Stub-runtime tests prove successful and failed startup/invocation logs contain
  stable event, stage, error/timing, build, policy, interface, and resource keys.
- Privacy tests inject identifying-looking filenames and metadata values, then
  assert that neither appears in emitted logs or exception responses.
- Timing tests use a controllable clock to prove finite nonnegative spans,
  success summaries, partial failure summaries, and correct stage association
  without sleeping.
- Runtime tests prove telemetry is bounded and does not emit per-window events.
- Container smoke tests invoke the exact built image through the external
  sidecar using an official-style MHA input and both official outputs, then
  assert HTTP 201, expected telemetry milestones, complete output validation,
  and original-source grid identity.
- The same built image is exercised with the NIfTI-only opaque fixture to retain
  dataset/interface/transport generality.
- Failure-injection container tests produce HTTP non-success, no accepted partial
  output set, and one actionable sanitized failure event.
- The Desktop live test converts one real non-patient-repository ISLES26 case
  into physically equivalent MHA/NIfTI representations, invokes the same
  DynUNet model, compares predictions within a declared tolerance, and verifies
  both MHA outputs against each original source grid.
- The relevant GC, inference, loader, evaluation, and training regression suites
  remain green before supervisory review.

### Acceptance criteria

- The hosted ISLES26 image-kind socket no longer depends on `.nii.gz` being
  present under `/input`; MHA is converted to canonical `.nii.gz` and TIFF is
  accepted when it is one scalar 3D image.
- NIfTI local/fixture inputs remain directly supported, and input
  canonicalization does not narrow the reusable builder to ISLES26.
- Source and canonical inputs are physically equivalent before preprocessing;
  final MHA probability and mask outputs match the original received input grid.
- No transport conversion reorientation, resampling, intensity processing, or
  model-space policy leaks outside the registered preprocessing adapter.
- Temporary canonical files are confined to `/tmp`, cleaned on success/failure,
  and cannot satisfy a later invocation as stale state.
- Startup readiness logs establish the exact image/runtime/model/interface/policy
  state and resource allocation.
- Every successful invocation reports the required stage durations, total time,
  peak GPU/host memory, scratch usage, input/output contracts, and validation
  outcomes.
- Every failed invocation reports a stable failed stage, error code/type,
  sanitized actionable detail, partial timing summary, and total elapsed time;
  `error_type` alone is insufficient.
- Logs contain no patient filenames, metadata values, tensors, or image contents
  and remain bounded independently of sliding-window count.
- The organizer-style MHA lifecycle succeeds from an external sidecar under the
  hosted 16 GB memory request and 300-second local invoke bound.
- Both required compressed-MHA outputs are reopened and validated before HTTP
  201; the existing NIfTI output route remains green.
- A new independently saved Docker archive is produced and paired with the
  already valid model archive for the next hosted try-out.

### Rollback

Retain commit `70cc6ca` and the Cut 10A NIfTI fixture evidence as the generic
container baseline. The known NIfTI-only Cut 10B state may be committed as an
explicit historical development checkpoint, but its image must not be
distributed as ISLES26-ready. If the broader image-kind schema or structured
logging implementation proves unsafe, reduce the implementation to the smallest
typed MHA-to-NIfTI canonicalizer and stage/error summary that satisfies the
hosted interface; do not bypass canonical preprocessing, weaken geometry
validation, embed the model in the image, or remove the already working
NIfTI/MHA output transports.

---

## 23. Cut 11: T4 resource qualification and ten-minute certification

### Context

Successful local correctness does not prove viability on the competition instance. Training occurs on an A100 40 GB, while submission runs on a T4 16 GB. Resource testing must use case batch one and deployment-equivalent settings.

### Dependencies

Cuts 0-10B-2.

### Affected files and components

- new or updated GC benchmark/test command
- `scripts/test_validation_memory.py` migrated or deprecated
- release manifests/reports
- synthetic and permissible representative validation inputs
- optional SLURM runner for T4-equivalent testing

### Desired changes

1. Consume the mandatory Cut 10B-2 startup/invocation timing schema and, through
   nonduplicative shared-pipeline observation seams where available, measure
   separately:
   - container/server startup;
   - model initialization;
   - input read;
   - preprocessing;
   - prediction;
   - spatial inversion;
   - native probability restoration;
   - postprocessing/segmentation derivation;
   - probability MHA write and reopen validation;
   - segmentation MHA write and reopen validation;
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
11. Measure the organizer-compatible external `/invoke` against both the
    300-second local call bound and the ten-minute total platform-job limit.
    The stricter eight-minute project target for the complete platform job
    remains recommended headroom.
12. Include both official outputs in host-RSS, disk-I/O, and repeated-invocation
    stability measurements; the probability map is part of the release
    workload, not an optional diagnostic artifact.

### Expected tests and testing components

- Baseline single-model FP16 prediction stays below 16 GB VRAM with meaningful margin.
- Host memory remains below the hosted 16 GB request with meaningful margin.
- Worst-case total per-case time remains below the project target and platform limit.
- Repeated invocations do not show unbounded memory growth.
- Both MHA outputs are written and reopened on every measured invocation; no
  timing result omits output conversion/validation.
- FP16 versus FP32 probability/metric differences are quantified.
- OOM is caught and reported as failure; no silent CPU fallback occurs in a GPU-certified release.
- Worker/IPC/ancdata failure modes are absent because the container path does not spawn data workers.

### Acceptance criteria

- A release report proves resource compliance for the exact image, model artifact, official ISLES26 manifest, inference policy, output space, runtime profile, and complete two-output artifact set.
- Optional enhancements that violate the budget are disabled even if they improve offline metrics.
- The selected release configuration has explicit runtime headroom.

### Rollback

Disable enhancements, retain `sw_batch_size=1`, or select a smaller model artifact. Do not change trained model attributes in deployment config to manufacture compatibility.

---

## 24. Cut 12: Platform dry run, release packaging, and documentation closure

### Context

The final cut validates the exact interface and platform behavior, not merely a local approximation. It also removes temporary duplicate paths only after the new route is certified.

### Dependencies

Cuts 0-11 and the Cut 10B manifest pinned to the official ISLES26 starter
template. Hosted platform access remains external.

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

1. Re-verify the checked-in official manifest against the then-current
   organizer template and record the pinned upstream commit. Do not defer
   multi-output/MHA implementation from Cut 10B into this release cut.
2. Run the organizer local lifecycle as written or through an equivalently
   strict external-sidecar wrapper with no algorithm runtime internet and both
   official outputs required.
3. Confirm through the hosted platform whether the T1 is materialized as the
   documented NIfTI. If it differs, update only the explicit input boundary
   adapter and manifest, retaining NIfTI and MHA transports already supported.
4. Save the Docker image archive and model tarball independently.
5. Verify both archives after reloading/extracting into clean temporary locations.
6. Upload or test through Grand Challenge's “Try Out Algorithm” facility when available.
7. Compare both returned output geometries with the uploaded test input and
   with each other; validate the binary segmentation and continuous
   probability domains independently.
8. Record image digest, archive hashes, model hashes, policy, code commit, runtime versions, and platform job identifier.
9. Record the active output space and prove that the production runtime rejected incompatible validation/evaluation options.
10. Document build, model replacement, native validation, container-diagnostic validation, platform test, troubleshooting, and rollback procedures.
11. Document the migration from `validation.inference` to top-level `inference` and its precedence rules.
12. Mark the older threshold-analysis implementation as delegated/deprecated once feature parity is confirmed.
13. Replace or mark the old memory script as legacy once the release benchmark supersedes it.
14. Retain compatibility wrappers for at least one stable release cycle where a demonstrated historical or external workflow requires them. This does not apply to the internal `valid_utils.py::build_validation_inferer()` path, which is removed and migrated atomically in Cut 4.
15. Document how a user creates an interface manifest by binding published socket slugs to canonical raw keys required by the saved model's registered dataset adapter; include one non-ISLES fixture/example to prevent the instructions from implying hardcoded T1/ISLES26 behavior.
16. Document the distinction between a dataset-agnostic builder/runtime implementation and the narrower dataset/model/interface combination certified in a particular release.

### Expected tests and testing components

- Clean-machine or clean-environment image load and invocation.
- Model-only replacement test: same image, new compatible singular model artifact.
- Image-only replacement test: same model artifact, new compatible code image.
- Deliberate incompatibility test fails during initialization.
- Grand Challenge try-out job completes successfully.
- Downloaded/returned segmentation has exact allowed values and expected
  geometry; the probability map is finite, continuous, within `[0, 1]`, and on
  the identical native grid.
- Documentation commands are executed as written.
- A fresh manifest using arbitrary fixture slugs can be validated against a registered non-ISLES26 model/dataset contract without Python code changes.

### Acceptance criteria

- Both upload artifacts are reproducible, independently replaceable, and verified.
- The exact phase interface is implemented; no placeholder slug remains.
- Both official MHA output values are accepted by the hosted platform and
  independently pass reopened physical-geometry validation.
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
  |
Cut-5-nnunet-precursor
       Space-aware 3D nnU-Net conversion/producer compliance
  |
Cut 5  Geometry-aware evaluation migration and dual-space validation
  |
Cut 6  Training validation/config migration
  |
Cut 7A Validation-only normalized record source
  |
Cut 7B Reusable typed evaluation-case producer
  |
Cut 7C Native-space restoration/output correctness
  |
Cut 7D Unified dual-space repository-model evaluator
  |
Cut 7E Live parity/full-valid-split evidence closure
  |
Cut 8  TTA/ensemble/postprocessing
  |
Cut 9  Model artifact builder
  |
Cut 10A Generic container runtime/image and NIfTI fixture lifecycle
  |
Cut 10B Official ISLES26 manifest, multi-output MHA, external lifecycle
  |
Cut 10B-2 Hosted image-input normalization and production observability
  |
Cut 11 T4 qualification
  |
Cut 12 Platform/release closure
```

Cut-5-nnunet-precursor is mandatory before Cut 5 so Cut 5 can remain confined
to the evaluator package. The Cut 7A-7E order is the acceptance order: record
discovery precedes rich case production; typed production precedes final
restoration certification; restoration and typed cases precede the unified
evaluator; and the finalized path precedes live evidence closure. Spatial
fixtures from Cut 7C may be developed earlier after Cut 4, but they do not make
the umbrella complete without Cuts 7A-7B and 7D-7E. Cut 8 and container release
work wait for all five Cut 7 subcuts. Cut 8 may be implemented incrementally;
no optional enhancement blocks a correct baseline container. Cut 10A is a
fixture-based implementation and evidence boundary. Cut 10B implements the
official output/socket contract; Cut 10B-2 is its mandatory hosted-input and
observability correction after the first platform try-out. Both precede Cut 11
so qualification includes canonicalization, telemetry overhead, and the
official two-output MHA workload. Cut 12 performs hosted/release closure rather
than postponing transport implementation.

The current reviewed-but-uncommitted spatial/output implementation maps to Cut
7C. Any exploratory `_iter_native_model_volume_samples`-style evaluator helper
is explicitly transitional and is replaced, not preserved, by Cut 7D. Existing
live evidence remains valuable input to Cut 7E, but final certification is
repeated through the finalized typed producer and unified evaluator.

Recommended pull-request granularity is one cut per PR, except very small scaffolding cuts may be combined if their tests and rollback boundaries remain clear.

---

## 26. Testing strategy

### 26.1 Unit tests

- policy validation;
- runtime-profile capability validation;
- top-level inference versus legacy validation-inference precedence;
- fixed output-space vocabulary and result metadata;
- model artifact manifest/hash validation at the release boundary;
- checkpoint prefix handling and strictness;
- predictor shape/domain checks;
- architecture-neutral predictor execution with backend-owned raw-output interpretation;
- sliding-window parameter validation;
- probability-before-overlap-blending order;
- early unsupported-generative failure naming the `ProbabilityPredictor` adapter extension point;
- TTA inversion;
- mean ensemble aggregation;
- connected-component filtering;
- interface dispatch and file discovery;
- registered dataset-preprocessing adapter dispatch;
- validation-only normalized record discovery without Dataset/DataLoader construction;
- validation-subset discovery with an absent/empty training partition;
- `load_labels` behavior and independence from `test_flag`;
- reusable producer construction and typed `PreprocessedCase`/`LabeledPreprocessedCase` contracts;
- independent native-label and transformed-label geometry validation;
- socket-slug to canonical raw-key binding validation;
- output-slug to explicit result-key binding and complete-output-set validation;
- required technical-JSON schema/type/nullability validation without model
  conditioning;
- parallel NIfTI and MHA image read/write validation;
- required `nnunet.export_space` composition and fixed-vocabulary validation
  for every supported 3D conversion preset;
- nnU-Net 3D source-versus-export geometry validation and explicit early
  rejection of uncertified 2D slice evaluation.

### 26.2 Spatial property tests

- preprocessing/inversion across generated shapes and affines;
- per-case native metadata capture across cases with deliberately different geometries;
- metadata retention before raw modality keys are merged/removed;
- world-coordinate preservation;
- continuous probability inversion before thresholding;
- output shape, affine, dtype, and allowed values;
- NIfTI qform/sform preservation on the existing NIfTI route;
- MHA size/spacing/origin/direction and RAS/LPS world-coordinate preservation;
- identical native grids for the ISLES26 probability and segmentation MHA
  outputs;
- explicit failure for geometry that cannot be losslessly represented by the
  selected transport;
- physical-volume filtering under anisotropic spacing.
- model-space result/reference pairing and native-space result/reference pairing;
- rejection of cross-space metric comparisons.
- nnU-Net native-grid export preservation and transformed-grid affine/spacing/
  orientation correctness;
- rejection of an nnU-Net `native_input` declaration when the exported grid
  differs from the selected raw reference;
- rejection of equal-shaped nnU-Net prediction/reference NIfTIs with different
  physical geometries.

### 26.3 Repository integration tests

- existing ISLES24 and ISLES26 loader, datalist, random-patch, and facade-routing tests;
- labeled-default and label-free preprocessing parity tests;
- characterization of the former validation-inferer behavior followed by direct shared-predictor consumer tests;
- absence of remaining `build_validation_inferer` imports/calls after Cut 4;
- evaluation model loader and volume producer tests;
- evaluation pipeline and threshold protocol tests;
- one unified repository-model loop selecting the correct reference from the declared result space;
- absence of evaluator-owned Dataset/Subset unwrapping, raw-record reconstruction, and native-only prediction orchestration;
- equal complete-case batch-one enforcement for model and native output, independently of sliding-window window batch;
- nnU-Net 3D conversion-config, volume-export, reopened-NIfTI, and compliant
  external-volume producer tests;
- early-error tests for repository and nnU-Net 2D inputs at the geometry-aware
  boundary while legacy conversion-only tests remain green;
- training validation smoke tests;
- old saved-run config translation tests;
- new top-level inference config composition tests;
- DP/DDP and checkpoint resume regressions.

### 26.4 Scientific parity tests

For a fixed model/checkpoint/input/config:

- old and new preprocessed inputs match;
- old and new model-space probability maps match within the predeclared tolerance;
- old and new sliding-window paths both produce patch probabilities before overlap blending;
- threshold-0.5 masks and metrics match;
- threshold sweep selection remains stable;
- native-space results are assessed against verified native labels;
- model-space and native-space evaluation consume the same typed preprocessed model input and predictor path;
- a completed evidence-only full-valid-split run exercises the finalized unified evaluator, with every source-geometry exclusion named and justified;
- evaluation of the same active inference config in native and container-test runtime profiles is numerically consistent;
- FP16 versus FP32 differences are reported rather than assumed negligible.

Exact tolerance should be established in Cut 0. CPU/GPU and FP16/FP32 comparisons may require different tolerances, but any metric-relevant disagreement requires investigation.

### 26.5 Container contract tests

- HTTP startup/health/invoke lifecycle;
- external-sidecar HTTP reachability; a self-probe alone is insufficient;
- invoke label;
- non-root execution;
- read-only input and writable output;
- offline network mode;
- model mount at `/opt/ml/model`;
- clean `/tmp` behavior;
- exact input/output interface paths;
- arbitrary-slug interface bindings to canonical dataset keys;
- image-kind discovery and MHA/TIFF/`.nii` canonicalization to downstream
  `.nii.gz`, with source/canonical physical-geometry equivalence;
- deterministic zero/multiple-file failures and invocation-local scratch
  cleanup;
- official ISLES26 slug bindings to `T1`, `mask`, and `probability` through the
  manifest rather than Python branches;
- complete two-output MHA materialization and reopened validation before HTTP
  201;
- preservation of the single-output NIfTI fixture lifecycle;
- organizer-compatible 300-second local invoke timeout;
- structured startup/invocation milestone, timing, resource, and sanitized
  failure-event assertions;
- privacy regression proving filenames, metadata values, tensors, and image
  contents are absent from logs;
- model-dataset adapter availability and raw-to-processed modality compatibility;
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
- model artifact hash;
- every config and checkpoint hash;
- repository commit and dirty-worktree status;
- runtime versions;
- interface manifest version;
- selected dataset preprocessing adapter and canonical input bindings;
- ordered output bindings, result keys, and transport formats;
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
**Mitigation:** One shared `src/models` lifecycle implementation, one `src/inference` predictor/orchestration path, registered loader-stack preprocessing adapters, and one typed evaluation-case producer reused across both repository-model result spaces; training and evaluation retain only their assessment/orchestration responsibilities. Compatibility wrappers exist only at demonstrated legacy boundaries, not around the removed internal validation inferer.

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

**Risk:** A dataset's original `.nii.gz` representation is confused with the
platform's image-kind serialization, causing the resolver to reject hosted MHA
or TIFF before preprocessing. Conversely, a competition output requirement may
be treated as a reason to replace the shared NIfTI output architecture.
**Mitigation:** Cut 10B-2 discovers one manifest-allowed image representation,
canonicalizes it to `.nii.gz` in `/tmp`, and verifies source/canonical physical
geometry before registered preprocessing. Preserve and test the existing NIfTI
writer and explicit MHA output adapters; never change trained preprocessing or
shared result semantics because of transport format.

### 28.11 Over-aggressive postprocessing

**Risk:** Removing small or secondary components deletes true lesions.
**Mitigation:** disabled by default, physical-volume parameters, held-out validation evidence, and policy-level rollback.

### 28.12 Invalid prediction/reference space pairing

**Risk:** Metrics compare a model-space prediction with a native-space label, or vice versa, producing errors or misleading results.
**Mitigation:** Every result and reference declares its space and geometry; evaluation validates both before metric computation.

### 28.13 Premature generic abstraction

**Risk:** Designing for every future model makes the critical path harder to validate.
**Mitigation:** Minimal probability predictor contract and explicit dataset-adapter registry; implement ISLES24/ISLES26 adapters from existing loaders and certify one ISLES26 discriminative backend first. Unknown datasets fail rather than triggering speculative generalization.

### 28.14 Dataset hardcoding in the container

**Risk:** The GC runtime copies ISLES26 transforms or assumes that a socket name is `T1`, so a compatible model from another registered dataset cannot be packaged without Python changes.
**Mitigation:** Select preprocessing from saved `dataset.id`; keep dataset-specific transforms in registered loader adapters; require an explicit interface-manifest binding from arbitrary socket slugs to canonical raw keys.

### 28.15 Label-mode coupling regresses existing validation

**Risk:** Reusing `test_flag` to mean blind inference removes labels from existing validation/test partitions, or image-only inference accidentally still requires a label.
**Mitigation:** Independent `load_labels: bool = True`, conditional datalist/transform keys, explicit false at blind-inference call sites, and a full flag-combination test matrix.

### 28.16 Native metadata is lost before restoration

**Risk:** Raw modality keys/meta are deleted during channel merging, leaving only a transformed tensor and no trustworthy path back to the case's input grid.
**Mitigation:** Capture native metadata per case and raw modality before preprocessing begins, retain an explicit trace/reference geometry, and test cases with different native grids.

### 28.17 Invalid interface-to-model modality binding

**Risk:** A valid platform socket is bound to the wrong canonical key or cannot produce the processed channels expected by the saved model, causing silent scientific input drift.
**Mitigation:** Validate binding uniqueness, adapter-recognized raw keys, and raw-to-processed coverage/order during initialization and before invocation; never infer bindings from slug text.

### 28.18 Model-specific output handling leaks into shared inference

**Risk:** `src.inference` begins branching on DynUNet/SwinUNETR or interpreting deep-supervision returns, coupling a supposedly reusable path to current architectures and duplicating backend logic.
**Mitigation:** Require an architecture-neutral `ProbabilityPredictor`; keep raw-output parsing, final-head selection, and activation in backend/model adapters; test shared execution with a generic mock predictor and reject unsupported backends at registration.

### 28.19 Sliding-window aggregation order changes scientific output

**Risk:** A refactor blends logits and applies sigmoid after overlap aggregation even though the accepted path predicts probabilities per patch and blends those probabilities, causing unmeasured probability and metric drift.
**Mitigation:** Characterize and preserve probability-before-blending in Cut 4 with parity/order regression tests. Treat logits-first blending only as a future explicit experimental policy with offline comparison and new release evidence.

### 28.20 Transformed nnU-Net tensor is written with a stale source affine

**Risk:** The 3D converter applies MONAI orientation or spacing transforms to an
image/label tensor but writes the resulting array with the raw input affine.
The file remains readable and may have plausible dimensions while representing
incorrect physical coordinates.

**Mitigation:** Cut-5-nnunet-precursor reads export geometry from the transformed
tensor grid, cross-checks image and label geometry, writes consistent
qform/sform, reopens spatial fixtures, and retains raw source geometry only as
separate provenance. Existing transformed exports remain uncertified until
inspected or regenerated.

### 28.21 2D slices are assembled into an unsupported 3D spatial claim

**Risk:** Repository or nnU-Net slice paths resize and export individual planes,
then `VolumeAssembler` stacks them by integer index without a verified
parent-volume affine, slice-plane placement, or invertible resize/preprocessing
trace. The resulting array could be mislabeled `native_input` or
`model_preprocessed` and used for invalid volume/surface metrics.

**Mitigation:** The geometry-aware evaluator supports 3D volume producers only
and rejects both repository and nnU-Net 2D inputs before assessment or assembly.
A later 2D compliance task must define typed parent/model geometry, slice axis
and placement, context-centre semantics, and an invertible reconstruction trace
with world-coordinate tests.

### 28.22 Rich native metadata inflates every training batch

**Risk:** Solving post-training native evaluation by adding raw records, untouched native labels, source NIfTIs, and full spatial traces to the ordinary training dataset item increases I/O, collation complexity, host memory, and worker-transfer costs for a training path that does not consume them.

**Mitigation:** Preserve the established lightweight training tuple. Cut 7B introduces a separate `LabeledPreprocessedCase` producer used only by post-training evaluation, diagnostics, and other consumers that explicitly require the rich payload.

### 28.23 Evaluator reconstructs cases behind the DataLoader contract

**Risk:** A native-only helper reaches through `dataloader.dataset`, unwraps `Subset`, resolves dataset records and paths, rebuilds preprocessing, and diverges from the model-space loop. It may work for one loader shape while silently coupling evaluation to training internals and duplicating scientific behavior.

**Mitigation:** Cut 7A exposes normalized validation records without constructing the training loader stack; Cut 7B turns them into typed rich cases; Cut 7D removes evaluator-owned record resolution and makes both spaces consume one typed prediction loop.

### 28.24 Complete-case batching is confused with sliding-window batching

**Risk:** An error describing `val_batch_size=1` as a native-space or GC-only rule suggests that native Python evaluation can process only one case per job, or leads users to reduce window throughput unnecessarily.

**Mitigation:** Enforce one complete case at a time for geometry-aware post-training 3D evaluation in both spaces, explain that a job may iterate the full split, and keep `inference.sliding_window.sw_batch_size` as the independent control for windows within the current case.

### 28.25 RAS/LPS or array-order error during MHA conversion

**Risk:** MONAI/nibabel tensor/affine conventions are copied directly into
SimpleITK, producing an apparently plausible MHA with flipped axes, incorrect
direction, or wrong origin.
**Mitigation:** Implement the conversion once at the transport boundary;
explicitly convert array order and RAS/LPS conventions; reopen with SimpleITK;
and test size, spacing, origin, direction, oblique landmarks, and world
coordinates. Fail on non-representable geometry.

### 28.26 Incomplete official output set

**Risk:** Segmentation writing succeeds but probability writing or validation
fails, while `/invoke` nevertheless reports HTTP 201 or a stale file satisfies
the test.
**Mitigation:** Treat outputs as one required artifact set, clear/reprovision
state, validate every declared binding, and return 201 only after the full set
passes. A partial write returns non-success.

### 28.27 Probability and mask semantics diverge

**Risk:** The exported probability is thresholded, normalized again, left on
the model grid, or differs spatially from the binary mask derived from it.
**Mitigation:** Bind both values explicitly to one `PredictionResult`; restore
the continuous probability before thresholding; derive the mask in the same
result space; and verify both outputs share the native T1 grid and expected
numeric domains.

### 28.28 Official metadata schema drift or leakage

**Risk:** Null or renamed technical fields cause an opaque runtime failure, or
metadata values are logged/consumed despite the unconditioned model contract.
**Mitigation:** Put field types/nullability in the manifest, fail with
transport-specific errors, avoid value logging, and keep metadata out of
preprocessing/prediction until a future saved model explicitly declares
conditioning support.

### 28.29 Localhost self-probe gives false HTTP confidence

**Risk:** Health/invoke succeeds only from inside the algorithm container while
Grand Challenge cannot reach port 4743 from its orchestrator.
**Mitigation:** Issue compatibility requests from an external tester sidecar on
the Docker network, retain `0.0.0.0:4743`, reject health redirects, and use the
organizer's 300-second local invoke bound.

### 28.30 Hosted failure is opaque after an expensive image upload

**Risk:** The container reports only an exception class, so socket inventory,
active stage, accepted file type, geometry contract, elapsed time, and resource
state are lost when the platform destroys the case container.
**Mitigation:** Emit bounded structured `GC_EVENT` startup/invocation milestones,
stable stage/error codes, sanitized detail, completed/failed stage durations,
build identity, and resource summaries. Assert those events in success and
failure container tests before upload.

### 28.31 Telemetry perturbs or floods inference

**Risk:** Per-window logging, repeated CUDA synchronization, unbounded config or
tensor serialization, or verbose third-party tracebacks consume time/memory and
may expose sensitive transport values.
**Mitigation:** Time only bounded enclosing stages with a monotonic clock; emit
one summary per major stage and no per-window events; log schemas and aggregate
contracts rather than values; sanitize known failures and report code locations
without uncontrolled exception payloads.

### 28.32 Input canonicalization exhausts scratch or leaves stale files

**Risk:** MHA/TIFF conversion duplicates a large input under `/tmp`, exceeds the
scratch allocation, or a failed invocation leaves a canonical NIfTI that a later
case reuses.
**Mitigation:** Check and log scratch capacity before conversion, use one
invocation-local directory, refuse insufficient capacity with an actionable
error, clean it in success/failure paths, and reprovision `/tmp` during
organizer-style repeated-invocation tests.

---

## 29. Definition of done for this CAP

This CAP is complete only when all of the following are true:

- [ ] `src/inference/` is the canonical architecture-neutral prediction implementation, with the 3D discriminative backend as the first supported implementation.
- [ ] Raw model-output/deep-supervision interpretation and activation remain backend-owned; shared inference has no model-architecture branching.
- [ ] Sliding-window inference preserves the certified probability-before-blending order unless a separately validated explicit policy is introduced.
- [ ] Training validation and live-model evaluation call `src.inference` directly; inference-specific symbols and the `build_validation_inferer()` façade are absent from `src/utils/valid_utils.py`.
- [ ] `src/models/` owns shared single-model construction and checkpoint loading; release-only strict validation is used by deployment without changing legacy resume behavior.
- [ ] Training validation, offline evaluation, native Python inference, container diagnostics, and GC deployment use one top-level `cfg.inference` schema.
- [ ] `cfg.inference_runtime` profiles enforce native-Python, diagnostic-container, and submission capabilities without silently rewriting requested policy.
- [ ] Existing saved runs containing only `cfg.validation.inference` remain supported through a tested compatibility translator.
- [ ] Explicit top-level inference config replaces rather than field-merges with historical validation inference settings.
- [ ] Existing evaluation uses it and passes parity tests.
- [ ] Training-time validation uses it without changing training behavior.
- [ ] ISLES24 and ISLES26 deterministic preprocessing are registered by saved `dataset.id` and shared by labeled native workflows and label-free inference without container-side transform copies.
- [ ] Dataset constructors default to `load_labels=True`; blind inference explicitly uses `False`; `test_flag` remains independent; and no dummy label is created.
- [ ] Native metadata is captured separately for every case and raw modality before preprocessing/channel merging, with sufficient geometry and trace for restoration.
- [ ] Post-training 3D evaluation discovers normalized validation records without constructing ordinary training datasets, DataLoaders, samplers, augmentation pipelines, workers, or requiring a non-empty training partition.
- [ ] `LabeledPreprocessedCase` provides one explicit rich boundary containing model input, trace, native input metadata, transformed model-space label/geometry, and untouched native label/geometry; label-free `PreprocessedCase` remains free of placeholder labels.
- [ ] Ordinary training and training-validation dataset items retain their lightweight established tuple contract.
- [ ] Every supported 3D nnU-Net conversion preset declares
  `nnunet.export_space`; native-grid declarations are verified, transformed
  tensors are written with transformed geometry, and existing source/export
  provenance records remain distinct.
- [ ] 3D nnU-Net prediction/reference producers validate both NIfTI geometries
  and enter evaluation through the same explicit space/geometry contract as
  repository-model volumes.
- [ ] Repository and nnU-Net 2D slice paths fail before geometry-aware
  evaluation until the deferred reconstruction contract is implemented and
  certified.
- [ ] The builder/runtime is dataset-agnostic across registered repository adapters and fails clearly for an unavailable adapter.
- [ ] Native-space restoration passes shape, affine, and world-coordinate tests.
- [ ] Repository-model post-training 3D evaluation uses one typed case/prediction loop for `model_preprocessed` and `native_input`; it contains no native-only record reconstruction, Dataset/Subset unwrapping, or duplicate restoration path.
- [ ] Geometry-aware post-training 3D evaluation enforces one complete case at a time in both spaces while keeping sliding-window `sw_batch_size` independent and allowing a job to process an entire split sequentially.
- [ ] During-training validation remains `model_preprocessed`; an attempted native-output request fails early and directs the user to post-training evaluation.
- [ ] The shared native NIfTI mask writer accepts only `native_input`, writes
  `uint8` `{0,1}`, and passes reopened-file geometry/qform/sform validation;
  adding ISLES26 MHA transport does not remove, wrap, or redirect this path.
- [ ] The finalized typed producer and unified evaluator pass live model-space parity, label-free native output, heterogeneous-grid cases, known-invalid-source fail-closed checks, and one completed evidence-only full-valid-split evaluation.
- [ ] A model tarball can be built from run-dir/checkpoint specifications and loads strictly from `/opt/ml/model/`.
- [ ] Docker image and model artifact archive build independently and together.
- [ ] The container implements the current Grand Challenge HTTP lifecycle and runs non-root/offline.
- [ ] The exact ISLES26 interface manifest is implemented with no placeholder socket slugs and explicitly binds each published input slug to its canonical raw dataset key.
- [ ] The official manifest maps `stroke-lesion-segmentation` to `mask` and
  `lesion-probability-map` to `probability` through explicit result keys rather
  than slug text or output order.
- [ ] Required stroke metadata fields and documented nullability are validated
  at the transport boundary without entering or conditioning the current model.
- [ ] Core preprocessing/inference receive canonical dataset keys only; arbitrary competition slug names remain isolated to the GC transport layer.
- [ ] The ISLES26 image-kind input accepts hosted scalar 3D MHA/TIFF and
  organizer-local NIfTI representations, canonicalizes non-`.nii.gz` inputs
  only beneath `/tmp`, and passes a verified `.nii.gz` path to registered
  preprocessing without reorientation, resampling, or intensity changes.
- [ ] Source/canonical geometry and asymmetric world-coordinate landmarks agree,
  and both final MHA outputs are independently verified against the originally
  received platform input grid.
- [ ] The official segmentation and probability outputs are both compressed
  MHA files on the identical native T1 physical grid; the segmentation is
  binary `uint8`, while the probability is continuous, finite floating point
  within `[0, 1]`.
- [ ] Production `/invoke` returns HTTP 201 only after every declared output has
  been written, reopened, and validated; partial/stale output sets fail.
- [ ] An external tester sidecar reaches health/invoke on port 4743 and the
  organizer-style local invocation completes within 300 seconds.
- [ ] Startup and every successful/failed invocation emit bounded privacy-safe
  structured events with exact build/runtime identity, stage/error outcome,
  required monotonic stage durations, total elapsed time, and GPU/host/scratch
  resource summaries; exception type alone is not accepted as diagnostics.
- [ ] Logs contain no platform-generated patient filename, metadata value,
  tensor, voxel content, or per-sliding-window event.
- [ ] A non-ISLES NIfTI fixture remains green, proving transport coexistence
  and preventing the official MHA requirement from narrowing the reusable
  builder/runtime to ISLES26.
- [ ] Native validation can explicitly and correctly evaluate `model_preprocessed` or `native_input` output against a reference in the same space.
- [ ] Container diagnostics can exercise either output space, while production `/invoke` accepts only `native_input`.
- [ ] The exact release candidate passes T4 memory and ten-minute qualification
  with headroom under the hosted 16 GB host-memory request.
- [ ] At least one Grand Challenge platform try-out succeeds before submission
  and returns both official outputs with independently verified geometry and
  numeric domains.
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
- model artifact manifest versioning and capability declarations.

A future diffusion implementation should therefore implement and register an adapter satisfying `src.inference.contracts.ProbabilityPredictor`, then extend model-artifact capability validation rather than fork the container or spatial pipeline. That adapter owns sampler invocation and conversion of backend-specific returns into finite `[B, C, *spatial]` probabilities; the shared inference package must not acquire DDPM/DDIM/OpenAI-diffusion branches.

The existing discriminative sliding-window contract predicts a probability for each patch and then blends overlapping probabilities. A future stochastic diffusion predictor may reuse that execution layer only after it defines and tests patch compatibility, noise seeding/correlation across windows, repeatability, seam behavior, and the meaning/calibration of its returned probabilities. Merely satisfying the tensor signature does not certify stochastic sliding-window inference.

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
9. **D8: Optional Grand Challenge enablement.** Permit a diffusion model artifact only after every previous cut passes and the artifact advertises the certified capability.

This future work should be its own PRD/CAP. The present document supplies the integration boundary and evidence pointers, but does not authorize implementation of 3D diffusion as part of the ISLES26 discriminative submission milestone.

---

## 34. Final implementation guidance

The safest first vertical slice is:

1. lock a real baseline;
2. load one DynUNet model artifact strictly at the release boundary;
3. preprocess one unlabeled T1 volume through the shared transforms;
4. produce one sliding-window probability map;
5. invert it to native space;
6. write and validate one binary NIfTI through the generic fixture transport;
7. compare that probability/mask against canonical offline evaluation;
8. place that same call behind the generic `/invoke` lifecycle;
9. reconcile the runtime with the official ISLES26 manifest by materializing
   the restored probability and its binary mask as two independently reopened,
   spatially identical compressed MHA outputs, while retaining the NIfTI route.

This vertical slice proves the most important claim of the architecture: the Docker container is a transport and runtime wrapper around the same inference implementation used to evaluate the model before submission.
