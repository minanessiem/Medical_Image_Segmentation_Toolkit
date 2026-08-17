# Grand Challenge submission builder

This package produces two independently replaceable release artifacts:

- a model archive whose contents are extracted directly under `/opt/ml/model/`;
- a Linux/amd64 container image containing the shared repository-model
  inference code, preprocessing adapters, CUDA/PyTorch runtime, and Grand
  Challenge HTTP service, but no model weights.

The initial certified runtime is for one 3D discriminative repository model. The
builder and transport are dataset-agnostic across registered repository datasets:
the saved model config selects the dataset adapter, while a user-supplied interface
manifest maps arbitrary platform socket slugs to that adapter's canonical raw keys.

## Operator release recipe

This is the recommended end-to-end procedure for producing and testing one
Grand Challenge image/model release pair. Run every command from the repository
root in Linux or WSL. Use a machine with Docker, the NVIDIA Container Toolkit,
an available NVIDIA GPU, and the repository's Python environment activated.
Budget disk space for the local Docker image, Docker's temporary uncompressed
`docker save` tar, and the approximately 3.7 GB compressed release archive; the
archive size alone is not a sufficient free-space estimate.

The procedure deliberately builds, tests, and saves in separate steps. This
ensures that the exact image/model pair is exercised through the external HTTP
lifecycle before the large image archive is created for upload.

### 1. Pin the release inputs

Start from a reviewed commit and a clean working tree:

```bash
git status --short
git rev-parse HEAD
```

Define release-specific paths and names. `GC_RUN_DIR` must be a complete training
run containing `.hydra/config.yaml` and the selected checkpoint. Use a new release
directory: the builders refuse to overwrite model archives, artifact directories,
or image archives, and the lifecycle tester rejects non-empty output directories.

```bash
export GC_RUN_DIR=/absolute/path/to/training-run
export GC_CHECKPOINT=best_model_step_040000_dice_3d_0.5724
export GC_RELEASE_ROOT=/absolute/path/to/new-release-directory
export GC_MODEL_OUTPUT="$GC_RELEASE_ROOT/model"
export GC_IMAGE_OUTPUT="$GC_RELEASE_ROOT/image"
export GC_TEST_INPUT=/absolute/path/to/platform-shaped-test-input
export GC_TEST_OUTPUT="$GC_RELEASE_ROOT/lifecycle-output"
export GC_IMAGE_NAME=medseg-diffusion-gc
export GC_IMAGE_TAG=isles26-YYYYMMDD-candidate01

mkdir -p "$GC_RELEASE_ROOT"
```

The checkpoint may be specified by its unique basename, with or without `.pth`,
when it occurs exactly once in the supported run checkpoint directories. If the
name is ambiguous, pass an exact path relative to `GC_RUN_DIR`. Release checkpoint
selection never searches outside the chosen run.

### 2. Build and strictly validate the model archive

The current certified baseline uses native-space FP32 inference:

```bash
python3 -m scripts.gc_submission_builder.cli build-model \
  --run-dir "$GC_RUN_DIR" \
  --checkpoint "$GC_CHECKPOINT" \
  --inference-policy configs/inference/sliding_window_native.yaml \
  --output-dir "$GC_MODEL_OUTPUT" \
  --validation-device cpu
```

Select the inference policy explicitly for every release. In particular, do not
substitute `sliding_window_native_fp16.yaml` until the recorded FP16
sliding-window accumulation issue has been resolved and certified.
The current release runtime also requires TTA, ensembling, and postprocessing to
remain disabled; do not hand-edit unsupported parameters into the packaged
policy before their shared Cut 8 implementations and validation evidence land.

The command reconstructs the model from the saved resolved training config,
strictly loads the selected weights, resolves the registered dataset adapter,
validates the production policy, and produces:

```text
<model-output>/
  algorithmmodel/
    artifact_manifest.json
    config.yaml
    inference_policy.yaml
    weights.pth
  <complete-training-run-directory-name>.tar.gz
  model_build_report.json
```

The `.tar.gz` archive already has the correct root-relative layout for Grand
Challenge expansion beneath `/opt/ml/model/`. Do not wrap it in another parent
directory or manually rebuild it from `algorithmmodel/` before upload.

### 3. Build the model-independent ISLES26 image

```bash
python3 -m scripts.gc_submission_builder.cli build-image \
  --image-name "$GC_IMAGE_NAME" \
  --image-tag "$GC_IMAGE_TAG" \
  --interface-manifest scripts/gc_submission_builder/configs/interfaces/isles26.yaml \
  --output-dir "$GC_IMAGE_OUTPUT"
```

The image is built for `linux/amd64`, contains no model weights, and is inspected
for the required non-root user, Grand Challenge API label, platform, and absence
of embedded checkpoint payloads. `container_build_report.json` records the image
ID, Dockerfile and dependency hashes, interface-manifest hash, and exact copied
source fingerprint.

### 4. Prepare a platform-shaped lifecycle input

For ISLES26, `GC_TEST_INPUT` must have this layout:

```text
<test-input>/
  inputs.json
  stroke-metadata.json
  images/
    t1-brain-mri/
      <one-file>.mha
```

The image may instead be one supported `.tif`, `.tiff`, `.nii`, or `.nii.gz`
file. There must be exactly one manifest-accepted image in the socket directory.
The platform-generated filename is intentionally irrelevant.

`inputs.json` contains the platform socket declarations:

```json
[
  {
    "socket": {
      "slug": "t1-brain-mri",
      "relative_path": "images/t1-brain-mri"
    }
  },
  {
    "socket": {
      "slug": "stroke-metadata",
      "relative_path": "stroke-metadata.json"
    }
  }
]
```

For an unconditioned lifecycle fixture, valid nullable metadata is sufficient:

```json
{
  "CENTER": null,
  "CHRONICITY": null,
  "DAYS_POST_STROKE": null
}
```

Do not commit patient data or platform-generated patient filenames as fixtures.

### 5. Test the exact image/model pair

`GC_TEST_OUTPUT` must be absent or empty. The lifecycle command mounts the
generated `algorithmmodel/` directory exactly as Grand Challenge mounts the
separately expanded model archive:

```bash
python3 -m scripts.gc_submission_builder.cli test \
  --image-name "$GC_IMAGE_NAME" \
  --image-tag "$GC_IMAGE_TAG" \
  --interface-manifest scripts/gc_submission_builder/configs/interfaces/isles26.yaml \
  --output-dir "$GC_IMAGE_OUTPUT" \
  --model-dir "$GC_MODEL_OUTPUT/algorithmmodel" \
  --input-dir "$GC_TEST_INPUT" \
  --test-output-dir "$GC_TEST_OUTPUT" \
  --readiness-timeout-seconds 300
```

A passing lifecycle requires all of the following:

- external-sidecar `GET /health` returns HTTP 200;
- external-sidecar `POST /invoke` returns HTTP 201 within 300 seconds;
- the container runs non-root, offline, with read-only input/model mounts;
- `stroke-lesion-segmentation/output.mha` is binary `uint8`;
- `lesion-probability-map/output.mha` is finite `float32` in `[0, 1]`;
- both outputs reopen successfully and match the native T1 physical grid;
- the runtime log contains successful `startup_completed`,
  `input_canonicalized`, and `invocation_completed` `GC_EVENT` records.

Do not proceed to image export after a failed or incomplete lifecycle. Diagnose
the stable `stage`, `error_code`, and sanitized `detail` fields in the final
`GC_EVENT` record first.

### 6. Save the tested image

Pass the same image identity, manifest, and output directory used for the
lifecycle test:

```bash
python3 -m scripts.gc_submission_builder.cli save \
  --image-name "$GC_IMAGE_NAME" \
  --image-tag "$GC_IMAGE_TAG" \
  --interface-manifest scripts/gc_submission_builder/configs/interfaces/isles26.yaml \
  --output-dir "$GC_IMAGE_OUTPUT"
```

The output is a deterministic-gzip Docker archive named from the image and tag,
for example:

```text
medseg-diffusion-gc-isles26-YYYYMMDD-candidate01.tar.gz
```

Saving re-inspects the image and refuses to overwrite an existing archive.

### 7. Verify and record the release pair

Before upload, inspect both reports and verify both compressed archives:

```bash
python3 -m json.tool "$GC_MODEL_OUTPUT/model_build_report.json"
python3 -m json.tool "$GC_IMAGE_OUTPUT/container_build_report.json"
gzip -t "$GC_MODEL_OUTPUT"/*.tar.gz
gzip -t "$GC_IMAGE_OUTPUT"/*.tar.gz
sha256sum "$GC_MODEL_OUTPUT"/*.tar.gz "$GC_IMAGE_OUTPUT"/*.tar.gz
tar -tzf "$GC_MODEL_OUTPUT"/*.tar.gz
```

The model archive listing must contain only the four model artifact files at its
root. Record at least:

- repository commit and whether the source tree was clean;
- complete training-run directory name and exact checkpoint path;
- inference policy path and policy hash;
- model archive filename, size, and SHA-256;
- image reference, image ID, copied-source SHA-256, archive filename, size, and
  SHA-256;
- selected interface-manifest hash;
- lifecycle input identity, outcome, invoke time, and peak memory;
- hosted Grand Challenge result ID after upload.

The image and model remain independently replaceable, so filenames alone are
not enough to identify a tested pair. Preserve the two build reports with the
release record.

### 8. Upload to Grand Challenge

Upload the two artifacts according to their distinct roles:

| Local artifact | Grand Challenge destination |
|---|---|
| Docker image `.tar.gz` from `GC_IMAGE_OUTPUT` | Algorithm container image |
| Training-run-named `.tar.gz` from `GC_MODEL_OUTPUT` | Algorithm model expanded beneath `/opt/ml/model/` |

Attach the model to the algorithm version that uses the matching image. A
successful hosted result should show the expected image source fingerprint,
artifact config/policy hashes, T4 identity, `output_space: native_input`, both
official output bindings, and a final successful `invocation_completed` event.

The Grand Challenge execution history includes provisioning, image download,
and result upload outside the container. Compare the platform's **Invoke
Duration**, rather than its total duration, with the internal
`invoke_total_seconds` telemetry.

### 9. Rebuilding or replacing one side

- To change only weights, checkpoint, or inference policy, build a new model
  archive and retest it against the unchanged local image before upload.
- To change runtime code, dependencies, Dockerfile, or interface handling,
  assign a new image tag, rebuild the image, and repeat the exact-pair lifecycle.
- Never infer compatibility merely because a previous image and a previous model
  each passed independently.
- Keep the last hosted-successful image/model pair available as the rollback
  baseline while testing a candidate replacement.

## Reference

## Interface manifests

`configs/interfaces/isles26.yaml` records the official ISLES26 contract from the
organizer template pinned by the CAP. It binds `t1-brain-mri` to the registered
dataset key `T1`, validates the required `stroke-metadata` JSON object, and emits
both official compressed-MHA outputs: a binary lesion segmentation and its
continuous probability map.

`configs/interfaces/fixture_single_nifti.yaml` remains an opaque, non-ISLES
development fixture. It proves that the same runtime can retain the original
single-output NIfTI transport when a different manifest selects it.

Every current image socket contains exactly one manifest-accepted scalar 3D
medical image. A manifest input declares:

- `slug`: the opaque Grand Challenge socket identifier;
- `dataset_key`: the repository adapter's canonical raw modality key;
- `relative_path`, `kind: image`, and single-value `cardinality`;
- `accepted_formats`, selected by the competition interface;
- `canonical_format: nii_gz`, required by registered preprocessing adapters.

The official ISLES26 manifest accepts hosted MHA/TIFF plus local `.nii` and
`.nii.gz` representations. The runtime discovers one accepted regular file
without depending on or logging its platform-generated name. MHA/TIFF are read
as scalar 3D SimpleITK images and written as invocation-local compressed NIfTI;
uncompressed NIfTI is gzip-normalized byte-for-byte; `.nii.gz` passes through
read-only. Source and canonical grid, dtype, voxel content, and asymmetric
world-coordinate landmarks are checked before preprocessing. Canonical scratch
is created only beneath `/tmp` and removed on success or failure. TIFF fidelity
is measured from the TIFF actually received—the runtime does not invent header
information that its serialization did not carry.

Socket slugs never enter shared preprocessing or model inference. The runtime reads
`/input/inputs.json`, selects the exact configured socket set, and passes only the
canonical raw-key-to-path mapping to `src.inference` with labels disabled.
Each manifest output binds an opaque socket slug and relative path to an explicit
`PredictionResult` key (`mask` or `probability`) and transport type (`nifti` or
`mha`). Output order is deterministic but carries no semantic meaning.

## Model archive

```bash
python3 -m scripts.gc_submission_builder.cli build-model \
  --run-dir /path/to/training-run \
  --checkpoint best_model_step_040000_dice_3d_0.5724 \
  --inference-policy configs/inference/sliding_window_native.yaml \
  --output-dir /path/to/model-artifacts
```

The default archive name is the exact training-run directory name plus `.tar.gz`.
The archive contains only `artifact_manifest.json`, `config.yaml`, `weights.pth`,
and the standalone `inference_policy.yaml`. The builder hash-checks and strictly
load-tests that extracted layout before reporting success.

## Container image

Container-only configuration lives in `configs/container.yaml`; it contains no
model architecture or checkpoint values.

```bash
python3 -m scripts.gc_submission_builder.cli build-image \
  --config scripts/gc_submission_builder/configs/container.yaml \
  --interface-manifest /path/inside/repository/official-interface.yaml
```

The Dockerfile pins the immutable PyTorch 2.6.0/CUDA 12.6 linux/amd64 base digest
and a minimal inference dependency lock. It copies no checkpoint. Image build and
save inspect the resulting filesystem and fail if `/opt/ml/model` is non-empty or
a PyTorch checkpoint-like file exists beneath `/opt/app`. At runtime the platform
mounts the separately extracted model directory read-only at `/opt/ml/model`.
The builder also hashes every source/config/dependency file copied into the
image. That SHA-256 is written to the build report, image label, and runtime
environment, so an image built from an uncommitted review tree remains exactly
identifiable.

Runtime logs use bounded single-line `GC_EVENT` JSON records. Startup records
dependency/GPU identity, artifact and policy hashes, adapter/interface bindings,
model summary, named stage durations, and resource capacity. Invocation records
safe input format/count/geometry contracts, canonicalization, required stage
durations, output validation, peak CUDA allocation/reservation, host RSS/cgroup
memory, and scratch usage. Failures carry a stable stage and error code with
sanitized detail; platform filenames, metadata values, tensors, voxel data, and
per-sliding-window events are never logged.

The equivalent thin shell entrypoint is:

```bash
bash scripts/gc_submission_builder/container/build.sh
```

## Local lifecycle test

The test command starts the image on an internal Docker network, a read-only root filesystem,
read-only input and model mounts, writable `/output`, transient `/tmp`, 16 GB host
memory, 8 CPUs, and one available GPU. A separate tester sidecar on that same
offline network requires an exact HTTP 200 from `/health` and invokes `/invoke`
with the organizer's 300-second local timeout. The test then reopens every declared
output independently. Official MHA outputs must match the native T1 size, spacing,
origin, direction, and physical landmarks; the fixture NIfTI route retains exact
shape, affine, qform/sform, and form-code checks.

```bash
python3 -m scripts.gc_submission_builder.cli test \
  --image-tag isles26-dev \
  --model-dir /path/to/extracted/algorithmmodel \
  --input-dir /path/to/platform-shaped/input \
  --test-output-dir /new/empty/output-directory
```

The input directory must contain `inputs.json`, one directory per image socket,
and any declared technical JSON paths. The output directory must be empty before
the lifecycle starts. Within an invocation, declared output directories are
cleared and the full set is staged, moved, reopened, and validated before HTTP 201;
a failed partial write is removed rather than reported as success.

## Save and combined build

```bash
python3 -m scripts.gc_submission_builder.cli save
python3 -m scripts.gc_submission_builder.cli build-all \
  --run-dir /path/to/training-run \
  --checkpoint best_model \
  --output-dir /path/to/model-artifacts \
  --container-output-dir /path/to/image-artifacts
```

`save` writes a gzip-compressed Docker archive and refuses to overwrite an existing
one. `build-all` builds the model artifact, builds the independent image, and saves
the image archive, but it does not run the exact-pair lifecycle test. Prefer the
staged operator recipe for release candidates. `container/save.sh` exposes the
same standalone save operation.

## Diagnostic result spaces

Production `/invoke` is fixed to the `gc_submission` runtime profile and refuses to
write anything except `native_input` output. Container diagnostics are separate:

```bash
docker run --rm --gpus all \
  -v /path/to/model:/opt/ml/model:ro \
  -v /path/to/input:/input:ro \
  -v /path/to/diagnostic:/diagnostic \
  IMAGE python -m scripts.gc_submission_builder.runtime.diagnostic \
    --output-space model_preprocessed \
    --retain
```

`--output-space` is a diagnostic-only override of the otherwise verified artifact
policy. It is accepted only by `gc_container_test`; the production
`gc_submission` profile rejects it. Diagnostics can therefore exercise either
declared result space without changing the archive or weakening `/invoke`.
Diagnostic output may not be `/output` or any path beneath it; the example writes
only beneath the separately mounted `/diagnostic` tree.

## Certification boundary

Desktop container smoke tests establish code, dependency, transport, and spatial
behavior. The current Cut 10B-2 baseline has also completed successful hosted T4
invocations with both official outputs, including a large oblique anisotropic
input. Hosted evidence is exact-image/exact-model evidence: it does not certify
untested checkpoints, a worst-case phase volume, or optional TTA, ensemble, and
postprocessing policies. Cut 11 retains formal worst-case resource qualification,
while Cut 12 records the final chosen image/model pair and platform result IDs.
