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
the image archive. `container/save.sh` exposes the same standalone save operation.

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
behavior. They do not certify AWS T4 peak memory or the ten-minute case deadline;
those measurements belong to the following resource-certification cut. The
official manifest and two-output lifecycle satisfy Cut 10B's interface boundary;
hosted image-input normalization and observability are completed by Cut 10B-2;
upload readiness still depends on its rebuilt-image lifecycle evidence, Cut 11
resource certification, and the final platform dry run in Cut 12.
