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

## Release gate: official interface values

`configs/interface_manifest.fixture.yaml` is deliberately a development fixture.
Its socket slugs and relative paths are not an ISLES26 release claim. Before upload,
replace it with a manifest containing the exact interface values published for the
active Grand Challenge phase, then rebuild and re-run the container lifecycle.

For the current NIfTI transport, every image socket must contain exactly one
`.nii.gz` file. A manifest input declares:

- `slug`: the opaque Grand Challenge socket identifier;
- `dataset_key`: the repository adapter's canonical raw modality key;
- `relative_path`, `file_type`, and single-value `cardinality`.

Socket slugs never enter shared preprocessing or model inference. The runtime reads
`/input/inputs.json`, selects the exact configured socket set, and passes only the
canonical raw-key-to-path mapping to `src.inference` with labels disabled.

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

The equivalent thin shell entrypoint is:

```bash
bash scripts/gc_submission_builder/container/build.sh
```

## Local lifecycle test

The test command starts the image with no network, a read-only root filesystem,
read-only input and model mounts, writable `/output`, transient `/tmp`, 32 GB host
memory, 8 CPUs, and one available GPU. It waits for model initialization, invokes
the HTTP endpoint from inside the isolated container, and validates the written
binary `uint8` NIfTI. For the single-input fixture, the external tester also
reopens the input and output and requires exact shape, affine, qform/sform, and
form-code agreement.

```bash
python3 -m scripts.gc_submission_builder.cli test \
  --image-tag cut10-dev \
  --model-dir /path/to/extracted/algorithmmodel \
  --input-dir /path/to/platform-shaped/input \
  --test-output-dir /new/empty/output-directory
```

The input directory must contain `inputs.json` and the socket directories declared
by the selected manifest. The output directory must be empty, preventing stale
predictions from satisfying the test.

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
those measurements belong to the following resource-certification cut. Likewise,
the fixture manifest does not satisfy the final official-interface release gate.
