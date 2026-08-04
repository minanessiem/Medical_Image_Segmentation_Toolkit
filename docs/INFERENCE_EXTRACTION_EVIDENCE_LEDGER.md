# Inference Extraction Evidence Ledger

**Governing document:** `docs/PRD_CAP_Inference_Package_Extraction_290726.md`

**Purpose:** Preserve durable, environment-dependent evidence for the inference
extraction CAP. The PRD defines what must be built, the remote-connection
runbook defines how tests are executed, and this ledger records which exact
source state was tested, what was observed, and what the result proves.

**Accepted coverage:** E001 records the Cut 0 legacy GPU baseline, E007 the
final Cut 1 contract, E009 the transition-only Cut 2 model-loading parity, and
E010 the current Cut 3 preprocessing snapshot. E011 records the Cut 4 shared
probability-execution contracts, and E012 records its real-model Desktop FP32
parity together with the unresolved FP16 finding. E013 records the accepted
Cut-5 nnU-Net precursor, and E014 records the accepted Cut 5 evaluator and its
live repository-model/nnU-Net evidence. E015 records the Cut 6 training-
validation config migration and live shared-predictor validation. E002-E006
and E008 are retained only as compact supersession records because they do not
represent accepted current designs. E016 retains the exploratory,
pre-decomposition Cut 7 restoration evidence. E017 records the completed Cut
7E verification of the finalized Cut 7A-7D architecture. E018 records the
pre-closure Cut 10A Desktop lifecycle, E019 records the committed Cut 10A
closure state, and E020 records the official Cut 10B interface reconciliation.
Cut 11 T4 qualification and Cut 12 platform/release gates remain open.

## Operating rule

An evidence item is accepted only when it records enough information to
distinguish the tested state from later or earlier revisions:

1. the question being answered;
2. the pinned code and, where relevant, model/config/input state;
3. the execution environment and exact command;
4. the result and durable artifact references or hashes;
5. the acceptance scope and invalidation conditions.

For a pending asynchronous job, also record the requested resources, expected
artifacts, outcome handling, and which dependency-safe work may continue. Once
the job is reconciled, remove obsolete queue forecasts and pending-work
instructions from the durable record.

Passing tests on one host does not imply authority for another environment.
An output is not accepted merely because a Slurm job or remote command exits
successfully; its provenance must match the pinned state.

---

## Persistent implementation observations and decisions

### O001: Test environments have different authority

| Environment | Observed runtime | Appropriate authority |
|---|---|---|
| Home desktop WSL | Python 3.10.12, PyTorch 2.8.0+cu128, OmegaConf 2.3.0 | Immediate unit, contract, synthetic-fixture, and consumer-regression feedback |
| LRZ standardized `.sqsh` | Python 3.12.3, PyTorch 2.6.0+cu124, MONAI 1.5.0 | Training-compatible dependency behavior, containerized CPU tests, and LRZ GPU characterization |
| Final Grand Challenge image on T4-equivalent hardware | Not yet built or certified | Release authority for CUDA compatibility, FP16/FP32 behavior, memory, timing, HTTP lifecycle, and output transport |

Desktop success accelerates development but does not establish `.sqsh` or T4
numerical/runtime parity. Every environment-sensitive acceptance statement
must name the environment that supplied it.

### O002: Desktop testing uses isolated worktrees

The existing desktop checkout contains unrelated user work and must not be
repurposed for CAP testing. Use a detached, cut-specific worktree built from a
pinned commit plus only the active cut's changes. The worktree remains
immutable while a test is running.

The authoritative connection, transfer, environment-activation, test-runner,
evidence-capture, and cleanup procedure is kept in the local, ignored remote
operations runbook. Endpoint-specific operational details are intentionally
not versioned in this public repository. Do not duplicate that procedure here.

### O003: Strict release loading is deferred beyond Cut 2

Cut 2 transferred the established single-model loading behavior into
`src/models/` without changing its strict-first/permissive-fallback semantics,
prefix handling, gradient state, or requested-device behavior. Training resume
and repository-model evaluation continue to use that compatible behavior.

The eventual model artifact and production runtime must add a separate strict
release boundary that rejects missing or unexpected state-dict keys. That work
belongs to the later artifact/release cuts and must not be retroactively
attributed to Cut 2.

### O004: Native FP16 probability-bound handling remains unresolved

E012 observed a real-model Gaussian sliding-window result with 139 voxels
slightly above the probability upper bound, reaching `1.0002766848`. Each
individual patch probability had already passed the `[0,1]` contract, and the
independent pre-extraction sliding-window expression reproduced the same
overshoot. The finding is therefore an existing finite-precision aggregation
boundary issue exposed by Cut 4 validation, not evidence of Cut 4 numerical
drift.

No behavior change is authorized as part of the current Cut 4 review. Before
the `native_fp16` profile can be certified, a later task must explicitly choose
and test the probability-bound policy. Candidate measures to evaluate are:

- limit autocast to backend/model prediction, cast each patch probability to
  FP32, and execute blending and normalization outside autocast;
- apply an observable final `[0,1]` clamp after patch validation, finite-value
  checks, and blending, with a guard against unexpectedly large corrections;
- determine whether both measures are required across representative cases and
  the T4-equivalent release environment.

That task must rerun real-model FP16 cases, verify thresholded-mask and metric
behavior against FP32, record correction counts and extrema, and measure memory
and runtime on the release-authoritative environment. Relaxing the validator
while leaving out-of-contract values in circulation is not an accepted
resolution. Until this work is completed, E011's synthetic FP16 test is only
contract coverage and `native_fp16` remains uncertified.

### O005: Geometry-aware 2D slice evaluation is deferred and must fail early

The current 2D evaluation path cannot make the spatial claims required by the
new geometry-aware evaluation contract. `SliceSample` carries tensors, case
and slice identifiers, and an untyped metadata mapping, but it does not require
prediction/reference spaces, parent-volume geometries, an invertible slice
trace, or verified physical placement within a 3D grid. `VolumeAssembler`
orders slices by integer index and stacks them into `[C,H,W,D]`; it does not
prove a common parent affine, spacing, orientation, slice axis, physical slice
position, or prediction/reference geometry.

The current producers also retain inconsistent and insufficient information:

- the online ISLES24 2D loader resizes native planes to the model image size
  without returning a geometry/transform record suitable for reconstruction;
- the online ISLES26 2D loader returns identity/path metadata but not the
  original and model-space geometries or an invertible record of its
  full-volume preprocessing and in-plane resize;
- the precomputed nnU-Net 2D loader records source metadata, but its returned
  tensor may be resized without a corresponding verified output-grid affine;
- the nnU-Net 2D converter writes each resized plane as an independent
  `[H,W,1]` NIfTI and derives a per-slice affine by scaling source in-plane
  vectors and offsetting along one presumed slice direction. Its JSONL record
  is useful provenance, but the conversion does not establish a typed,
  invertible mapping for reassembling all predictions into either the exact
  source parent grid or the full model-preprocessed parent grid;
- the nnU-Net 2D conversion naming convention turns every slice into an
  independent nnU-Net case. Neither nnU-Net prediction output nor the current
  adapter proves that all returned slices are complete, consistently ordered,
  drawn from one parent geometry, or still associated with the correct centre
  plane for context-slice inputs;
- the nnU-Net slice-file adapter can identify a slice and read a NIfTI affine,
  but does not currently verify prediction/reference affines or establish a
  common parent-volume geometry for the assembled stack;
- context-slice inputs still predict the centre plane, but that centre-plane
  placement is not represented as a spatial contract.

Consequently, equal 2D shapes or a common `volume_id` do not establish that
slice predictions and references occupy the same physical plane, nor that a
stacked result occupies the native or model-preprocessed 3D grid. Allowing
volume metrics, physical-volume metrics, surface distances, exports, or
native-space claims from these samples would risk spatially plausible but
scientifically unsupported results.

The project decision is therefore:

- 2D slice geometry compliance is a separate follow-up task outside the
  current inference-extraction PRD;
- the geometry-aware evaluation path introduced by the current work supports
  3D volumes only;
- a request with `data_mode.dim=2d`, or a `SliceSample`/assembled volume that
  reaches the new geometry-aware path without the future typed spatial
  contract, must fail before prediction assessment or volume assembly with an
  actionable error naming the missing 2D geometry/reconstruction support;
- `VolumeAssembler` must not manufacture a parent affine, assume that
  first-slice metadata describes the stack, or treat index order as physical
  geometry;
- no compatibility default may label unidentified 2D slices as
  `model_preprocessed` or `native_input` merely to keep an old path running;
- existing standalone 2D code is not certified by this CAP and must not be
  presented as compliant with the new result/reference-space contract;
- existing nnU-Net 2D conversion may remain available for legacy standalone
  workflows, but its converted datasets and predictions must fail early when
  submitted to the new geometry-aware evaluator.

The active 3D scope is narrower and explicit. Live repository-model volumes
may be evaluated in `model_preprocessed` space now and in `native_input` space
after Cut 7 restoration. Full-volume nnU-Net prediction/reference pairs may be
evaluated only when their originating conversion preset explicitly declares
their common space as `model_preprocessed` or `native_input`, as recorded in
O006; their complete NIfTI geometries must agree independently of that
declaration.

A future 2D compliance task must cover repository slice loaders, precomputed
slice loaders, nnU-Net 2D conversion, nnU-Net prediction adapters, and volume
assembly together. It should define a typed slice spatial contract that
retains, at minimum, original and model parent-volume geometries, slice axis,
centre-slice index, ordering, pre/post-resize plane shapes, and an invertible
in-plane/full-volume transform trace. It must then update each 2D producer,
verify prediction/reference plane geometry, reconstruct floating-point
probabilities before thresholding, and test axis permutations, reverse order,
anisotropic spacing, oblique affines, missing/duplicate slices, context-slice
placement, and world-coordinate preservation. Only after those acceptance
tests pass may 2D slice inputs enter geometry-aware volume evaluation.

### O006: 3D nnU-Net space is conversion-owned and export geometry must be truthful

A NIfTI affine describes the physical grid encoded by a file, but it does not
establish whether this project should classify that grid as the untouched
`native_input` grid or as a `model_preprocessed` grid. The repository's nnU-Net
conversion pipeline can export volumes after applying MedSegDiff preprocessing.
A conversion may resample or reorient a source case before writing the
nnU-Net-compatible NIfTI. Conversely, another conversion may preserve the raw
input grid. File extension, filename, array shape, affine validity, or nnU-Net
ownership cannot distinguish these semantic cases reliably.

The earlier proposal made space classification a separate user-supplied
evaluation value. Repository inspection established a better source of truth:
`evaluate_nnunet_results.py` already composes the exact nnU-Net conversion
preset with the evaluation preset. The complete 3D conversion preset therefore
owns required `nnunet.export_space`, using the same fixed vocabulary as shared
inference: `model_preprocessed` or `native_input`. It has no inferred or
compatibility default, and the evaluator consumes rather than redeclares it.

Current presets divide into two groups that the precursor must assign
explicitly:

- `isles24_cluster_3d_baseline` and `isles24_local_3d_baseline` receive
  `native_input`, because their base config disables orientation and spacing
  transformations and disables full-volume padding;
- `isles26_cluster_3d_t1raw`, `isles26_local_3d_t1raw`, and
  `isles26_atlas30_cluster_3d_t1raw` receive `model_preprocessed`, because the
  inherited ISLES26 preprocessing reorients to RAS and resamples to 1 mm
  isotropic spacing;
- `isles26_local_3d_t1raw_native` receives `native_input`: it retains T1-raw
  intensity handling but disables orientation and spacing, explicitly allows
  native spacing, retains the no-padding full-volume policy, and uses a
  distinct nnU-Net dataset identity. `T1_RAW` alone does not imply native
  spatial geometry.

The inspection also exposed an exporter defect. `VolumeExportStrategy` obtains
the image and label tensors after repository preprocessing but currently loads
the raw source image and assigns its affine to the exported arrays. For the
ISLES26 presets, that can write a reoriented/resampled array with an
untransformed affine. The resulting file is loadable but its physical geometry
is untrustworthy. Cut-5-nnunet-precursor must instead obtain export geometry
from the transformed MONAI image/label grid, cross-check aligned image and
label geometry, and write that verified affine/qform/sform. Raw source geometry
remains separate provenance.

The configured semantic declaration must be checked against actual output. A
`native_input` export must match the selected raw reference shape and physical
geometry. A `model_preprocessed` export must match the transformed tensor grid.
The existing `dataset.json` source context and `export_provenance.jsonl` should
record `export_space` plus distinct source/export geometry; no additional hash
or artifact-provenance subsystem is required.

This declaration does not replace evaluation-time geometry validation. The
3D nnU-Net producer must still read prediction and reference NIfTI geometry
independently and reject shape, affine, spacing, or orientation mismatches
before metrics. Geometry agreement proves that the pair occupies the same
encoded grid; `nnunet.export_space` states what that grid means within this
repository's result-space contract. The selected space and verified geometry
must enter the shared `VolumeSample` contract and reports so model-space and
native-space analyses cannot be accidentally combined.

All nnU-Net config, conversion, affine-writing, 3D producer, and 2D early-error
work belongs to Cut-5-nnunet-precursor. The finalized Cut 5 consumes the
compliant 3D producer and changes only the common evaluator package. Previously
generated spatially transformed nnU-Net datasets are not certified merely by
adding the config key; they require inspection and should be regenerated if
their arrays were written with stale source affines.

### O007: Paired model-space image/reference grids are not yet authoritative

Cut 5 live validation exposed a preprocessing issue that had previously been
hidden by the evaluation path's loss of spatial metadata. The historical
ISLES26 pipeline transforms the image and label independently and later treats
their tensors as index-aligned. For most cases this produces matching grids,
but it does not guarantee that both outputs share one exact physical target
grid. Cut 5 now retains the metadata long enough to reject that unsupported
assumption before calculating metrics.

A complete data-loader-only audit of the 64-case `val_fast` split found two
model-space mismatches:

- `sub-r065s005`: the raw image and label have the same
  `[184, 512, 512]` shape and nominal `(0.9993909, 0.5, 0.5)` mm spacing. Their
  raw affines agree at the evaluator tolerance (`2.98e-8` maximum difference),
  but a tiny direction-matrix difference is amplified when orientation and
  spacing are applied independently. Both transformed arrays have shape
  `[184, 257, 256]` and RAS orientation, while their transformed affines differ
  by as much as `0.4835456` mm.
- `sub-r069s031`: the raw image and label have the same
  `[160, 224, 224]` shape and nominal spacing, but their raw affines already
  differ by approximately `0.0007935` mm. The independently transformed
  `[160, 230, 230]` RAS grids retain a maximum affine difference of
  approximately `0.0007941` mm.

This evidence does not establish mistaken lesion annotation. It establishes a
header/grid and independent-resampling problem: equal shapes and nearly equal
source headers are insufficient to prove that separately transformed arrays
occupy the same model-space grid. Merely copying the image affine onto the
label, loosening evaluator tolerances, or ignoring the mismatch would conceal
rather than correct the spatial relationship.

The accepted future solution is to make the transformed image grid the single
authoritative model-space target grid for each case:

1. Apply the trained image orientation/spacing preprocessing and capture its
   exact resulting shape and affine.
2. Resample the untouched native label directly onto that exact target grid
   with nearest-neighbour interpolation.
3. Preserve the untouched native label and native image metadata separately;
   do not rewrite an affine without resampling voxel values.
4. Require the resulting model-space image and label to have identical shape
   and physical geometry before metrics.
5. Continue to use the image transform trace as the authority when Cut 7
   restores floating-point predictions to `native_input` space.

This is justified because the model consumes the transformed image grid, so
that grid is the only defensible reference for model-space labels and
predictions. Nearest-neighbour interpolation preserves discrete label values,
while direct resampling from the native label avoids compounding interpolation
through an intermediate independently chosen grid.

Before changing production preprocessing, the follow-up must characterize the
full dataset, compare old and corrected masks (changed voxels, Dice, lesion
volume, and visual alignment for affected cases), prove that image tensors and
model prediction hashes are unchanged, and rerun the complete evaluation split.
The strict Cut 5 geometry rejection must remain in place. Its error should also
be enhanced later with the case identifier and compared affine values.

This remediation is deliberately deferred. Blind Grand Challenge inference has
no reference label and therefore does not encounter this paired-reference-grid
problem; it does not block the basic container-generation path or the current
competition test phase. It does block claiming complete, scientifically
certified model-space evaluation for affected cases and must be resolved before
full-dataset spatial certification or final Cut 7 native-space certification.
The exact Desktop evidence and hashes are recorded in E014.

---

## Evidence item E001: Legacy four-case GPU baseline

### Status

| Field | Value |
|---|---|
| Evidence ID | `E001` |
| Slurm job | `5722967` |
| Submitted | `2026-07-31T00:52:51+02:00` |
| Final state | `COMPLETED` |
| Exit code | `0:0` |
| Elapsed | `00:02:00` |
| Result | `ACCEPTED` |
| CAP relationship | Accepted measured-output portion of Cut 0; parity source for Cut 4 |

### Question being answered

Can the unmodified legacy 3D discriminative evaluation path load the selected
raw DynUNet checkpoint and produce probability volumes for four locked cases
within a 16 GB GPU-memory ceiling, while preserving artifacts suitable for
later comparison with the shared predictor?

This is a legacy characterization run, not T4 or Grand Challenge release
certification.

### Pinned code state

| Concern | Pinned value |
|---|---|
| Repository | `/dss/dsshome1/0D/di38tap/code/medseg-diffusion_ISLES24`, mounted at `/mnt/code/medseg-diffusion_ISLES24` |
| Execution-time HEAD | `88d9b55dd2e3fa52df0c842e356b7e96a25b7b0b` |
| Submitted batch-script SHA-256 | `771ba4d871d4b80c3b65e1d758545da5535017ef49377d9d3fceb8478a60fcb9` |
| Tracked worktree state | No tracked or staged modifications |
| Documented unrelated untracked state | Existing logs, `packages.txt`, and `configs/loss/.multitask.yaml.swp`; none shadowed an E001 import or config input |

The job used a live checkout. The accepted manifest confirmed the execution
HEAD and tracked worktree state against these pinned values.

### Pinned model and training provenance

| Concern | Value |
|---|---|
| Run | `discriminative_dynunet_isles26_atlas30_3d_randompatch_280726/dynunet_128_3d_k3-3-3-3_f32-64-128-256_b3_p5n1_adamw2e4_wcos10_s100K_ldicefocal100log_dsup2_t1RAW_augSPAT3D_disc_e1_2026-07-28_20-13-22` |
| Saved config SHA-256 | `7152715bded1f11cc244a8a4270b6cc592b8a0e555f40d6d40af5b2924cc918a` |
| Saved overrides SHA-256 | `6cdd2234aae0716431c17287dd35f6d3f2ebaf1211470e1ed8896c5df4761f30` |
| Checkpoint | `models/best/best_model_step_040000_dice_3d_0.5724.pth` |
| Checkpoint SHA-256 | `120c93ee6a32f79829bf8a6b2d3ab7db59861f865ad1e8a37889f87ab0c82441` |
| Training revision | `60e24ecfa8984673d20586b48a361cace7095bfd`, strongly inferred from LRZ reflog because the historical run did not record it |
| Durable baseline record | This E001 ledger item; large measured artifacts remain on LRZ |

The run used the repaired p5n1 saved config and raw, non-EMA checkpoint. The
baseline record describes the config repair and pre-repair backup.

### Pinned inputs and inference policy

Focused split:

```text
analysis/cut0/gpu_preflight_5722939/focused_split.json
SHA-256: e977975bd82c9f123b89651f6a58a1b4b42c34b05d1c65521f145098047272e7
```

Cases:

- `sub-r049s033`: small/fast case;
- `sub-r004s006`: median-foreground case;
- `sub-r041s108`: empty-label case;
- `sub-r035s015`: largest selected volume and resource-stress case.

Legacy policy:

- output space: `model_preprocessed`;
- case batch size: `1`;
- DataLoader workers: `0`;
- precision: FP32;
- sliding-window ROI: `128 x 128 x 128` from the saved model config;
- sliding-window batch size: `4`;
- overlap: `0.5`;
- blend mode: `gaussian`;
- padding mode: `constant`;
- probability threshold: `0.5`;
- seed: `42`.

### Execution environment

| Resource | Value |
|---|---|
| Container | `$SSD_STORE/MedSegDiff_nnUNet_010226.sqsh` |
| Partition / QOS | `lrz-v100x2` / `gpu` |
| GPU | One NVIDIA Tesla V100, 16 GB |
| CPU | 2 CPUs |
| Host memory request | 8 GB |
| Wall-clock limit | 10 minutes |

The V100 matched the target T4's 16 GB VRAM capacity but does not establish T4
architecture, precision, or timing parity.

### Accepted artifacts and observations

Slurm logs:

```text
<run>/analysis/cut0/gpu_baseline_5722967.out
<run>/analysis/cut0/gpu_baseline_5722967.err
```

Evidence directory:

```text
<run>/analysis/cut0/baseline_raw_step040000_5722967/
  baseline_manifest.json
  resolved_evaluation_config.yaml
  <case_id>_probability.npy
  <case_id>_label.npy
```

The accepted manifest records the repository, checkpoint, split and config
hashes; all four case IDs; per-case artifact hashes; finite probability
summaries in `[0, 1]`; thresholded-mask hashes and foreground counts; metrics;
timings; and GPU-memory measurements.

Observed resource result:

- peak reserved GPU memory: 7.578 GiB;
- total case inference time: 16.691 seconds;
- main container step MaxRSS: approximately 2.24 GiB;
- batch step MaxRSS: approximately 4 MiB.

The stderr contained dependency warnings and a generator-finalization exception
after all four cases and the complete manifest had been written. This is an
accepted non-fatal legacy-harness cleanup defect, not clean runtime behavior.
It should disappear when the legacy generator path is migrated.

### Acceptance scope and invalidation

E001 is the legacy numerical/resource baseline for Cut 4 comparison. Preserve
the large arrays on LRZ and keep only their hashes and summaries in Git. It is
invalidated as parity evidence if the checkpoint, saved config, focused split,
legacy policy, or source commit differs. It does not certify FP16/BF16, T4,
Docker, native-space restoration, or the ten-minute Grand Challenge lifecycle.

---

## Superseded Cut 1 evidence index

These runs remain historical proof that intermediate contracts executed, but
they do not define the accepted Cut 1 design and must not be used to restore
temporary configuration or API choices.

| Evidence | Host/job | Result | Superseded because |
|---|---|---|---|
| E002 | LRZ `.sqsh`, job `5722997` | 19 tests passed | Initial focused policy/contracts snapshot |
| E003 | LRZ `.sqsh`, job `5722998` | 49 tests passed | Expanded compatibility suite on the E002-era snapshot |
| E004 | LRZ `.sqsh`, job `5723000` | 20 tests passed | Added runtime-profile assertion, still before final Cut 1 contract |
| E005 | Desktop WSL | 50 tests passed | Pre-hardening desktop snapshot |
| E006 | Desktop WSL | 73 tests passed | Contained the rejected temporary `data_mode.roi_key` design |

The LRZ runs establish that earlier additive contracts could execute under the
standardized training environment. They do **not** certify the exact final E007
snapshot under `.sqsh`; E007's final-state verification was performed on the
desktop environment described below.

---

## Evidence item E007: Final Cut 1 model-owned ROI contract

### Status

| Field | Value |
|---|---|
| Evidence ID | `E007` |
| Execution host | Home desktop WSL2 |
| Final state | `COMPLETED` |
| Exit code | `0` |
| Result | `ACCEPTED` |
| Tests | 74 passed |
| Test execution time | 22.576 seconds, excluding environment startup |

### Question being answered

Does the final Cut 1 contract resolve a mandatory ROI from the existing saved
model/data dimensionality without adding ROI fields to inference YAML or
changing established data-mode configs, while preserving affected consumers?

### Pinned source state

| Concern | Value |
|---|---|
| Base commit | `88d9b55dd2e3fa52df0c842e356b7e96a25b7b0b` |
| Transferred snapshot SHA-256 | `fb39572fc61635d19874aaaa01e6a10b3db60c83b49c6947a74ba0db5b264023` |
| Recorded desktop snapshot | `/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut1_final_20260731a` |

The snapshot contained the final Cut 1 `src/inference/`, inference and runtime
configs, policy/contract/runtime tests, and the then-current PRD/ledger. No
established `configs/data_mode/*.yaml` file differed from the base commit.

The final resolver uses existing `model.spatial_dims` / `data_mode.dim` to
select `dataset.preprocessing_configs.roi.slice_2d` or `volume_3d`. New
inference YAML contains no ROI field.

### Environment and tests

```text
Python 3.10.12
torch 2.8.0+cu128
OmegaConf 2.3.0
Hydra 1.3.2
```

The runner activated:

```bash
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate
```

Test command:

```bash
python -m unittest \
  tests.test_inference_contracts \
  tests.test_inference_policy \
  tests.test_inference_runtime \
  tests.test_evaluation_io_model_volumes \
  tests.test_evaluation_model_config \
  tests.test_training_runtime_contracts \
  tests.test_discriminative_adapter \
  tests.test_discriminative_output_domains \
  tests.test_data_contract_generalization \
  tests.test_dynunet_phase5_profiles \
  -v
```

All 74 tests passed. The explicit exit marker was `0`, and the log contained no
`FAIL` or `ERROR` result.

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut1_final_20260731a/cut1_final_tests.log
SHA-256: 3903d1694f02ee6502df22a4f1b27fb59a312c2f9371f8092b6bf1e1ddec9d71
```

### Acceptance scope and invalidation

E007 accepts the final Cut 1 contract on the recorded desktop snapshot. Changes
to the resolver, contracts, inference/runtime configs, or exercised consumer
contracts require fresh verification. It is not `.sqsh`, GPU, Docker, spatial
inversion, or release-runtime certification.

---

## Evidence item E008: Rejected Cut 2 prototype tombstone

E008 tested a strict artifact-oriented and multi-model prototype that was
rejected as premature for the transition-only Cut 2. E009 supersedes it
completely for Cut 2 acceptance. Its strict loading, artifact hashing, config
projection, multi-model API, CPU-first preparation, and parameter-freezing
behavior must not be restored from this ledger.

One exploratory observation remains potentially useful for a later release
cut: the selected p5n1 config and raw checkpoint could be reconstructed on the
desktop RTX 4070 Ti SUPER, all 52 state tensors matched the characterized
legacy path, and the model could be placed on CUDA. This was not measured
through the accepted Cut 2 implementation and is not T4, inference, artifact,
or release certification.

Pinned exploratory inputs:

| Concern | Value |
|---|---|
| Config SHA-256 | `7152715bded1f11cc244a8a4270b6cc592b8a0e555f40d6d40af5b2924cc918a` |
| Checkpoint SHA-256 | `120c93ee6a32f79829bf8a6b2d3ab7db59861f865ad1e8a37889f87ab0c82441` |
| Parameters | 5,641,315, stored as FP32 |

The full rejected implementation and test details remain available in Git
history rather than as an active architectural record.

---

## Evidence item E009: Cut 2 transition-only model-loading parity

### Status

| Field | Value |
|---|---|
| Evidence ID | `E009` |
| Execution host | Home desktop WSL2 |
| Final state | `COMPLETED` |
| Exit code | `0` |
| Result | `ACCEPTED` |
| Tests | 106 passed |
| Test execution time | 7.924 seconds |

### Question being answered

Can model-owned loading behavior move out of evaluation/training utility
locations while preserving the current evaluation API, checkpoint fallback and
diagnostics, training checkpoint behavior, requested device mapping, model
parameters, evaluation state, and gradient state?

### Pinned code state

| Concern | Value |
|---|---|
| Base commit | `a29dca5` (`feat(inference): define shared inference contracts`) |
| Base archive SHA-256 | `e666b04cee2a2e502c70ee24e1b91dbe009366db7e03875b83dc5e9700bb90d9` |
| Revised Cut 2 overlay SHA-256 | `3c908fc40088260d23f15e96a93655f2c10f47d92b220d6e637732762841b12d` |
| Recorded desktop snapshot | `/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_transition_20260731c` |
| Python environment | `/mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate` |
| Python / PyTorch | `3.10.12` / `2.8.0+cu128` |

The snapshot was the committed Cut 1 base plus only the revised Cut 2 files. It
did not modify or depend on the desktop repository checkout.

### Regression tests

```bash
python -m unittest \
  tests.test_model_loader \
  tests.test_evaluation_model_loader \
  tests.test_checkpoint_state \
  tests.test_inference_contracts \
  tests.test_training_runtime_contracts \
  tests.test_discriminative_adapter \
  tests.test_discriminative_output_domains \
  tests.test_data_contract_generalization \
  tests.test_dynunet_phase5_profiles \
  tests.test_evaluation_model_config \
  tests.test_evaluation_io_model_volumes \
  tests.test_evaluation_pipeline \
  -v
```

All 106 tests passed.

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_transition_20260731c/cut2_transition_tests.log
SHA-256: 2a94a3dcee1dd6e314880d3553d00c0b533b58915065ef411e3ebbd350dacc18
```

### Real selected-model parity

The retained p5n1 resolved config and checkpoint were reconstructed once
through the exact pre-transfer evaluation sequence and once through
`src.models.model_loader.load_model()` on CPU.

| Check | Result |
|---|---|
| Missing/unexpected keys, legacy path | `0 / 0` |
| Missing/unexpected keys, moved path | `0 / 0` |
| State-dict tensors | All 52 exactly equal |
| State-dict key order | Identical |
| Preparation state | Both `eval()` |
| Gradient state | Both retain `requires_grad=True` |
| Device | Both CPU for this parity run |

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_transition_20260731c/cut2_real_model_parity.log
SHA-256: 72c740d94bc386015dca74aa1fdb7f11eb866f80699238272fa2b68c1d60d7c5
```

### Acceptance scope and invalidation

E009 accepts the physical ownership transfer only. It introduces no strict
release mode, artifact abstraction, hashing policy, multi-model API, config
projection, CPU-first preparation, parameter freezing, or new prefix handling.
Changes to the moved loader/checkpoint/config helpers or their compatibility
facades require new parity evidence.

---

## Evidence item E010: Cut 3 label-optional preprocessing extraction

### Status

| Field | Value |
|---|---|
| Evidence ID | `E010` |
| Execution host | Home desktop WSL2 |
| Final state | `COMPLETED` |
| Exit code | `0` |
| Result | `ACCEPTED_FOR_PINNED_CPU_SNAPSHOT` |
| Focused/affected tests | 95 passed |
| Repository discovery | 411 passed; one pre-existing import error |

### Question being answered

Can the existing ISLES24 and ISLES26 deterministic preprocessing sources of
truth support both legacy labeled consumers and blind inference without dummy
labels, while exposing per-case native NIfTI geometry before MONAI orientation,
spacing, channel processing, and raw-key cleanup?

### Pinned code state

| Concern | Value |
|---|---|
| Base commit | `bb915902a16a9da654f70dc5f443a71642eabbb6` (`refactor(models): centralize single-model loading`) |
| Cut 3 file-manifest SHA-256 | `b6c4f9eebbda638aaacbbf2d32989fa31cfc225172776463be5f7f4379b13998` |
| Manifest definition | Sorted `relative_path=sha256(file_bytes)` lines, joined with LF and hashed as UTF-8 |
| Execution worktree | `/mnt/c/Users/minanessiem/Development/medseg-diffusion-cut3-test` |
| Isolation | Disposable Git worktree; the dirty desktop checkout was not modified |

The final 20-file manifest covered:

```text
configs/dataset/isles24_base.yaml
configs/dataset/isles26_base.yaml
src/data/loader_stack/__init__.py
src/data/loader_stack/contracts.py
src/data/loader_stack/isles24_loader.py
src/data/loader_stack/isles26_loader.py
src/data/loader_stack/preprocessing.py
src/data/loader_stack/registry.py
src/data/loaders.py
src/inference/__init__.py
src/inference/contracts.py
src/inference/preprocessing.py
tests/test_inference_preprocessing.py
tests/test_isles24_dataset_randompatch3d.py
tests/test_isles24_label_optional.py
tests/test_isles26_datalist_parser.py
tests/test_isles26_dataset2d.py
tests/test_isles26_dataset3d.py
tests/test_isles26_dataset_randompatch3d.py
tests/test_loader_stack_routing.py
```

`tests/test_inference_contracts.py` was exercised from the committed Cut 2 base
but was not changed or included in the Cut 3 manifest. The PRD, ledger,
prior-cut files, and temporary test runner were also excluded.

### Execution environment

The runner used the authoritative WSL activation command:

```bash
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate
```

`command -v python` resolved to:

```text
/mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/python
```

No `.bashrc`, Windows Python, `sudo`, GPU, or LRZ allocation was used. The
original result did not record a complete preprocessing dependency inventory or
a durable log hash. E010 therefore supports the pinned desktop CPU snapshot but
must not be treated as dependency-qualified cross-environment evidence.

### Focused and affected regression evidence

```bash
python -m unittest \
  tests.test_loader_stack_routing \
  tests.test_isles26_datalist_parser \
  tests.test_isles26_dataset3d \
  tests.test_isles26_dataset2d \
  tests.test_isles26_dataset_randompatch3d \
  tests.test_isles24_dataset_randompatch3d \
  tests.test_isles24_label_optional \
  tests.test_isles26_random_patch_pipeline \
  tests.test_isles26_loader_refactor_suite \
  tests.test_loaders_facade_isles24_routing \
  tests.test_loaders_facade_isles26_routing \
  tests.test_inference_contracts \
  tests.test_inference_preprocessing \
  -v
```

All 95 tests passed in 0.665 seconds after environment import/startup. The suite
covered adapter routing, labeled and label-free behavior, label-guided
random-patch rejection, compatibility facades, native metadata capture,
labeled/unlabeled preprocessing parity, orientation separation, and one raw
modality producing multiple processed channels.

### Repository-wide discovery

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The run executed 412 tests in 31.520 seconds: 411 passed and one module failed
to import. The sole error was the pre-existing
`tests/test_summarize_threshold_calibration.py`, which imports the absent
`scripts.summarize_threshold_calibration` module. The test exists at the pinned
base while the referenced script does not. Every Cut 3 and affected
loader/inference test passed in both runs.

### Supplemental live label-mode and spatial evidence

A second isolated desktop worktree at the same base commit and then-current
18-file source/test manifest exercised the repository against real ISLES26
volumes, not only mocked loader records or small unit fixtures:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut3_live_labels_20260801a
```

The run used Python 3.10.12, PyTorch 2.8.0+cu128, MONAI 1.5.0, and nibabel
5.3.2 from `MedSegDiff_env`. It directly iterated the full-volume dataset with
`test_flag=True` in both label modes and called the shared registered ISLES26
preprocessing adapter on the same records. No DataLoader worker pool was used.

The `val_full` split contained 193 records; header inspection found 124 whose
native orientation or spacing required a configured RAS/1 mm transform. Two
such cases with distinct native geometry were processed through all relevant
contracts. For each case:

- `load_labels=True` returned the legacy `(image, label, case_id)` item;
- `load_labels=False` returned `(image, case_id)`, and its normalized record
  contained no `label` key;
- the legacy labeled image, legacy blind image, shared labeled image, and
  shared blind image were numerically identical (`max_abs_delta = 0.0`);
- transformed image and label shapes and affines matched exactly;
- the retained native label was voxel-exact with the source NIfTI, and its
  retained affine matched exactly;
- the source T1 and label shape/affine matched within each case.

One real case changed from native shape `[218, 216, 252]`, RAS orientation,
and spacing approximately `[0.8594, 0.999516, 0.8594]` mm to model shape
`[187, 216, 217]`, RAS orientation, and spacing approximately
`[1.0, 0.999516, 1.0]` mm. Its native and transformed labels therefore had
different grids and foreground counts while each remained binary. A second
real case retained shape `[188, 256, 256]` while changing orientation from LAS
to RAS. These results demonstrate that the transformed label follows the model
image grid while the native label remains in source geometry.

A controlled anisotropic LAS fixture passed through the same production
adapter as a nontrivial cross-check. Its native label remained `[8, 7, 6]` at
`[2.0, 1.5, 3.0]` mm in LAS orientation, while the jointly transformed label
became `[15, 10, 16]` at `[1.0, 1.0, 1.0]` mm in RAS orientation and matched the
transformed image affine exactly.

```text
cut3_live_label_modes.log
SHA-256: 6c43e6b6366a5218da520b173762660fdb4e4e12c6f1cfd3968abb62bb781a83

cut3_live_label_modes.json
SHA-256: 435b280805ea921f55c4bac1faa4fb932ccfdfca6bbb2bb938313a833a22459d
```

The supplemental run exited `0` with `CUT3_LIVE_LABEL_MODES_PASS`. The raw log
and JSON were copied to the laptop temporary evidence directory before the
disposable worktree was removed; the hashes above are the durable ledger
identifiers.

### Explicit base-config default verification

The final Cut 3 snapshot added `load_labels: true` to the ISLES24 and ISLES26
base dataset configs without changing preprocessing implementation files. This
makes the labeled default part of newly resolved model provenance while the
existing callable defaults preserve historical configs that lack the field.

The isolated desktop worktree
`/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut3_config_defaults_20260801a`
verified all of the following:

- direct OmegaConf loading resolves both base declarations to Boolean `True`;
- representative ISLES24 and ISLES26 Hydra experiment composition retains
  `dataset.load_labels=true`;
- `get_dataloaders`, `ISLES24Dataset3D`, and `ISLES26Dataset3D` retain
  `load_labels=True` callable defaults for historical saved configs;
- the complete 95-test Cut 3 focused/affected suite passes on the expanded
  20-file manifest.

```text
cut3_config_defaults_full.log
SHA-256: c8a41c6ee9df7baee18df97c397309c496472c0754e4f1d00eb83d1ea50133e1

cut3_config_defaults_tests.log
SHA-256: d15246c6e00a6f6161a79fbb95c68690c415e53e8973b0b55db7183d4536c0c1
```

### Acceptance scope and invalidation

E010 establishes for its pinned CPU snapshot that:

- `load_labels=True` preserves labeled behavior by default;
- current ISLES24/ISLES26 base configs declare the default explicitly, while
  historical resolved configs without the field retain the runtime fallback;
- `test_flag` remains independent from label loading;
- label-free 2D and 3D loaders create no fabricated label;
- label-guided random-patch loaders reject `load_labels=False` immediately;
- ISLES24 and ISLES26 reuse their dataset-owned transform builders;
- native shape, dtype, affine, spacing, orientation, qform/sform and codes are
  captured per case and raw modality before preprocessing;
- model-space and native-space labels exist only on the explicit labeled
  result contract;
- Cut 3 added no cross-modality alignment precheck, prediction, inversion, TTA,
  ensemble, postprocessing, or Grand Challenge transport behavior.

Changes to a manifest-covered file require fresh verification. Changes to
MONAI, nibabel, NumPy, or other preprocessing dependencies require a new
dependency-qualified run against the supplemental versions recorded above.
E010 is not GPU/T4, `.sqsh`, container, spatial-inversion,
numerical-performance, or ten-minute runtime certification.

---

## Evidence item E011: Cut 4 shared probability execution

### Status and question

| Field | Value |
|---|---|
| Evidence ID | `E011` |
| Status | `ACCEPTED` for Desktop contract and integration evidence |
| Base commit | `e9c404a707c062edc2a622113a704812aef46e83` |
| Tested source-manifest SHA-256 | `b284bcd081d9a87b9951ddc9a85f452fe5748b7ee1f2720e8a1803f560eab0ab` |
| Deferred gate | Exact E001 numerical parity in a later same-environment LRZ run |

E011 asks whether the Cut 4 snapshot moves direct and sliding-window
probability generation into the architecture-neutral `src.inference` path,
preserves backend-owned output interpretation and probability-before-blending,
migrates training validation and repository-model evaluation directly, and
rejects unsupported generative/native-space execution clearly.

### Execution environment and results

The tests ran in the isolated Desktop WSL worktree
`cut4_impl_20260801a` using:

```bash
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate
python -m unittest <grouped modules> -v
```

Python resolved to the `MedSegDiff_env` interpreter and reported version
3.10.12. Tests were split into isolated processes to bound retained model
memory:

- 57 shared inference, config/runtime, evaluation-consumer,
  training-validation-consumer, and training-runtime tests passed;
- 6 registered preprocessing regression tests passed;
- 3 existing discriminative-adapter tests passed, including 2D, 3D, and
  deep-supervision adapter paths;
- 11 discriminative deep-supervision tests passed;
- all changed Cut 4 production modules passed `python -m py_compile`.

The CUDA FP16 sliding-window test executed rather than being skipped. It used
window batch one and passed the same probability-domain and shape checks as the
FP32 paths.

Durable log identifiers:

```text
cut4_shared_tests.log          d65bd0a36512db92b879af1d5bc007a037b76db0703c9ce23826a7c1eecdc696
cut4_preprocessing_tests.log   479f9e937a9bf5618988248da8f11a45a4a4d1c7607ebe33e7067bab08813a1b
cut4_adapter_tests.log         14429b4464c51a220aa751acb257e852c6cabdbf84d43bdc5ce62c34795f1dfd
cut4_deep_supervision_tests.log adfa14529cdd7d597014dce4541c71d599ac554563adcc9792698d1d906410f2
```

### Acceptance scope and deferred parity

E011 establishes for its pinned Desktop snapshot that:

- a generic probability predictor drives both direct and MONAI
  sliding-window execution;
- every patch is converted to a probability by the backend before overlap
  blending, and the established ROI, window batch, overlap, blend mode,
  padding mode, and progress controls are forwarded;
- `DiscriminativeAdapter` remains responsible for raw-return parsing,
  deep-supervision final-head selection, and sigmoid activation;
- shared inference contains no DynUNet, SwinUNETR, diffusion-sampler, label,
  metric, NIfTI, or Grand Challenge socket branches;
- configured generative execution fails before prediction with the future
  `ProbabilityPredictor` adapter-registration hook;
- training validation and live repository-model evaluation use the shared
  executor directly, and the removed inferer symbols are absent from
  `src/utils/valid_utils.py` and all consumers;
- Cut 4 rejects `native_input` rather than relabeling model-space output before
  Cut 7 restoration exists.

E001 was produced on LRZ hardware in the standardized container. Per the user
decision on 2026-08-01, no cross-hardware numerical-parity claim is made from
the Desktop result and no new LRZ job is submitted now. Exact comparison with
E001 remains a later same-environment LRZ acceptance item. E011 also does not
certify T4 resource use, Grand Challenge runtime behavior, native-space
restoration, or the ten-minute case limit.

---

## Evidence item E012: Cut 4 real-model Desktop execution

### Status and supersession

| Field | Value |
|---|---|
| Evidence ID | `E012` |
| Status | `ACCEPTED` for real-model FP32 parity and consumer integration; FP16 policy blocked |
| Base commit | `e9c404a707c062edc2a622113a704812aef46e83` plus the uncommitted Cut 4 source manifest below |
| Tested source-manifest SHA-256 | `d85fc4a7e83bc92c390b9f9b882977f87d5d2b6cdfc1fd8fee3c47c1199bfac9` |
| Deferred gate | Exact E001 numerical parity in a later same-environment LRZ run |

E012 supplements E011's synthetic contract coverage with a saved 3D
DynUNet checkpoint and two real ISLES26 cases. It asks whether the shared
executor reproduces the pre-extraction probability expression, whether both
migrated consumers receive the same probability tensor, and whether the
proposed native-FP16 profile satisfies the real full-volume probability
contract.

### Pinned model, config, and input state

The run loaded the saved resolved config and checkpoint through
`src.models.model_loader.load_model`. Loading reported no missing or unexpected
keys.

```text
resolved config SHA-256
7152715bded1f11cc244a8a4270b6cc592b8a0e555f40d6d40af5b2924cc918a

checkpoint SHA-256
120c93ee6a32f79829bf8a6b2d3ab7db59861f865ad1e8a37889f87ab0c82441
```

The real labeled cases were `sub-r049s033` and `sub-r035s015`. Their native
spatial shapes were respectively `96x160x160` and `192x512x512`; model-space
shapes were `143x240x240` and `192x256x256`. The explicit Cut 4 policy resolved
to ROI `128x128x128`, window batch one, overlap `0.5`, Gaussian blending, and
constant padding. The legacy saved policy was also resolved independently and
retained its window batch of four, proving policy-source selection rather than
using it for the explicit run.

The uncommitted Cut 4 source files were copied into an isolated detached
Desktop worktree. Their per-file hashes and aggregate are retained in the
result JSON; the aggregate SHA-256 is pinned in the table above.

### Execution environment and command

The run used the Desktop WSL `MedSegDiff_env` environment, Python 3.10.12,
PyTorch 2.8.0+cu128, MONAI 1.5.0, CUDA 12.8, and an NVIDIA GeForce RTX 4070 Ti
SUPER with 15.992 GiB reported device memory. The disposable harness invoked:

```bash
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate
python -u .cut4_live_harness.py \
  --config <pinned-resolved-config> \
  --checkpoint <pinned-checkpoint> \
  --data-root <desktop-isles26-training-raw> \
  --case sub-r049s033 \
  --case sub-r035s015 \
  --device cuda:0 \
  --output-json cut4_live_results.json
```

The reference path independently evaluated the pre-extraction expression:
MONAI sliding-window inference with `model.sample` as its patch predictor.
Both paths therefore retained the established probability-before-blending
semantics.

### Result and durable artifacts

Both FP32 full-volume comparisons were bit-identical at the recorded tensor
representation: maximum and mean absolute differences were zero, probability
hashes matched, threshold-0.5 masks matched with zero disagreeing voxels, and
Dice deltas were zero.

| Case | Patch calls | Shared/reference probability SHA-256 | Dice | Shared / reference time | Peak reserved |
|---|---:|---|---:|---:|---:|
| `sub-r049s033` | 18 | `53f365d8983acd3203ade2455185f1bcba82858d4142e85182bce64e4e667653` | 0.6826347113 | 1.817 s / 1.229 s | 2.146 GiB |
| `sub-r035s015` | 18 | `0a9c92a1b0c094dca64063b171788eb09263de291802038387867a8335c725c8` | 0.7961137891 | 1.510 s / 1.221 s | 2.193 GiB |

On a single ROI, direct shared execution was also bit-identical to the backend
call. The backend and shared timings were 0.086 s and 0.084 s. The repository-
model evaluation consumer and training-validation consumer each reproduced the
retained fast-case probability hash exactly. Their maximum/mean differences
and threshold disagreements were zero; training validation reported the same
Dice and returned the model to training mode as expected by the training-loop
consumer.

Unsupported branches failed before invoking the backend. `native_input`
reported that restoration belongs to Cut 7, and configured DDPM execution
reported the future `ProbabilityPredictor` adapter-registration boundary.

The real-model FP16 run did **not** satisfy the final probability-domain
contract. Every patch passed its probability check, but half-precision
Gaussian overlap accumulation produced 139 values above one, with maximum
`1.0002766848` and overshoot `0.0002766848`. An independent run of the old
sliding-window expression reproduced the same raw maximum, demonstrating that
this is not numerical drift introduced by the shared executor. Relative to
FP32 its maximum and mean absolute differences were `0.0071218312` and
`5.2308502e-7`; it used 1.473 GiB peak reserved memory and took 0.702 s. No
production clamp or other correction was added without an explicit inference-
policy decision.

```text
cut4_live_results.json
SHA-256: e86f2258269634f64f3e89d588ab12c377420d815c9793e517d7c8bb4a868420

cut4_live.log
SHA-256: d2d69ade9afaf6399a13548efe6fee76faa0c17cc08f260a9e0a5452149c4d15

disposable harness
SHA-256: be26c5377c213d41c98f1e715f4569085b172913428ecb6eec621aa06cf75b75
```

### Acceptance scope and invalidation

E012 establishes real-model, real-volume FP32 parity between the shared Cut 4
executor and the old expression on the pinned Desktop environment. It also
establishes that evaluation and training validation consume that same result.
It does not certify cross-hardware equality with E001, LRZ/`.sqsh` execution,
T4 resources, native-space restoration, Grand Challenge transport, or the
ten-minute case limit.

The E011 CUDA FP16 unit result remains valid for its synthetic fixture, but it
is insufficient to certify a native-FP16 release profile. The real-model
finding blocks such certification until the project explicitly chooses and
tests an accumulation/output-domain policy. Changes to a source-manifest file,
the checkpoint/config, preprocessing dependencies, precision behavior, MONAI
sliding-window behavior, or the selected policy invalidate the corresponding
claims and require rerunning this evidence.

---

## Evidence item E013: Cut-5 nnU-Net precursor spatial export

### Status and question

| Field | Value |
|---|---|
| Evidence ID | `E013` |
| Status | `ACCEPTED` for bounded Desktop conversion and spatial-contract evidence |
| Base commit | `1f8c8c4ad6813dd198ea7c8789fd866be9f8240e` plus the uncommitted Cut-5 precursor diff |

E013 asks whether the updated 3D nnU-Net converter preserves a declared native
grid, writes a genuinely transformed grid with its transformed affine, records
source/export geometry separately, and produces geometry-complete 3D nnU-Net
evaluation samples.

### Execution environment and command scope

Tests ran in the isolated Desktop WSL worktree
`cut5_nnunet_precursor_20260801a` using Python 3.10.12 from
`MedSegDiff_env`. Focused spatial/config tests reported 23 passes after the
native ISLES26 preset extension. Existing
nnU-Net conversion and directly affected evaluation regressions reported 36
passes; a broader nnU-Net and evaluation-consumer run reported 16 and 34
passes respectively. These groups overlap and must not be summed as a unique
test count.

Real conversion used the ordinary Hydra entrypoint with:

```text
nnunet.test=true
nnunet.test_max_slices=1
nnunet.parallel.enabled=false
```

This exported one training and one validation/test case for both the local
ISLES24 3D baseline and local ISLES26 3D T1-raw presets. It did not convert a
complete dataset. A header-only scan then selected one additional ISLES26 case
whose source grid was not already 1 mm and passed only that case through the
same dataset transform and `VolumeExportStrategy` path.

The native ISLES26 extension used
`nnunet/convert/isles26_local_3d_t1raw_native` with `nnunet.test=true`,
`nnunet.test_max_slices=10`, both export splits enabled, and parallel export
disabled. Thus the converter wrote 10 training and 10 validation/test cases,
not a complete dataset.

The transformed ATLAS30 extension used the existing
`nnunet/convert/isles26_atlas30_cluster_3d_t1raw` preset with only the desktop
environment/runtime, isolated output path, 10-case test limit, and sequential
export overridden. It likewise wrote 10 training and 10 `val_full` cases,
not a complete dataset. Because those bounded cases were already RAS/1 mm, a
second probe disabled the local loader smoke cap, scanned the same configured
train/`val_full` subsets by NIfTI header, and exported one source that actually
required resampling through the same configured loader and
`VolumeExportStrategy`.

### Result and acceptance scope

- The ISLES24 native export preserved its source shape, affine, spacing, and
  LAS orientation, and recorded `native_input`.
- The initially selected ISLES26 smoke cases were already RAS/1 mm. They prove
  loader, `Subset`, writer, and provenance integration but not a grid change.
- The targeted real ISLES26 case changed from spacing `(1.1, 1.0, 1.0)` and
  shape `(160, 196, 200)` to RAS/1 mm and shape `(176, 196, 200)`.
  Source/export affines differed as expected.
- Reopened image and label NIfTIs matched the recorded transformed shape and
  affine; qform and sform were both set and agreed with that export affine.
- Synthetic fixtures additionally cover nontrivial transformed affine,
  false-native rejection, image/label affine disagreement, wrapped-dataset
  source lookup, and equal-shape prediction/reference affine mismatch.
- The local `isles26_local_3d_t1raw_native` preset inherited T1-raw intensity
  handling through the dedicated dataset/data-profile layers, disabled
  orientation and spacing, allowed native spacing, retained no full-volume
  padding, and used the distinct nnU-Net identity
  `Dataset264_isles26_t1_raw_native`.
- A bounded run exported 10 training and 10 validation/test cases. All 20
  records and reopened image/label NIfTIs preserved their source shape,
  affine, spacing, and orientation; qform/sform were set, labels were binary
  `uint8`, and provenance declared `native_input`.
- Because those first 20 cases were already RAS/1 mm, a targeted additional
  native-preset probe selected a real source with spacing `(1.1, 1.0, 1.0)`
  and shape `(160, 196, 200)`. The exported image and label retained that exact
  spacing, shape, orientation, and affine, demonstrating that the native
  preset does not resample a non-unit grid.
- Ten geometry-identical prediction fixtures entered the real 3D nnU-Net
  evaluator through the composed native conversion preset. It matched 10/10
  pairs, processed 10 volumes and 1,915 derived slices, and recorded
  `volume_space=native_input`. Exact-reference volume Dice and surface Dice
  were 1.0; HD95 and absolute volume differences were 0.
- The named ATLAS30 transformed preset resolved to dataset 265, `val_full`, RAS
  orientation, 1 mm isotropic spacing, and `model_preprocessed` export space
  while the desktop environment override changed only filesystem/runtime
  concerns.
- All 20 bounded ATLAS30 records and reopened image/label NIfTIs agreed with
  their recorded export shape, affine, spacing, and RAS orientation; qform and
  sform were set, labels were binary `uint8`, and provenance declared
  `model_preprocessed`. The first 20 source grids were already RAS/1 mm, so
  they exercise the transformed preset without independently proving a grid
  change.
- The targeted ATLAS30 probe found a source with spacing
  `(1.100000023841858, 1.0, 1.0)` and shape `(160, 196, 200)`. The configured
  preprocessing exported RAS/1 mm image and label volumes with shape
  `(176, 196, 200)` and a changed affine. Reopened qform/sform and both export
  grids agreed with the transformed affine.
- Ten geometry-identical ATLAS30 prediction fixtures entered the real 3D
  nnU-Net evaluator through the named transformed conversion preset. It
  matched 10/10 pairs, processed 10 volumes and 1,850 derived slices, and
  recorded `volume_space=model_preprocessed`. Exact-reference volume Dice and
  Surface Dice were 1.0; all reported HD95 variants and absolute volume
  differences were 0.

E013 does not certify previously generated nnU-Net datasets, a complete
dataset conversion, nnU-Net planning/training/prediction, 2D reconstruction,
or the finalized Cut 5 evaluator. Changes to conversion transforms, exporter
geometry handling, spatial contracts, MONAI/nibabel versions, or the six 3D
conversion presets require this evidence to be rerun.

---

## Evidence item E014: Cut 5 geometry-aware offline evaluation

### Status and question

| Field | Value |
|---|---|
| Evidence ID | `E014` |
| Status | `ACCEPTED` for Cut 5 Desktop repository-model and 3D nnU-Net evaluation |
| Base commit | `215801e1813b63ad48ed14aff1c76562b06bdc1b` plus the uncommitted Cut 5 diff |
| Deferred finding | O007 paired model-space reference-grid remediation |

E014 asks whether the post-training evaluator now uses the shared inference
policy/runtime and shared probability executor for a live 3D repository model,
requires explicit result/reference space and verified geometry, produces
metrics and reports from the verified common grid, accepts compliant native and
model-preprocessed 3D nnU-Net volumes, and fails rather than assessing a pair
whose physical grids disagree.

### Pinned model, config, and input state

The Desktop run used the same p5n1 3D DynUNet model as E001 and E012. The files
were copied from LRZ and verified after transfer:

```text
saved resolved config
7152715bded1f11cc244a8a4270b6cc592b8a0e555f40d6d40af5b2924cc918a

saved overrides
6cdd2234aae0716431c17287dd35f6d3f2ebaf1211470e1ed8896c5df4761f30

best_model_step_040000_dice_3d_0.5724.pth
120c93ee6a32f79829bf8a6b2d3ab7db59861f865ad1e8a37889f87ab0c82441

four-case focused split
e977975bd82c9f123b89651f6a58a1b4b42c34b05d1c65521f145098047272e7
```

The four locked cases were `sub-r004s006`, `sub-r035s015`, `sub-r041s108`,
and `sub-r049s033`. The composed evaluation used the `native` runtime, an
explicit top-level `sliding_window_model_space` policy, FP32, ROI
`128x128x128` resolved from the model config, case/window batch one, overlap
`0.5`, Gaussian blending, constant padding, and fixed threshold `0.5`. The
recorded policy source was `explicit_top_level`; the post-training evaluation
policy superseded the saved training-validation policy rather than field-
merging with it.

### Execution environment and command scope

Execution used the isolated Desktop WSL worktree
`cut5_20260802a`, Python 3.10.12 from `MedSegDiff_env`, and an NVIDIA GeForce
RTX 4070 Ti SUPER with 16,376 MiB reported VRAM. The live repository-model run
used the ordinary `scripts.evaluation.evaluate_model` entrypoint with the
pinned model directory, checkpoint, focused split, and explicit Cut 5
evaluation/inference overrides.

Before live execution, the Cut 5 snapshot passed these Desktop test groups:

- 82 focused inference/evaluation contract tests;
- 81 remaining evaluation tests;
- 75 shared-inference and 3D-loader tests;
- a final 47-test focused rerun after integration review.

These groups overlap and must not be summed as a unique test count.
`git diff --check` also passed for the Cut 5 source diff.

The 3D nnU-Net evaluator was exercised with isolated prediction/reference
fixtures copied from the accepted E013 bounded conversions. No trained nnU-Net
model was required because Cut 5 concerns ingestion and assessment of already
produced prediction volumes, not nnU-Net training or prediction correctness.

### Result and durable artifacts

#### Live repository-model evaluation

All four locked volumes completed through the public evaluator. Each reported
matching prediction/reference shapes, affines, spacing, orientation, and
`model_preprocessed` space before metric calculation.

| Case | Cut 5 Dice | E001 Dice | Delta |
|---|---:|---:|---:|
| `sub-r004s006` | `0.5875658393` | `0.5876157284` | approximately `-4.99e-5` |
| `sub-r035s015` | `0.7961137891` | `0.7961222529` | approximately `-8.46e-6` |
| `sub-r041s108` | `0.0` | `0.0` | `0.0` |
| `sub-r049s033` | `0.6826347113` | `0.6826347113` | `0.0` |

The Cut 5 mean Dice was `0.5165785849`, compared with E001's
`0.5165931731` (approximately `-1.46e-5`). E001 ran on LRZ/V100 while E014 ran
on the Desktop/RTX 4070 Ti SUPER; per the existing project decision, this small
cross-hardware difference is not treated as a parity failure.

Durable hashes from `cut5_live_artifacts/evaluation_output_locked4`:

```text
canonical_results.json
4e79edd968954c006fa62308f4e17408dd5a3b39f883bd0cdc853cff66f5a36d

volume_metrics CSV
e277f5aaff58a78e2b254a136be3b3bd401c0e83b6b484a3f49ba39d169728c2

per-case CSV
eb3390046467f85bb652ae3940a59f6b933b825035a2dfe1e2485db5cff6f10d

resolved evaluation config
86f465fa9f27bb05247fd29090dfef74eb21791131b284bf57ecd4138b69119d

evaluation summary
0c3db4ee2fd11f32d9de48e064e515280fd8b62094dcf9f309e5a90dc5fa13a0

four-case live log
6d3e751504413cc3a3324e6a7855600db1c127c9aac042f34f438fb4c567ccc1
```

#### Strict spatial failure and O007 diagnosis

A broader 64-case `val_fast` run completed 59 volumes and then stopped at
`sub-r065s005` with the intended error:

```text
VolumeSample prediction/reference affine mismatch: equal tensor shapes do not
establish a shared physical grid.
```

A data-loader-only audit, which avoided rerunning model inference, examined all
64 cases and found exactly the two mismatches documented in O007:
`sub-r065s005` and `sub-r069s031`. The diagnostic record includes raw and
transformed shapes, affines, spacing, orientation, and maximum affine deltas:

```text
geometry diagnosis JSON
1cec0991e37a0f16bc4b7797fb624e5ff46e85d7a39de3c67e007bacd3c66a24

E001 baseline manifest copied for comparison
c2aec4acdadddc06d3f7fa71829474a28b06f72dfa458554e6cda45f59274c31
```

The failure is accepted evidence that Cut 5 enforces its spatial contract. It
must not be converted into a tolerance fallback. O007 records the separate
preprocessing remediation required for complete full-split certification.

#### 3D nnU-Net producer/evaluator matrix

The real 3D nnU-Net evaluation path produced these outcomes:

- 10/10 `native_input` identity pairs succeeded with Dice and surface Dice
  `1.0`, HD95 `0`, and absolute volume difference `0`;
- 10/10 `model_preprocessed` identity pairs produced the same perfect metrics;
- changing one foreground prediction to an empty mask while preserving its
  geometry succeeded and reduced aggregate Dice to `0.9`, demonstrating that
  image values, rather than filenames alone, were assessed;
- shifting one prediction affine by 2 mm failed before the first volume was
  assessed with an actionable affine-mismatch error.

Fixture and log identifiers:

```text
fixture manifest
334405b678c43795385aaf07863ceab8f1ae4537543ae55e442742795abdcf68

native identity log
ff86e06c0b1f0b58525a489b201fb07f523087cacfc6fd13cdebca4c2700da5b

model-preprocessed identity log
344e7fdf81b6b3d833bc7f899e0eb6ec8a51d727e3fba2a8fe432461bb5f7df1

altered-mask log
71a810135ae453727488180016b26c0931522050bee2dc4c7e30496ef63bf4b8

bad-affine log
fe6876d2c8535ba060967c91cea43a14b06ee7e8c6865c80cd3807cd654d61ae
```

The corresponding canonical-result hashes are
`d43b42a671db19cc339327a946411084c8b605de66d974f17642af11a94cce75`
for native identity,
`cef03708717cf013794601ddc663851b1a6b195622d9539600670524eb7742b6`
for model-preprocessed identity, and
`754d67fabaa5135afffde53c75bf8d1d67b5f0f585c828f5430b7e17ca854dfb`
for the altered-mask run.

All E014 files remain preserved outside Git under:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/
  cut5_20260802a/cut5_live_artifacts/
```

### Acceptance scope and invalidation

E014 completes Cut 5 for its declared 3D scope. It establishes that the common
post-training evaluator consumes the shared inference/runtime policy, assesses
live repository-model and file-backed nnU-Net results through explicit space
and geometry contracts, records the selected policy/space in its results, and
rejects unsupported physical-grid mismatches before metrics. The O007 finding
does not invalidate Cut 5: discovering and stopping at that mismatch is the
new evaluator behaving correctly, and blind Grand Challenge inference does not
load or compare a reference label.

E014 does not certify 2D evaluation, complete 64-case ISLES26 model-space
evaluation, corrected paired-grid preprocessing, native-space inversion,
Grand Challenge transport/container behavior, T4 performance, native FP16, or
the ten-minute case limit. Those remain governed by O004, O005, O007, Cut 7,
and the later container/release cuts. Changes to the shared inference executor,
evaluation contracts, result/reference geometry checks, producer adapters,
model/checkpoint/config, or relevant MONAI/nibabel behavior require the
corresponding evidence to be rerun.

---

## Evidence item E015: Cut 6 training-validation migration

### Status and question

| Field | Value |
|---|---|
| Evidence ID | `E015` |
| Status | `ACCEPTED` for Cut 6 Desktop validation and live replays |
| Base commit | `b6e07bb0ba9e985fd722dd788b3c4a0c1d7c875a` plus the uncommitted Cut 6 diff |
| Preserved legacy contract | Saved configs with only `validation.inference` remain translated when top-level `inference` is absent |

E015 asks whether new training configs obtain prediction behavior from the
shared top-level inference policy, whether current 2D and 3D profiles preserve
their direct and sliding-window behavior respectively, and whether the
training-validation consumer can execute the accepted shared probability path
without changing model state or established metrics.

### Config and runtime contract

The base `local` and `cluster` training configs compose
`inference=direct_model_space` and `inference_runtime=native`. The direct
policy inherits the complete model-space policy, including model-owned ROI,
and changes only `sliding_window.enabled=false`. Each current 3D training
profile explicitly overrides this with `sliding_window_model_space`. Existing
cluster 3D profiles retain window batch four, while local 3D profiles retain
the policy default of one. Prediction fields no longer live in new validation
presets.

Training validation validates that the runtime permits access to ground truth
before placing the model in evaluation mode or executing a prediction. The
trainer remains responsible for labels, metric lifecycle, progress, logging,
and restoring training mode. The shared executor remains responsible for
probability generation. Mean ensembling is explicitly dimension agnostic;
the current 2D-only soft-STAPLE implementation now fails clearly when selected
for 3D data.

### Desktop verification

Verification ran in the isolated Desktop WSL worktree `cut6_20260802a` using
Python 3.10.12 from `MedSegDiff_env`. The focused suite passed 81/81 tests and
covered training config composition (including the named 2D DDP
profiles), policy resolution, runtime guards, ensemble semantics, trainer
validation, and the evaluation consumers affected by validation-group
ownership. The affected regression suite passed 79/79
tests and covered checkpoint selection/save/load/resume, discriminative and
ablation config composition, inference contracts/predictors/sliding window,
the model-volume evaluator, evaluation pipeline/SLURM routing, and run naming.
After review removed the empty legacy `validation/sliding_window` group, the
24 directly affected training/evaluation composition tests passed again with
the two 3D metric bundles inheriting `validation/default` directly.

Durable test-log hashes:

```text
cut6_focused_tests_final.log
0de6a5c0da273f0085f40c6dad206e00e7d798f683f3feab8012769a9de56e8c

cut6_affected_tests.log
3940f068568812a96360331ac6dec755264755f65685b85a15805e986a0e3c8d
```

### Live repository-model validation

The live exercise used the pinned E014 p5n1 DynUNet checkpoint and the same
four locked cases, through `src.training.trainer.validate_one_epoch` and the
real shared model probability executor. It ran on an NVIDIA GeForce RTX 4070
Ti SUPER with FP32, ROI `128x128x128`, model-preprocessed output, overlap
`0.5`, and the migrated cluster window batch of four.

The four-case mean Dice was `0.5165785551`; E014 recorded
`0.5165785849`, a difference of approximately `-2.98e-8`. The run completed in
13.95 seconds. The resolved policy source was `explicit_top_level`, the
runtime profile was `native`, all four validation batches completed, model
training mode was restored, and the complete model-state digest was identical
before and after validation:

```text
model state before and after
5417c2a9d5be69d4319fc171c04904995b0d5e4666fd2655a52bac0a2cbd7ff4

cut6_live_validation.json
75f1a16c2841e38bacdc6d8ddc6ffcd21e475546c617ff8bea0e160338037aa9

cut6_live_validation.log
807539ba2a4b86e94b01979ab4de60e4669edbbb50311da5e9c1b04720531cfc
```

### Legacy-policy and 3D mean-ensemble replays

Two additional real-model replays exercised the remaining Cut 6 migration
boundaries with the same checkpoint, four cases, FP32 policy, and window batch
four.

The canonical run resolved from `cfg.inference` with source
`explicit_top_level`; the legacy run removed that top-level policy and resolved
the pinned saved run's unmodified `validation.inference` with source
`legacy_validation`. Both resolved to exactly the same typed policy. All four
floating-point probability volumes were byte-identical between the two paths,
including their shapes and per-case SHA-256 digests. Every aggregate metric
was also identical; both runs produced Dice `0.5165785551`, HD95
`12.4736900330`, and surface Dice `0.7291332483`.

The 3D mean-ensemble replay used two deterministic DynUNet predictions per
case. For every case, both member probability volumes were byte-identical to
the canonical single prediction, the mean output was byte-identical to those
members, and the complete metric result matched the canonical run. This
exercises the real `[N,B,C,D,H,W]` mean path rather than only its synthetic
unit fixture.

Across the canonical, legacy, and ensemble executions, the complete model
state retained digest
`5417c2a9d5be69d4319fc171c04904995b0d5e4666fd2655a52bac0a2cbd7ff4`
and training mode was restored after every replay. Durable replay artifacts:

```text
cut6_replay_tests.json
b199ade9dc97e690eac0deb7c60aa0d3b25be2aec4bb5308fdf94ef8147ae3b5

cut6_replay_tests.log
e3adc5ef1ed5699f69c2d30885c85db2ae4629a10c172f10df663a7fa29aad95
```

Artifacts remain outside Git under:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut6_20260802a/
```

### Acceptance scope and invalidation

E015 establishes on the pinned Desktop snapshot that canonical training
profiles compose the new policy/runtime ownership correctly, training
validation uses the same probability executor as evaluation, ground-truth
runtime incompatibility fails before prediction, existing metrics are
numerically preserved on the locked live cases, and validation does not mutate
model state. It additionally establishes byte-level probability parity between
the canonical and legacy policy sources and correct real-volume 3D mean
ensembling for the deterministic discriminative backend. The Cut 6 diff does
not alter `train_one_epoch`, optimization,
loss, scheduler, EMA, DP/DDP construction, or checkpoint-writing logic; their
affected regression tests remain green.

E015 does not certify a new end-to-end optimization run, LRZ/`.sqsh` parity,
T4 execution, FP16, 2D geometry-aware evaluation, native-space inversion, or
Grand Challenge container behavior. Changes to training config composition,
runtime assessment, shared probability execution, ensemble semantics, the
pinned model/input state, or relevant PyTorch/MONAI numerical behavior require
the corresponding evidence to be rerun.

---

## Evidence item E016: Cut 7 native-space restoration and NIfTI correctness

### Status and question

| Field | Value |
|---|---|
| Evidence ID | `E016` |
| Status | `HISTORICAL`; superseded by E017 for the finalized Cut 7 architecture |
| Base commit | `0d9ae4460133f785b97d338e31af75e6312cdf55` plus the uncommitted Cut 7 diff |
| Environment | Desktop WSL `MedSegDiff_env`, MONAI 1.5.0, NVIDIA GeForce RTX 4070 Ti SUPER with 16,376 MiB |

E016 asks whether a real repository DynUNet probability can traverse the
case-aware, label-free preprocessing path, be inferred on its trained model
grid, be continuously restored before thresholding, and be written as a
binary NIfTI on the exact original T1 grid. It also asks whether the same
restoration contract can supply native-grid repository-model evaluation
without weakening the strict prediction/reference geometry checks from Cut 5.

### Preprocessing trace characterization

Before defining the inverse boundary, a four-case probe exercised the pinned
ISLES26 validation records used by E014. The cases jointly covered identity
geometry, an LAS-to-RAS left-right flip with 0.5 mm-to-1 mm resampling, an
oblique affine, and 1.5 mm-to-1 mm resampling. For every case:

- label-free and labeled preprocessing produced byte-identical image tensors;
- the returned image remained a `MetaTensor` whose affine matched the recorded
  model geometry;
- native shape, affine, spacing, orientation, qform/sform, and their codes were
  captured before preprocessing;
- the applied MONAI history retained `Orientation` and `SpatialResample` where
  applicable and contained no pending operations.

Durable trace evidence:

```text
cut7_preprocessing_trace_probe.json
7559deeb2910bf6aac0ea991bc8b1fe8eb2ab0a9b366be31c9caea44004056c0

cut7_preprocessing_trace_probe.log
036d36ec9ebd198c0437709cd81ed1a91486c3e4aec81c75b1b75be190641dde
```

### Contract and regression verification

The synthetic Cut 7 fixtures cover identity, permutation, flip, anisotropic
spacing, translation, odd shapes, oblique geometry, an exact world-space
landmark, continuous probability restoration versus threshold-first
restoration, independently varying case grids, corrupted/incomplete traces,
empty/full masks, distinct qform/sform semantics, and refusal to write a
model-space result through the native writer. Written files are reopened and
checked for exact grid, header-form codes, `uint8` dtype, and binary values.

The final focused native spatial/output suite passed 16/16 tests. Before the
last four focused assertions were added, the complete inference regression
family passed 61/61 and the complete evaluation regression family passed
147/147. The added assertions exercise only the already-passing spatial
landmark, padding, distinct-form, and fail-closed writer paths; they
subsequently passed in the focused 16-test run.

```text
cut7_inference_regression.log
8931a629c47c13ad32f8a35629dc58aee4590f692e134b9cb48b44a10ea2c298

cut7_evaluation_regression.log
4eedd6d758d61b42d946c73df80740e0419d3d05389942416089284758212e7e
```

The regression commands were:

```bash
python -m unittest discover -s tests -p 'test_inference*.py' -v
python -m unittest discover -s tests -p 'test_evaluation*.py' -v
```

### Live label-free native-output replay

The live replay used the pinned p5n1 DynUNet checkpoint
`best_model_step_040000_dice_3d_0.5724` and case `sub-r035s015`. Its checkpoint
SHA-256 was
`120c93ee6a32f79829bf8a6b2d3ab7db59861f865ad1e8a37889f87ab0c82441`.
The model predicted in RAS model space with shape `[192,256,256]` and
approximately 1 mm spacing. Cut 7 restored the floating probability to the
raw T1's LAS grid with shape `[192,512,512]` and spacing
`[0.99989998,0.5,0.5]`, then applied threshold `0.5` and wrote the native mask.

After reopening, native shape, effective affine, qform, sform, qform/sform
codes, `uint8` dtype, and binary value set all passed. End-to-end time,
including preprocessing, model loading, inference, restoration, writing, and
reopening, was 8.91 seconds. Inference plus restoration took 4.36 seconds and
peak allocated GPU memory was 1,869,463,040 bytes. This is Desktop evidence,
not T4 release certification.

```text
cut7_live_native.log
b1240bf1f650723c07f9f3a468291a63d241699c9a4515054d49c16795c8a185

cut7_live_native_output/report.json
f71a7de24656b2337d596f6119d53e89d8aa293b06f69f9d21f7c98b79adaff4

cut7_live_native_output/segmentation.nii.gz
632ca5ddc9bee3123bfbef568b16b82154507312093f306464abe95869045731
```

### Live native evaluator replay

The repository evaluator was also composed with
`inference=sliding_window_native`, the native runtime, FP32, validation batch
one, and the focused E014 split. The configured `val_fast` subset contained
one case, `sub-r049s033`, whose oblique 1.5 mm input exercises a materially
different geometry from the label-free replay. The evaluator produced one
strictly geometry-matched native `VolumeSample`, completed the fixed-threshold
metric path, and reported Dice `0.714777` without using a compatibility
fallback.

```text
cut7_live_native_evaluation.log
0cf4bf4a31b0398d45ee53997f0281ca886d3400e41531859578b2cdccbfce22

cut7_live_native_evaluation/canonical_results.json
294c6f6313bd2005253c6afe0b488169ec9c00fd844cefca4214ad571b1871ff
```

### Expanded `val_full` native evaluator replay

An additional Desktop replay selected the original ISLES26 `val_full` subset,
which resolved to 193 three-dimensional volumes. It used the same pinned
DynUNet checkpoint, `inference=sliding_window_native`, FP32, case batch one,
and zero validation workers. The evaluator completed 184 volumes across their
independent native grids without an OOM, worker-process failure, restoration
failure, or prediction/reference geometry mismatch. At 14 minutes 51 seconds
it inferred and restored the next case, `sub-r069s031`, then deliberately
stopped before metric calculation with:

```text
Error: VolumeSample prediction/reference affine mismatch: equal tensor shapes do not establish a shared physical grid.
```

This is the known O007 source-data case, not a newly introduced Cut 7 failure.
Its T1 and lesion mask both have native shape `[160,224,224]`, but their native
affines differ by approximately `0.0007935` mm. The replay therefore confirms
both broad case-specific restoration over 184 heterogeneous volumes and the
required fail-closed behavior when a native reference label does not occupy
the prediction's verified T1 grid. The remaining eight volumes were not
reached, and this run must not be represented as a completed 193-case metric
evaluation or as resolution of O007.

The exact command selected `dataset.active_subsets.val=val_full`,
`validation.val_batch_size=1`, `data_runtime.num_valid_workers=0`,
`inference=sliding_window_native`, and `inference.precision=fp32`. Durable
evidence:

```text
isles26_nested_15_5_best_2026-07-28.json
349f890a1d64e578b7ac258668d903a4b7861899cf64ab6f215b62d85825576b

cut7_val_full_native_evaluation.log
e4d5f2a76f72191ca034b066706d3d35902077280a4926c4aebc4bd8140190f0
```

To complete case-path coverage without needlessly repeating the 184 successful
cases, a follow-up evidence-only split retained the original loader-compatible
training partition but selected only the eight validation cases after
`sub-r069s031`:

```text
sub-r069s032  sub-r069s036  sub-r069s037  sub-r070s005
sub-r071s014  sub-r071s018  sub-r071s019  sub-r071s024
```

The temporary split explicitly contained neither `sub-r065s005` nor
`sub-r069s031`; the authoritative split was not modified. All eight cases
completed in 33 seconds, produced exactly eight unique per-case rows carrying
`native_input` prediction/reference metadata, and wrote the complete canonical
artifact set. The fixed-threshold (`0.5`) mean Dice over this remaining subset
was `0.423900`. Artifact-level verification established that the first log
contains 184 successful cases (including `sub-r065s005`) before the deliberate
`sub-r069s031` rejection, while the second result contains exactly the eight
previously unreached cases. Excluding both O007 cases from the certification
claim therefore gives complete case-path coverage of the other 191/191
`val_full` volumes: 183 from the first run plus eight from the follow-up.

```text
cut7_remaining_after_r069s031_v2.json
e805e2f623bf15a74e348c3bf95d48baa7630f4c689b585f102992662761d00a

cut7_remaining_native_evaluation_v2.log
1cad1c1e34a471670bc152d1abea14c26e54f3f79726cd88bb5a4b86e99e0874

cut7_remaining_native_evaluation_v2/canonical_results.json
370480d3a07d2f19162d5d526e9f03fea1cac4d48c817e1e08845e60dead1ca4

cut7_remaining_native_evaluation_v2/resolved_evaluation_config.yaml
7e2a77839d4f08c8b445065c25ed5574f2acdcaa50c6d82f55c383c292537db6

cut7_remaining_native_evaluation_v2/volume_metrics_per_threshold.csv
e4a44e01dfcd60152c2ecc07edfd06a75b027278c9334e78795558577a4dc50c

cut7_remaining_native_evaluation_v2/per_case_threshold_metrics.csv
bfadeb4821185fb8831dec44861455c31ae51166c9b57518835cd8452c60c161
```

All E016 artifacts remain outside Git under:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut7_20260802a/
```

### Acceptance scope and invalidation

E016 established on its pinned Desktop snapshot that the exploratory Cut 7
implementation restored floating
probabilities before thresholding; uses the case-specific T1 reference rather
than a dataset-wide geometry; preserves the original shape, affine,
orientation, qform/sform and codes; writes only validated native `uint8`
binary masks through the native writer; and supplies both model-preprocessed
and native-input repository evaluation without silently pairing different
spaces.

It does not certify the Grand Challenge socket lifecycle, Docker image,
AWS T4 timing/memory, LRZ `.sqsh` parity, native FP16 (O004), 2D reconstruction
(O005), or completed full-split labeled native metrics while O007 remains
unresolved. The expanded replay now directly demonstrates that O007 blocks
that certification at `sub-r069s031` after 184 successful native cases, while
the follow-up establishes case-path coverage for every otherwise-valid
`val_full` volume.
The blind label-free result is not blocked by O007. Changes to preprocessing,
spatial trace capture, interpolation conventions, threshold ordering, NIfTI
materialization, the pinned model/input, or relevant MONAI/nibabel behavior
require the corresponding evidence to be rerun. E016 does not certify the
post-decomposition typed producer and unified evaluator; E017 replaces that
acceptance role while retaining E016 as the discovery record that motivated
Cuts 7A-7D.

---

## Evidence item E017: Finalized Cut 7A-7E typed native-restoration milestone

### Status and question

| Field | Value |
|---|---|
| Evidence ID | `E017` |
| Status | `PASS`; Cut 7E evidence is complete and pending supervisory review/commit |
| Base commit | `b9b734db3fe5ee22934da955806819025b5715cf` |
| Environment | Desktop WSL `MedSegDiff_env`, NVIDIA GeForce RTX 4070 Ti SUPER, FP32 |

E017 asks whether the architecture finalized by Cuts 7A-7D, rather than the
superseded exploratory native-volume helper, can use one typed preprocessing
producer and the shared predictor for both model-preprocessed and native-input
output policies; preserve the accepted probability baseline; write a valid
label-free native NIfTI; fail closed on known source-grid violations; and
complete one evaluation over every other `val_full` case through the unified
Cut 7D evaluator.

### Pinned code, model, config, and input state

All evidence was generated from a clean tracked worktree at exact commit
`b9b734db3fe5ee22934da955806819025b5715cf`. The non-versioned worktree and
durable artifacts remain at:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut7e_20260803a/
```

The live runs used the pinned p5n1 DynUNet run and its complete saved config:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut5_20260802a/
  cut5_live_artifacts/pinned_p5n1/
```

The selected checkpoint was
`best_model_step_040000_dice_3d_0.5724.pth`, SHA-256
`120c93ee6a32f79829bf8a6b2d3ab7db59861f865ad1e8a37889f87ab0c82441`.
The four-case parity/model-space replay used the accepted focused split from
E014. The complete native replay derived an evidence-only split from:

```text
isles26_nested_15_5_best_2026-07-28.json
349f890a1d64e578b7ac258668d903a4b7861899cf64ab6f215b62d85825576b

cut7e_valid191_split.json
77607ad4d25ac31d6d82c07fe86693db39ab764df99284952003fab094ad03d4
```

The source split contains 193 validation records. The derived split preserves
the loader-compatible structure, selects 191 unique validation records, and
excludes exactly `sub-r065s005` and `sub-r069s031`. Neither the source split nor
patient data was modified or added to Git.

### Execution environment and commands

The Desktop environment reported Python 3.10.12, PyTorch 2.8.0+cu128 (CUDA
12.8), MONAI 1.5.0, nibabel 5.3.2, NumPy 1.26.4, and OmegaConf 2.3.0. CUDA was
available on an NVIDIA GeForce RTX 4070 Ti SUPER with 17,170,956,288 bytes of
device memory. This host establishes scientific and spatial behavior only; it
is not the AWS T4 or Grand Challenge runtime.

The committed-state regression commands were:

```bash
python -m unittest discover -s tests -p 'test_inference_*.py' -v
python -m unittest discover -s tests -p 'test_evaluation_*.py' -v
python -m unittest \
  tests.test_training_validation_inference \
  tests.test_training_inference_config_migration \
  tests.test_loader_stack_record_source \
  tests.test_nnunet_volume_spatial_contract \
  tests.test_nnunet_precursor_config \
  tests.test_nnunet_converter_affine \
  tests.test_nnunet_converter_roundtrip -v
```

They passed 74/74 inference tests, 149/149 evaluation tests, and 27/27
training/config/record-source/nnU-Net integration tests. The combined log
SHA-256 is
`fc201752fe5b09a0f75baa39293aafd2f6ae3b42ae300e05c3df25fcb869318d`.

The public evaluator was invoked with the pinned run directory and checkpoint,
`data_mode=full_volumes_3d`, `dataset.active_subsets.val=val_full`, zero
validation workers, `validation.val_batch_size=1`, FP32, and either
`inference=sliding_window_model_space` or
`inference=sliding_window_native`. The four-case model-space replay retained
the accepted baseline `inference.sliding_window.sw_batch_size=4`; the complete
native replay used the native profile's batch-one sliding-window policy. The
full native command selected `cut7e_valid191_split.json` and wrote to
`cut7e_valid191_native_evaluation/`. Disposable evidence scripts performed
split auditing, earliest-representation hash comparison, label-free writing,
and post-run artifact verification; they are not production interfaces and
remain outside Git.

The two public evaluator invocations were:

```bash
python -u -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut5_20260802a/cut5_live_artifacts/pinned_p5n1 \
  evaluation.model_name=best_model_step_040000_dice_3d_0.5724 \
  evaluation.output_dir=/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut7e_20260803a/cut7e_model_space_evaluation \
  evaluation.device=cuda:0 \
  data_mode=full_volumes_3d \
  environment.dataset.data_root=/mnt/c/Users/minanessiem/Development/isles26_combined/atlas21_training_raw/Training_Raw \
  environment.dataset.split_file=/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut5_20260802a/cut5_live_artifacts/focused_split.json \
  dataset.active_subsets.val=val_full \
  data_runtime.num_valid_workers=0 \
  validation=sliding_window_3d_metrics_subset \
  validation.val_batch_size=1 \
  inference=sliding_window_model_space \
  inference.precision=fp32 \
  inference.sliding_window.sw_batch_size=4

python -u -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut5_20260802a/cut5_live_artifacts/pinned_p5n1 \
  evaluation.model_name=best_model_step_040000_dice_3d_0.5724 \
  evaluation.output_dir=/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut7e_20260803a/cut7e_valid191_native_evaluation \
  evaluation.device=cuda:0 \
  data_mode=full_volumes_3d \
  environment.dataset.data_root=/mnt/c/Users/minanessiem/Development/isles26_combined/atlas21_training_raw/Training_Raw \
  environment.dataset.split_file=/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut7e_20260803a/cut7e_valid191_split.json \
  dataset.active_subsets.val=val_full \
  data_runtime.num_valid_workers=0 \
  validation=sliding_window_3d_metrics_subset \
  validation.val_batch_size=1 \
  inference=sliding_window_native \
  inference.precision=fp32
```

### Shared model-grid probability and model-space evaluation

One reusable typed producer yielded the same preprocessed case tensor to the
same prepared predictor under both output policies. Across `sub-r004s006`,
`sub-r035s015`, `sub-r041s108`, and `sub-r049s033`:

- every model-policy probability hash exactly matched the corresponding
  native-policy probability hash at their shared model-grid executor boundary;
- all four hashes exactly matched the accepted Cut 6 baseline digests;
- both policies reported `explicit_top_level` as their policy source; and
- the replay completed in 16.835 seconds.

The finalized Cut 7D public evaluator then completed the same four records in
`model_preprocessed` space and wrote the canonical report set. Its fixed 0.5
threshold mean Dice was `0.516579`; this metric is recorded as an execution
result, while the earlier probability hashes are the stronger parity boundary.

```text
cut7e_model_probability_parity.json
29d7494f0fb372a1a323875e68a1e625ec38f03dee7cb2c1edecbcdecf3229d5

cut7e_model_probability_parity.log
b1b2428d58983f5b4b0c116ba39d6a8cbe0cb37034dbb668014a5321e1f8d0ab

cut7e_model_space_evaluation/canonical_results.json
9b69e6deabb99bdb116a07ad2c564044fc1a596bca146c8e8ecc5f9b137d4ef

cut7e_model_space_evaluation.log
2af2d5ab4672a1f3a7e38ba00b8c8fdd4fadfafb047fe0028f5236ac29f4b79c
```

### Label-free native-output validation

Case `sub-r035s015` was loaded from a record containing no label. The shared
producer and predictor inferred on the model grid, restored the floating
probability, thresholded in `native_input` space, and wrote
`segmentation.nii.gz`. Reopening the result proved exact native input shape,
affine, qform, sform, and form codes; `uint8` dtype; and value set `{0,1}`. The
native result has shape `[192,512,512]`, spacing
`[0.99989998,0.5,0.5]`, and LAS orientation.

Inference plus restoration took 3.930 seconds; the complete load/infer/write/
reopen check took 8.011 seconds. Peak allocated GPU memory was 1,869,463,040
bytes. These measurements are descriptive Desktop evidence, not a T4 resource
certificate.

```text
cut7e_label_free_native_output/report.json
e9a00c6bd7322aaa495a2f8caa5b0146a11495c4dd3fe05df8d2d76c85a5e081

cut7e_label_free_native_output/segmentation.nii.gz
632ca5ddc9bee3123bfbef568b16b82154507312093f306464abe95869045731

cut7e_label_free_native.log
b6d2942dbab7bb6f06881445f021743d64535fc9930e4dec90724c0c4ea4f366
```

### Invalid-source audit and completed 191-case native evaluation

Before the complete run, each excluded O007 case was independently passed to
the finalized reusable labeled-case producer. Both failed before metrics with
`SpatialRestorationError` because the model label and image did not occupy the
same model-space physical grid. No tolerance was weakened and neither case was
silently resampled to obtain a complete result.

```text
cut7e_split_and_invalid_audit.json
39553c777a88cade9d758da3d6e2252564c7b4dd9eaed2edba01ddd5bbbe0cf7

cut7e_split_and_invalid_audit.log
b773b2c577a9000ac3cab5c082c6794673ade4319faecd992beba8ab7565168
```

The single evidence-only native evaluation then completed all 191 selected
cases through the unified Cut 7D typed-volume loop in 879 seconds. It produced
191 unique spatial-contract samples and 191 unique per-case metric rows, with
the exact expected case-ID set, zero unaccounted failures, and the complete
canonical artifact set. Every prediction/reference pair declared
`native_input`, had identical shapes and orientations, and passed the
production `rtol=1e-5`, `atol=1e-5` affine and spacing contract. The maximum
absolute affine-element difference was 0.0009003 mm and maximum spacing
difference was 0.000005725 mm; the affine maximum occurs on translated
coordinates where the contract's relative term applies (`sub-r069s032`, affine
element `[0,3]`, `-101.6347809` versus `-101.6338806`).

The first disposable post-run verifier compared serialized geometry mappings
for exact floating-point equality and therefore rejected this valid
prediction/reference pair after the production evaluation had completed. The
verifier was corrected to mirror the already-enforced production tolerance,
then passed all 191 unchanged canonical samples. No inference or evaluation
artifact was regenerated, no source case was added, and no production
tolerance changed.

The completed selection exercises 54 native shapes, 49 spacings when rounded
to 1e-6, LAS/PSR/RAS orientations, and 94 oblique cases. At fixed threshold
0.5 it reported mean Dice `0.575165`, mean Surface Dice `0.522845`, and mean
HD95 `53.768511`. These metrics confirm complete evaluator traversal but do
not supersede the competition's eventual held-out scoring.

```text
cut7e_valid191_native_evaluation.log
ae49ca6c789e80e068ac2b28c6efd0f8a1f0102ca7ff6e04d40c3674c6850362

cut7e_valid191_native_evaluation/canonical_results.json
6b819e1d0ff6ef8714370f27f39670e8d4b09d5940e47ac710c206e0c04fdb7c

cut7e_valid191_native_evaluation/resolved_evaluation_config.yaml
31de960935cb61a8a9764893cd4029f5d2777a833927177fddb1a50d9e7518b4

cut7e_valid191_native_evaluation/volume_metrics_per_threshold.csv
822404ed1953a3f6c4b666f013dd2a2df4449ad68edb73009fb8facd96ce4c3d

cut7e_valid191_native_evaluation/per_case_threshold_metrics.csv
7bb37b4a0ccc5da4ff18a58021082868c9b58a47bbf1885071171ccb251041d3

cut7e_valid191_native_evaluation/evaluation_summary.txt
f00e4475c8ee462e0db4da12cbed33038709dbe5159e34b0e3e7b8452526a4f5

cut7e_valid191_verification.json
84fbacc5e8492f77306086fe7f9521f589b45b93f853ccbbc51ff2656fb7b148
```

### Acceptance scope and invalidation

E017 satisfies Cut 7E's Desktop evidence contract for the finalized Cuts
7A-7D implementation: the typed producer and unified evaluator have real-model
evidence in both output spaces; the earliest shared probability representation
retains exact accepted parity; blind native output is a reopened, exact-grid
binary NIfTI; heterogeneous case-specific geometry is exercised broadly; the
two source violations fail closed; and every other `val_full` record completes
in one audited run.

E017 does not certify Docker construction, Grand Challenge socket I/O, AWS T4
memory or timing, native FP16 (O004), 2D reconstruction (O005), LRZ `.sqsh`
parity, or correction of the two O007 source cases. Those remain explicit
future/release gates. Changes to the typed case producer, saved-model loading,
probability executor, restoration/interpolation, output-space policy,
threshold order, NIfTI writer, evaluation spatial tolerance, relevant
MONAI/nibabel behavior, pinned checkpoint, or selected data invalidate the
corresponding portion of E017 and require it to be rerun.

---

## Evidence item E018: Cut 10 Grand Challenge container lifecycle

### Status and supersession

| Field | Value |
|---|---|
| Evidence ID | `E018` |
| Status | `PASS` for the pre-closure Cut 10A Desktop development lifecycle; superseded for final Cut 10A closure by E019 |
| Base commit | `8d4f87851fd812ae43a5dac8392372544f64aab2` plus the uncommitted Cut 10 overlay |
| Environment | Desktop WSL/Docker, NVIDIA GeForce RTX 4070 Ti SUPER, model FP32 |

E018 supersedes the Cut 7 ledger's statement that Docker construction and Grand
Challenge HTTP/socket I/O were untested. It does not supersede E017's scientific
and spatial evidence, and it is not the Cut 11 AWS T4 resource certificate or
the Cut 12 official-platform release certificate.

### Question being answered

E018 asks whether a model-independent Linux/amd64 image can initialize the
separately mounted Cut 9 model artifact before readiness; dispatch arbitrary
platform socket slugs through an explicit manifest; reuse registered label-free
preprocessing and `src.inference`; restore one prediction to the native input
grid; write and reopen the required binary NIfTI beneath `/output`; survive a
fresh-container replay; fail health closed when model initialization fails; and
save as an independently replaceable Docker archive without embedding weights.

### Pinned code and image state

The isolated Desktop worktree is:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut10_20260803a/
```

It is detached at Cut 9 commit
`8d4f87851fd812ae43a5dac8392372544f64aab2` with the exact uncommitted Cut 10
overlay under supervisory review. The image is built from the immutable base:

```text
pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
sha256:f894dae26e1ee8557c544f9cfdb9dc011b1552bf3c1e656b422f2e221d380e40
```

The resulting image is `medseg-diffusion-gc:cut10-dev`, image ID
`sha256:aab9461440a7f00ed08a9ad557e6fd6cc18ac661b62eb7c6c1eb31607c351724`.
Docker inspection proved `linux/amd64`, configured user
`algorithm:algorithm`, and label
`org.grand-challenge.api-method=invoke`. An independent filesystem audit found
no `.pth`, `.pt`, or `.ckpt` under `/opt/app` and an empty `/opt/ml/model` in the
image. The Dockerfile, pip lock, and fixture manifest SHA-256 values are:

```text
d46a66175b8ec15c5280223699b22a6ef72efa44e7757c0a245b1b95034c2512  Dockerfile
60091acb5f6512a04b713d01f4170bf679fd1b3154a00e200766fbd9e420ba83  requirements.lock
44d0c98774ead4f5d876a47d982fbf92077780b9fbf7d2461274c62b39bd2ffd  interface_manifest.fixture.yaml
```

The image import audit reported Python 3.11, PyTorch `2.6.0+cu126`, CUDA 12.6,
MONAI 1.5.0, nibabel 5.3.2, NumPy 1.26.4, OmegaConf 2.3.0, SimpleITK 2.5.2,
FastAPI 0.116.1, and Uvicorn 0.35.0. The base digest pins the preinstalled
Python/PyTorch/CUDA layer; every additional pip-installed inference dependency
is pinned in `requirements.lock`.

### Pinned model/config/input state

The test used the same p5n1 DynUNet checkpoint as E017:

```text
best_model_step_040000_dice_3d_0.5724.pth
120c93ee6a32f79829bf8a6b2d3ab7db59861f865ad1e8a37889f87ab0c82441
```

The independently generated Cut 9 artifact was mounted read-only at
`/opt/ml/model`. Its resolved model config and standalone inference-policy hashes
were respectively
`7152715bded1f11cc244a8a4270b6cc592b8a0e555f40d6d40af5b2924cc918a`
and
`74848c03ce60616a5a94541d5d53c04c1627bd4c933d494cf5dfea2e2f586520`.
The model archive is:

```text
cut10_live_artifacts/fp32_model/pinned_p5n1_fp32.tar.gz
d58b6e1a97e7b905cbccaf3c0ab302d609a029d8346d62565df9c511bf84cd51
```

The container proof deliberately used the supported native FP32 policy. It does
not close deferred issue O004 concerning native FP16 sliding-window accumulation.

The input was a generated, non-patient 3D NIfTI with shape `[48,56,64]`,
anisotropic spacing `[1.5,1.2,2.0]`, ALS orientation, nontrivial permutation and
translation, qform code 1, and sform code 2:

```text
cut10_live_artifacts/input/images/fixture-input/synthetic-t1.nii.gz
1dc4ad45094f6fe4e9c5bb02d510cafddbf19a14551d9709b94e7b0a3b60cfd7
```

The fixture manifest intentionally uses opaque, noncanonical socket slugs and
maps its image socket to repository raw key `T1`. These values are development
fixtures, not claimed ISLES26 phase slugs.

### Execution environment and lifecycle

The production-equivalent smoke command ran the image with:

- `--network none` and `--read-only`;
- read-only `/input` and `/opt/ml/model` bind mounts;
- writable `/output` and a bounded transient `/tmp` tmpfs;
- 8 CPUs, 32 GB host-memory limit, 1 GB shared memory, and one GPU;
- non-root UID verification before invoke.

The runtime initialized the artifact, strict model loader, registered ISLES26
adapter, unlabeled case producer, shared `ProbabilityPredictor`, and production
`gc_submission` policy before `/health` returned HTTP 200. POST `/invoke`
returned HTTP 201. A separate run with an empty model mount served `/health` as
HTTP 503 and logged only the error type. Multiple fresh containers completed,
showing that success did not depend on prior `/tmp` state.

The final structured run reported 0.479 seconds for transport resolution,
preprocessing, shared prediction, native restoration, and output writing, with
peak allocated CUDA memory 1,990,069,760 bytes. These Desktop values are
descriptive only. They do not establish T4 memory headroom or the ten-minute
platform deadline.

The reopened final output proved exact input shape, affine, qform, sform, and
form-code equality; `uint8` dtype; and allowed-value subset `{0,1}`:

```text
cut10_live_artifacts/smoke_output_structured/images/fixture-output/output.nii.gz
65a7334c1d3eab159b5164e08cf2a8e4b378f7ffea0b22a866279a2574f3033f
```

### Regression and archive results

The focused Cut 10 interface/image/runtime/diagnostic/builder suite passed
28/28 tests. The final dependency-wide matrix then passed 290/290 tests:

```bash
python -m unittest discover -s tests -p 'test_inference_*.py' -v       # 74
python -m unittest discover -s tests -p 'test_evaluation_*.py' -v      # 149
python -m unittest discover -s tests -p 'test_gc_*.py' -v              # 36
python -m unittest \
  tests.test_training_validation_inference \
  tests.test_training_inference_config_migration \
  tests.test_loader_stack_record_source \
  tests.test_nnunet_volume_spatial_contract \
  tests.test_nnunet_precursor_config \
  tests.test_nnunet_converter_affine \
  tests.test_nnunet_converter_roundtrip \
  tests.test_model_loader -v                                           # 31
```

The combined relevant-regression log SHA-256 is
`b712fc955f565902e881550fea396194bc364c62d99109af193b56daa0ea59f0`.
Repository-wide `unittest` discovery additionally passed 519 tests and exposed
one pre-existing collection error: `tests/test_summarize_threshold_calibration.py`
imports absent `scripts.summarize_threshold_calibration`. Neither file is
changed by Cut 10; the missing module already exists at the pinned Cut 9 base.
The full-discovery log SHA-256 is
`e492e185c4e2ea54ff5cd6d0d5d8daddd98995ddedc25f91021a94eca55c8ca1`.

The saved image archive passed `gzip -t` and is:

```text
cut10_live_artifacts/image_export/medseg-diffusion-gc-cut10-dev.tar.gz
d81ef1cb2066b7340bc9d75a700d8b6b460f3faec4a6f4907f6c886da0ae74ba
approximately 3.2 GB
```

### Acceptance scope and invalidation

E018 established the initial Cut 10A development boundary: the model and image
are independently produced; no weights are embedded; the container has a pinned
offline runtime; arbitrary socket slugs remain transport-only; preprocessing and
prediction delegate to the shared repository path with labels disabled; native
NIfTI output is reopened and spatially validated; invalid initialization prevents
readiness; and the built image is saveable as a valid gzip stream.

E018 does not certify the official ISLES26 interface reconciliation, a
platform-hosted try-out, AWS g4dn.2xlarge/T4 resource behavior, the ten-minute
deadline, native FP16 stability/parity, clean reload of both release archives,
or final release documentation. Those remain Cut 10B, Cut 11, and Cut 12 gates.
Changes to the Dockerfile/base
digest, inference lock, interface manifest, artifact contents, model loader,
dataset adapter, shared inference/restoration/writer path, HTTP lifecycle, or
container build/test/save implementation invalidate the corresponding evidence.

---

## Evidence item E019: Cut 10A bounded closure

### Status and supersession

| Field | Value |
|---|---|
| Evidence ID | `E019` |
| Status | `PASS` for bounded Cut 10A closure; committed as `70cc6caae0a88a140e771d8cf3aa1b593510b313` |
| Base commit | `8d4f87851fd812ae43a5dac8392372544f64aab2` plus the uncommitted final Cut 10A overlay |
| Environment | Desktop WSL/Docker, NVIDIA GeForce RTX 4070 Ti SUPER, model FP32 |

E019 retains E018's pinned model, synthetic input, base-image digest, dependency
lock, and fixture manifest. It supersedes E018 only for the tightened Cut 10A
runtime, diagnostic, image-audit, lifecycle-test, and documentation contracts.

### Question being answered

E019 asks whether the bounded Cut 10A implementation closes its remaining
development-contract gaps without entering official-interface work: production
must retain the artifact's native-output policy; the diagnostic runtime may
explicitly recompose that policy for model-space inspection; diagnostics must
never write to `/output` or a descendant; the versioned FastAPI application
must be tested in its real image environment; image build and save must audit
that no model payload is embedded; and the lifecycle tester must independently
compare the sole NIfTI input and output geometries.

### Pinned code and image state

The isolated Desktop worktree is:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut10a_closure_20260803a/
```

It is detached at Cut 9 commit
`8d4f87851fd812ae43a5dac8392372544f64aab2` with the exact uncommitted Cut 10A
overlay under review. The rebuilt image is
`medseg-diffusion-gc:cut10a-closure-dev`, image ID
`sha256:455965c8f20a68c7c658d5d80447fe039a461f0e6cbe222c59a8e3dabd9bdcf2`.
Its build report records `linux/amd64`, user `algorithm:algorithm`, API method
`invoke`, `model_payload_audited: true`, and `model_embedded: false`:

```text
cut10a_live_artifacts/image_build/container_build_report.json
ef2dd8747285c14a33b76dc5edad7e3f3bb96ea90000f585ba84f47687ed3c12
```

The audit executed against the built image with networking disabled and a
read-only root filesystem. It required `/opt/ml/model` to contain no files and
rejected checkpoint-like `.pth`, `.pt`, or `.ckpt` payloads beneath `/opt/app`
or `/opt/ml/model`. The same audit ran again before image export.

### Execution and results

The complete Grand Challenge and shared-inference regression families ran in
the desktop `MedSegDiff_env`: 117 tests completed with 114 passes and three
expected skips because FastAPI is intentionally installed in the GC image, not
the host research environment. Those three application tests were then mounted
read-only into the rebuilt image with the platform-equivalent writable `/tmp`
tmpfs and passed 3/3. They directly exercised successful readiness/invocation,
failed initialization as HTTP 503, and inference failure as generic HTTP 500:

```text
cut10a_live_artifacts/app_test/test_gc_app.log
faf56e1437d7400ebf4bda7385f70d40c5b0fa57edea634e605c95e687030a2d
```

The production `gc_submission` replay mounted the unchanged E018 model and
input read-only, ran non-root with no network and a read-only root filesystem,
returned HTTP 201, and independently reopened the sole input and output. Shape,
affine, qform, sform, and form codes matched; the output remained binary
`uint8`. Runtime provenance reported `policy_origin=artifact`, peak allocated
CUDA memory `1,990,069,760` bytes, and 0.747 seconds for the synthetic
invocation. The deterministic output hash remained identical to E018:

```text
cut10a_live_artifacts/smoke_output/images/fixture-output/output.nii.gz
65a7334c1d3eab159b5164e08cf2a8e4b378f7ffea0b22a866279a2574f3033f
```

The diagnostic profile explicitly selected `model_preprocessed`, retained
probability and mask arrays of shape `[1,1,67,72,127]`, recorded
`inference_policy_origin=diagnostic_output_space_override`, and wrote only
beneath `/diagnostic`. Unit contracts reject that override for production and
reject both `/output` and every descendant as a diagnostic destination:

```text
cut10a_live_artifacts/diagnostic/diagnostic_report.json
a73a12e1cc4af3579aa4910fe7ab15a2a425309359fd9353cf660e33c07b3c3c
```

The independently saved image passed `gzip -t`:

```text
cut10a_live_artifacts/image_export/medseg-diffusion-gc-cut10a-closure-dev.tar.gz
7e0f237d39d3be0bf7e5a4f42bceb62653daf04ae14b3a12f6bf724898c722f5
```

### Acceptance scope and invalidation

E019 closes the bounded Cut 10A development contract. It demonstrates a
model-independent, audited, saveable NIfTI fixture image; artifact-owned
production inference policy; explicit diagnostic-only output-space override;
versioned HTTP application behavior inside the real image; and external
single-input native-grid validation. It does not make the image an official
ISLES26 submission.

Cut 10B still owns the published ISLES26 socket manifest, ordered probability
and segmentation outputs, compressed MHA conversion/validation, official
technical JSON schema, external tester-sidecar HTTP lifecycle, and 300-second
organizer-style invoke bound. Cut 11 owns T4 resource and ten-minute
qualification. Cut 12 owns hosted-platform try-out and release closure. Changes
to the audited image contents, runtime policy composition, diagnostic boundary,
HTTP application, NIfTI lifecycle validation, image build/save implementation,
or the pinned E018 model/input invalidate the corresponding portion of E019.

---

## Evidence item E020: Cut 10B official ISLES26 interface reconciliation

### Status and supersession

| Field | Value |
|---|---|
| Evidence ID | `E020` |
| Status | `PASS` for Cut 10B implementation and Desktop lifecycle; pending supervisory review/commit |
| Base commit | `70cc6caae0a88a140e771d8cf3aa1b593510b313` plus the uncommitted Cut 10B overlay |
| Environment | Desktop WSL/Docker, NVIDIA GeForce RTX 4070 Ti SUPER, model FP32 |

E020 supersedes E019 only for the interface manifest, output set, MHA
transport, and external HTTP-probe evidence. E019 remains the historical Cut
10A proof for the generic image boundary, model-independent build, diagnostic
path, and original NIfTI lifecycle at its exact source state.

### Question being answered

E020 asks whether the generic Cut 10A image can be reconciled with official
ISLES26 template commit
`5e25bfc36b1dc6d9c04c8c364f53fb75c6afad32` without creating a second
prediction path: accept the exact T1 and metadata input sockets, validate the
published nullable metadata fields without conditioning the model, derive both
official outputs from one restored `PredictionResult`, write them as compressed
MHA on the native T1 grid, and return HTTP 201 only after the complete output
set validates. It also asks whether HTTP is exercised from outside the
algorithm container with a 300-second local invoke bound while both containers
remain on an internal, no-internet Docker network.

### Pinned code and image state

The isolated Desktop worktree is:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut10b_20260804a/
```

It is detached at committed Cut 10A state
`70cc6caae0a88a140e771d8cf3aa1b593510b313` with the exact uncommitted Cut
10B overlay under review. The official image is
`medseg-diffusion-gc:cut10b-dev`, image ID
`sha256:5f5f87e2b401cb2cea66ac2900c22224a9d5bf36bce967446bd292be565c3030`.
Its build report retains `linux/amd64`, non-root `algorithm:algorithm`, API
method `invoke`, and the no-embedded-model audit:

```text
cut10b_live_artifacts_20260804a/image_build/container_build_report.json
af44417b6f67ff9e170462d9fc6c61d237450f91f7cc267cfc467efc28e9ccea
```

The base-image digest and dependency lock are unchanged from E019. Current
transport-boundary hashes are:

```text
7de961b2a4ff90aab3baf79c3fa156cea0008b02d5d141e27af06cb06c867d45  Dockerfile
60091acb5f6512a04b713d01f4170bf679fd1b3154a00e200766fbd9e420ba83  requirements.lock
d676aca5ecb753a8e81b00cda0c1f8d5875346e974ae1df850e5e52f7a277db2  interfaces/isles26.yaml
2b98514d5067c3d39ac3e0c61e1a2b09b16bd964d01a788c68924d70b261af0b  interfaces/fixture_single_nifti.yaml
```

### Pinned model/config/input state

The live lifecycle reused the E018/E019 p5n1 DynUNet artifact without
repackaging or mutation. Its checkpoint, resolved config, and standalone FP32
native-output inference-policy hashes remain those recorded in E018. The same
non-patient synthetic NIfTI was placed beneath the official input path; its
shape, anisotropic spacing, ALS orientation, permutation, translation, qform,
and sform therefore remain unchanged:

```text
cut10b_live_artifacts_20260804a/input/images/t1-brain-mri/synthetic-t1.nii.gz
1dc4ad45094f6fe4e9c5bb02d510cafddbf19a14551d9709b94e7b0a3b60cfd7
```

`inputs.json` declared exactly `t1-brain-mri` and `stroke-metadata`; the
metadata fixture supplied null values for `CENTER`, `CHRONICITY`, and
`DAYS_POST_STROKE`, exercising the organizer-documented nullable contract.

### Execution environment and commands

The Desktop `MedSegDiff_env` ran the focused and dependency-wide tests. The
image builder then selected `configs/interfaces/isles26.yaml`, mounted the
unchanged model and input read-only, allocated writable `/output` plus bounded
tmpfs, and started the algorithm non-root on a newly created Docker network
using `docker network create --internal`. Health and invoke requests came from
a separate ephemeral tester container on that network, not from `docker exec`;
the client uses `http.client` so redirects are not followed and requires exact
status 200/201. The invoke request and its host-side process guard were bounded
at 300 and 305 seconds respectively.

The relevant regression matrix was:

```bash
python -m unittest discover -s tests -p 'test_inference_*.py' -q
python -m unittest discover -s tests -p 'test_evaluation_*.py' -q
python -m unittest discover -s tests -p 'test_gc_*.py' -q
python -m unittest \
  tests.test_training_validation_inference \
  tests.test_training_inference_config_migration \
  tests.test_loader_stack_record_source \
  tests.test_nnunet_volume_spatial_contract \
  tests.test_nnunet_precursor_config \
  tests.test_nnunet_converter_affine \
  tests.test_nnunet_converter_roundtrip \
  tests.test_model_loader -q
```

### Results and durable artifacts

The matrix passed 307 tests: 74 inference, 149 evaluation, 53 GC, and 31
training-validation/loader/nnU-Net/model-loader tests. Three GC application
tests were expected host skips because FastAPI is image-only; the same tests
then passed 3/3 inside the official rebuilt image. A final focused MHA suite
passed 9/9 after adding explicit axis-flip coverage, and the final sidecar
status/timeout suite passed 10/10. The stable combined matrix log is:

```text
cut10b_live_artifacts_20260804a/relevant_regression.log
6a0eb14f2deff75df6b14a1565d22123fc69825d9161d3c70fc3fa20d8e99d30
```

The official lifecycle returned HTTP 201 and produced both required files:

```text
cut10b_live_artifacts_20260804a/output_final_replay/images/stroke-lesion-segmentation/output.mha
402c0e785a6b660a2fa7864b8513d53d3fd7490935ea9445bf3ef8c21ec24ef9

cut10b_live_artifacts_20260804a/output_final_replay/images/lesion-probability-map/output.mha
e147d33a831b435c46e7db010e3a61753fd0786350f3c328e2105f46e5853702
```

Both reopened as compressed MHA with size `[48,56,64]`, spacing
`[1.5,1.2000000476837158,2.0]`, origin `[-20,10,5]`, and direction
`[0,1,0,-1,0,0,0,0,1]`, exactly matching the native T1 physical grid. The
segmentation is binary `uint8`; the probability map is finite `float32` in
`[0,1]`; and an independent array comparison proved the segmentation equals
the same native probability thresholded at `0.5`. Runtime provenance reported
`policy_origin=artifact`, peak allocated CUDA memory `1,990,069,760` bytes,
and 0.682 seconds for the final synthetic Desktop replay. The earlier full
value/threshold inspection produced the same two byte hashes.

The opaque NIfTI fixture was rebuilt as
`medseg-diffusion-gc:cut10b-nifti-fixture`, image ID
`sha256:c775b5fe041398794784ca657d71f52424bed124dd13e9a4a78973430ed80597`.
Its external-sidecar lifecycle passed and reproduced the E018/E019 binary
NIfTI byte-for-byte:

```text
cut10b_live_artifacts_20260804a/nifti_fixture_output/images/fixture-output/output.nii.gz
65a7334c1d3eab159b5164e08cf2a8e4b378f7ffea0b22a866279a2574f3033f
```

The official image was independently saved, passed `gzip -t`, and is
approximately 3.2 GB:

```text
cut10b_live_artifacts_20260804a/image_export/medseg-diffusion-gc-cut10b-dev.tar.gz
12bbd1db2b4daa8d0bc4dd3c83f4a896bab901898315ba94e39508cf8c3124af
```

### Acceptance scope and invalidation

E020 closes the Desktop development evidence for Cut 10B: the exact official
socket contract is configuration-owned; shared preprocessing receives only
canonical `T1`; technical metadata is validated but unconsumed; one native
probability result supplies the ordered, explicit mask/probability output set;
RAS/LPS and array-order conversion preserve physical geometry; unsupported
shear fails; partial/stale declared output sets cannot return success; HTTP is
tested from outside the algorithm container on an internal network; and the
legacy NIfTI transport remains live and byte-stable.

This is not Cut 11 qualification. The synthetic Desktop timing and RTX memory
figures do not establish T4 memory headroom, worst-case-volume runtime, or the
ten-minute job limit. It is also not the Cut 12 hosted Grand Challenge try-out
or final upload release. Changes to the official manifest, metadata schema,
result binding, MHA conversion/validation, complete-output transaction,
external sidecar lifecycle, image dependency state, shared native restoration,
or the pinned model/input invalidate the corresponding portion of E020.

---

## Ledger update template

Use this compact structure for completed evidence:

```text
## Evidence item E###: <name>
### Status and supersession
### Question being answered
### Pinned code state
### Pinned model/config/input state (when applicable)
### Execution environment and command
### Result and durable artifacts
### Acceptance scope and invalidation
```

For a pending asynchronous run, additionally record:

```text
### Requested runtime
### Expected artifacts
### Reconciliation and outcome handling
### Work authorized while pending
```
