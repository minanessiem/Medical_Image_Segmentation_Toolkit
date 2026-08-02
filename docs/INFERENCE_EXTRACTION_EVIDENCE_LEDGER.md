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
live repository-model/nnU-Net evidence. E002-E006 and E008 are retained only
as compact supersession records because they do not represent accepted current
designs.

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
| Versioned baseline record | `tests/fixtures/inference/isles26_dynunet_p5n1_baseline.json` |

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
