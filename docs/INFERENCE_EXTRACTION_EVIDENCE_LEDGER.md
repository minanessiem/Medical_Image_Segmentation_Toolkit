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
parity together with the unresolved FP16 finding. E002-E006 and E008 are
retained only as compact supersession records because they do not represent
accepted current designs.

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
