# Inference Extraction Evidence Ledger

**Governing document:** `docs/PRD_CAP_Inference_Package_Extraction_290726.md`
**Purpose:** Track asynchronous, environment-dependent evidence without blocking dependency-safe implementation work.
**Current CAP state:** Cut 0 is open for GPU baseline reconciliation; Cut 1 may proceed using the locked records, fixtures, and contracts that do not depend on the pending numerical result.

## Operating rule

Cluster, Docker, Grand Challenge, and hardware-qualification jobs are asynchronous evidence gates. Submitting one does not freeze development unless its result could change the interface currently being implemented.

Every asynchronous run must record:

1. the question it is intended to answer;
2. the exact code, config, model, data selection, and runtime request;
3. the expected artifacts and their locations;
4. objective acceptance and invalidation rules;
5. the follow-up action for success, scientific disagreement, infrastructure failure, timeout, or OOM;
6. which CAP work may continue while it is pending.

An output is not accepted merely because the Slurm job exits successfully. Its manifest and provenance must be reconciled against this ledger.

---

## Persistent implementation observations and decisions

These observations affect more than one evidence job or CAP cut. They should be re-checked when their underlying environment or implementation changes.

### O001: Test environments have different authority

| Environment | Observed runtime | Appropriate authority |
|---|---|---|
| Home desktop WSL | Python 3.10.12, PyTorch 2.8.0+cu128, OmegaConf 2.3.0 | Immediate unit, contract, synthetic-fixture, and consumer-regression feedback |
| LRZ standardized `.sqsh` | Python 3.12.3, PyTorch 2.6.0+cu124, MONAI 1.5.0 | Training-compatible dependency behavior, containerized CPU tests, and LRZ GPU characterization |
| Final Grand Challenge image on T4-equivalent hardware | Not yet built/certified | Release authority for CUDA compatibility, FP16/FP32 behavior, memory, timing, HTTP lifecycle, and output transport |

**Decision:** Desktop success accelerates development but does not establish `.sqsh` or T4 numerical/runtime parity. Environment-sensitive cut acceptance must identify which environment supplied the evidence.

### O002: Desktop cut testing must not use the existing desktop checkout

The desktop checkout was observed at commit `66458c1` with numerous unrelated tracked and untracked user changes. The laptop/LRZ base for the current work is `88d9b55dd2e3fa52df0c842e356b7e96a25b7b0b`, which was not present in the desktop repository object database.

**Decision:** Do not pull, reset, overlay, or otherwise repurpose the existing desktop checkout for this CAP. Create a fresh cut-specific snapshot under:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/
```

Build that snapshot from the intended laptop commit plus only the active cut's files, record and verify its archive hash, and explicitly activate:

```bash
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate
```

### O003: Slurm `--test-only` start times are advisory

`sbatch --test-only` is an LRZ Slurm scheduler forecast based on the state visible when queried. It is not a reservation or a lower bound. The Cut 1 CPU forecast indicated a multi-day wait, while the actual one-CPU jobs backfilled and completed almost immediately.

**Decision:** Submit one small, bounded, well-provenanced CPU job when standardized-runtime evidence is useful. Do not create duplicate jobs across queues solely because a test-only estimate is pessimistic. Record the real job state and accounting.

### O004: Queued environment evidence does not block dependency-safe refactoring

Cut 0 GPU job E001 characterizes the legacy numerical/resource baseline. Its result is required before certifying Cut 4 parity or removing the legacy prediction path, but it does not determine the shape of Cut 1 contracts or the strict-loading boundary in Cut 2.

**Decision:** Continue cuts whose interfaces do not depend on the pending measurement. Each queued job must declare what it blocks and what may proceed.

### O005: Deployment checkpoint loading requires a new strict boundary

Current repository evidence:

- `src/training/checkpoint_utils.py::load_model_state_dict_compat()` first attempts strict loading, then falls back to `strict=False` and returns missing/unexpected keys;
- training resume depends on that legacy compatibility behavior;
- `scripts/evaluation/core/model_loader.py` uses the compatibility loader and logs missing/unexpected keys rather than treating them as a deployment-fatal condition;
- `scripts/analysis/threshold_analysis.py` contains another older permissive loading path that should not become the new deployment source of truth.

**Decision for Cut 2:** Preserve training-resume defaults. Add a deployment/release loader that normalizes only known DP/DDP prefixes and then fails on every missing or unexpected key. Evaluation may delegate to the shared construction path while retaining its run-directory/checkpoint-selection interface, but release loading must never accept a partial state dict.

### O006: A test snapshot is immutable while a test is running

LRZ and desktop tests consume isolated snapshots. Changing an overlaid file after submission can otherwise make submission-time provenance differ from execution-time code.

**Decision:** Do not update a snapshot while a job/process is running. After a review change, wait for the current test to finish, update the snapshot, and rerun the relevant focused suite. Record which job or desktop run tested the final file state.

---

## Evidence item E001: Legacy four-case GPU baseline

### Status

| Field | Value |
|---|---|
| Evidence ID | `E001` |
| Slurm job | `5722967` |
| State | `ACCEPTED` |
| Submitted | `2026-07-31T00:52:51+02:00` |
| Scheduled start last observed | `2026-07-31T03:30:47+02:00` |
| Time limit | 10 minutes |
| CAP relationship | Completes the remaining measured-output portion of Cut 0 |
| Development blocking | No: Cuts 1-3 contracts, model loading, and preprocessing can proceed |
| Migration blocking | No: accepted baseline artifacts are available for Cut 4 parity |

Scheduler start times are advisory. Query Slurm rather than assuming the timestamp above remains current.

The job later completed with state `COMPLETED`, exit code `0:0`, and elapsed
time `00:02:00`. Its batch step reported approximately 4 MiB MaxRSS and the
main container step approximately 2.24 GiB MaxRSS. Reconciliation accepted the
result because the manifest recorded the pinned commit, checkpoint and focused
split hashes, all four requested cases, finite probability summaries within
`[0, 1]`, per-case artifacts, metrics, timings, and GPU-memory measurements.

Peak reserved GPU memory across the four cases was 7.578 GiB on the 16 GB V100;
total case inference time was 16.691 seconds. The stderr contained dependency
warnings and a generator-finalization exception emitted after all cases and
the complete manifest were written. This is classified as a non-fatal legacy
harness cleanup defect, not ignored as clean behavior; it should be eliminated
when the evaluation generator is migrated, but it does not invalidate the
captured tensors.

### Question being answered

Can the unmodified legacy 3D discriminative evaluation path load the selected raw DynUNet checkpoint and produce probability volumes for the four locked cases within a 16 GB GPU memory ceiling, while preserving enough artifacts to compare the later shared predictor against the legacy implementation?

This is a characterization run, not a release certification. A V100 result does not replace the final T4 Docker qualification required by Cut 11.

### Pinned code state

| Concern | Pinned value |
|---|---|
| Repository | `/dss/dsshome1/0D/di38tap/code/medseg-diffusion_ISLES24` mounted at `/mnt/code/medseg-diffusion_ISLES24` |
| Submission-time HEAD | `88d9b55dd2e3fa52df0c842e356b7e96a25b7b0b` |
| Submission-time branch | `main` |
| Local development HEAD | `88d9b55dd2e3fa52df0c842e356b7e96a25b7b0b` on `codex/inference-package-prd-cap` |
| Submitted batch-script SHA-256 | `771ba4d871d4b80c3b65e1d758545da5535017ef49377d9d3fceb8478a60fcb9` |
| Tracked LRZ worktree state at submission | No tracked or staged modifications observed |
| Untracked LRZ state | Existing logs, `packages.txt`, and `configs/loss/.multitask.yaml.swp`; none is an intended import or config input for E001 |

The job mounts a live checkout rather than an immutable archive. Its manifest therefore records execution-time `git rev-parse HEAD` and `git status --porcelain`. Accept E001 only when execution-time HEAD equals the pinned submission-time HEAD and there are no tracked changes. Unrelated untracked files may be documented and accepted only after confirming they did not shadow an imported module or selected config.

### Pinned model and training provenance

| Concern | Value |
|---|---|
| Run | `discriminative_dynunet_isles26_atlas30_3d_randompatch_280726/dynunet_128_3d_k3-3-3-3_f32-64-128-256_b3_p5n1_adamw2e4_wcos10_s100K_ldicefocal100log_dsup2_t1RAW_augSPAT3D_disc_e1_2026-07-28_20-13-22` |
| Saved config SHA-256 | `7152715bded1f11cc244a8a4270b6cc592b8a0e555f40d6d40af5b2924cc918a` |
| Saved overrides SHA-256 | `6cdd2234aae0716431c17287dd35f6d3f2ebaf1211470e1ed8896c5df4761f30` |
| Checkpoint | `models/best/best_model_step_040000_dice_3d_0.5724.pth` |
| Checkpoint SHA-256 | `120c93ee6a32f79829bf8a6b2d3ab7db59861f865ad1e8a37889f87ab0c82441` |
| Training code revision | `60e24ecfa8984673d20586b48a361cace7095bfd` strongly inferred from the LRZ reflog; not logged by the historical run |
| Baseline record | `tests/fixtures/inference/isles26_dynunet_p5n1_baseline.json` |

The repaired p5n1 saved config and its pre-repair backup are described in the baseline record. E001 must use the repaired config and raw, non-EMA checkpoint.

### Pinned input selection and inference policy

Focused split:

```text
analysis/cut0/gpu_preflight_5722939/focused_split.json
SHA-256: e977975bd82c9f123b89651f6a58a1b4b42c34b05d1c65521f145098047272e7
```

Cases:

- `sub-r049s033`: small/fast case;
- `sub-r004s006`: median foreground case;
- `sub-r041s108`: empty-label case;
- `sub-r035s015`: largest selected volume and spatial/resource stress case.

Legacy policy:

- result space: `model_preprocessed`;
- case batch size: `1`;
- validation loader workers: `0`;
- precision: FP32 legacy validation path;
- sliding-window ROI: `128 x 128 x 128` from the saved model config;
- sliding-window batch size: `4`;
- overlap: `0.5`;
- blend mode: `gaussian`;
- padding mode: `constant`;
- probability threshold for recorded metrics/mask: `0.5`;
- seed: `42`.

### Requested runtime

| Resource | Request |
|---|---|
| Container | `$SSD_STORE/MedSegDiff_nnUNet_010226.sqsh` |
| Partition / QOS | `lrz-v100x2` / `gpu` |
| GPU | One NVIDIA Tesla V100, 16 GB |
| CPU | 2 CPUs |
| Host memory | 8 GB |
| DataLoader workers | 0 |
| Wall clock | 10 minutes |

The V100 was selected because its 16 GB VRAM matches the Grand Challenge T4 ceiling and it had the earliest safe LRZ estimate among the inspected queues. The current validation forward path is FP32; this run does not claim FP16, BF16, or T4 numerical equivalence.

### Expected artifacts

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

The manifest is written before model construction and after every completed case so that OOM, timeout, or later-case failure can still provide bounded diagnostic evidence. It should contain:

- execution-time repository HEAD and worktree status;
- runtime and GPU identity;
- checkpoint, split, and resolved-config hashes;
- model and loader construction timings;
- dataset sizes, batch size, and worker counts;
- per-case probability and label file hashes;
- probability summary statistics;
- thresholded-mask raw-byte hash and foreground voxel count;
- metrics at threshold `0.5`;
- per-case inference time;
- peak allocated and reserved GPU memory.

### Reconciliation procedure

After Slurm leaves `PENDING`/`RUNNING`:

1. Read `sacct -j 5722967` and record state, exit code, elapsed time, allocation, and `MaxRSS`.
2. Read both Slurm logs. Warnings must be classified; stderr being non-empty is not automatically a failure.
3. Locate and parse `baseline_manifest.json`.
4. Verify execution-time HEAD and tracked worktree state against the pinned code state.
5. Verify checkpoint, focused split, and resolved-config hashes.
6. Verify exactly the four expected case IDs were yielded once each.
7. Verify probability tensors are finite, in `[0, 1]`, and have the expected model-space rank/channel contract.
8. Verify hashes, metrics, timings, and GPU-memory records exist for each completed case.
9. Update `tests/fixtures/inference/isles26_dynunet_p5n1_baseline.json` with the measured outputs and final accounting.
10. Mark E001 `ACCEPTED`, `PARTIAL_DIAGNOSTIC`, `INVALID_PROVENANCE`, or `FAILED`, with the reason and follow-up.

Do not copy the large `.npy` files into git. Preserve them on LRZ and version their hashes and summary values.

### Outcome handling

| Outcome | Required action |
|---|---|
| Complete and provenance-valid | Accept as the legacy Cut 0 probability baseline; use it for Cut 4 parity. |
| Complete but HEAD/config/checkpoint/split mismatch | Mark invalid provenance and rerun from a pinned checkout; do not compare numerically. |
| CUDA OOM | Preserve partial manifest; record the failing case and peak memory; rerun the legacy characterization with `sw_batch_size=1` as a separate non-parity resource diagnostic. Do not rewrite the historical baseline policy. |
| Ten-minute timeout | Preserve completed cases; identify the active stage/case; use the result to constrain the T4-safe policy. Rerun only the missing characterization case if scientifically sufficient. |
| Infrastructure/container failure | Diagnose from logs, correct the evidence harness, and rerun without changing model or inference policy. |
| Numerical/domain/shape failure | Treat as a legacy-path correctness finding; investigate before Cut 4 migration. |

### Work authorized while E001 is pending

- Cut 1: contracts, policy parsing, runtime capability profiles, and predictor output validation;
- Cut 2: strict model-bundle and single-device loading contracts, provided no numerical parity is claimed;
- Cut 3: deterministic preprocessing extraction and synthetic spatial tests;
- documentation and fixtures that do not encode E001's unmeasured outputs.

Do not while E001 is pending:

- remove or materially alter the legacy evaluation prediction path;
- declare Cut 0 or Cut 4 numerical parity complete;
- choose numerical tolerances based on unobserved V100/T4 behavior;
- treat V100 completion as final T4/container certification.

---

## Evidence item E002: Cut 1 focused CPU contract tests

### Status

| Field | Value |
|---|---|
| Evidence ID | `E002` |
| Slurm job | `5722997` |
| Final state | `COMPLETED` |
| Exit code | `0:0` |
| Elapsed | `00:01:19` |
| Peak RSS | `591404K` for the container step |
| Result | `ACCEPTED` |

### Question being answered

Do the additive Cut 1 inference contracts, strict policy parser, legacy-policy resolver, runtime profiles, and probability validator execute correctly inside the standardized project `.sqsh` environment?

### Pinned code and test snapshot

The job used a detached LRZ worktree rooted at commit:

```text
88d9b55dd2e3fa52df0c842e356b7e96a25b7b0b
```

Only these laptop-authored Cut 1 paths were overlaid into that isolated worktree:

```text
src/inference/
configs/inference/
configs/inference_runtime/
tests/test_inference_contracts.py
tests/test_inference_policy.py
tests/test_inference_runtime.py
```

The live LRZ training checkout was not modified. The isolated worktree is:

```text
$SSD_STORE/codex_test_worktrees/cut1_contracts_20260731a
```

### Requested runtime

- partition/QOS: `lrz-cpu` / `cpu`;
- CPU: `1`;
- memory: `4G`;
- wall clock: `00:10:00`;
- container: `$SSD_STORE/MedSegDiff_nnUNet_010226.sqsh`;
- CPU threads: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

Before submission, `sbatch --test-only` forecast a start at `2026-08-02T06:11:39+02:00`. The real short job instead backfilled and completed immediately. Scheduler forecasts are therefore advisory and must not be treated as a reason to avoid submitting small bounded test jobs.

### Command and artifacts

```bash
python3 -m unittest \
  tests.test_inference_contracts \
  tests.test_inference_policy \
  tests.test_inference_runtime -v
```

Logs:

```text
$SSD_STORE/codex_test_runs/cut1_contracts_20260731a/5722997.out
$SSD_STORE/codex_test_runs/cut1_contracts_20260731a/5722997.err
```

### Accepted result

All 19 tests passed. The accepted behavior includes:

- explicit output-space declarations;
- finite `[0, 1]` probability, rank, channel, and spatial-shape checks;
- explicit rejection of current 3D non-discriminative diffusion;
- conservative policy defaults and strict unknown-key rejection;
- explicit top-level inference policy replacing, not merging with, legacy validation inference;
- legacy policy translation when no top-level policy exists;
- parsing of all canonical Cut 1 policy files;
- repository and diagnostic-container support for both output spaces;
- production rejection of model-space output, batch size above one, workers, labels, threshold sweeps, intermediate artifacts, and uncertified BF16;
- prevention of config-based relaxation of production hard constraints.

---

## Evidence item E003: Cut 1 nearest-consumer compatibility regression

### Status

| Field | Value |
|---|---|
| Evidence ID | `E003` |
| Slurm job | `5722998` |
| Final state | `COMPLETED` |
| Exit code | `0:0` |
| Elapsed | `00:02:11` |
| Peak RSS | `1993304K` for the container step |
| Result | `ACCEPTED` |

### Scope and result

The isolated E002 snapshot was tested against the nearest existing consumers and capability guardrails:

```bash
python3 -m unittest \
  tests.test_inference_contracts \
  tests.test_inference_policy \
  tests.test_inference_runtime \
  tests.test_evaluation_io_model_volumes \
  tests.test_evaluation_model_config \
  tests.test_training_runtime_contracts \
  tests.test_discriminative_adapter \
  tests.test_discriminative_output_domains -v
```

All 49 tests passed. Existing evaluation volume contracts, saved/evaluation config composition, 3D diffusion guardrails, discriminative adapters, and probability-domain behavior remained green. The `.sqsh` emitted pre-existing TensorFlow/protobuf compatibility warnings; they did not fail or alter the tests and remain an environment-lock concern rather than a Cut 1 regression.

Logs:

```text
$SSD_STORE/codex_test_runs/cut1_contracts_20260731a/5722998.out
$SSD_STORE/codex_test_runs/cut1_contracts_20260731a/5722998.err
```

---

## Evidence item E004: Cut 1 final canonical-profile coverage

### Status

| Field | Value |
|---|---|
| Evidence ID | `E004` |
| Slurm job | `5723000` |
| Final state | `COMPLETED` |
| Exit code | `0:0` |
| Elapsed | `00:01:17` |
| Peak RSS | `591812K` for the container step |
| Result | `ACCEPTED` |
| Requested resources | 1 CPU, 4 GB RAM, 10 minutes on `lrz-cpu` |

### Question and expected result

After E003, review made `gc_container_test.require_cuda` explicit and added a test that asserts the repository, diagnostic-container, and production profiles' CUDA, worker, case-batch, and timeout declarations. Job E004 runs the final 20-test focused suite against that exact isolated snapshot.

E004 exited `0:0` and all 20 focused tests passed. This is the exact final Cut 1 snapshot, including the canonical-profile declaration assertion.

Expected logs:

```text
$SSD_STORE/codex_test_runs/cut1_contracts_20260731a/5723000.out
$SSD_STORE/codex_test_runs/cut1_contracts_20260731a/5723000.err
```

---

## Evidence item E005: Desktop Cut 1 and compatibility suite

### Status

| Field | Value |
|---|---|
| Evidence ID | `E005` |
| Execution host | Home desktop WSL2 |
| Final state | `COMPLETED` |
| Exit code | `0` |
| Result | `ACCEPTED` |
| Tests | 50 passed |
| Test execution time | 2.639 seconds, excluding SSH/WSL startup |

### Environment

The runner explicitly activated:

```bash
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate
```

Observed runtime:

```text
Python 3.10.12
torch 2.8.0+cu128
OmegaConf 2.3.0
```

The existing desktop repository was not used as the test source because it was at commit `66458c1` and contained unrelated tracked and untracked user changes. No desktop repository files were modified.

### Isolated source snapshot

The laptop's tracked tree at commit `88d9b55dd2e3fa52df0c842e356b7e96a25b7b0b` was archived and overlaid with only the current Cut 1 paths:

```text
src/inference/
configs/inference/
configs/inference_runtime/
tests/test_inference_contracts.py
tests/test_inference_policy.py
tests/test_inference_runtime.py
docs/INFERENCE_EXTRACTION_EVIDENCE_LEDGER.md
```

Transferred archive SHA-256:

```text
74d56c4ba1472ad08bc9f3cb5fc82687466e0f1209402e502b51887d51a61f35
```

The hash was verified on the desktop before extraction. The retained test snapshot is:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut1_contracts_20260731a
```

The transferred archive and temporary runner were removed after the test. The extracted snapshot remains available for inspection and subsequent additive test overlays.

### Test scope

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
  -v
```

All 50 tests passed. TensorFlow printed informational oneDNN/CPU-feature messages during import; they were not test failures.

### Supersession note — 2026-07-31

E005 remains valid evidence for its exact archived snapshot, but it does not
validate the subsequent Cut 1 contract hardening. The current working tree now
includes:

- sliding-window enabled as the new-policy default;
- mandatory non-null ROI equality with the saved model/data contract;
- dimension-agnostic inference presets using `data_mode.roi_key`;
- specialized bundle, input, prediction, spatial, and resource errors;
- probability/mask/result-to-geometry validation;
- renamed `native` runtime policy and revised policy tests.

These changes require a new isolated desktop snapshot and fresh focused plus
compatibility tests before Cut 1 can again be called verified. Per user
instruction, no desktop synchronization or execution was attempted while the
contract was still under review. Evidence item E006 closes this verification
gap for the subsequently approved hardened contract.

---

## Evidence item E006: Hardened Cut 1 policy and contract verification

### Status

| Field | Value |
|---|---|
| Evidence ID | `E006` |
| Execution host | Home desktop WSL2 |
| Final state | `COMPLETED` |
| Exit code | `0` |
| Result | `ACCEPTED` |
| Tests | 73 passed |
| Test execution time | 22.865 seconds, excluding environment startup |

### Pinned source state

The isolated source snapshot combined base commit:

```text
88d9b55dd2e3fa52df0c842e356b7e96a25b7b0b
```

with the current Cut 1 paths:

```text
src/inference/
configs/inference/
configs/inference_runtime/
configs/data_mode/full_volumes_3d.yaml
configs/data_mode/nnunet_slices_2d.yaml
configs/data_mode/online_slices_3d_to_2d.yaml
configs/data_mode/random_patches_3d.yaml
tests/test_inference_contracts.py
tests/test_inference_policy.py
tests/test_inference_runtime.py
docs/PRD_CAP_Inference_Package_Extraction_290726.md
docs/INFERENCE_EXTRACTION_EVIDENCE_LEDGER.md
```

Transferred archive SHA-256:

```text
391d7cb5bcabcd324c054ce07b82ab60d3a6fda1df647f776bbe815712fa7a7a
```

The hash was verified on the desktop before extraction. The retained source
snapshot is:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut1_hardened_20260731a
```

### Environment

The runner explicitly activated:

```bash
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate
```

Observed runtime:

```text
Python 3.10.12
torch 2.8.0+cu128
OmegaConf 2.3.0
Hydra 1.3.2
```

### Test scope

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

All 73 tests passed. The explicit exit-status marker contained `0`.

Retained desktop log:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut1_hardened_20260731a/cut1_hardened_tests.log
```

Log SHA-256:

```text
6a40811c82eb2add0f12b9ada54393b6a6c873c5279bc3eba0b0256446f0d87f
```

An initial pseudo-terminal invocation separately printed green summaries for
31 focused tests and 42 compatibility tests, but its local SSH wrapper reached
its timeout during terminal cleanup. E006 therefore uses the subsequent
non-interactive combined run and explicit exit marker as the authoritative
result.

### Supersession note

E006 remains valid for its exact snapshot, but E007 supersedes it for the final
Cut 1 contract. E007 removes the temporary `data_mode.roi_key` design and keeps
all established data-mode configs unchanged.

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

### Pinned source state

The isolated source snapshot combined base commit:

```text
88d9b55dd2e3fa52df0c842e356b7e96a25b7b0b
```

with the final Cut 1 paths:

```text
src/inference/
configs/inference/
configs/inference_runtime/
tests/test_inference_contracts.py
tests/test_inference_policy.py
tests/test_inference_runtime.py
docs/PRD_CAP_Inference_Package_Extraction_290726.md
docs/INFERENCE_EXTRACTION_EVIDENCE_LEDGER.md
```

No established `configs/data_mode/*.yaml` file differs from the base commit.
The inference policy resolver uses existing `model.spatial_dims` /
`data_mode.dim` to select `dataset.preprocessing_configs.roi.slice_2d` or
`volume_3d`. New inference YAML contains no ROI field.

Transferred archive SHA-256:

```text
fb39572fc61635d19874aaaa01e6a10b3db60c83b49c6947a74ba0db5b264023
```

The hash was verified before extraction. The retained source snapshot is:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut1_final_20260731a
```

### Environment

```text
Python 3.10.12
torch 2.8.0+cu128
OmegaConf 2.3.0
Hydra 1.3.2
```

The runner explicitly activated:

```bash
source /mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate
```

### Test scope

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

All 74 tests passed. The explicit exit marker contained `0`, and the log
contained no `FAIL` or `ERROR` result.

Retained desktop log:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut1_final_20260731a/cut1_final_tests.log
```

Log SHA-256:

```text
3903d1694f02ee6502df22a4f1b27fb59a312c2f9371f8092b6bf1e1ddec9d71
```

---

## Evidence item E008: Superseded Cut 2 strict/artifact prototype

### Status

Superseded by the agreed transition-only Cut 2 boundary. The retained desktop
results remain useful evidence that the selected DynUNet config and checkpoint
can be reconstructed on CPU/CUDA, but the tested bundle, hashing, strict-loader,
config-projection, and multi-model implementation is intentionally not part of
Cut 2 and must not be cited as acceptance evidence for the replacement code.

### Question being answered

Historical question only: could a proposed strict artifact-oriented loader be
made to work? After review, this question was found to anticipate later release
and ensemble cuts rather than characterize the ownership transfer required by
Cut 2.

### Pinned code state

| Concern | Value |
|---|---|
| Base commit | `a29dca5` (`feat(inference): define shared inference contracts`) |
| Base archive SHA-256 | `e666b04cee2a2e502c70ee24e1b91dbe009366db7e03875b83dc5e9700bb90d9` |
| Cut 2 overlay SHA-256 | `28c2a8dc7c8ac5ff39f32e5f2ce0fb6dfae64b0e93d412b5c2d88d21d6e7076e` |
| Desktop snapshot | `/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_dev_20260731b` |
| Python environment | `/mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate` |
| Python | `3.10.12` |
| PyTorch | `2.8.0+cu128` |
| OmegaConf | `2.3.0` |

The overlay contains only the active Cut 2 implementation, tests, CAP
correction, and its directly required Cut 1 contract relocation. It does not
contain the untracked Cut 0 utilities or fixtures.

### Tests

```bash
python -m unittest \
  tests.test_model_bundle \
  tests.test_model_loader \
  tests.test_inference_contracts \
  tests.test_evaluation_model_loader \
  tests.test_checkpoint_state \
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

All 116 tests passed in 8.193 seconds.

Retained desktop log:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_dev_20260731b/cut2_regression_tests.log
```

Log SHA-256:

```text
4c81d958b5e836b0c0c39843e86cc308e9ed50ead0234b3e0f64a1b2c1aeb0ed
```

### Covered acceptance evidence

- bundle paths are confined beneath the model root and config/weight hashes
  are checked before a descriptor is returned;
- full saved configs are resolved and retained as read-only provenance;
- construction receives a mutable copy with validation/evaluation/inference
  policy removed;
- model input-channel synchronization still derives from the existing data
  contract without mutating saved provenance;
- unwrapped, leading `module.`, and embedded `model.module.` checkpoints load
  strictly;
- missing keys fail strict deployment loading and remain reportable through
  the explicit permissive compatibility mode;
- stored BF16 tensors do not force BF16 execution dtype;
- prepared models are placed on one explicit device, set to evaluation mode,
  and have gradients disabled;
- the existing evaluation checkpoint-discovery and loader interface remains
  green through its compatibility facade;
- training checkpoint save/load behavior and 2D/diffusion runtime guardrails
  remain green.

### Real selected-model desktop smoke

The pinned p5n1 config and 22 MB raw checkpoint were copied from LRZ storage to
an isolated desktop artifact directory. No LRZ compute job was submitted for
this check.

| Concern | Value |
|---|---|
| Artifact directory | `/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_model_artifacts_20260731a` |
| Test snapshot | `/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_real_model_20260731a` |
| Config SHA-256 | `7152715bded1f11cc244a8a4270b6cc592b8a0e555f40d6d40af5b2924cc918a` |
| Checkpoint SHA-256 | `120c93ee6a32f79829bf8a6b2d3ab7db59861f865ad1e8a37889f87ab0c82441` |
| Final overlay SHA-256 | `95fd3afd35ff72aa7cb5ae6c5da9ea3e3aa6373f2653a5fd6e71e8995e549e84` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB |
| Runtime | Python `3.10.12`, PyTorch `2.8.0+cu128` |
| Strict key transform | `unwrapped` |
| Missing/unexpected keys | `0 / 0` |
| State tensors | All 52 exactly equal to the legacy construction path |
| Parameters | 5,641,315, stored as FP32 |
| Prepared state | `eval()`, gradients disabled, all parameters on `cuda:0` |
| Strict CPU preparation | 39.656 seconds including cold imports/construction |
| Strict CUDA preparation | 0.395 seconds after imports were warm |
| Peak CUDA allocated/reserved | 22,567,936 / 27,262,976 bytes |

Retained result:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_real_model_20260731a/strict_model_smoke.json
SHA-256: 5da4bbc1f7cd89c454025db62704870cd160da18ee264983db2357b508e80194
```

Retained log:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_real_model_20260731a/strict_model_smoke.log
SHA-256: 3a7b3292115ad39348e76f37b94dbf55e83ab7cee8c4d157944daa5b8725b034
```

This establishes real artifact reconstruction and CUDA placement on the
no-queue desktop. It is not a T4 performance certification; that remains Cut
11 and must use the final container/runtime candidate.

### Desktop testing procedure for subsequent cuts

1. Never update or overlay the dirty desktop repository for Codex cut testing.
2. Build a fresh laptop snapshot from the intended base commit plus only the active cut's files.
3. Record and verify the archive SHA-256 before extraction.
4. Extract to a new cut-specific directory under:

   ```text
   /mnt/c/Users/minanessiem/Development/codex_test_worktrees/
   ```

5. Run a script that changes into that snapshot and explicitly sources `MedSegDiff_env`.
6. Run tests narrow-to-broad; record exact commands and results here.
7. Remove transfer archives and runners, retaining only useful test snapshots.

---

## Evidence item E009: Cut 2 transition-only model loading parity

### Status

Complete for the revised Cut 2 boundary: existing single-model evaluation
construction, checkpoint-to-model loading, and channel-contract helpers were
physically transferred into `src/models/` without introducing release,
artifact, ensemble, hash, compatibility-signature, or preparation policy.

### Question being answered

Can model-owned loading behavior move out of the evaluation/training utility
locations while preserving the current evaluation API, checkpoint fallback and
diagnostics, training checkpoint behavior, requested device mapping, model
parameters, evaluation state, and gradient state?

### Pinned code state

| Concern | Value |
|---|---|
| Base commit | `a29dca5` (`feat(inference): define shared inference contracts`) |
| Base archive SHA-256 | `e666b04cee2a2e502c70ee24e1b91dbe009366db7e03875b83dc5e9700bb90d9` |
| Revised Cut 2 overlay SHA-256 | `3c908fc40088260d23f15e96a93655f2c10f47d92b220d6e637732762841b12d` |
| Desktop snapshot | `/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_transition_20260731c` |
| Python environment | `/mnt/c/Users/minanessiem/Development/MedSegDiff_env/bin/activate` |
| Python | `3.10.12` |
| PyTorch | `2.8.0+cu128` |

The snapshot is the committed Cut 1 base plus only the revised Cut 2 files. It
does not modify or depend on the desktop repository checkout.

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

All 106 tests passed in 7.924 seconds.

Retained log:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_transition_20260731c/cut2_transition_tests.log
SHA-256: 2a94a3dcee1dd6e314880d3553d00c0b533b58915065ef411e3ebbd350dacc18
```

### Real selected-model parity

The retained p5n1 resolved config and selected checkpoint were reconstructed
once through the exact pre-transfer evaluation sequence and once through
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

Retained log:

```text
/mnt/c/Users/minanessiem/Development/codex_test_worktrees/cut2_transition_20260731c/cut2_real_model_parity.log
SHA-256: 72c740d94bc386015dca74aa1fdb7f11eb866f80699238272fa2b68c1d60d7c5
```

This evidence supersedes E008 for Cut 2 acceptance. E008 remains a historical
prototype record only; its strict, bundle, hash, multi-model, config-projection,
CPU-first, and frozen-parameter behaviors were removed from the active cut.

---

## Ledger update template

For subsequent asynchronous evidence, copy the following headings:

```text
## Evidence item E###: <name>
### Status
### Question being answered
### Pinned code state
### Pinned model/config/input state
### Requested runtime
### Expected artifacts
### Acceptance and invalidation rules
### Reconciliation procedure
### Outcome handling
### Work authorized while pending
```
