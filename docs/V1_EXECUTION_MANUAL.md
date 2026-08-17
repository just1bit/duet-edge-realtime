# Duet-EDGE Realtime V1 Execution Manual

This manual provides the environment setup and execution workflow for a complete V1 GPU acceptance run or a local fake/core validation run. Numbered scripts perform repeatable work and archive machine-generated evidence. One optional Markdown file is available for operator notes.

## 1. Workspace and Configuration

The project root is the directory that contains both repositories and the acceptance assets:

```text
PROJECT_ROOT/
├── duet-edge/
├── duet-edge-realtime/
└── data+checkpoint/
```

Both repositories use the latest code from `main` for this acceptance run. Run the commands below from `PROJECT_ROOT/duet-edge-realtime`.

Review the operational settings once:

```
PROJECT_ROOT/duet-edge-realtime/scripts/v1_execution/acceptance.conf
```

The acceptance configuration contains the acceptance profile, relative paths, expected hashes, the Python command, the Viewer HTTP port, and the GPU sampling interval. Model, stream, queue, server, and SLO values remain in `configs/v1.cuda.json` and `configs/v1.fake.json`.

### Profiles

`ACCEPTANCE_PROFILE=gpu` is the default and produces formal V1 GPU acceptance evidence. `ACCEPTANCE_PROFILE=local` produces fake/core/Viewer evidence with the local optional dependency set.

For a local run, change `ACCEPTANCE_PROFILE` to `local` before running.

| Stage | GPU profile | Local profile |
|---|---|---|
| 01 | Execute | Execute |
| 02 | Execute runtime verification and CUDA smoke | Execute runtime verification; CUDA smoke is Skipped |
| 03–06 | Execute | Execute |
| 07–08 | Execute | Skipped |
| 09 | Execute | Execute fake Viewer review |
| 10–13 | Execute as applicable | Skipped |
| 14 | Execute | Execute automatic local evidence/report; final CUDA check is Skipped |

Local preflight records GPU checks as not applicable. Browser capability may be `unknown` when the environment reports no launcher; this remains advisory. A successful local run establishes evidence for core streaming, protocol, fake inference, and Viewer behavior. The GPU profile adds CUDA inference, GPU performance, real-model motion quality, and ten-minute session evidence.

### Runtime Installation

Create and activate the environment:

```bash
cd PROJECT_ROOT/duet-edge-realtime
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

For Local acceptance, install the local environment:

```bash
python -m pip install -e '.[local]'
```

For GPU acceptance, ensure `g++` is available, then install CUDA 12.8 torch, PyTorch3D in order:

```bash
python -m pip install 'torch==2.7.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[gpu]'
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
```

Reactivate `.venv` in each new environment. Follow the workflow issue instructions to back to this section for fixing.

## 2. Evidence Model

Stage 01 creates an active run such as:

```text
outputs/acceptance-20260814-143000/
├── run-metadata.json
├── acceptance-notes.md
├── acceptance-report.md
├── logs/
├── stage-results/
├── evidence/
├── candidate-configs/
├── p1-fake/
├── real-smoke/
└── final-10min/
```

After Stage 01 initialization, every executed or skipped action creates a numbered log and JSON result. Repeated actions create new attempts and preserve earlier evidence. Automatic metrics, hashes, test results, and artifact inventories stay in generated evidence. `acceptance-notes.md` serves as an independent operator record, while automatic validation uses generated evidence.

## 3. Stage Overview

| Stage | Purpose | Applicability | Human contribution |
|---|---|---|---|
| 01 | Create or resume an acceptance run | Start every run here | — |
| 02 | Verify the runtime and run smoke testing | Verify every run | — |
| 03 | Capture machine and asset preflight | Every run | — |
| 04 | Prepare the AIST++ input | Every new run | — |
| 05 | Exercise the streaming core | New code/runtime evidence | — |
| 06 | Exercise the fake end-to-end path | Every run | — |
| 07 | Exercise real CUDA streaming | GPU profile | — |
| 08 | Export the real fixture | GPU profile, after Stage 07 | — |
| 09 | Review the Viewer | Every run | Visual observations |
| 10 | Measure the 50-step baseline | GPU performance run | Baseline decision |
| 11 | Explore candidates | GPU profile, when candidate evidence adds value | Candidate selection |
| 12 | Apply and validate manual adjustments | GPU profile, after performance selection | Changes and rationale |
| 13 | Run the ten-minute session | GPU profile, after configuration validation | Qualitative monitoring |
| 14 | Validate and index all evidence | At completion | Final decision |

## 4. Stage 01 — Initialize or Resume

Create a new run:

```bash
bash scripts/v1_execution/01_initialize.sh
```

Resume an existing run:

```bash
bash scripts/v1_execution/01_select_run.sh outputs/acceptance-<timestamp>
```

Completion signal: the terminal prints the active run and the path to `acceptance-notes.md`.

## 5. Stage 02 — Runtime Smoke

Verify the active environment:

```bash
bash scripts/v1_execution/02_verify_runtime.sh
```

Run the deterministic real-model CUDA smoke test. The local profile records this action as Skipped:

```bash
bash scripts/v1_execution/02_cuda_smoke.sh
```

The GPU verification log captures Python 3.10, PyTorch 2.7, CUDA 12.8, RTX 5090 capability, PyTorch3D, project imports, checkpoint integrity, and model loading. Local verification checks Python 3.10 or newer, fake/core dependencies, project imports, and the fake fixture. A ready environment can serve later runs while each run receives fresh verification evidence.

## 6. Stage 03 — Machine and Asset Preflight

```bash
bash scripts/v1_execution/03_preflight.sh
```

The generated preflight JSON records the profile, paths, asset hashes, GPU applicability/details, disk capacity, Viewer ports, and browser capability. Follow any printed action, then repeat the script to create a new attempt.

## 7. Stage 04 — Prepare Input

```bash
bash scripts/v1_execution/04_prepare_input.sh
```

The script converts the configured raw AIST++ motion into `${RUN_ROOT}/input_motion.pkl` and validates frame count, shape, scale, and finite values.

## 8. Stage 05 — Core Tests

Run the standard suite:

```bash
bash scripts/v1_execution/05_unit_tests.sh
```

Run the local WebSocket integration explicitly:

```bash
bash scripts/v1_execution/05_network_tests.sh
```

Each command has a separate log and result. Selecting the original run preserves and reuses its existing stage evidence in the generated report.

## 9. Stage 06 — Fake End-to-End Run

Run the virtual-clock fake backend and validate its output:

```bash
bash scripts/v1_execution/06_run_fake.sh
bash scripts/v1_execution/06_check_fake.sh
```

The validator checks protocol v2, lifecycle order, frame continuity, joint shape, commit intervals, queue bounds, and summary alignment.

## 10. Stage 07 — Real CUDA Smoke Run

Run real streaming and validate its output:

```bash
bash scripts/v1_execution/07_run_real.sh
bash scripts/v1_execution/07_check_real.sh
```

The validator confirms the CUDA backend, checkpoint evidence, inference samples, frame continuity, and normal lifecycle completion.

## 11. Stage 08 — Export the Real Fixture

```bash
bash scripts/v1_execution/08_export_fixture.sh
```

The output `${RUN_ROOT}/real_fixture.npz` contains normalized lead motion, generated motion, lead and generated joints, and model metadata. Later stages reuse this compact fixture.

## 12. Stage 09 — Viewer Review

Start the realtime stream in terminal A:

```bash
bash scripts/v1_execution/09_viewer_stream.sh
```

Start the web Viewer in terminal B:

```bash
bash scripts/v1_execution/09_viewer_web.sh
```

Open the printed HTTP address and connect to the WebSocket endpoint from `configs/v1.fake.json`. Review:

1. Two simultaneously visible, labelled skeletons: blue lead and cyan companion;
2. Both skeletons are upright in the default Z-up view rather than lying on the ground;
3. Limbs change relative to each root continuously; root translation alone or a frozen pose is a failure;
4. State transitions through buffering, playing, tail commit, and completion;
5. Reconnection and state recovery;
6. Fake NDJSON replay;
7. Real NDJSON replay for orientation, drift, flips, ground axis, naturalness, and window boundaries.

Record an explicit pass/fail for every item plus screenshot/timestamp references under `Viewer Review` in `acceptance-notes.md`. Stage-script evidence and this visual judgment are separate acceptance inputs. The GPU profile includes real NDJSON review. The service and web-server logs are archived automatically; stopping the web server with Ctrl-C is an accepted completion.

## 13. Stage 10 — 50-Step Baseline

Run and summarize the canonical baseline:

```bash
bash scripts/v1_execution/10_run_baseline.sh
bash scripts/v1_execution/10_summarize_baseline.sh
```

Review `${RUN_ROOT}/evidence/benchmarks/benchmark.json`. The automatic summary calculates:

```text
p99_ms + safety_margin_ms < hop_period_ms
recommended_playout_delay_s >= (p99_ms + safety_margin_ms) / 1000
```

When useful, record the baseline or candidate decision and evidence paths under `Performance` in `acceptance-notes.md`.

## 14. Stage 11 — Candidate Tuning

Use this stage when candidate evidence supports the performance decision. Replace `<steps>` with the selected sampling steps:

```bash
bash scripts/v1_execution/11_prepare_candidate.sh <steps>
bash scripts/v1_execution/11_run_candidate.sh <steps>
bash scripts/v1_execution/11_compare_quality.sh <steps>
bash scripts/v1_execution/11_summarize_candidates.sh
```

Repeat the first three actions for each candidate. Candidate JSON files stay under `${RUN_ROOT}/candidate-configs/`. Performance summaries and quality comparisons stay under `${RUN_ROOT}/evidence/benchmarks/`.

Review deadline status, latency, memory, LMA, PFC, boundary motion, and Viewer quality. The selected candidate and rationale may be recorded under `Performance` in the notes.

## 15. Stage 12 — Manual Adjustment and Validation

Display calculated recommendations for the selected steps:

```bash
bash scripts/v1_execution/12_show_recommendation.sh <steps>
```

Apply the reviewed values to the relevant project files or `configs/v1.cuda.json`. Record useful tuning details under `Manual Changes` in `acceptance-notes.md`, including:

- Files adjusted;
- Previous and updated values;
- Decision rationale and supporting evidence;
- Stages selected for another execution;
- The resulting conclusion.

Validate the canonical CUDA configuration:

```bash
bash scripts/v1_execution/12_validate_config.sh
```

Run the selected stages again. Every attempt receives its own automatic evidence.

## 16. Stage 13 — Ten-Minute Final Run

Start resource monitoring in terminal A:

```bash
bash scripts/v1_execution/13_monitor_gpu.sh
```

Start the final service in terminal B:

```bash
bash scripts/v1_execution/13_run_final.sh
```

The final service repeats the five-second fixture 120 times and produces ten minutes of motion. Stop resource monitoring after the service completes; Ctrl-C is an accepted monitor completion. Record useful qualitative conclusions under `Final Review` in `acceptance-notes.md`.

## 17. Stage 14 — Final Validation and Report

Run the final stream check, evidence check, and report builder:

```bash
bash scripts/v1_execution/14_check_final.sh
bash scripts/v1_execution/14_check_evidence.sh
bash scripts/v1_execution/14_build_report.sh
```

The final validator checks duration, frame count, inference performance, playout timing, queue bounds, jitter, lifecycle, and EOS. The evidence checker confirms automatic artifacts for the selected profile and records the presence of `acceptance-notes.md`; its content remains an independent operator review. The report builder creates `${RUN_ROOT}/acceptance-report.md` with links to stage results, logs, and indexed evidence.

Add any final judgment or follow-up to `acceptance-notes.md`, then run the report builder. One invocation refreshes the report after its own archived result is written.

## 18. Completion Evidence

Retain the complete `RUN_ROOT`, the reviewed `configs/v1.cuda.json`, and the selected repository copies. The run directory contains reproducible automatic evidence alongside optional acceptance notes.
