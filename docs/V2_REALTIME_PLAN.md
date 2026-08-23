# Duet-EDGE Near-Realtime System V2: Final Implementation Plan

## 1. Goal

V2 delivers one runnable near-realtime system:

```text
continuous AIST lead-motion replay at 30 FPS
  -> resident Duet-EDGE CUDA inference
  -> 150/75 rolling windows
  -> online motion continuity
  -> timestamped playout
  -> live lead and companion visualization
  -> replayable run evidence
```

The operator follows the staged operation manual, then opens the Viewer address
printed by the stream-start Stage in a browser (default:
`http://127.0.0.1:8080`). The Viewer connects to the current run and displays the
CUDA, Recorded, or Fake backend, run state, and live lead and companion skeletons.
The final CUDA preset uses the epoch-1800 checkpoint, lead-only guidance, 50
sampling steps, and one resident model on one GPU.

### Deliverables

1. A staged CUDA run and recorded rehearsal workflow with explicit operator checks.
2. A continuous real-motion input timeline lasting at least ten minutes.
3. Distinct and spatially coherent lead and companion skeleton streams at 30 FPS.
4. A Viewer supporting live display, reconnect, local replay, and diagnostic views.
5. Structured timing, continuity, motion-quality, client, and resource evidence.
6. Automated functional, motion, browser, and ten-minute GPU acceptance.

## 2. V2 Runtime Design

### 2.1 Input timeline

The run-local `config.json` records the input path, hash, frame range, and transition
parameters. The input adapter preprocesses it with the checkpoint normalizer and
emits one continuous `MotionFrame` sequence. Stage 07 validates and locks the
selected input; later run, verification, and export stages reuse that recorded path
and hash unless an explicit input override is supplied.

At a clip transition:

1. align the next clip's ground-plane translation and heading;
2. blend root position with a raised-cosine curve;
3. slerp joint rotations through the transition;
4. regenerate and validate the canonical 151D representation;
5. attach clip and transition metadata to emitted frames.

`data+checkpoint/` provides three purpose-specific input sets:

- `baseline_input/` for the baseline and default formal test;
- `smoke_input/` for runtime and CUDA smoke checks;
- `stitched_long_input/` for endurance, continuity, and release-acceptance runs.

### 2.2 Authoritative lead path

Lead joints come directly from FK of the canonical input timeline. Adjacent window
overlaps are validated, and every source frame is committed once. Source-transition
metrics and model-window metrics are reported separately.

An acceptance test compares emitted lead joints with direct source FK frame by
frame.

### 2.3 Companion continuity

V2 separates continuity into model, parameter, and timeline layers. The release
default is:

```text
causal-overlap DDIM latent handoff
  -> local relative-root correction
  -> raised-cosine position blend + shortest-path quaternion slerp
  -> FK
  -> 75-frame exactly-once commit
```

#### 2.3.1 Causal-overlap DDIM

`CausalOverlapDDIMSampler` uses the diffusion model prediction, schedule, and DDIM
parameters. While generating `W_i`, it saves the last half after every non-final
DDIM update:

```text
handoff_i[k] = x_i[k, 75:150, :]
```

While generating `W_(i+1)`, it restores those values into the current prefix at the
corresponding step:

```text
x_(i+1)[k, 0:75, :] = handoff_i[k]
```

Handoff occurs after the DDIM update and before the next model prediction, using
the long-mode timestep pairs, eta, guidance ramp, and music/lead guidance weights.
The first window establishes state and each successor consumes and updates it. A
50-step configuration retains about 2.3 MB. The final clean window proceeds to
parameter-space blending.

#### 2.3.2 Local relative-root correction and blending

For each unnormalized generated window:

```text
relative_root[t] = companion_root[t] - lead_root[t]
```

The continuity processor blends overlapping relative-root trajectories and joint
rotations, then reconstructs the companion in the continuous lead coordinate frame:

```text
companion_world_root[t] = lead_world_root[t] + blended_relative_root[t]
```

The seam correction applies only to the current window's first 75 frames:

```text
delta = pending_relative_root[0] - current_relative_root[0]
corrected_current[t] = current_relative_root[t]
                       + (1 - raised_cosine[t]) * delta,  0 <= t < 75
```

Frame `t=0` aligns the overlap start and `t=74` returns to the generated
relative-root. `lead_world_root` comes from the authoritative direct-FK path at the
same global source frame.

The processor uses:

- raised-cosine blending over the 75-frame overlap;
- shortest-path quaternion slerp for 24 joints;
- relative-root seam correction decayed through the overlap;
- robust filtering of relative-root outliers;
- deterministic per-window seeds and per-session state reset;
- bounded diagnostic samples for failed quality checks.

Validation uses direct lead FK, offline stitching, and original-repository
`long_ddim_sample` clips.

#### 2.3.3 Authoritative timeline commit

The service commits 75-frame batches: the first window emits its first half, each
successor emits the resolved overlap, and EOF flushes the valid tail.

### 2.4 Sampling state

The CUDA backend session owns handoff state and the serialized inference worker
updates it. State is cleared after warmup, at session boundaries, and when the input
sequence restarts. Each consumption validates window ID, shape, sampling steps,
timestep schedule, dtype, and device. Metrics record handoff use, reset, and size.

### 2.5 Scheduling and playout

The runtime uses one model instance, one serialized inference worker, bounded
inference/output queues, and a monotonic playout clock.

Initial RTX 5090 release targets are:

```text
sampling steps: 50
inference p99: <= 650 ms
inference max: <= 700 ms
hop period: 2,500 ms
playout delay: 0.75 s, finalized by machine calibration
fixed latency: approximately 5.717 s plus Viewer render buffering
```

The summary records model load, window inference, queue residence, batch-ready
time, send lateness, playout jitter, frame latency, underflow, and resource trends.

### 2.6 Stream protocol

V2 uses `duet-edge-stream/v3` with schema `3.0.0`.

`hello` contains:

- run, session, and stream identity;
- backend, checkpoint hash, guidance mode, and sampling steps;
- causal-overlap and continuity-correction information;
- source timeline identity;
- FPS, skeleton hierarchy, coordinate system, and latency budget.

Each `frame` contains:

- frame and source identity;
- source, target playout, emission, and latency timestamps;
- window and boundary metadata;
- `lead_joints[24][3]`;
- `companion_joints[24][3]`.

Control messages are `hello`, `state`, `metrics`, `frame`, `eos`, and `error`.

### 2.7 Viewer

The Viewer receives frames into a one- or two-frame render buffer and draws through
`requestAnimationFrame` with timestamp-based interpolation.

The default shared-world view uses one camera transform for both dancers and shows
the ground plane, duet spacing, and root trails. A side-by-side root-relative view
supports pose diagnostics.

The page provides:

- automatic connection and exponential-backoff reconnect;
- Live, Replaying, Completed, Failed, and Disconnected states;
- pause, resume, restart, seek, playback speed, and NDJSON selection;
- shared-world, side-by-side, and camera controls;
- backend, model mode, checkpoint, steps, input clip, latency, inference, jitter,
  frame age, drop, and reconnect displays;
- a visible fake/recorded/CUDA badge;
- in-page replay and connection guidance.

The resident runtime also provides the existing `web/` assets and health endpoints.
Stage 04 loads and warms the model, Stage 05 readies the stream component, and
Stage 06 starts the Viewer in `waiting_for_input`. Stage 08 starts the formal session.

### 2.8 Motion and client metrics

Bounded online metrics include:

- root position, velocity, and acceleration discontinuity;
- joint angular and positional discontinuity;
- source-transition and model-boundary distributions;
- body-centered lead/companion pose distance;
- horizontal and vertical relative-root envelope and trend;
- foot velocity during contact and ground penetration;
- continuity-correction magnitude;
- per-window normalized-overlap disagreement and latent-handoff use, reset, and
  state size;
- connected clients, connection duration, drops, reconnects, render FPS, frame
  age, and visible stalls;
- GPU utilization/memory, CPU, and RSS trends.

## 3. Run, Configuration, and Asset Conventions

V2 run directories use `outputs/run-xxxxxx-xxx/`. Configuration instances and run
evidence are stored with their corresponding run:

```text
PROJECT_ROOT/
├── duet-edge/
├── duet-edge-realtime/
│   ├── configs/
│   │   └── example.json            # configuration template
│   └── outputs/
│       └── run-xxxxxx-xxx/
│           ├── config.json         # finalized by Stage 03
│           ├── config.sha256
│           ├── calibration.json
│           ├── input-manifest.json # locked by Stage 07
│           ├── run-metadata.json
│           ├── logs/
│           ├── evidence/
│           ├── stream.ndjson
│           ├── summary.json
│           ├── report.md
│           └── fixtures/           # fixtures exported by Stage 10
└── data+checkpoint/
    ├── train-1800.pt
    ├── baseline_input/
    ├── smoke_input/
    └── stitched_long_input/
```

`configs/example.json` is the run initialization template. Stage 01 copies it to
`${RUN_ROOT}/config.json`, then resolves absolute paths, asset hashes, and the run ID
into that instance. Stage 03 runs the fixed-step real-clock baseline, automatically
finalizes the latency parameters, and locks the result with `config.sha256`. Stage 07
records the formal input separately in `input-manifest.json`, so the configuration
already loaded by the resident runtime does not change.

## 4. V2 Operations Manual

The maintained [V2 operation manual](V2_EXECUTION_MANUAL.md) defines environment
setup, the single primary command for each Stage, expected artifacts, and continuation
criteria. Stage numbers are the execution order:

1. Init / Resume
2. Runtime Check & Smoke
3. Baseline & Auto-config
4. Model Service Ready
5. Realtime Stream Ready
6. Viewer Web Ready
7. Prepare & Check Input
8. Input & Run
9. Verify & Report
10. Export Fixture (optional)

Stages 04–06 ready the resident runtime without consuming input. Stage 07 locks an
input manifest, and Stage 08 verifies its hashes before starting the formal session.

## 5. Implementation Phases

### Phase 1 — Motion analysis and input timeline

1. Extend the run checker with seam, drift, distinctness, foot, and
   boundary analysis.
2. Verify the analyzer with regression samples for periodic joins, identical dancer
   motion, and root drift.
3. Validate the manifests and hashes in `baseline_input/`, `smoke_input/`, and
   `stitched_long_input/`.
4. Implement run-local input loading, multi-clip replay, and transition blending in
   `input_adapters.py`.
5. Generate direct-FK and offline reference excerpts.

Exit results:

- the long input timeline has annotated, quality-checked transitions;
- emitted lead matches direct source FK within the frozen tolerance;
- the analyzer detects all three frozen motion-quality regression samples.

### Phase 2 — Companion continuity

1. Freeze original-repository long-mode and V1 GPU evidence as references.
2. Implement the causal-overlap sampler, handoff state, and session reset.
3. Separate direct lead commits from companion stitching.
4. Implement local-decay relative-root overlap correction.
5. Retain raised-cosine/slerp/FK and record correction magnitude.
6. Add two-window, boundary, spatial-envelope, periodic-input, reset, and deadline
   tests.

Exit results:

- periodic input produces bounded lead/companion separation;
- boundary metrics stay within the frozen offline reference envelope;
- overlap disagreement and continuity correction stay within frozen thresholds;
- CUDA and recorded companion poses pass the distinctness gate;
- foot and ground metrics stay within the accepted reference tolerance;
- per-window inference fits the hop budget and the endurance-run handoff sequence
  remains complete.

### Phase 3 — Viewer and staged operation

1. Integrate static asset and health serving into the service.
2. Add render buffering, interpolation, world-space display, and replay controls.
3. Add backend/model/source status and client telemetry.
4. Implement the Stage-based workflow under `scripts/v2_execution/`.
5. Exercise live, completion, reconnect, and replay flows in browser tests.

Exit results:

- the operation manual exposes each startup step, its purpose, and its result;
- the page automatically connects and identifies the current backend;
- completion and network failure have distinct states;
- browser rendering has no unplanned stall over 100 ms in the acceptance excerpt;
- automated screenshots and a short run capture are generated.

### Phase 4 — Final GPU acceptance

Run two release jobs:

1. A natural-motion CUDA quality run with the live Viewer connected.
2. A ten-minute CUDA endurance run with at least 18,000 frames and full resource,
   motion, server, and browser telemetry.

Package the effective config, source hashes, summary, NDJSON, gate results, resource
series, tests, screenshots, video, and operator review into one evidence archive.

## 6. Final Acceptance Matrix

### 6.1 Functional

| Gate | Target |
|---|---|
| Backend | CUDA, expected checkpoint SHA256, lead-only, 50 steps |
| Continuity | Causal-overlap DDIM + local relative-root correction |
| Input | Hashed multi-clip timeline with source identity |
| Duration | At least 18,000 frames / 10 minutes |
| Sequence | Input = committed = recorded; contiguous IDs |
| Lifecycle | starting -> buffering -> playing -> draining -> finished |
| Numerical output | Finite 24x3 lead and companion joints |
| Runtime errors | 0 overloads, sequence errors, unhandled errors, and recorder loss |
| Handoff state | Established by the first window, then consumed/produced with counts matching the window sequence |
| Handoff residency | Handoff remains on the CUDA device inside the DDIM loop; profiler shows no per-step D2H/H2D copy |
| Session isolation | With identical input and seed, new-session first-window output is unaffected by warmup and a prior session, within frozen numerical tolerance |
| Sequence restart | Source restart clears handoff; the new sequence starts from first-window state with contiguous IDs |

### 6.2 Performance and delivery

| Gate | Target |
|---|---:|
| Output FPS | 29.7–30.3 |
| Server playout jitter p95 | <= 10 ms |
| Frame-send interval max | < 50 ms while playing |
| Underflows | 0 |
| Inference p99 | <= 650 ms on the validated RTX 5090 |
| Inference max | <= 700 ms on the validated RTX 5090 |
| Compute budget | p99 + safety margin < 2,500 ms |
| Lookahead | The first complete 150-frame window triggers the first output batch without waiting for its successor |
| Handoff overhead | Copy time, state bytes, and batch-ready time are recorded, and total inference remains within the compute budget |
| GPU resource sampling | <= 100 ms during GPU benchmarks and endurance runs, covering inference bursts |
| Fixed latency | Observed p95 within 20 ms of the declared value |
| Viewer drops | 0 on the acceptance client |
| Browser stalls | 0 intervals over 100 ms while playing |
| Reconnect | Automatic recovery after an injected interruption |

### 6.3 Motion quality

Threshold values are frozen from direct-FK and offline reference excerpts before
continuity tuning.

| Gate | Target |
|---|---|
| Lead fidelity | Emitted lead matches direct source FK within tolerance |
| Source transitions | Position, angle, velocity, and acceleration within the source envelope |
| Model boundaries | Root, joint, angular, and acceleration metrics within the offline envelope |
| Overlap consistency | Normalized-overlap disagreement and parameter-layer correction within frozen thresholds |
| Clean-step boundary | Pre-blend clean-overlap error and post-blend boundary error are both recorded and remain within frozen envelopes |
| Drift stress | Bounded relative-root envelope under periodic input |
| Spatial coherence | Relative-root p99 and trend within the reference envelope |
| Transition response | After an annotated source transition, companion pose/velocity response latency and repeated-frame duration remain within reference envelopes |
| Propagation recovery | After a one-window injected anomaly, overlap correction and boundary metrics return to reference envelopes within the frozen window count |
| Long-mode reference | Boundary, LMA, and PFC remain within the original-repository long-mode reference envelopes |
| Distinctness | Body-centered lead/companion distance above the frozen lower bound |
| Feet and ground | Foot skating and penetration within the accepted reference tolerance |
| Visual review | No periodic pop, freeze, teleport, or unbounded travel |

### 6.4 Release evidence

The release archive contains:

- clean source revisions and environment details;
- checkpoint and input hashes;
- effective config and source timeline definition;
- complete NDJSON, summary, and machine-readable gate result;
- GPU/CPU/RSS/server/client time series;
- automated test results;
- live and replay screenshots plus a short demo video;
- completed operator review.

## 7. Definition of Done

V2 is complete when the staged CUDA workflow presents a smooth and spatially
coherent duet, supports reliable live and replay operation, completes the ten-minute
run, and passes motion-quality gates that detect the repeated-input seam,
lead/companion equality, and accumulated root drift. Causal latent handoff and
relative-root stitching jointly satisfy overlap, motion-quality, and latency
thresholds.
