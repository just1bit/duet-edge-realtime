# Duet-EDGE Realtime

Duet-EDGE Realtime turns a continuous lead-motion timeline into a timestamped
lead-and-companion skeleton stream. The runtime keeps one inference worker alive,
processes overlapping motion windows causally, preserves companion continuity,
records a replayable event stream, and serves a browser Viewer from the same run.

## System shape

The repository contains four cooperating parts:

- an input layer that validates and identifies the source timeline;
- a model and continuity layer that generates one companion stream from rolling
  lead-motion windows;
- a playout and protocol layer that emits ordered frames and records run evidence;
- a Viewer that supports live display, reconnect, local replay, and diagnostics.

Every execution is represented by a run directory. Its effective configuration,
source identities, logs, stream recording, metrics, gate results, and report stay
together so a result can be inspected or replayed later.

## Runtime profiles

The backend is selected in the run configuration. CUDA is the release path;
Recorded replays a captured model result through the same streaming stack; Fake is
used for deterministic local and automated checks. All profiles share the protocol,
playout, evidence, and Viewer layers.

## Operation and reference

Follow the [V2 operation manual](V2_EXECUTION_MANUAL.md) for the current
environment setup, Stage commands, execution order, expected artifacts, and
acceptance procedure.

The wire contract is documented in [PROTOCOL.md](PROTOCOL.md). The architecture,
quality gates, and definition of done are recorded in
[V2_REALTIME_PLAN.md](V2_REALTIME_PLAN.md).

The optional camera-to-model path is documented in
[MEDIAPIPE_INPUT.md](MEDIAPIPE_INPUT.md).
