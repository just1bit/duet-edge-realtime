# Duet-EDGE Realtime

Duet-EDGE Realtime turns file or live-camera lead motion into a timestamped
lead-and-companion skeleton stream. The runtime keeps one inference worker alive,
processes overlapping motion windows causally, preserves companion continuity,
records a replayable event stream, and serves a browser Viewer from the same run.

## System shape

The repository contains four cooperating parts:

- an input layer for validated files and a MediaPipe camera producer;
- a model and continuity layer that generates one companion stream from rolling
  lead-motion windows;
- a playout and protocol layer that emits ordered frames and records run evidence;
- a Viewer that supports live display, reconnect, local replay, and diagnostics.

The resident service can switch between file and MediaPipe input without reloading
the CUDA model. The MediaPipe producer owns camera capture and pose detection; the
service performs resampling, SMPL24 retargeting, checkpoint normalization, inference,
playout, recording, and Viewer delivery.

Every execution is represented by a run directory. Its effective configuration,
source identities, logs, stream recording, metrics, gate results, and report stay
together so a result can be inspected or replayed later.

## Runtime profiles

The backend is selected in the run configuration. CUDA powers file and MediaPipe
production input. Recorded replays a captured model result through the same streaming
stack, while Fake provides deterministic local and automated checks. All profiles
share the protocol, playout, evidence, and Viewer layers.

## Operation and reference

Follow the [Final service manual](FINAL_SERVICE_MANUAL.md) for environment setup,
service commands, input execution, outputs, and troubleshooting.

The wire contract is documented in [PROTOCOL.md](PROTOCOL.md).

The live camera workflow is documented in [MEDIAPIPE_INPUT.md](MEDIAPIPE_INPUT.md).
