# Duet-EDGE Stream Protocol V3

The wire protocol is `duet-edge-stream/v3`; every message uses schema `3.0.0`.
NDJSON stores one JSON object per line and WebSocket transports one object per text
message.

## Lifecycle

```text
Viewer pre-session: hello + state(waiting_for_input)
Formal session:
hello
state(starting -> buffering -> playing -> draining -> finished)
frame / metrics / degraded / backpressure ...
eos
```

A Viewer that connects after Stage 06 receives the pre-session state. Stage 08
publishes a complete session `hello` before the formal lifecycle begins. The NDJSON
recording starts with that complete session `hello` and contains no pre-session data.

A failed session publishes `state(failed)` followed by `error`. The NDJSON file
retains every frame committed before that event.

## `hello`

`hello` defines the complete stream identity:

- `run_id`, `session_id`, and `stream_id`;
- `backend`, `backend_badge`, `model_mode`, checkpoint name/hash, guidance, and
  sampling steps;
- causal-overlap residency and continuity-correction mode;
- source timeline identity, path, hash, selected frame range, and clip count;
- 30 FPS, 24-joint hierarchy, right-handed Z-up coordinates, and latency budget;
- recorder, Viewer, queue, and exactly-once delivery semantics.

The fixed latency is `(window_frames - 1) / fps + playout_delay_s`. Stage 03
automatically finalizes `playout_delay_s` from the real-clock baseline; the Viewer
render buffer is additional.

## `frame`

Every `frame` includes:

| Group | Fields |
|---|---|
| Identity | `run_id`, `session_id`, `stream_id`, `frame_id`, `seq` |
| Source | `source_id`, `source_sha256`, `source_time_s`, `clip_id`, `clip_frame` |
| Transition | `transition_id`, `in_transition`, `boundary`, `flags` |
| Timing | `target_playout_offset_s`, `emitted_monotonic_offset_s`, `emitted_wall_time_s`, `frame_latency_ms`, `send_lateness_ms` |
| Commit | `window_id`, `commit_start_frame_id`, `commit_end_frame_id`, `commit_kind` |
| Motion | `lead_joints[24][3]`, `companion_joints[24][3]` |

`frame_id` is contiguous and each source frame is committed once. Commit intervals
are half-open. `stable` identifies a resolved 75-frame region and `tail` identifies
the valid EOF tail. `joints` remains an alias of `companion_joints` for V1 clients.

The authoritative lead skeleton comes directly from FK of the canonical source
timeline. Companion joints come from causal-overlap generation followed by local
relative-root correction, raised-cosine blending, shortest-path quaternion SLERP,
and FK.

## Control and telemetry messages

| Type | Purpose |
|---|---|
| `state` | Service lifecycle |
| `metrics` | Inference, queues, jitter, clients, frame age, and render health |
| `backpressure` | Bounded input waiting for inference capacity |
| `overload` | Queue policy event |
| `degraded` | Inference deadline event |
| `eos` | Normal timeline completion |
| `error` | Structured session failure |

Viewer clients send `client_metrics` messages containing render FPS, frame age,
visible stalls, and connect/reconnect events. Each Viewer owns a bounded
latest-frame lane, while the NDJSON recorder preserves the complete timeline.
