# Duet-EDGE Stream Protocol V2

The protocol name is `duet-edge-stream/v2`, and the message schema version is `2.0.0`. Each NDJSON line contains one JSON message, and each WebSocket text message contains one JSON object.

## Identifiers and Timebases

| Field | Meaning |
|---|---|
| `run_id` | Identifies both the run directory and a single execution |
| `session_id` | Identifies the input, inference, and playout lifecycle; in V1, it is identical to `run_id` |
| `stream_id` | Identifies the output motion stream, in the form `<run_id>:companion-motion` |
| `source_time_s` | Event time measured from input source frame 0 |
| `target_playout_offset_s` | Target playout time measured from the service's monotonic-clock origin |
| `emitted_monotonic_offset_s` | Actual emission time measured from the service's monotonic-clock origin |
| `emitted_wall_time_s` | Unix epoch seconds used to correlate logs across services |

## Message Sequence

A successful file-based run produces the following sequence:

```text
hello
state(starting)
state(buffering)
state(playing)
frame / metrics / degraded / backpressure ...
state(draining)
frame / metrics ... (queued windows and the tail continue playing)
state(finished)
eos
```

A failed run produces `state(failed)` and `error`. Frames committed before the failure remain in the NDJSON output.

## hello

```json
{
  "type": "hello",
  "protocol": "duet-edge-stream/v2",
  "schema_version": "2.0.0",
  "run_id": "demo",
  "session_id": "demo",
  "stream_id": "demo:companion-motion",
  "fps": 30,
  "joint_count": 24,
  "joint_names": ["root"],
  "parents": [-1],
  "coordinate_system": {
    "handedness": "right",
    "x": "lateral",
    "y": "depth",
    "z": "up",
    "units": "model-space"
  },
  "axis": "x=lateral,y=depth,z=up",
  "timebases": {
    "source_time_s": "seconds from source frame 0",
    "target_playout_offset_s": "monotonic seconds from service start",
    "emitted_wall_time_s": "Unix epoch seconds",
    "emitted_monotonic_offset_s": "monotonic seconds from service start"
  },
  "fixed_latency_s": 6.9666666667,
  "latency_budget": {
    "window_fill_s": 4.9666666667,
    "playout_delay_s": 2.0,
    "hop_period_s": 2.5,
    "inference_slo_ms": 1900.0,
    "safety_margin_ms": 100.0,
    "jitter_slo_ms": 20.0
  },
  "delivery": {
    "timeline": "contiguous-exactly-once-commit",
    "recorder": "complete",
    "viewer": "latest-frame-wins",
    "inference_queue_policy": "block"
  }
}
```

In an actual `hello` message, `joint_names` and `parents` each contain 24 entries.

## frame

```json
{
  "type": "frame",
  "schema_version": "2.0.0",
  "run_id": "demo",
  "session_id": "demo",
  "stream_id": "demo:companion-motion",
  "frame_id": 75,
  "seq": 75,
  "source_time_s": 2.5,
  "motion_time_s": 2.5,
  "target_playout_offset_s": 9.4666666667,
  "emitted_monotonic_offset_s": 9.4668,
  "emitted_wall_time_s": 1786380000.0,
  "end_to_end_latency_ms": 6966.8,
  "window_id": 1,
  "commit_start_frame_id": 75,
  "commit_end_frame_id": 150,
  "commit_kind": "stable",
  "flags": ["generated", "stable"],
  "lead_joints": [[-0.5, 0.0, 1.0]],
  "companion_joints": [[0.5, 0.0, 1.0]],
  "joints": [[0.5, 0.0, 1.0]]
}
```

`frame_id` and `seq` increase contiguously. Commit ranges use the half-open interval `[commit_start_frame_id, commit_end_frame_id)`. A `commit_kind` of `stable` identifies the stable region of a sliding window, while `tail` identifies the final valid tail after the input ends. Both `lead_joints` and `companion_joints` contain 24 three-dimensional, Z-up coordinates. `joints` remains as a compatibility alias for `companion_joints`, and their contents must be identical.

## Status and Diagnostics

| `type` | Key fields | Purpose |
|---|---|---|
| `state` | `state`, `wall_time_s`, `monotonic_offset_s` | Lifecycle transitions |
| `metrics` | p95, queues, total/per-client dropped frames, underflows, deadline misses, backpressure waits | Real-time runtime status |
| `backpressure` | `window_id`, `policy`, `wait_ms` | Input waiting for inference capacity |
| `overload` | `window_id`, `policy`, `reason` | Activation of the `fail` policy |
| `degraded` | `window_id`, `observed_ms`, `slo_ms` | Inference SLO miss |
| `eos` | `frames`, `reason` | Normal completion |
| `error` | `error` | Structured error |

After connecting, a Viewer first receives `hello`, followed by the current `state`, the latest `metrics`, any terminal message, and live frames. `degraded`, `backpressure`, and `overload` are delivered as real-time events, while NDJSON preserves their complete history. Each Viewer mailbox coalesces pending `state`, `metrics`, `degraded`, `backpressure`, and `overload` messages by type. When the frame channel reaches capacity, it retains the latest frame. Clients can use `frame_id` to detect gaps between displayed frames and use NDJSON to recover the complete committed sequence.
