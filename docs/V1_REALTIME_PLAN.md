# Duet-EDGE Near-Realtime System V1: Modular Monolith Plan

> Basis: `TASK_ANALYSIS_CN.md` and its English counterpart `TASK_ANALYSIS_EN.md`  
> Realtime repository: [just1bit/duet-edge-realtime](https://github.com/just1bit/duet-edge-realtime.git)  
> Model repository: [just1bit/duet-edge](https://github.com/just1bit/duet-edge.git)  
> Target environment: laboratory GPU workstation covering protocol, scheduling, visualization, inference, benchmarking, and acceptance

## 1. Goal and System Position

V1 establishes a continuous near-realtime path from lead-dancer motion to browser visualization:

```text
Motion replay
  -> canonical input frames
  -> 150/75 sliding windows
  -> Duet-EDGE conditional inference
  -> online alignment and overlap blending
  -> continuous timeline commits
  -> playout buffer
  -> NDJSON / WebSocket
  -> browser visualization
```

The system uses a fixed startup buffer. Inference begins when the first window becomes available, and generated frames emit at target playout times. V1 serves one input stream with one resident model instance on one GPU. The modular monolith keeps deployment simple while its interfaces support later camera input, music conditioning, an independent inference process, and multiple sessions.

At 30 FPS, each 150-frame window supplies about five seconds of context. A 75-frame hop starts inference every 2.5 seconds. Run summaries and acceptance evidence quantify latency, throughput, playout stability, and motion continuity.

## 2. Design Principles

1. **Canonical data contracts:** input frames, inference windows, committed batches, and output frames use explicit schemas.
2. **Unified time semantics:** source time, ingest time, target playout time, monotonic time, and wall time have distinct meanings.
3. **Continuous commit semantics:** every output frame enters one continuous timeline exactly once after overlap blending.
4. **Channel-specific delivery:** the recorder preserves the complete stream while the Viewer prioritizes the newest frame.
5. **Explicit capacity policies:** every bounded queue has a configured capacity and full-queue behavior.
6. **Observable lifecycle:** state, inference latency, queue watermarks, playout jitter, and end-to-end latency flow into structured evidence.
7. **Light runtime coupling:** the realtime repository loads the external model through `DUET_EDGE_ROOT`, preserving separate model and service projects.

## 3. Repositories and Model Adapter

### 3.1 Peer Repositories

```text
workspace/
├── duet-edge/               # Model project
└── duet-edge-realtime/      # Streaming service
```

The runtime connection is:

```text
duet-edge-realtime
  -> CudaDuetEdgeBackend
  -> DUET_EDGE_ROOT
  -> EDGE, normalizer, diffusion, SMPLSkeleton
```

Real-backend startup validates the model directory, core modules, checkpoint structure, CUDA runtime, and finite numerical output. The run summary records the model path, checkpoint SHA256, inference parameters, PyTorch/CUDA versions, and GPU properties.

The realtime service uses Duet-EDGE's parameterized DDIM interface and existing runtime structure. Real-model smoke tests, protocol tests, numerical tests, and quality comparisons exercise the complete inference path.

### 3.2 Backend Interface

`FakeInferenceBackend` and `CudaDuetEdgeBackend` share this lifecycle:

```text
warmup()
infer(MotionWindow) -> GeneratedChunk
unnormalize(motion)
version_info()
close()
```

The service loads and warms the model during startup, then reuses it for every window. A dedicated executor thread serializes backend calls while the event loop continues input scheduling, playout, WebSocket delivery, and metric collection.

## 4. Runtime Architecture

```text
Input Adapters
  -> canonical MotionFrame
  -> Session / Timeline Coordinator
  -> SlidingWindowBuffer
  -> bounded InferenceQueue
  -> InferenceBackend
  -> OnlineContinuityProcessor
  -> TimelineCommitter
  -> bounded OutputQueue
  -> PlayoutClock
  -> FanOut
       ├── NDJSON recorder: complete stream
       ├── WebSocket Viewer: newest frames first
       └── metrics / summary.json: metrics and run summary
```

One Python service process hosts the system. Data objects and small interfaces connect its modules. `InferenceBackend` can later become a separate process or remote service while the existing input, commit, playout, and Viewer contracts remain stable.

### 4.1 Input and Windowing

`MotionFrame` contains:

| Field | Meaning |
|---|---|
| `seq` | Contiguous frame number within the source |
| `source_time_s` | Event time measured from source frame zero |
| `ingest_monotonic_s` | Monotonic clock value at service ingestion |
| `source_id` | Input-source identifier |
| `schema_version` | Data-contract version |
| `motion_151` | Canonical normalized lead-motion vector |

The window buffer retains the newest 150 frames and creates a `MotionWindow` every 75 frames. Each window records sequence and source-time ranges, trigger time, valid tail length, and deterministic seed. At end of input, the final frame extends the context and `valid_frames` defines the final commit range.

### 4.2 Online Continuity and Commit

Each generated window follows this sequence:

1. Restore motion parameters through the checkpoint normalizer.
2. Align root positions over the overlap.
3. Blend root trajectories with raised-cosine weights.
4. Blend joint rotations with quaternion slerp.
5. Run SMPL forward kinematics to produce `[N, 24, 3]` joints.
6. Retain the second half as overlap state for the next window.
7. Submit the stable region or final tail to `TimelineCommitter`.

`TimelineCommitter` maintains `next_frame_id` and accepts each batch at that exact position. `CommittedBatch` carries the half-open interval `[start_frame_id, end_frame_id)`, source window, commit type, and joints. Structural validation guides correction of repeated or skipped intervals and preserves one commit per frame.

Offline long-sequence Duet-EDGE generation shares neighboring state during DDIM denoising. The online path generates one window at a time and applies parameter-space continuity afterward. Both paths share motion representation, normalizer, blending principles, and forward kinematics. Representative clips provide boundary, numerical, and perceptual comparison evidence.

### 4.3 Playout Timeline

The protocol uses these clocks:

| Field | Time basis |
|---|---|
| `source_time_s` | Event time from source frame zero |
| `target_playout_offset_s` | Target time from the service monotonic origin |
| `emitted_monotonic_offset_s` | Actual send time from the monotonic origin |
| `emitted_wall_time_s` | Unix wall time for cross-process correlation |

First-frame latency is:

```text
first_frame_latency
  = (window_frames - 1) / fps
  + playout_delay_s
```

Steady-state compute and buffer budgets are:

```text
inference_p99_ms + safety_margin_ms < hop_frames / fps * 1000
playout_delay_s * 1000 >= inference_p99_ms + safety_margin_ms
```

`stream.safety_margin_ms` starts at 100 ms and is calibrated from measured GPU variation, workload, and acceptance evidence. The playout module schedules against absolute deadlines and records jitter, underflow, and end-to-end latency. Inference wall latency includes sampling, CUDA synchronization, and transfer of the generated window to CPU, ending when continuity processing can begin.

## 5. Backpressure, Overload, and Channels

| Channel | Bounded policy | Capacity behavior | Recorded metrics |
|---|---|---|---|
| Input to inference | `inference_queue_size` | `block` waits for capacity; `fail` completes with diagnostics | Watermark, waits, wait time, overload |
| Inference to playout | `output_queue_size` | Waits for playout while preserving complete committed batches | Watermark and waits |
| Playout to NDJSON | Complete recording | Call completion acknowledges receipt | Frames, contiguous sequence, write status |
| Playout to Viewer | `viewer_queue_frames` | Drops the oldest client frame and retains current control state | Per-client dropped-frame summary |

File replay uses `block`, pausing input while inference capacity refills. Fixed-rate capture sessions can use `fail` and let an external workflow start a new session. The effective configuration and summary record policy, capacity, and observed waits.

An inference duration above `inference_slo_ms` emits a `degraded` event and increments deadline metrics. `deadline_miss_policy=continue` preserves the active playout timeline. `deadline_miss_policy=fail` completes the session and writes current records, state history, and structured diagnostics.

## 6. Lifecycle and Recovery

```text
starting -> buffering -> playing -> draining -> finished
    |           |           |           |
    +-----------+-----------+-----------+-> failed
```

| State | Meaning |
|---|---|
| `starting` | Start sinks and publish protocol/run identity |
| `buffering` | Receive input and form the first window |
| `playing` | Emit committed batches on the playout clock |
| `draining` | Complete the tail and drain queues |
| `finished` | Publish EOS, summary, and complete recording |
| `failed` | Preserve generated evidence and publish structured diagnostics |

State transitions flow to NDJSON, WebSocket, and `summary.json`. A reconnecting Viewer receives hello, current state, and current metrics, restoring skeleton definition and run context.

## 7. Output Protocol V2

The hello message declares:

- Protocol and schema versions: `duet-edge-stream/v2` and `2.0.0`;
- Run identity: `run_id`, `session_id`, and `stream_id`;
- Motion definition: FPS, joint names, hierarchy, and coordinate system;
- Playout conventions: time basis, latency budget, and delivery policy.

Each frame includes:

```text
schema_version
run_id / session_id / stream_id
frame_id / seq
source_time_s
target_playout_offset_s
emitted_monotonic_offset_s
emitted_wall_time_s
end_to_end_latency_ms
window_id
commit_start_frame_id / commit_end_frame_id / commit_kind
flags
joints[24][3]
```

`seq`, `motion_time_s`, and `wall_time_s` remain compatibility fields for the Viewer and analysis tools. [PROTOCOL.md](PROTOCOL.md) defines complete messages and examples.

## 8. Configuration

```json
{
  "backend": "cuda",
  "paths": {
    "duet_edge_root": "<model-repository-path>",
    "checkpoint": "<checkpoint-path>",
    "input_motion": "<input-motion-path>",
    "output_dir": "<run-output-path>"
  },
  "model": {
    "guidance_music": 0.0,
    "guidance_lead": 2.0,
    "sampling_steps": 50,
    "eta": 1.0,
    "seed": 1234
  },
  "stream": {
    "fps": 30,
    "window_frames": 150,
    "hop_frames": 75,
    "playout_delay_s": 2.0,
    "inference_queue_size": 1,
    "output_queue_size": 2,
    "viewer_queue_frames": 150,
    "inference_queue_policy": "block",
    "inference_slo_ms": 1900.0,
    "safety_margin_ms": 100.0,
    "deadline_miss_policy": "continue",
    "jitter_slo_ms": 20.0
  },
  "server": {"bind_host": "127.0.0.1", "port": 8765}
}
```

Path values use command line, environment variable, then JSON precedence. Command-line options also support backend, root scaling, sampling steps, and playout delay. Every service run writes `effective_config.json`, `stream.ndjson`, and `summary.json` in an independent output directory.

## 9. Repository Structure

```text
duet-edge-realtime/
├── pyproject.toml
├── docs/
│   ├── README.md
│   ├── PROTOCOL.md
│   ├── TASK_ANALYSIS_CN.md
│   ├── TASK_ANALYSIS_EN.md
│   ├── V1_REALTIME_PLAN.md
│   └── V1_EXECUTION_MANUAL.md
├── configs/
│   ├── v1.fake.json
│   └── v1.cuda.json
├── src/duet_edge_realtime/
├── web/
├── tests/
└── scripts/
    ├── v1_execution/
    └── development/
```

## 10. Observability and Run Records

`summary.json` aggregates:

- Effective configuration and model runtime metadata;
- Load, warmup, wall/CUDA inference p50/p95/p99;
- Input count, observed FPS, sequence status, and window ranges;
- Inference SLO, hop period, headroom, queue watermarks, waits, and deadlines;
- Output count, committed batches, jitter, underflow, Viewer drops, and end-to-end latency;
- State history, diagnostics, exit reason, and SLO outcomes.

Metrics use bounded samples so memory remains stable during long sessions. The acceptance framework adds per-attempt logs, machine-readable stage results, preflight evidence, benchmark summaries, GPU resource evidence, and a generated Markdown index.

## 11. Testing and Acceptance

### 11.1 Local Automated Validation

- Input sequence, window boundaries, tail padding, and source time;
- Continuity alignment, rotation blending, forward kinematics, and finite values;
- Continuous timeline commits and interval validation;
- V2 hello/frame/state/EOS/error protocol and lifecycle;
- Continue/fail deadline policies;
- Viewer latest-frame-wins behavior and control-state retention;
- Long-run metrics, queue, and memory bounds;
- Virtual-clock end-to-end and realtime WebSocket integration.

### 11.2 Laboratory GPU Validation

- Model directory, checkpoint, CUDA, warmup, and deterministic single-window inference;
- Continuous windows and tail processing;
- Warm inference p50/p95/p99, CUDA time, and peak GPU memory;
- Compute budget: `p99_ms + safety_margin_ms < hop_period_ms`;
- Playout budget: `playout_delay_ms >= p99_ms + safety_margin_ms`;
- Target 30 FPS, jitter p95, underflow, and queue watermarks;
- Boundary continuity, naturalness, and coordinate orientation;
- Ten-minute sequence integrity and resource trend.

The complete operator workflow lives in [V1_EXECUTION_MANUAL.md](V1_EXECUTION_MANUAL.md).

## 12. Definition of Done

V1 is complete when:

1. The realtime repository installs independently and the local fake path passes.
2. The laboratory workstation starts the GPU backend from configured model, checkpoint, and input paths.
3. Versioned contracts cover input, windows, commits, and output; effective configuration and summaries capture lifecycle, backpressure, deadlines, and playout behavior.
4. Automated tests, realtime visualization, NDJSON replay, and structured acceptance pass.
5. Continuous-window, performance, quality, and final-duration evidence is archived and reproducible.
