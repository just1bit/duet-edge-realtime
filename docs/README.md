# Duet-EDGE Realtime V1

Duet-EDGE Realtime is a near-realtime streaming partner-dance system built around the external Duet-EDGE model repository. The V1 path runs 30 FPS motion replay through 150/75 sliding windows, lead-conditioned inference, online continuity processing, continuous timeline commits, playout buffering, NDJSON/WebSocket delivery, and a Canvas Viewer.

V1 uses a fixed-latency timeline. The default first-frame budget is `149 / 30 + 2.0 ≈ 6.97 seconds`, and steady-state inference starts every 2.5 seconds. GPU benchmarks determine production `sampling_steps`, `inference_slo_ms`, and `playout_delay_s`; `stream.safety_margin_ms` supplies the shared safety margin.

> **Required model repository:** This repository provides the streaming integration layer, not the Duet-EDGE model implementation or its runtime assets. To run the model, also clone [Duet-EDGE](https://github.com/just1bit/duet-edge). The environment setup and execution workflow are documented in [V1_EXECUTION_MANUAL.md](V1_EXECUTION_MANUAL.md).

## Workspace Layout

```text
workspace/
├── duet-edge/             # Model repository and runtime assets
└── duet-edge-realtime/    # Streaming service
```

Clone both repositories into the same parent directory:

```bash
git clone https://github.com/just1bit/duet-edge.git
git clone https://github.com/just1bit/duet-edge-realtime.git
```

The real backend loads the model at runtime from `DUET_EDGE_ROOT`. Startup verifies the model files, checkpoint structure, and CUDA runtime. The checkpoint normalizer is loaded as a complete Python object with `weights_only=False`, and acceptance assets are identified by SHA256. Each summary records model paths, checkpoint SHA256, inference parameters, PyTorch/CUDA versions, and GPU details.

## Getting Started

Use [V1_EXECUTION_MANUAL.md](V1_EXECUTION_MANUAL.md) for the environment setup and execution workflow.

Each run creates `paths.output_dir/<run-id>/` with:

- `effective_config.json`: the final merged command-line, environment, and JSON configuration;
- `summary.json`: versions, lifecycle, windows, inference, queues, commits, playout, and SLO results.

Selecting the `ndjson` sink also creates `stream.ndjson` with hello, state, frame, metrics, degraded/backpressure, EOS, and error messages.

Path precedence is:

```text
--duet-edge-root > DUET_EDGE_ROOT > paths.duet_edge_root
--checkpoint     > EDGE_CHECKPOINT > paths.checkpoint
--input          > EDGE_INPUT_MOTION > paths.input_motion
--output-dir     > EDGE_OUTPUT_DIR > paths.output_dir
```

## Viewer

```bash
# Terminal 1
python -m duet_edge_realtime.service \
  --config configs/v1.fake.json \
  --clock realtime \
  --sink websocket,ndjson

# Terminal 2
python3 -m http.server 8080 --directory web
```

Open `http://127.0.0.1:8080` and connect to `ws://127.0.0.1:8765`. The Viewer also supports local `stream.ndjson` replay.

The canvas shows the lead (blue) and generated companion (cyan) side by side in the Z-up coordinate system. The horizontal display separation is visual only; the streamed joint coordinates remain unchanged.

Each WebSocket client owns an independent latest-frame-wins queue, while control messages retain the latest state by type. NDJSON records the complete committed timeline. See [PROTOCOL.md](PROTOCOL.md) for protocol fields and examples.

## CUDA Backend

```bash
export DUET_EDGE_ROOT=/data/user/duet-edge
export EDGE_CHECKPOINT=/data/user/train-1800.pt
export EDGE_INPUT_MOTION=/data/user/aist_plusplus_final/motions/example.pkl
export EDGE_OUTPUT_DIR=/data/user/realtime-runs

python -m duet_edge_realtime.service \
  --config configs/v1.cuda.json \
  --root-scaled false \
  --clock virtual \
  --sink ndjson
```

Raw `motions/*.pkl` inputs use `--root-scaled false`. Inputs already scaled to model units under `motions_sliced/*.pkl` use `--root-scaled true`. The parameterized DDIM interface maps `model.sampling_steps` to `sampling_timesteps` and forwards `model.eta`; it supports the 50-step baseline and lower-step performance/quality candidates.

## Queues and Deadlines

Key settings live under `stream`:

- `inference_queue_policy=block`: input waits for inference capacity during complete file replay;
- `inference_queue_policy=fail`: queue capacity produces overload diagnostics and a completed run record;
- `deadline_miss_policy=continue`: the service records an inference SLO event and continues playout;
- `deadline_miss_policy=fail`: an inference SLO event produces structured diagnostics and a completed run record;
- `output_queue_size`: complete committed batches between inference and playout;
- `viewer_queue_frames`: newest frames retained for each Viewer client.

Production configuration satisfies:

```text
inference_p99_ms + safety_margin_ms < hop_period_ms
playout_delay_ms >= inference_p99_ms + safety_margin_ms
```

See [V1_REALTIME_PLAN.md](V1_REALTIME_PLAN.md) for the complete design and [V1_EXECUTION_MANUAL.md](V1_EXECUTION_MANUAL.md) for the laboratory acceptance workflow.
