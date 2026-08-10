# Duet-EDGE Realtime V1

基于外部 Duet-EDGE 模型仓库的近实时流式伴舞系统：30 FPS 文件回放 → 150/75 滑窗 → lead-only 推理 → 在线连续性处理 → 连续时间线提交 → 播放缓冲 → NDJSON/WebSocket/Canvas Viewer。

V1 使用固定延迟时间线。默认首帧预算为 `149 / 30 + 2.0 ≈ 6.97 秒`，稳态每 2.5 秒触发一个推理窗口。GPU 基准负责确定正式 `sampling_steps`、`inference_slo_ms` 和 `playout_delay_s`。

## 仓库关系

```text
workspace/
├── duet-edge/             # 模型仓库与运行环境
└── duet-edge-realtime/    # 流式服务
```

真实后端通过 `DUET_EDGE_ROOT=/absolute/path/to/duet-edge` 在运行时加载模型。启动过程检查核心模型文件、checkpoint 结构和 CUDA 环境；运行摘要记录实际模型路径、remote、commit、工作区状态、checkpoint SHA256、PyTorch/CUDA 和 GPU 信息。

## 本地快速开始

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
pytest

python -m duet_edge_realtime.service \
  --config configs/v1.fake.json \
  --clock virtual \
  --sink ndjson \
  --run-id local-smoke
```

每次运行创建 `paths.output_dir/<run-id>/` 并写入：

- `effective_config.json`：命令行、环境变量和 JSON 合并后的最终配置；
- `stream.ndjson`：hello、state、frame、metrics、degraded/backpressure、EOS/error；
- `summary.json`：版本、生命周期、窗口、推理、队列、提交、播放和 SLO 摘要。

路径配置优先级：

```text
--duet-edge-root > DUET_EDGE_ROOT > paths.duet_edge_root
--checkpoint     > EDGE_CHECKPOINT > paths.checkpoint
--input          > EDGE_INPUT_MOTION > paths.input_motion
--output-dir     > EDGE_OUTPUT_DIR > paths.output_dir
```

## Viewer

```bash
# 终端 1
python -m duet_edge_realtime.service \
  --config configs/v1.fake.json \
  --clock realtime \
  --sink websocket,ndjson

# 终端 2
python3 -m http.server 8080 --directory web
```

浏览器打开 `http://127.0.0.1:8080`，连接 `ws://127.0.0.1:8765`。Viewer 也支持选择 `stream.ndjson` 进行本地回放。

WebSocket 客户端拥有独立 latest-frame-wins 队列；控制消息按类型保留最新状态。NDJSON 通道记录完整提交时间线。协议字段见 [PROTOCOL.md](PROTOCOL.md)。

## CUDA 后端

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

原始 `motions/*.pkl` 使用 `--root-scaled false`；已按模型单位缩放的 `motions_sliced/*.pkl` 使用 `--root-scaled true`。默认 Duet-EDGE API使用 50-step、eta=1；带采样参数的优化运行时可使用其他候选值进行性能与质量基准。

## 队列与 deadline

关键配置位于 `stream`：

- `inference_queue_policy=block`：输入等待推理容量，适用于完整文件回放；
- `inference_queue_policy=fail`：队列满时结束 session 并保存诊断；
- `deadline_miss_policy=continue`：记录推理 SLO miss 并继续播放；
- `deadline_miss_policy=fail`：SLO miss 结束 session；
- `output_queue_size`：推理和播放之间的完整提交批次容量；
- `viewer_queue_frames`：每个 Viewer 客户端保留的最新帧容量。

正式配置同时满足：

```text
inference_p99 + 100ms < hop_period
playout_delay >= inference_p99 + 100ms
```

完整单机验收步骤和质量门槛见
[V1_ACCEPTANCE_EXECUTION_CN.md](V1_ACCEPTANCE_EXECUTION_CN.md)。
