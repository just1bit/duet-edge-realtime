# Duet-EDGE Realtime V1

基于外部 Duet-EDGE 模型仓库的近实时流式伴舞系统：30 FPS 文件回放 → 150/75 滑窗 → lead-only 推理 → 在线连续性处理 → 24 关节坐标 → NDJSON/WebSocket/Canvas Viewer。

V1 是固定延迟流，时间线延迟约为 `5 秒输入窗口 + playout_delay`。`sampling_steps=50` 与 `playout_delay_s=2.0` 在 GPU 基准前仍是占位值。

## 两个独立仓库

```text
workspace/
├── duet-edge/             # 模型仓库，独立 clone 和环境
└── duet-edge-realtime/    # 本仓库，不含模型代码或 submodule
```

真实后端通过 `DUET_EDGE_ROOT=/absolute/path/to/duet-edge` 加载模型。兼容版本由 [compat/duet-edge.lock.json](compat/duet-edge.lock.json) 锁定；正式运行拒绝 commit 不符或 Python 源码 dirty 的引擎。开发时可用 `--allow-engine-mismatch`，但 summary 会标记 `non_reproducible=true`。

## Mac 快速开始

```bash
git clone https://github.com/just1bit/duet-edge-realtime.git
cd duet-edge-realtime
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
pytest

python -m duet_edge_realtime.service \
  --config configs/v1.fake.json --clock virtual --sink ndjson
```

Fake 配置使用仓库内小型 fixture，并将运行产物写到仓库外的 `/tmp/duet-edge-realtime-runs/<run_id>/`。每次运行创建新目录并写入：

- `effective_config.json`：CLI > 环境变量 > JSON 合并后的最终配置。
- `stream.ndjson`：hello、frame、metrics/degraded、eos/error。
- `summary.json`：版本、窗口、推理、队列、jitter、underflow 和错误。

指定路径时优先级为 CLI > 环境变量 > JSON：

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
  --config configs/v1.fake.json --clock realtime --sink websocket,ndjson

# 终端 2
python3 -m http.server 8080 --directory web
```

打开 `http://127.0.0.1:8080` 并连接 `ws://127.0.0.1:8765`。Viewer 也可直接回放 NDJSON，无 npm、构建步骤或外部 CDN。

## 外部 CUDA 后端

服务器复用已有 EDGE conda 环境：

```bash
export DUET_EDGE_ROOT=/data/zliu753/EDGE
export EDGE_CHECKPOINT=/data/zliu753/EDGE/runs/train/exp9/weights/train-1800.pt
export EDGE_INPUT_MOTION=/data/zliu753/EDGE/data/val/motions/example.pkl
export EDGE_OUTPUT_DIR=/data/zliu753/realtime-runs

python -m duet_edge_realtime.service \
  --config configs/v1.cuda.json --root-scaled false --clock virtual --sink ndjson
```

原始 `motions/*.pkl` 使用 `--root-scaled false`；已经缩放的 `motions_sliced/*.pkl` 使用 `true`。系统不会猜测尺度。默认 50-step/eta=1 可调用 lock 中未修改的引擎；非默认 DDIM 参数仅在模型仓库已有兼容优化且 lock 已更新时启用。

完整服务器操作、性能与质量门槛见 [ACCEPTANCE_EXECUTION_CN.md](ACCEPTANCE_EXECUTION_CN.md)。
