# Duet-EDGE Realtime V1

基于 Duet-EDGE 的近实时流式伴舞系统。V1 链路为：AIST++/标准化动作文件按 30 FPS 回放 → 150/75 滑窗 → 常驻 lead-only 推理 → 在线根位置对齐、raised-cosine 与 quaternion slerp → 24 关节坐标 → NDJSON/WebSocket → 浏览器 Canvas Viewer。

V1 的“近实时”是固定延迟流：约 `5 秒输入窗口 + playout_delay`，不是低延迟交互。配置中的 `sampling_steps=50` 和 `playout_delay_s=2.0` 是服务器基准前的占位值，正式验收必须用 GPU 实测值替换。

## 目录

- `src/duet_edge_realtime/`：输入、滑窗、后端、连续性、输出、指标和 CLI。
- `third_party/duet-edge/`：锁定的模型引擎 submodule。
- `web/`：无需构建、无 CDN 的实时/NDJSON Viewer。
- `tests/`：Mac 可运行测试及 opt-in CUDA smoke。
- `scripts/slurm/`：三轮 GPU 作业脚本。
- `ACCEPTANCE_EXECUTION_CN.md`：从干净 clone 到最终签字的完整验收步骤。

## Mac 快速开始

```bash
git clone --recurse-submodules <realtime-repository-url>
cd duet-edge-realtime
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/make_fake_fixture.py --frames 900 --output /tmp/fake-motion.npz
python -m duet_edge_realtime.service \
  --backend fake --input /tmp/fake-motion.npz --clock virtual \
  --sink ndjson --output-dir outputs/fake-virtual
python scripts/check_run.py \
  --summary outputs/fake-virtual/summary.json \
  --ndjson outputs/fake-virtual/stream.ndjson
```

不安装项目也可以运行测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Viewer

终端 1 启动服务：

```bash
python -m duet_edge_realtime.service \
  --backend fake --input /tmp/fake-motion.npz --clock realtime \
  --sink ndjson,websocket --output-dir outputs/viewer
```

终端 2 提供静态页面：

```bash
python3 -m http.server 8080 --directory web
```

打开 `http://127.0.0.1:8080`，连接默认 `ws://127.0.0.1:8765`。也可选择服务生成的 `stream.ndjson` 做本地回放。WebSocket 仅绑定 loopback；服务器展示使用 SSH tunnel。

## 输入契约

标准化 fixture 是 `.npz`，必须包含 `[N,151] float32` 的 `motion_151`（兼容键 `motion`），且 `N >= 150`。AIST++ pickle 必须显式声明尺度状态：

- 原始 `motions/*.pkl`：`--root-scaled false`，服务执行 `pos / scale[0]`。
- 已切片 `motions_sliced/*.pkl`：`--root-scaled true`，不得再次缩放。

系统拒绝猜测尺度，也拒绝重复、乱序和缺失的输入序号。

## 产物

每次运行生成统一 `run_id` 的：

- `stream.ndjson`：hello、frame、metrics/degraded 和 eos/error。
- `summary.json`：版本、配置、窗口、推理分位数、队列、jitter、underflow、错误和 GPU 信息。

正式 GPU 操作、质量回归、Viewer 断线重连和验收门槛见 [ACCEPTANCE_EXECUTION_CN.md](ACCEPTANCE_EXECUTION_CN.md)。
