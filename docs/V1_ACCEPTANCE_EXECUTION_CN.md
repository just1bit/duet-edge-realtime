# Duet-EDGE Realtime V1 单机验收手册

这份手册用于实验室的一台 NVIDIA GPU 电脑。全部命令直接在这台电脑上运行。

验收目标是确认：模拟链路正常、真实模型可以连续推理、浏览器可以展示、推理速度满足 2.5 秒步长，最后稳定运行 10 分钟。

## 阶段计划

| 阶段 | 输入 | 要做的事 | 输出 | 通过标准 |
|---|---|---|---|---|
| P0：准备环境 | 两个仓库、checkpoint、AIST++ 动作 | 设置路径，转换一份测试动作 | `input_motion.pkl` | 路径和 CUDA 检查通过 |
| P1：测试流式核心 | 仓库自带 fake fixture | 跑自动测试和模拟全链路 | fake NDJSON、summary | 测试通过，检查结果 `passed: true` |
| S1：接入真实模型 | checkpoint、`input_motion.pkl` | 运行真实 CUDA 推理并导出 fixture | real NDJSON、`real_fixture.npz` | 帧连续、坐标有效、运行正常结束 |
| P2：检查 Viewer | fake fixture | 启动服务和网页 | 浏览器骨架动画 | 动画、状态、重连和文件回放正常 |
| S2：测量性能 | `real_fixture.npz` | 测量 50-step 推理 p99 | `benchmark.json` | `deadline_candidate: true` |
| P3：写入正式配置 | benchmark 结果 | 更新 steps、播放缓冲和推理 SLO | `configs/v1.cuda.json` | 配置值来自实测结果 |
| S3：最终验收 | 正式配置、real fixture | 按真实时钟运行 10 分钟 | final NDJSON、summary | FPS、延迟、序列和稳定性全部通过 |

阶段按表格顺序执行。每一阶段完成后再进入下一阶段。

---

## P0：准备环境

### 输入

- `duet-edge-realtime`：流式服务仓库；
- `duet-edge`：模型仓库；
- `train-1800.pt`：模型 checkpoint；
- 一份 AIST++ `motions/*.pkl` 动作。

下面假设两个仓库和数据目录是并列的：

```text
工作目录/
├── duet-edge-realtime/
├── duet-edge/
└── data+checkpoint/
    ├── train-1800.pt
    └── aist_plusplus_final/
```

### 操作 1：进入仓库并激活环境

```bash
cd /实际路径/duet-edge-realtime
conda activate edge
python -m pip install -e '.[dev]'
```

这里的 `edge` 是能够运行原 Duet-EDGE 的 conda 环境。

### 操作 2：设置本次验收使用的路径

```bash
export REALTIME_ROOT="$(pwd)"
export DUET_EDGE_ROOT="$(cd ../duet-edge && pwd)"
export EDGE_CHECKPOINT="$(cd ../data+checkpoint && pwd)/train-1800.pt"
export AIST_RAW="$(cd ../data+checkpoint/aist_plusplus_final/motions && pwd)/gJB_sBM_cAll_d07_mJB0_ch06.pkl"
export RUN_ROOT="${REALTIME_ROOT}/outputs/acceptance-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${RUN_ROOT}"
```

`RUN_ROOT` 是本次验收的总输出目录。时间戳让每次验收使用一个新目录。

如果示例动作文件不存在，把 `AIST_RAW` 最后一段替换成 `motions/` 中实际存在的文件名。

### 操作 3：检查路径和 GPU

```bash
test -f "${DUET_EDGE_ROOT}/EDGE.py"
test -f "${EDGE_CHECKPOINT}"
test -f "${AIST_RAW}"
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0))'
```

预期看到：

```text
CUDA available: True
GPU: <实验室电脑的 GPU 名称>
```

`test` 命令没有输出并返回终端提示符，表示文件存在。

### 操作 4：转换测试动作

AIST++ 原始文件使用 `smpl_trans/smpl_poses/smpl_scaling`。下面的命令把它转换成流式输入适配器使用的 `pos/q/scale`：

```bash
python scripts/prepare_aist_motion.py \
  --input "${AIST_RAW}" \
  --output "${RUN_ROOT}/input_motion.pkl"
```

### 输出与预期结果

输出文件：

```text
${RUN_ROOT}/input_motion.pkl
```

终端会显示原始帧数和 `root_scaled=false`。文件存在即表示 P0 完成：

```bash
test -f "${RUN_ROOT}/input_motion.pkl"
```

---

## P1：测试流式核心

这个阶段使用 fake backend，不调用 GPU 模型。它用于快速检查滑窗、连续性处理、提交时间线、状态机、协议和结果记录。

### 输入

```text
tests/fixtures/fake_motion.npz
```

### 操作 1：运行自动测试

```bash
pytest
```

默认会跳过 CUDA 和需要绑定本机端口的 WebSocket 集成测试，其余测试应全部通过。
随后显式运行 WebSocket 集成测试：

```bash
RUN_NETWORK_TESTS=1 pytest tests/test_websocket_integration.py
```

这条命令也必须通过；如果实验室安全策略禁止绑定 `127.0.0.1`，记录限制并在 P2
用浏览器连接完成等价验证，不能把默认的 `skipped` 当成通过。

### 操作 2：运行模拟全链路

```bash
python -m duet_edge_realtime.service \
  --config configs/v1.fake.json \
  --output-dir "${RUN_ROOT}" \
  --clock virtual \
  --sink ndjson \
  --run-id p1-fake
```

`virtual` 时钟会直接推进播放时间，因此这一步通常几秒内完成。

### 操作 3：自动检查结果

```bash
python scripts/check_run.py \
  --summary "${RUN_ROOT}/p1-fake/summary.json" \
  --ndjson "${RUN_ROOT}/p1-fake/stream.ndjson"
```

### 输出与预期结果

输出目录：

```text
${RUN_ROOT}/p1-fake/
├── effective_config.json
├── stream.ndjson
└── summary.json
```

预期检查结果：

```json
{
  "passed": true,
  "failures": []
}
```

这表示协议为 v2、输出帧连续、每帧有 24 个关节、提交帧数一致、生命周期正常结束。

---

## S1：接入真实 CUDA 模型

这个阶段把 P0 转换的主舞动作送入真实 Duet-EDGE，并把结果保存成后续性能测试使用的轻量 fixture。

### 输入

- `${RUN_ROOT}/input_motion.pkl`；
- `${EDGE_CHECKPOINT}`；
- `${DUET_EDGE_ROOT}`。

### 操作 1：运行真实流式推理

```bash
python -m duet_edge_realtime.service \
  --config configs/v1.cuda.json \
  --duet-edge-root "${DUET_EDGE_ROOT}" \
  --checkpoint "${EDGE_CHECKPOINT}" \
  --input "${RUN_ROOT}/input_motion.pkl" \
  --input-format aist \
  --root-scaled false \
  --output-dir "${RUN_ROOT}" \
  --clock virtual \
  --sink ndjson \
  --run-id s1-real-smoke
```

第一次启动会加载 checkpoint，并执行模型预热。终端返回提示符表示运行结束。

### 操作 2：检查真实流

```bash
python scripts/check_run.py \
  --summary "${RUN_ROOT}/s1-real-smoke/summary.json" \
  --ndjson "${RUN_ROOT}/s1-real-smoke/stream.ndjson" \
  --require-backend cuda
```

### 操作 3：运行 CUDA 单窗口确定性测试

```bash
RUN_CUDA_TESTS=1 \
DUET_EDGE_CHECKPOINT="${EDGE_CHECKPOINT}" \
DUET_EDGE_ROOT="${DUET_EDGE_ROOT}" \
pytest tests/test_cuda_smoke.py
```

### 操作 4：导出真实 fixture

```bash
python scripts/export_fixture.py \
  --checkpoint "${EDGE_CHECKPOINT}" \
  --duet-edge-root "${DUET_EDGE_ROOT}" \
  --motion "${RUN_ROOT}/input_motion.pkl" \
  --root-scaled false \
  --output "${RUN_ROOT}/real_fixture.npz"
```

### 输出与预期结果

主要输出：

```text
${RUN_ROOT}/s1-real-smoke/stream.ndjson
${RUN_ROOT}/s1-real-smoke/summary.json
${RUN_ROOT}/real_fixture.npz
```

通过标准：

- `check_run.py` 输出 `passed: true`；
- CUDA 单窗口确定性测试通过；
- `summary.json` 中 `backend.backend` 为 `cuda`；
- `exit_reason` 为 `input_complete`；
- `real_fixture.npz` 成功生成。

---

## P2：检查浏览器 Viewer

Viewer 检查使用 fake backend，便于快速重复。需要两个终端窗口。

### 输入

```text
tests/fixtures/fake_motion.npz
```

### 操作 1：在终端 A 启动实时流

```bash
cd "${REALTIME_ROOT}"
conda activate edge
python -m duet_edge_realtime.service \
  --config configs/v1.fake.json \
  --output-dir "${RUN_ROOT}" \
  --loop 10 \
  --clock realtime \
  --sink websocket,ndjson \
  --run-id p2-viewer
```

`--loop 10` 会把测试动作重复十次，为浏览器检查留出足够时间。

### 操作 2：在终端 B 启动网页

```bash
cd "${REALTIME_ROOT}"
python -m http.server 8080 --directory web
```

浏览器打开：

```text
http://127.0.0.1:8080
```

点击连接，默认 WebSocket 地址为：

```text
ws://127.0.0.1:8765
```

### 操作 3：检查五项功能

1. 骨架持续运动；
2. 状态显示“缓冲中”“播放中”“尾段提交”“已完成”；
3. 断开后重新连接，页面恢复骨架定义和当前状态，并继续接收新画面；
4. 选择 `${RUN_ROOT}/p2-viewer/stream.ndjson`，文件可以回放。
5. 再选择 `${RUN_ROOT}/s1-real-smoke/stream.ndjson`，确认真实伴舞没有明显骨架翻转、
   地面轴错误、持续漂移或窗口边界跳变；记录观察结论和使用的模型 commit。

检查完成后等待终端 A 正常结束，并在终端 B 按 `Ctrl+C` 关闭网页服务。

### 输出与预期结果

输出为浏览器动画和 `${RUN_ROOT}/p2-viewer/` 下的运行记录。五项功能均正常表示 P2 完成。

---

## S2：测量真实推理性能

这个阶段使用 S1 生成的规范化真实动作，重复 51 次，得到 101 个窗口延迟样本。
`virtual` 时钟让测试只测计算速度，不等待真实播放时间。少于 100 个样本时，
`summarize_benchmark.py` 会拒绝生成推荐配置。

### 输入

```text
${RUN_ROOT}/real_fixture.npz
```

### 操作 1：运行 50-step 基准

```bash
python -m duet_edge_realtime.service \
  --config configs/v1.cuda.json \
  --duet-edge-root "${DUET_EDGE_ROOT}" \
  --checkpoint "${EDGE_CHECKPOINT}" \
  --input "${RUN_ROOT}/real_fixture.npz" \
  --input-format fixture \
  --output-dir "${RUN_ROOT}" \
  --loop 51 \
  --sampling-steps 50 \
  --clock virtual \
  --sink ndjson \
  --run-id s2-benchmark-50
```

### 操作 2：生成简短的基准摘要

```bash
python scripts/summarize_benchmark.py "${RUN_ROOT}" \
  --pattern "s2-benchmark-*/summary.json" \
  --min-samples 100 \
  --output benchmark.json
python -m json.tool "${RUN_ROOT}/benchmark.json"
```

### 输出与预期结果

输出：

```text
${RUN_ROOT}/s2-benchmark-50/summary.json
${RUN_ROOT}/benchmark.json
```

重点看 `benchmark.json` 的 `recommended_baseline`：

```json
{
  "recommended_baseline": {
    "deadline_candidate": true,
    "recommended_playout_delay_s": 1.234
  }
}
```

- `deadline_candidate: true` 表示推理 p99 加 100 ms 后仍小于 2.5 秒，可以持续跟上输入；
- `recommended_playout_delay_s` 是正式配置建议使用的播放缓冲。

50-step 达标后直接进入 P3。若 `deadline_candidate` 为 `false`，先切换到明确支持
`sampling_timesteps`/`eta` 且带 lead-only CFG 快路径的已审核 Duet-EDGE commit，记录
该 commit，再用新的 run ID 重复本阶段。实时后端会拒绝在不支持参数化 DDIM 的引擎上
运行非 50-step，不能绕过这个检查。

切换优化引擎后先验证 lead-only CFG 快路径数值等价：

```bash
python scripts/check_cfg_equivalence.py \
  --checkpoint "${EDGE_CHECKPOINT}" \
  --duet-edge-root "${DUET_EDGE_ROOT}" \
  --output "${RUN_ROOT}/cfg-equivalence.json"
```

必须得到 `passed: true`、`fast_path_forward_calls: 2`，否则不能测试低 steps 候选。

对 25/20/10 等低 steps 候选，必须同时运行 50-step 基线并执行质量回归：

```bash
python scripts/compare_quality.py \
  --fixture "${RUN_ROOT}/real_fixture.npz" \
  --baseline-ndjson "${RUN_ROOT}/s2-benchmark-50/stream.ndjson" \
  --candidate-ndjson "${RUN_ROOT}/s2-benchmark-25/stream.ndjson" \
  --duet-edge-root "${DUET_EDGE_ROOT}" \
  --output "${RUN_ROOT}/quality-25.json"
```

每个低 steps 候选分别执行一次。只有 `deadline_candidate: true` 且质量 JSON
`passed: true` 的候选可以进入 P3；在合格候选中选择 steps 最大者。质量 JSON 与
benchmark 一起归档。

---

## P3：写入正式配置

### 输入

- S2 通过的 `steps`；
- `benchmark.json` 中对应的 `recommended_playout_delay_s`。

先把最终选择写入环境变量，值必须来自 benchmark 和质量选择结果：

```bash
export SAMPLING_STEPS=50
export PLAYOUT_DELAY_S=2.0
```

上面的数值只是格式示例，不能直接照抄。

### 操作

```bash
python scripts/update_runtime_config.py \
  --config configs/v1.cuda.json \
  --sampling-steps "${SAMPLING_STEPS}" \
  --playout-delay-s "${PLAYOUT_DELAY_S}" \
  --safety-margin-ms 100
python -m json.tool configs/v1.cuda.json
```

脚本会同时更新：

- `model.sampling_steps`；
- `stream.playout_delay_s`；
- `stream.inference_slo_ms`。

`inference_slo_ms` 等于播放缓冲和 2.5 秒 hop 周期中的较小值，再减去 100 ms 安全余量。

### 输出与预期结果

输出文件：

```text
configs/v1.cuda.json
```

终端会打印最终 steps、播放缓冲、推理 SLO 和安全余量。确认这些值与 S2 选择一致，P3 即完成。

---

## S3：运行 10 分钟最终验收

`real_fixture.npz` 包含 150 帧，也就是 5 秒动作。重复 120 次得到 600 秒，即 10 分钟。

### 输入

- P3 更新后的 `configs/v1.cuda.json`；
- `${RUN_ROOT}/real_fixture.npz`。

### 操作 1：按真实时钟运行

```bash
python -m duet_edge_realtime.service \
  --config configs/v1.cuda.json \
  --duet-edge-root "${DUET_EDGE_ROOT}" \
  --checkpoint "${EDGE_CHECKPOINT}" \
  --input "${RUN_ROOT}/real_fixture.npz" \
  --input-format fixture \
  --output-dir "${RUN_ROOT}" \
  --loop 120 \
  --clock realtime \
  --sink websocket,ndjson \
  --run-id s3-final-10min
```

这条命令会实际运行约 10 分钟。运行期间可以打开 P2 的网页，实时观察骨架。

### 操作 2：自动检查最终结果

```bash
python scripts/check_run.py \
  --summary "${RUN_ROOT}/s3-final-10min/summary.json" \
  --ndjson "${RUN_ROOT}/s3-final-10min/stream.ndjson" \
  --duration-min 10 \
  --require-backend cuda \
  --min-inference-samples 100 \
  --require-performance
```

### 输出与预期结果

最终输出：

```text
${RUN_ROOT}/s3-final-10min/
├── effective_config.json
├── stream.ndjson
└── summary.json
```

预期检查结果：

```json
{
  "passed": true,
  "failures": [],
  "frames": 18000
}
```

这个结果代表：

- 动作时间达到 10 分钟；
- 18000 帧连续提交且每帧只有一次；
- 推理 p99 满足正式 SLO；
- 推理速度可以覆盖 2.5 秒 hop；
- 播放缓冲覆盖 p99 和 100 ms 余量；
- 输出帧率在 29.7–30.3 FPS；
- jitter p95 满足配置；
- underflow、overload 和输入序列错误均为 0；
- 生命周期以 `finished` 结束，NDJSON 最后一条为 EOS。

---

## 验收完成后保留什么

保留整个 `${RUN_ROOT}`，其中已经包含本次验收的全部证据：

```text
input_motion.pkl
real_fixture.npz
benchmark.json
quality-*.json（仅使用低 steps 候选时）
cfg-equivalence.json（仅使用优化引擎时）
p1-fake/
s1-real-smoke/
p2-viewer/
s2-benchmark-50/
s3-final-10min/
```

同时保留最终的：

```text
configs/v1.cuda.json
```

如需重新验收，从 P0 创建一个新的 `RUN_ROOT`，然后按阶段顺序执行即可。
