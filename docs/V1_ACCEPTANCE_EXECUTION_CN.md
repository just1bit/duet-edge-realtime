# Duet-EDGE Realtime V1 单机验收手册

这份手册用于实验室的一台 NVIDIA GPU 电脑。全部命令直接在这台电脑上运行。

验收目标是确认：模拟链路正常、真实模型可以连续推理、浏览器可以展示、推理速度满足 2.5 秒步长，最后稳定运行 10 分钟。

## 本次实验室机器基线（2026-08-14 preflight）

本手册下面的路径和机器检查已经按
`acceptance-preflight-20260814-134121.txt` 更新。后续若仓库、checkpoint、驱动或环境发生
变化，应重新运行 `scripts/collect_acceptance_preflight.sh`，不能继续引用本节结果。

| 项目 | 本机实测值 | 验收状态 |
|---|---|---|
| 主机 / 系统 | `robot-System-Product-Name`，Ubuntu 24.04.4 LTS x86_64 | 已确认 |
| CPU / 内存 | Ryzen 9 9900X3D，12 核 24 线程；123 GiB RAM，8 GiB swap | 充足 |
| GPU | NVIDIA GeForce RTX 5090，32607 MiB；compute capability 应为 12.0 | 硬件已确认，仍须由 PyTorch 实算确认 |
| NVIDIA 驱动 | 595.84；`nvidia-smi` 显示 CUDA 13.2 | 驱动已确认；这不是 PyTorch 自带 CUDA runtime 的版本 |
| 磁盘 | 根分区约 3.4 TiB 可用 | 充足 |
| 时间 | Pacific/Auckland；NTP 已同步 | 已确认 |
| 端口 | 8080、8765、18765 当时均空闲 | 每次运行前仍须复查 |
| 浏览器 | Firefox `/usr/bin/firefox` | 已确认 |
| GPU 占用 | 仅桌面图形进程，约 614 MiB；无 compute 进程 | 基线良好；正式验收时保持无其他计算任务 |
| Python / CUDA 推理环境 | 尚未运行兼容性脚本 | **先完成 C0，成功后再继续** |

### 当前仍待关闭的问题

当前只先处理 C0 兼容性门。C0 脚本会安装固定依赖，并验证 Python、PyTorch、CUDA、RTX
5090、PyTorch3D、Duet-EDGE 导入和 checkpoint 加载。脚本未显示 `COMPATIBILITY GATE:
PASSED` 时，不执行 P0 或任何后续验收。

本次验收固定输入如下：

```text
REALTIME_ROOT=/home/robot/data/ethan/duet-edge-realtime-v1/duet-edge-realtime
DUET_EDGE_ROOT=/home/robot/data/ethan/duet-edge-realtime-v1/duet-edge
EDGE_CHECKPOINT=/home/robot/data/ethan/duet-edge-realtime-v1/data+checkpoint/train-1800.pt
AIST_RAW=/home/robot/data/ethan/duet-edge-realtime-v1/data+checkpoint/aist_plusplus_final/motions/gKR_sBM_cAll_d28_mKR2_ch06.pkl
```

用于证据核对的版本和哈希：

```text
duet-edge-realtime commit: 01fe13f72acbde637d975efc6d0e36520f7ec3c6
duet-edge commit:          0556b1a0d6e2c7546a81c6155eb0fab77e7bb0bb
checkpoint SHA256:         2c948e74400ba78dbec469880746a78dfbb10ed56597917ba2e406cfeb8f9e15
AIST_RAW SHA256:            54968522c7a41457a91d5c916e0f504db27c1afd696ca30bd2bc98dc12294e21
```

preflight 中 `duet-edge` 工作树干净；`duet-edge-realtime` 只有生成的 preflight 报告未跟踪。
开始验收前再次运行 `git status --short`，并把任何新的代码或配置差异记入验收记录。

## 最短开始方式

三个新增脚本的使用关系：

| 脚本 | 是否手动运行 | 用途 |
|---|---|---|
| `setup_acceptance_runtime.sh` | 是，首次运行 C0 | 安装依赖并完成兼容性门 |
| `verify_acceptance_runtime.py` | 否 | 由上一个脚本自动调用，执行只读兼容性检查 |
| `start_acceptance_run.sh` | 是，C0 通过后运行 P0 | 创建本次验收目录并准备输入 |

实际开始时只需要按顺序执行：

```bash
cd /home/robot/data/ethan/duet-edge-realtime-v1/duet-edge-realtime
bash scripts/setup_acceptance_runtime.sh
source scripts/start_acceptance_run.sh
```

第二条必须先显示 `COMPATIBILITY GATE: PASSED`，才能执行第三条。第三条必须使用 `source`。
后面的 P1–S3 按各章节继续。

## 阶段计划

| 阶段 | 输入 | 要做的事 | 输出 | 通过标准 |
|---|---|---|---|---|
| C0：兼容性门 | 当前 Python 环境、RTX 5090、checkpoint | 脚本安装依赖并自动验证 | 环境证据 | 显示 `COMPATIBILITY GATE: PASSED` |
| P0：准备验收输入 | 两个仓库、checkpoint、AIST++ 动作 | 设置路径、复采 preflight、转换测试动作 | post-env preflight、`input_motion.pkl` | 路径、哈希和输入转换通过 |
| P1：测试流式核心 | 仓库自带 fake fixture | 跑自动测试和模拟全链路 | fake NDJSON、summary | 测试通过，检查结果 `passed: true` |
| S1：接入真实模型 | checkpoint、`input_motion.pkl` | 运行真实 CUDA 推理并导出 fixture | real NDJSON、`real_fixture.npz` | 帧连续、坐标有效、运行正常结束 |
| P2：检查 Viewer | fake fixture | 启动服务和网页 | 浏览器骨架动画 | 动画、状态、重连和文件回放正常 |
| S2：测量性能 | `real_fixture.npz` | 测量 50-step 推理 p99 | `benchmark.json` | `deadline_candidate: true` |
| P3：写入正式配置 | benchmark 结果 | 更新 steps、播放缓冲和推理 SLO | `configs/v1.cuda.json` | 配置值来自实测结果 |
| S3：最终验收 | 正式配置、real fixture | 按真实时钟运行 10 分钟 | final NDJSON、summary、资源趋势 | 自动指标和人工检查全部通过 |

阶段按表格顺序执行。每一阶段完成后再进入下一阶段。

---

## C0：先解决 PyTorch / CUDA 兼容性

`setup_acceptance_runtime.sh` 按 `acceptance-runtime-requirements.txt` 安装依赖，然后自动调用
只读验证脚本和真实 CUDA smoke test。

### 操作 1：运行安装和验证脚本

```bash
cd /home/robot/data/ethan/duet-edge-realtime-v1/duet-edge-realtime
bash scripts/setup_acceptance_runtime.sh
```

脚本会检查 Python 3.10、PyTorch 2.7/CUDA 12.8、RTX 5090、PyTorch3D、checkpoint、依赖
冲突和真实单窗口推理。当前链路不需要 `torchvision` 或 `torchaudio`。

成功时最后必须显示：

```text
COMPATIBILITY GATE: PASSED
You may now continue with P0 in the acceptance guide.
```

只有看到这两行才进入 P0。如果脚本失败，保留从 `[1/7]` 开始的完整终端输出，先解决该错误；
不要跳到 P1、S1 或性能测试。

脚本不会自动运行需要管理员权限的系统安装。如果它报告缺少 `g++`，只需让管理员执行：

```bash
sudo apt update
sudo apt install build-essential
```

然后重新运行安装脚本。

安装脚本保存的环境证据位于：

```text
outputs/environment-evidence/pip-freeze.txt
outputs/environment-evidence/torch-environment.txt
```

---

## P0：准备验收输入

在刚才运行 C0 的同一终端执行：

```bash
source scripts/start_acceptance_run.sh
```

脚本会设置本次运行路径、核对仓库/checkpoint/AIST 文件及 SHA256、复采 preflight，并转换测试
动作。成功时显示：

```text
P0 PREPARATION: PASSED
RUN_ROOT=/home/robot/data/ethan/duet-edge-realtime-v1/duet-edge-realtime/outputs/acceptance-<时间戳>
```

必须使用 `source`，这样后续步骤才能继续使用 `${RUN_ROOT}` 等路径变量。

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
  --require-backend cuda \
  --min-inference-samples 2
```

### 操作 3：导出真实 fixture

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
- C0 的 CUDA 单窗口确定性测试已经通过；
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

本机已确认安装 Firefox，也可以在图形桌面终端执行：

```bash
firefox http://127.0.0.1:8080
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
   地面轴错误、持续漂移或窗口边界跳变；记录观察结论、checkpoint SHA256 和有效配置。

把五项结果、checkpoint SHA256 和有效配置路径写入
`${RUN_ROOT}/acceptance-observations.md`。检查完成后等待终端 A 正常结束，并在终端 B
按 `Ctrl+C` 关闭网页服务。

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

- `deadline_candidate: true` 表示推理 p99 加配置中的 `safety_margin_ms` 后仍小于 2.5 秒，可以持续跟上输入；
- `recommended_playout_delay_s` 是正式配置建议使用的播放缓冲。

50-step 达标后直接进入 P3。低-step 调优不属于本次 V1 主线验收。

---

## P3：写入正式配置

### 操作

```bash
python scripts/update_runtime_config.py \
  --config configs/v1.cuda.json \
  --benchmark "${RUN_ROOT}/benchmark.json" \
  --sampling-steps 50
python -m json.tool configs/v1.cuda.json
```

脚本会校验 50-step 基准的 deadline、安全余量和推荐播放延迟，然后更新：

- `model.sampling_steps`；
- `stream.playout_delay_s`；
- `stream.inference_slo_ms`。

`inference_slo_ms` 等于播放缓冲和 2.5 秒 hop 周期中的较小值，再减去
`stream.safety_margin_ms`。基准汇总、配置更新、运行摘要和最终检查共同读取这一配置项。

### 输出与预期结果

输出文件：

```text
configs/v1.cuda.json
```

终端会打印 benchmark、quality（如适用）、最终 steps、播放缓冲、推理 SLO 和安全余量。
确认这些值与 S2 选择一致，P3 即完成。

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

### 操作 2：记录 GPU 资源趋势

服务运行期间，在终端 C 每 5 秒记录一次 GPU 利用率、显存、功耗和温度：

```bash
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv \
  -l 5 > "${RUN_ROOT}/s3-gpu-resources.csv"
```

终端 A 的服务结束后，在终端 C 按 `Ctrl+C` 停止采样。检查 CSV 覆盖完整运行时段，
并把显存趋势和 Viewer 观察结论写入 `${RUN_ROOT}/acceptance-observations.md`。

### 操作 3：自动检查最终结果

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

这个自动检查结果代表：

- 动作时间达到 10 分钟；
- 18000 帧连续提交且每帧只有一次；
- 推理 p99 满足正式 SLO；
- 推理速度可以覆盖 2.5 秒 hop；
- 播放缓冲覆盖 p99 和配置的安全余量；
- 输出帧率在 29.7–30.3 FPS；
- jitter p95 满足配置；
- underflow、overload 和输入序列错误均为 0；
- 生命周期以 `finished` 结束，NDJSON 最后一条为 EOS。

最终验收还包括：

- P2 的真实动作方向、漂移、自然度和窗口边界观察记录；
- `s3-gpu-resources.csv` 覆盖完整运行时段，显存趋势稳定；
- `acceptance-observations.md` 记录 Viewer 丢帧、资源趋势和人工检查结论。

---

## 验收完成后保留什么

保留整个 `${RUN_ROOT}`，其中已经包含本次验收的全部证据：

```text
input_motion.pkl
real_fixture.npz
benchmark.json
acceptance-observations.md
s3-gpu-resources.csv
p1-fake/
s1-real-smoke/
p2-viewer/
s2-benchmark-50/
s3-final-10min/
```

同时保留最终的：

```text
configs/v1.cuda.json
outputs/environment-evidence/pip-freeze.txt
outputs/environment-evidence/torch-environment.txt
```

如需重新验收，从 P0 创建一个新的 `RUN_ROOT`，然后按阶段顺序执行即可。
