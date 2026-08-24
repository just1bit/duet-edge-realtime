# Duet-EDGE Realtime V2 操作手册

## 1. Environment Setup & Runtime Installation

标准工作区结构如下：

```text
workspace/
├── duet-edge/
├── duet-edge-realtime/
└── data+checkpoint/
    ├── train-1800.pt
    ├── baseline_input/
    ├── smoke_input/
    └── stitched_long_input/
```

Create and activate the environment:

```bash
cd PROJECT_ROOT/duet-edge-realtime
python3.10 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
```

For GPU acceptance, ensure `g++` is available, then install CUDA 12.8 torch, PyTorch3D in order:

```bash
python3 -m pip install 'torch==2.7.0' --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -e '.[gpu]'
python3 -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
```

Reactivate `.venv` in each new terminal. When a runtime check reports a dependency
issue, return to this section and repeat the corresponding installation command.

## 2. Stage 顺序

| Stage | 名称 | 主命令 | 完成结果 |
|---:|---|---|---|
| 01 | Init / Resume | `bash scripts/v2_execution/01_run.sh` | 创建或恢复独立 `RUN_ROOT`。 |
| 02 | Runtime Check & Smoke | `bash scripts/v2_execution/02_runtime_smoke.sh` | 运行时、测试、输入和资产通过检查。 |
| 03 | Baseline & Auto-config | `bash scripts/v2_execution/03_baseline.sh` | 用固定采样步数完成真实时钟基线并自动定稿配置。 |
| 04 | Model Service | `bash scripts/v2_execution/service.sh model start` | 模型加载、预热并进入 Ready。 |
| 05 | Realtime Stream | `bash scripts/v2_execution/service.sh stream start` | 流处理组件进入 Ready，等待输入。 |
| 06 | Viewer Web | `bash scripts/v2_execution/service.sh viewer start` | Viewer 与 WebSocket 进入 Ready，页面显示等待输入。 |
| 07&08 | Input & Run | `bash scripts/v2_execution/service.sh test <path>` | 校验和锁定输入，注入并运行。 |
| 09 | Verify & Report | `bash scripts/v2_execution/09_verify_report.sh` | 生成验收结果和报告。 |
| 10 | Export Fixture（可选） | `bash scripts/v2_execution/10_export_fixture.sh` | 导出回归和 Recorded 回放数据。 |

Stage 01 将当前运行路径写入 `outputs/.run-current`，后续命令默认使用该运行。切换已有运行时，
在支持的命令后添加 `--run outputs/run-...`；Stage 01 使用
`--resume outputs/run-...` 恢复当前指针。

## 3. 各 Stage 操作

### Stage 01 — Init / Resume

```bash
bash scripts/v2_execution/01_run.sh
```

该命令从 `configs/example.json` 创建带时间戳的 `RUN_ROOT`，解析运行路径，记录代码版本、
机器信息、checkpoint 和初始输入身份，并创建：

- `config.json`：Stage 03 定稿前的运行配置；
- `run-metadata.json`：机器、Python、仓库状态和资产身份；
- `calibration.json`：初始状态为 `pending-baseline`；
- `logs/`、`evidence/`、`fixtures/`。

继续条件：终端打印 `Active run selected` 和 `Stage 01 SUCCESS`；终端已打印选中的绝对 `RUN_ROOT`，且该目录中存在
`config.json`。

恢复已有运行：

```bash
bash scripts/v2_execution/01_run.sh --resume outputs/run-...
```

### Stage 02 — Runtime Check & Smoke

```bash
bash scripts/v2_execution/02_runtime_smoke.sh
```

该命令运行完整 pytest，检查当前输入结构，核对三套数据清单及运行资产哈希。CUDA 后端还会
使用 smoke 输入执行 5-step 短推理。输入检查结果写入 `evidence/input-check.json`，不会锁定
正式输入，也不会修改配置。

继续条件：测试、输入、测试资产、后端 smoke 和配置资产检查依次完成，
最后显示 `Runtime environment ready` 和 `Stage 02 SUCCESS`；同时终端已打印
`Runtime, tests, input, and asset hashes are ready.`。

### Stage 03 — Baseline & Auto-config

```bash
bash scripts/v2_execution/03_baseline.sh
```

该命令使用 `config.json` 中固定的默认 `sampling_steps` 和真实时钟运行基线。CUDA 默认先用
单轮 12 秒输入建立动作质量基线，再运行 5 轮计时基线，以获得足够的正式推理窗口且不把循环
边界计入动作阈值；可通过 `V2_BASELINE_LOOPS` 调整计时轮数。

基线自动完成以下工作：

1. 记录真实推理 p99 和动作质量统计；
2. 根据固定余量计算 `inference_slo_ms`；
3. 根据 SLO、安全余量和默认下限计算 `playout_delay_s`；
4. 验证推理预算可落入 2.5 秒窗口步长；
5. 自动写回 `config.json`，不改变默认采样步数；
6. 生成 `config.sha256`，从此锁定运行参数。

产物位于：

```text
calibration.json
config.sha256
evidence/baseline-runs/baseline/summary.json
evidence/baseline-runs/baseline/stream.ndjson
evidence/baseline-runs/timing-baseline/summary.json
```

继续条件：终端最后显示 `Configuration calibrated and locked` 和
`Stage 03 SUCCESS`；`calibration.json.status` 为 `finalized`，并且 `config.sha256` 存在。

### Stage 04 — Model Service

```bash
bash scripts/v2_execution/service.sh model start
```

该命令启动一个常驻 Runtime 进程。CUDA 后端在此阶段创建 EDGE 模型、加载 checkpoint、执行
warmup，并保持唯一模型实例驻留。正式输入尚未创建或读取。

Runtime PID 写入 `runtime.pid`，日志写入 `logs/runtime.log`，模型证据写入
`evidence/model-service.json`。

继续条件：warmup 的窗口与 sampling-step 两层真实进度完成并显示 `ready`，
随后终端显示 `Model service ready` 和 `Stage 04 SUCCESS · Model Service · start`；
`runtime.pid` 对应的进程存活，且运行时状态中 `model.state` 为 `ready`。如需复核，运行
`bash scripts/v2_execution/service.sh status`，或检查 `evidence/model-service.json.status` 为 `ready`。

### Stage 05 — Realtime Stream

```bash
bash scripts/v2_execution/service.sh stream start
```

该命令激活流处理组件，确认推理队列、播放队列和策略已经按定稿配置就绪。此时不会启动输入、
推理或播放任务。

证据写入 `evidence/stream-service.json`。

继续条件：`Preparing realtime stream service` 显示 `ready`，随后终端显示 `Realtime stream service ready` 和
`Stage 05 SUCCESS · Realtime Stream Service · start`；运行时状态中 `stream.state` 为 `ready`，
`session.state` 仍为 `idle`，且 `evidence/stream-service.json.status` 为 `ready`。可用
`bash scripts/v2_execution/service.sh status` 复核。

### Stage 06 — Viewer Web

```bash
bash scripts/v2_execution/service.sh viewer start
```

该命令启动 Viewer 静态页面和 WebSocket 服务。默认 Viewer 地址为
`http://127.0.0.1:8080`。浏览器可以在正式运行前连接，页面状态显示
`Waiting for input`。

证据写入 `evidence/viewer-service.json`。

继续条件：`Starting Viewer Web` 显示 `ready`，终端打印
`Viewer ready and waiting for input: ...`，随后显示 `Viewer URL generated` 和
`Stage 06 SUCCESS · Viewer Web · start`；运行时状态中 `viewer.state` 为 `ready`，浏览器能够打开
Viewer、连接 WebSocket，并显示 `Waiting for input`；`evidence/viewer-service.json.status` 为
`ready`，其 URL 与终端输出一致。

### Stage 07&08 — Input & Run

```bash
bash scripts/v2_execution/service.sh test <path>
```

测试长输入：

```bash
bash scripts/v2_execution/service.sh test '../data+checkpoint/stitched_long_input/stitched_long_input.pkl'
```

首先确认 Model、Stream 和 Viewer 全部 Ready，并确认当前没有正在执行的测试；然后校验输入，计算 SHA-256、帧数、时长、格式和时间线身份，生成并锁定 `input-manifest.json`。省略 `path` 时使用 `config.json` 中 `paths.input_motion` 指定的默认输入。

CUDA AIST 输入需要对齐的 `pos`/`q` 序列，至少包含 300 个 60 FPS 原始帧；Recorded/Fake 输入需要带 `motion_151` 的 `.npz`。

输入清单记录 `run_id`、配置哈希、输入哈希和锁定时间。证据副本写入 `evidence/input.json`。

锁定input后，向 Runtime 提交正式session。Runtime 在此时才创建输入适配器，并依次进入`buffering → playing → draining → finished`。

脚本持续等待正式输入处理完毕。同 `RUN_ROOT` 可以串行执行多次 `test`。新输入校验并锁定成功后，脚本会覆盖上次结果，每次只保留最新结果；如果新输入校验失败，则保留上次结果。

继续条件：窗口与 sampling-step 两层真实进度完成后，playout 按已输出帧数到达 100% 并显示 `ready`；终端打印 `Formal run completed: ...`、`Run evidence written` 和 `Stage 08 SUCCESS`；运行时 `session.state` 为 `finished`，且 `summary.json` 和 `stream.ndjson` 均存在。

### Stage 09 — Verify & Report

```bash
bash scripts/v2_execution/09_verify_report.sh
```

该命令从锁定输入、完整事件流和汇总数据验证协议、生命周期、帧数、帧序、时序、handoff、
动作质量、资源状态和 Viewer 指标。

结果写入 `gate-results.json` 和 `report.md`。长输入验收使用：

```bash
bash scripts/v2_execution/09_verify_report.sh --long-input
```

不加参数时执行常规验收，检查协议、生命周期、帧数与帧序、时序、handoff、动作质量、
资源状态和 Viewer 指标。加 `--long-input` 时会执行全部常规检查，并额外要求：输出不少于
18,000 帧且时长不少于 600 秒、后端为 CUDA、Viewer 至少连接过一次、Viewer 零丢帧，
以及浏览器零可见卡顿。该参数只验收 Stage 08 已生成的结果，不会自动选择或运行长输入。

继续条件：运行证据定位和自动验收门检查完成，最后显示
`Acceptance report generated` 和 `Stage 09 SUCCESS`；`gate-results.json.passed` 为
`true`，且 `report.md` 中的适用验收项全部通过。若门检失败，脚本会以非零状态退出并打印
`Stage 09 FAILED`，不得以已生成报告作为通过依据。

### Stage 10 — Export Fixture（可选）

```bash
bash scripts/v2_execution/10_export_fixture.sh
```

该命令先有序停止常驻 Runtime，释放模型和 GPU，再使用锁定输入导出三窗口 fixture 与
Recorded 回放数据：

```text
fixtures/fixture.npz
fixtures/recorded_fixture.npz
```

它们用于回归测试、离线复现和无 CUDA 的完整流式彩排。

完成条件：常驻 Runtime 停止、模型与输入身份解析和 fixture 导出均完成，
最后显示 `Exported files verified` 和 `Stage 10 SUCCESS`；
`fixtures/fixture.npz` 与 `fixtures/recorded_fixture.npz` 均存在。这是可选的最后阶段，
完成后无需再进入下一 Stage。

## 4. 状态和日常控制

查看整个 Runtime 状态：

```bash
bash scripts/v2_execution/service.sh status
```

状态包含 Model、Stream、Viewer 和正式 session 四部分。停止 Runtime：

```bash
bash scripts/v2_execution/service.sh stop
```

一次完成的运行包含：

```text
outputs/run-.../
├── config.json
├── config.sha256
├── calibration.json
├── input-manifest.json
├── run-metadata.json
├── runtime.pid
├── effective_config.json
├── stream.ndjson
├── summary.json
├── gate-results.json
├── report.md
├── logs/
│   ├── runtime.log
│   └── stage-XX.log
├── evidence/
└── fixtures/
```

每个 Stage 的完整 stdout/stderr 保存在对应的 `logs/stage-XX.log`；再次执行同一 Stage 时覆盖
该 Stage 的旧日志。常驻服务的历次
启动输出追加到 `logs/runtime.log`，不会因重启覆盖。服务状态保存在
`evidence/runtime-status.json`，协议内容位于 `stream.ndjson`，最终验收结论位于
`gate-results.json` 和 `report.md`。

## 5. 运行后端

仓库只维护一个规范模板 `configs/example.json`。Stage 01 可通过
`--template /path/to/config.json` 使用自定义模板。

| 后端 | `backend` | 输入 | 用途 |
|---|---|---|---|
| CUDA | `cuda` | AIST `.pkl`，包含 `pos` 和 `q` | 模型质量、性能和长输入验收 |
| Recorded | `recorded` | 带 `motion_151` 的 V2 `.npz` | 重放已捕获推理结果并验证流式链路 |
| Fake | `fake` | 带 `motion_151` 的测试 `.npz` | 确定性开发和自动化验证 |

三个后端共享相同的 Stage 顺序、协议、播放、证据和 Viewer 流程。
