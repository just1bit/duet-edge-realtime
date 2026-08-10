# Duet-EDGE Realtime V1 完整验收执行步骤

本文覆盖本地非算力验收、GPU smoke、性能选型、质量/连续性回归、10 分钟稳态、Viewer 和最终归档。任何一项失败都不能宣称 V1 完成。`configs/realtime_v1.json` 中的采样步数和播放延迟在性能选型前只是占位值。

## 0. 验收记录表

开始前新建验收记录，至少填写：日期、执行人、主仓库 commit/dirty、submodule commit/dirty、checkpoint 绝对路径/大小/SHA-256、数据路径、Slurm job id、节点/GPU、最终 steps、最终 playout delay、所有产物路径和每项 Pass/Fail。

## 1. 干净环境与版本检查（Mac + 登录节点）

```bash
git clone --recurse-submodules <realtime-repository-url>
cd duet-edge-realtime
git status --short
git submodule status
git -C third_party/duet-edge status --short
git -C third_party/duet-edge rev-parse HEAD
shasum -a 256 /absolute/path/train-1800.pt
```

预期：两个 worktree 均无输出；submodule 没有 `-`/`+` 前缀；模型引擎 commit 与主仓库 gitlink 一致。记录 checkpoint SHA-256，后续三个作业必须相同。计算节点不能访问 GitHub时，在登录节点提前 clone/update，作业本身不访问网络。

## 2. Mac 非算力自动验收

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
RUN_NETWORK_TESTS=1 python -m unittest tests.test_websocket_integration -v
python scripts/make_fake_fixture.py --frames 54000 --output /tmp/duet-v1-30min.npz
python -m duet_edge_realtime.service \
  --backend fake --input /tmp/duet-v1-30min.npz --clock virtual \
  --sink ndjson --output-dir outputs/local-30min
python scripts/check_run.py \
  --summary outputs/local-30min/summary.json \
  --ndjson outputs/local-30min/stream.ndjson
```

检查 `summary.json`：输入/输出都是 54000 帧；序号错误、overload、NaN/Inf 为 0；推理与输出队列高水位有界；窗口数和范围正确。虚拟时钟不用于判断真实 jitter/FPS。

实际时钟运行 5 分钟（fixture 9000 帧）：

```bash
python scripts/make_fake_fixture.py --frames 9000 --output /tmp/duet-v1-5min.npz
python -m duet_edge_realtime.service \
  --backend fake --input /tmp/duet-v1-5min.npz --clock realtime \
  --sink ndjson,websocket --output-dir outputs/local-5min
python scripts/check_run.py \
  --summary outputs/local-5min/summary.json \
  --ndjson outputs/local-5min/stream.ndjson
```

Mac 调度环境可能受休眠/系统负载影响；记录 jitter 与 underflow，若超门槛先在接电、禁止休眠、低负载条件复跑。GPU 最终验收仍以计算节点结果为准。

## 3. Viewer 功能验收（Mac）

服务运行时另开终端：

```bash
python3 -m http.server 8080 --directory web
```

打开 `http://127.0.0.1:8080`，完成：

1. 连接 `ws://127.0.0.1:8765`，确认骨架 24 关节、Z 向上、三种视角、帧号/动作时间/固定延迟/推理 P95/队列/丢帧状态更新。
2. 播放至少 5 分钟；中途断开再连接，确认服务和 NDJSON 不停，重连先收到 hello 和最新状态。
3. 选择本次 `stream.ndjson` 回放，确认实时与文件路径的骨架连接和坐标方向一致。
4. 模拟后台标签页/慢客户端；允许 `dropped_view_frames` 增长，但 NDJSON frame seq 必须连续。

## 4. GPU 环境与真实 fixture（第一次 Slurm 作业）

服务器安装项目与原 Duet-EDGE CUDA/PyTorch3D 依赖，确认 PyTorch 2.1.x/CUDA 11.8 环境可用。所有计算用 Slurm，不在登录 shell 推理。

```bash
export CHECKPOINT=/absolute/path/train-1800.pt
export AIST_MOTION=/absolute/path/aist_plusplus_final/motions/<sequence>.pkl
export ROOT_SCALED=false
sbatch --export=ALL scripts/slurm/run_realtime_smoke.sh
```

若输入来自 `motions_sliced/`，必须改为 `ROOT_SCALED=true`。记录 job id，作业完成后检查日志和 `outputs/slurm-smoke-<jobid>/`：

- EMA checkpoint 只加载一次，`repr_dim=151`、`horizon=150`。
- CUDA、PyTorch3D、GPU 正常；输出无 NaN/Inf。
- `real-fixture.npz` 包含 normalized lead、generated normalized/unnormalized、FK joints 和元数据。
- summary 中保存引擎 commit/dirty、checkpoint SHA-256、Torch/CUDA/GPU 和峰值显存。

随后执行 opt-in CUDA 测试：

```bash
RUN_CUDA_TESTS=1 \
DUET_EDGE_CHECKPOINT="$CHECKPOINT" \
DUET_EDGE_ROOT=third_party/duet-edge \
python -m unittest tests.test_cuda_smoke -v
```

固定 seed 两次输出应在 `atol/rtol=1e-6` 内一致。

## 5. CFG 快路径数值等价

在同一 checkpoint、同一随机 `x/cond/t` 上，分别计算原三分支公式与快路径：

```text
reference = eps_unc + 0*(eps_music-eps_unc) + 2*(eps_lead-eps_unc)
fast      = guided_forward(..., w_music=0, w_lead=2)
```

记录最大/平均绝对误差，要求 float32 最大误差 `<=1e-6`；同时用 profiler/调用计数确认 reference 为 3 pass、fast 为 2 pass。再覆盖 `(0,0)` 1-pass、`(2,0)` 2-pass、`(2,2)` 3-pass。此项只证明代数与实现等价，不替代质量评估。

## 6. 性能选型（第二次 Slurm 作业）

使用至少 30 窗的真实标准化 fixture；脚本对 150 帧 smoke fixture 默认 `--loop 16`，若换成长 fixture，按实际长度调整到稳态窗口数不少于 30。

```bash
export CHECKPOINT=/absolute/path/train-1800.pt
export FIXTURE=/absolute/path/real-fixture-or-longer-normalized-input.npz
sbatch --export=ALL scripts/slurm/run_realtime_benchmark.sh
```

作业比较 50/25/20/10 steps。每项排除模型加载和前三次 warm-up，记录 wall/CUDA p50/p95/p99、峰值显存、NaN/Inf。若需比较原 3-pass，可在锁定基线引擎上单独跑相同输入并把结果并入报告，不能把两个 dirty worktree 的结果混用。

性能候选必须先满足 `p99 + 100ms < 2.5s`。然后对 50-step 基线和每个候选以相同 3 个代表片段、固定 seed 计算：

- LMA 绝对下降 `<=0.02`。
- PFC 恶化 `<=10%`。
- 边界连续性（根位移、关节最大位移、速度、加速度，单列每 75 帧边界）恶化 `<=10%`。
- 无 NaN/Inf，显存后段无持续上升。

满足全部条件后选 p99 最低者；性能相近时选 steps 更多者。若 50 steps 已达标，优先保留 50。最终设置：

```text
measured_p99 + 0.1s <= playout_delay < 2.5s
```

把选定 `sampling_steps` 和 `playout_delay_s` 回填配置并提交；保存 `benchmark.json`、原始 summary、质量表和选择理由。不得仅依据脚本的 `deadline_candidate` 跳过质量门槛。

## 7. 在线/离线拼接等价与人工连续性

用相同生成 chunks 同时调用实时 `OnlineContinuityProcessor` 和 Duet-EDGE `render.py::_stitch_and_fk()`，比较完整 `[T,24,3]`：目标最大绝对误差 `<=1e-5`；跨设备差异必须解释且上限不得高于 `1e-4`。同时人工观看至少 3 段、每段至少 30 秒，确认无固定 2.5 秒跳变、冻结、回跳或轴翻转，并把视频/NDJSON 名称写入验收记录。

## 8. 真实 10 分钟端到端（第三次 Slurm 作业）

先保证 fixture × `LOOP_COUNT` 至少 10 分钟；循环只重置输入源，不重载模型。

```bash
export CHECKPOINT=/absolute/path/train-1800.pt
export FIXTURE=/absolute/path/long-normalized-input.npz
export SAMPLING_STEPS=<步骤6选定值>
export PLAYOUT_DELAY_S=<步骤6实测值>
export LOOP_COUNT=<保证至少10分钟的整数>
sbatch --export=ALL scripts/slurm/run_realtime_acceptance.sh
```

若允许 SSH 转发，在作业运行期间于 Mac 执行：

```bash
ssh -N -L 8765:127.0.0.1:8765 <login-host>
```

集群若需经计算节点二跳，按管理员要求配置 `ProxyJump`/端口转发。连接 Viewer 至少 5 分钟并断开重连一次。禁止端口转发时，WebSocket 展示不作为阻塞项；复制 NDJSON 回 Mac 用同一 Viewer 回放。

## 9. 最终机器门槛

稳态统计排除加载和前三次 warm-up：

- 单窗推理 `p99 + 100ms < 2.5s`，且 `p99 + 100ms <= playout_delay < 2.5s`。
- 推理队列没有等待旧窗口；10 分钟核心缺帧、underflow、NaN/Inf 均为 0。
- 输出 `30 ± 0.3 FPS`，单帧 jitter p95 `<=20ms`。
- NDJSON seq 从 0 连续到 EOS，每帧恰好 24×3 有限坐标，输出帧数等于输入有效时长。
- 后段 GPU 显存无持续增长；Viewer 丢帧不影响核心记录。
- 正常 EOF flush；模型异常/OOM 产生部分 summary、error 消息和非零退出，不静默 CPU 回退。

重新执行：

```bash
python scripts/check_run.py \
  --summary outputs/slurm-acceptance-<jobid>/summary.json \
  --ndjson outputs/slurm-acceptance-<jobid>/stream.ndjson \
  --duration-min 10 --require-performance
```

脚本通过只是必要条件；还须核对实际持续时间、GPU 显存趋势、质量/连续性和人工观看。

## 10. 最终归档与签字

归档主/引擎 commit、两个 clean 状态、checkpoint SHA-256、三个 Slurm job 日志、真实 fixture、所有 summary、benchmark.json、质量/连续性表、最终 stream.ndjson、Viewer 截图/观看记录和失败时最慢窗口列表。验收报告逐条映射功能、性能、连续性门槛，标记 Pass/Fail/N/A 与证据路径。只有所有强制项 Pass，才将 V1 标为完成；摄像头、实时音乐、模型结构降延迟属于 V2。
