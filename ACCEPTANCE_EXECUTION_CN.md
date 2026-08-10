# Duet-EDGE Realtime V1 验收操作手册

这不是检查项列表，而是一份从零开始的操作手册。请严格按章节顺序执行。除非某一步明确写着“可以跳过”，否则前一步没有通过时不要继续。

## 0. 先看懂整个流程

本次操作会用到两台机器：

- **Mac**：运行不需要 CUDA 的测试、查看结果、提交代码。
- **GPU 服务器**：准备模型环境和数据，通过 Slurm 运行原始 Duet-EDGE、实时 smoke、GPU benchmark 和最终 10 分钟验收。

服务器上有两个并列且彼此独立的 Git 仓库：

```text
/data/<你的用户名>/duet-v1/
├── duet-edge/             # 原模型仓库
└── duet-edge-realtime/    # 实时服务仓库
```

实时仓库不会复制模型源码，也没有 `third_party` 或 Git submodule。运行时通过 `DUET_EDGE_ROOT` 使用旁边的模型仓库。

必须按下面的顺序执行：

```text
Mac 非 GPU 自测
  ↓
服务器环境和数据准备
  ↓
原始 Duet-EDGE 独立 smoke（证明模型本身能运行）
  ↓
Realtime GPU smoke（生成统一 benchmark fixture）
  ↓
原模型、50-step 基线 benchmark
  ├─ 达标 → 不修改模型 → 回填 50 steps 和实测 delay
  └─ 不达标 → 应用模型快路径 → 更新 lock → 复跑原模型
                → CFG 等价测试 → 跑 50/25/20/10 → 质量比较 → 回填结果
  ↓
10 分钟真实墙钟验收
  ↓
归档全部证据
```

其中最重要的决策门是：

```text
推理 p99 + 100 ms < 2500 ms
```

只有原始 50-step 不满足这个条件，才进入模型优化路线。不能在 GPU 基线测量前凭经验修改 steps。

---

## 1. Mac：完成不需要 GPU 的自测

### 1.1 打开终端并进入实时仓库

以下命令中的路径请替换成你自己的实际路径：

```bash
cd "/你的路径/duet-edge-realtime"
pwd
```

`pwd` 的最后一段必须是 `duet-edge-realtime`。

### 1.2 创建 Python 3.10 虚拟环境

先确认 Python 版本：

```bash
python3.10 --version
```

预期看到 `Python 3.10.x`。如果提示 command not found，先安装 Python 3.10，再继续。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

以后每次新开 Mac 终端，都要先执行：

```bash
cd "/你的路径/duet-edge-realtime"
source .venv/bin/activate
```

### 1.3 运行自动测试

```bash
pytest
RUN_NETWORK_TESTS=1 python -m unittest tests.test_websocket_integration -v
```

通过标准：

- `pytest` 没有 `failed` 或 `error`；CUDA 测试在 Mac 上显示 skipped 是正常的。
- WebSocket 测试最后显示 `OK`。

任何一条失败都先停止。保存完整终端输出并修复，不能带着失败结果进入 GPU 阶段。

### 1.4 运行 30 分钟虚拟时钟稳定性测试

虚拟时钟不会真的等待 30 分钟，只验证 30 分钟数据量的完整性。

```bash
export LOCAL_TAG=$(date +%Y%m%d-%H%M%S)
export LOCAL_OUTPUT="/tmp/duet-v1-${LOCAL_TAG}"
python scripts/make_fake_fixture.py \
  --frames 54000 \
  --output "/tmp/duet-v1-30min-${LOCAL_TAG}.npz"
python -m duet_edge_realtime.service \
  --config configs/v1.fake.json \
  --input "/tmp/duet-v1-30min-${LOCAL_TAG}.npz" \
  --output-dir "${LOCAL_OUTPUT}" \
  --run-id local-30min \
  --clock virtual \
  --sink ndjson
python scripts/check_run.py \
  --summary "${LOCAL_OUTPUT}/local-30min/summary.json" \
  --ndjson "${LOCAL_OUTPUT}/local-30min/stream.ndjson" \
  --duration-min 30
```

最后必须看到：

```json
"passed": true
```

并检查帧数：

```bash
python -m json.tool "${LOCAL_OUTPUT}/local-30min/summary.json"
```

`output.frames` 应为 `54000`，`queues.overloads` 和错误数必须为 `0`。

### 1.5 运行 5 分钟真实墙钟测试并打开 Viewer

这一步真的需要约 5 分钟。保留当前终端，先准备输入：

```bash
python scripts/make_fake_fixture.py \
  --frames 9000 \
  --output "/tmp/duet-v1-5min-${LOCAL_TAG}.npz"
```

在当前终端启动服务：

```bash
python -m duet_edge_realtime.service \
  --config configs/v1.fake.json \
  --input "/tmp/duet-v1-5min-${LOCAL_TAG}.npz" \
  --output-dir "${LOCAL_OUTPUT}" \
  --run-id local-5min \
  --clock realtime \
  --sink ndjson,websocket
```

服务运行期间，新开第二个 Mac 终端：

```bash
cd "/你的路径/duet-edge-realtime"
python3 -m http.server 8080 --directory web
```

浏览器打开 `http://127.0.0.1:8080`，连接地址填写 `ws://127.0.0.1:8765`。确认骨架在运动；断开一次再重新连接，服务不能因此退出。

第一个终端运行结束后执行：

```bash
source .venv/bin/activate
python scripts/check_run.py \
  --summary "${LOCAL_OUTPUT}/local-5min/summary.json" \
  --ndjson "${LOCAL_OUTPUT}/local-5min/stream.ndjson" \
  --duration-min 5 \
  --require-performance
```

必须看到 `"passed": true`。此时 Mac 非算力自测完成。

---

## 2. GPU 服务器：准备目录和固定变量

### 2.1 登录服务器并准备两个独立仓库

从 Mac 登录服务器：

```bash
ssh <你的用户名>@<登录节点地址>
```

在服务器登录节点执行：

```bash
mkdir -p /data/<你的用户名>/duet-v1
cd /data/<你的用户名>/duet-v1
git clone https://github.com/just1bit/duet-edge.git
git clone https://github.com/just1bit/duet-edge-realtime.git
```

如果目录已经存在，不要重复 clone，改为：

```bash
git -C /data/<你的用户名>/duet-v1/duet-edge fetch origin
git -C /data/<你的用户名>/duet-v1/duet-edge-realtime fetch origin
```

### 2.2 每次登录服务器都先设置这些变量

把下面所有尖括号内容替换成真实值。不要原样保留 `<...>`。

```bash
export WORK_ROOT=/data/<你的用户名>/duet-v1
export DUET_EDGE_ROOT=${WORK_ROOT}/duet-edge
export REALTIME_ROOT=${WORK_ROOT}/duet-edge-realtime
export EDGE_ENV_INIT=/data/<你的用户名>/init_env.sh
export EDGE_CONDA_ENV=edge
export EDGE_CHECKPOINT=/data/<你的用户名>/checkpoints/train-1800.pt
export AIST_ROOT=/data/<你的用户名>/datasets/aist_plusplus_final
export EDGE_OUTPUT_DIR=/data/<你的用户名>/duet-v1-runs
mkdir -p "${EDGE_OUTPUT_DIR}"
```

说明：

- `EDGE_ENV_INIT` 是服务器原有的环境初始化脚本。如果服务器不需要它，可创建一个只包含 conda 初始化的脚本，但这个文件必须能被 `source`。
- `EDGE_CHECKPOINT` 必须是 duet 模型的 `train-1800.pt`，不是 EDGE solo checkpoint。
- `AIST_ROOT` 目录下面必须直接有 `motions/` 和 `wavs/`。
- 运行产物放在 `EDGE_OUTPUT_DIR`，不要放入任何 Git 仓库。

立即验证路径：

```bash
test -d "${DUET_EDGE_ROOT}" && echo "OK duet-edge"
test -d "${REALTIME_ROOT}" && echo "OK realtime"
test -f "${EDGE_ENV_INIT}" && echo "OK env init"
test -f "${EDGE_CHECKPOINT}" && echo "OK checkpoint"
test -d "${AIST_ROOT}/motions" && echo "OK AIST motions"
test -d "${AIST_ROOT}/wavs" && echo "OK AIST wavs"
```

六行都应以 `OK` 开头。如果 checkpoint 或 AIST++ 还没上传，请先上传或解压，再继续。例如 AIST++ 是 `.7z` 时：

```bash
mkdir -p "${AIST_ROOT}"
7z x /data/<你的用户名>/datasets/aist_plusplus_final.7z -o"${AIST_ROOT}"
```

解压后如果变成了多套一层目录，例如 `${AIST_ROOT}/aist_plusplus_final/motions`，应把 `AIST_ROOT` 改为真正直接包含 `motions` 和 `wavs` 的那一层。

### 2.3 验证仓库确实是并列外部依赖结构

```bash
cd "${REALTIME_ROOT}"
test ! -e .gitmodules && echo "OK no submodule"
test ! -e third_party/duet-edge && echo "OK no vendored model"
cat compat/duet-edge.lock.json
git -C "${DUET_EDGE_ROOT}" rev-parse HEAD
git -C "${DUET_EDGE_ROOT}" status --short -- '*.py'
```

要求：

- lock 文件中的 `commit` 与 `git rev-parse HEAD` 完全相同。
- 最后一条命令不能输出任何内容；输出内容代表模型 Python 源码有未提交修改。

如果 commit 不同，先切回 lock 指定版本：

```bash
export BASELINE_COMMIT=$(python -c 'import json; print(json.load(open("compat/duet-edge.lock.json"))["commit"])')
git -C "${DUET_EDGE_ROOT}" switch --detach "${BASELINE_COMMIT}"
```

记录版本和 checkpoint 摘要：

```bash
git -C "${REALTIME_ROOT}" rev-parse HEAD
git -C "${DUET_EDGE_ROOT}" rev-parse HEAD
sha256sum "${EDGE_CHECKPOINT}"
```

把这三行保存到验收记录中。

---

## 3. GPU 服务器：安装并验证运行环境

这部分在登录节点执行，不占用 GPU；如果单位规定不能在登录节点安装软件，请改用单位允许的交互/安装节点。

### 3.1 优先复用现有 `edge` conda 环境

```bash
source "${EDGE_ENV_INIT}"
conda activate "${EDGE_CONDA_ENV}"
python --version
python -c 'import torch, pytorch3d, jukemirlib, accelerate; print("torch", torch.__version__); print("imports OK")'
```

预期 Python 为 3.10，最后输出 `imports OK`。

### 3.2 仅在环境不存在时创建环境

如果上一节 `conda activate edge` 或 import 失败，才执行：

```bash
source "${EDGE_ENV_INIT}"
cd "${DUET_EDGE_ROOT}"
bash setup_env.sh
conda activate edge
python -c 'import torch, pytorch3d, jukemirlib, accelerate; print("torch", torch.__version__); print("imports OK")'
```

`setup_env.sh` 会下载较大的依赖，可能需要数分钟到数十分钟。网络或磁盘问题必须在这里解决。

### 3.3 把实时服务安装进同一个环境

```bash
cd "${REALTIME_ROOT}"
python -m pip install -e .
python -c 'import duet_edge_realtime; print("realtime import OK")'
```

必须输出 `realtime import OK`。

---

## 4. GPU 服务器：准备两类输入数据

这里很容易混淆：实时 smoke 和原模型独立 smoke 需要的数据不同。

- **实时 smoke** 只需要一个 AIST++ motion pickle，不需要音乐。
- **原模型独立 smoke** 需要模型仓库中已经切片的 val motion、wav 和 Jukebox 特征。

两类都要准备。

### 4.1 准备实时 smoke 的单个 motion

在登录节点执行：

```bash
cd "${REALTIME_ROOT}"
export RAW_MOTION=$(find "${AIST_ROOT}/motions" -type f -name '*.pkl' | sort | head -n 1)
echo "${RAW_MOTION}"
test -f "${RAW_MOTION}" && echo "OK raw motion"
python scripts/prepare_aist_motion.py \
  --input "${RAW_MOTION}" \
  --output "${EDGE_OUTPUT_DIR}/inputs/lead.pkl"
export EDGE_INPUT_MOTION=${EDGE_OUTPUT_DIR}/inputs/lead.pkl
export ROOT_SCALED=false
```

预期最后看到 `root_scaled=false`。这个转换输出来自原始 motion，所以后面必须一直使用 `ROOT_SCALED=false`。

### 4.2 检查原模型 val 数据是否已经存在

```bash
test -f "${DUET_EDGE_ROOT}/data/splits/duet_pairs_val.json" && echo "OK pairs"
find "${DUET_EDGE_ROOT}/data/val/motions_sliced" -type f -name '*.pkl' 2>/dev/null | head
find "${DUET_EDGE_ROOT}/data/val/jukebox_feats" -type f -name '*.npy' 2>/dev/null | head
find "${DUET_EDGE_ROOT}/data/val/wavs_sliced" -type f -name '*.wav' 2>/dev/null | head
```

如果三个 `find` 都至少显示一个文件，跳到第 5 章。

如果任一目录为空，就必须先提交数据预处理作业。Jukebox 特征提取很慢，并且需要 GPU：

```bash
cd "${REALTIME_ROOT}"
mkdir -p logs
export DATA_JOB_ID=$(sbatch --parsable --export=ALL scripts/slurm/prepare_model_data.sh)
echo "DATA_JOB_ID=${DATA_JOB_ID}"
```

查看排队状态：

```bash
squeue -j "${DATA_JOB_ID}"
```

作业开始后查看日志：

```bash
tail -f "logs/edge-v1-data-${DATA_JOB_ID}.out"
```

按 `Ctrl-C` 只会退出日志查看，不会取消作业。检查最终状态：

```bash
sacct -j "${DATA_JOB_ID}" --format=JobID,State,Elapsed,ExitCode
```

主作业必须是 `COMPLETED` 且 `ExitCode` 为 `0:0`。然后重新执行本节开头的三个 `find`。如果仍有目录为空，停止并检查日志，不能运行原模型 smoke。

---

## 5. 第一次 GPU 作业：先独立运行原始 Duet-EDGE

这一章不经过实时服务，目的是证明 checkpoint、环境和模型原有推理流程本身可用。不要用仓库中的 `run_smoke_test.sh` 代替；那个脚本实际会做训练 smoke，而且带有旧硬编码路径。

重新确认变量还在当前 shell：

```bash
echo "${DUET_EDGE_ROOT}"
echo "${EDGE_CHECKPOINT}"
echo "${EDGE_OUTPUT_DIR}"
echo "${EDGE_ENV_INIT}"
```

若其中任何一行为空，回到 2.2 重新 export。

提交作业：

```bash
cd "${REALTIME_ROOT}"
mkdir -p logs
export ORIGINAL_JOB_ID=$(sbatch --parsable --export=ALL scripts/slurm/original_model_smoke.sh)
echo "ORIGINAL_JOB_ID=${ORIGINAL_JOB_ID}"
```

等待并查看：

```bash
squeue -j "${ORIGINAL_JOB_ID}"
tail -f "logs/edge-v1-original-${ORIGINAL_JOB_ID}.out"
```

作业结束后：

```bash
sacct -j "${ORIGINAL_JOB_ID}" --format=JobID,State,Elapsed,ExitCode,MaxRSS
test -f "${EDGE_OUTPUT_DIR}/original-model-smoke-${ORIGINAL_JOB_ID}/eval/summary.json" \
  && echo "PASS original model smoke"
python -m json.tool \
  "${EDGE_OUTPUT_DIR}/original-model-smoke-${ORIGINAL_JOB_ID}/eval/summary.json"
```

通过标准：主作业 `COMPLETED`、`ExitCode=0:0`，并看到 `PASS original model smoke`。如果失败，停在这里；此时问题在原模型环境、checkpoint 或预处理数据，不应继续排查实时服务。

Slurm 脚本在模型仓库实际执行的原始模型命令是：

```bash
python eval/run_ep1800_cfg_sweep.py \
  --checkpoint "${EDGE_CHECKPOINT}" \
  --data_dir "${DUET_EDGE_ROOT}/data" \
  --smoke \
  --no_render
```

它运行 1 个 CFG 配置、1 对 val 舞蹈和数值评估；`--no_render` 只是跳过视频渲染，不会跳过模型推理。

---

## 6. 第二次 GPU 作业：Realtime smoke 并生成 benchmark fixture

### 6.1 提交 smoke

```bash
cd "${REALTIME_ROOT}"
mkdir -p logs
test -f "${EDGE_INPUT_MOTION}" && echo "OK input motion"
export SMOKE_JOB_ID=$(sbatch --parsable --export=ALL scripts/slurm/smoke.sh)
echo "SMOKE_JOB_ID=${SMOKE_JOB_ID}"
```

### 6.2 监控和验收

```bash
squeue -j "${SMOKE_JOB_ID}"
tail -f "logs/duet-v1-smoke-${SMOKE_JOB_ID}.out"
```

结束后执行：

```bash
sacct -j "${SMOKE_JOB_ID}" --format=JobID,State,Elapsed,ExitCode,MaxRSS
export SMOKE_DIR=${EDGE_OUTPUT_DIR}/smoke-${SMOKE_JOB_ID}
ls -lh "${SMOKE_DIR}"
python scripts/check_run.py \
  --summary "${SMOKE_DIR}/summary.json" \
  --ndjson "${SMOKE_DIR}/stream.ndjson"
```

必须看到 `"passed": true`，目录内必须有：

```text
effective_config.json
summary.json
stream.ndjson
real-fixture.npz
```

检查 fixture 内容：

```bash
python -c 'import os,numpy as np; p=os.environ["SMOKE_DIR"]+"/real-fixture.npz"; d=np.load(p); print({k:d[k].shape for k in d.files if hasattr(d[k],"shape")})'
```

至少应看到：

- `motion_151` 的形状为 `(150, 151)`；
- `lead_joints` 为 `(150, 24, 3)`；
- `generated_joints` 为 `(150, 24, 3)`。

如果没有 `real-fixture.npz`，不能进入 benchmark。

设置后续统一输入：

```bash
export FIXTURE=${SMOKE_DIR}/real-fixture.npz
```

---

## 7. 第三次 GPU 作业：只跑原模型 50-step 基线

第一次 benchmark 必须只测 lock 对应的原模型和 50 steps，不能同时测 25/20/10。

```bash
cd "${REALTIME_ROOT}"
mkdir -p logs
export BENCHMARK_STEPS=50
export BENCH_JOB_ID=$(sbatch --parsable --export=ALL scripts/slurm/benchmark.sh)
echo "BENCH_JOB_ID=${BENCH_JOB_ID}"
```

查看状态和日志：

```bash
squeue -j "${BENCH_JOB_ID}"
tail -f "logs/duet-v1-bench-${BENCH_JOB_ID}.out"
```

结束后：

```bash
sacct -j "${BENCH_JOB_ID}" --format=JobID,State,Elapsed,ExitCode,MaxRSS
export BENCHMARK_JSON=${EDGE_OUTPUT_DIR}/benchmark-${BENCH_JOB_ID}.json
python -m json.tool "${BENCHMARK_JSON}"
```

这个 JSON 会直接给出以下二者之一：

```json
"decision": "baseline_pass"
```

或者：

```json
"decision": "optimization_required"
```

判断逻辑已经固定为 `p99_ms + 100 < 2500`，不要人工修改标准。

### 7.1 如果是 `baseline_pass`

执行下面命令读取建议值：

```bash
python -c 'import json,os; d=json.load(open(os.environ["BENCHMARK_JSON"])); r=d["recommended_baseline"]; print("SAMPLING_STEPS=",r["steps"]); print("PLAYOUT_DELAY_S=",r["recommended_playout_delay_s"])'
```

应得到 `SAMPLING_STEPS=50` 和一个小于 2.5 秒的 delay。将它们 export，例如实际输出 delay 是 2.1 时：

```bash
export SAMPLING_STEPS=50
export PLAYOUT_DELAY_S=2.1
```

然后直接跳到第 9 章。**不要修改模型仓库，也不要更新 lock。**

### 7.2 如果是 `optimization_required`

不要提交最终验收作业。继续第 8 章，根据这次真实 GPU 结果回头更新模型代码。

---

## 8. 仅当基线失败：应用模型优化并重新验收

本章就是“GPU 先跑一次，再根据结果回头更新代码”的完整闭环。基线已经通过时禁止执行本章。

### 8.1 在模型仓库应用已有的 V1 推理快路径提交

V1 快路径提交为：

```text
e6a731106b912c1a4a8856b2a082d58cd9b93d3d
```

它包含两项改动：DDIM steps/eta 可配置，以及 lead-only CFG 从 3 次 forward 减为保持数值等价的 2 次 forward。它必须先存在于模型仓库远端；如果 `git fetch` 后找不到，请由开发者先从保存该提交的本地 `duet-edge` 仓库 push，验收执行者不要手工重写模型代码。

在 GPU 服务器执行：

```bash
git -C "${DUET_EDGE_ROOT}" fetch origin
git -C "${DUET_EDGE_ROOT}" cat-file -e e6a731106b912c1a4a8856b2a082d58cd9b93d3d^{commit}
git -C "${DUET_EDGE_ROOT}" switch --detach e6a731106b912c1a4a8856b2a082d58cd9b93d3d
git -C "${DUET_EDGE_ROOT}" status --short -- '*.py'
```

最后一条不能输出内容。

如果 `cat-file` 报 `Not a valid object name`，停止并让开发者先在拥有该提交的机器执行：

```bash
cd "/拥有优化提交的路径/duet-edge"
git push origin e6a731106b912c1a4a8856b2a082d58cd9b93d3d:refs/heads/realtime-v1-fastpath
```

然后服务器重新 `git fetch origin`。

### 8.2 更新实时仓库的外部引擎锁

```bash
cd "${REALTIME_ROOT}"
python scripts/update_engine_lock.py --duet-edge-root "${DUET_EDGE_ROOT}"
cat compat/duet-edge.lock.json
git diff -- compat/duet-edge.lock.json
```

lock 中的 commit 必须变成 `e6a731106b912c1a4a8856b2a082d58cd9b93d3d`。正式测试禁止使用 `--allow-engine-mismatch`。

这次 lock 修改需要进入实时仓库版本记录：

```bash
git add compat/duet-edge.lock.json
git commit -m "Pin Duet-EDGE V1 inference fast paths"
```

如果服务器只负责验收、不允许提交，就在 Mac 做相同的 lock 更新和 commit，再将服务器实时仓库切到该 commit；原则是任何正式结果都必须能对应到一个包含新 lock 的 realtime commit。

### 8.3 优化后再次运行原模型独立 smoke

这一步用于证明快路径改动没有破坏原有离线入口：

```bash
cd "${REALTIME_ROOT}"
mkdir -p logs
export ORIGINAL_FAST_JOB_ID=$(sbatch --parsable --export=ALL scripts/slurm/original_model_smoke.sh)
echo "ORIGINAL_FAST_JOB_ID=${ORIGINAL_FAST_JOB_ID}"
```

结束后检查：

```bash
sacct -j "${ORIGINAL_FAST_JOB_ID}" --format=JobID,State,Elapsed,ExitCode
test -f "${EDGE_OUTPUT_DIR}/original-model-smoke-${ORIGINAL_FAST_JOB_ID}/eval/summary.json" \
  && echo "PASS optimized original model smoke"
```

必须 `COMPLETED` 和 `ExitCode=0:0`，否则回退并修复模型提交，不能继续。

### 8.4 用 GPU 验证 CFG 快路径数值等价

```bash
cd "${REALTIME_ROOT}"
mkdir -p logs
export CFG_JOB_ID=$(sbatch --parsable --export=ALL scripts/slurm/cfg_equivalence.sh)
echo "CFG_JOB_ID=${CFG_JOB_ID}"
```

结束后：

```bash
sacct -j "${CFG_JOB_ID}" --format=JobID,State,Elapsed,ExitCode
python -m json.tool "${EDGE_OUTPUT_DIR}/cfg-equivalence-${CFG_JOB_ID}.json"
```

必须同时满足：

```json
"max_abs_error" <= 0.000001
"fast_path_forward_calls": 2
"passed": true
```

### 8.5 在同一块 GPU 上比较 50/25/20/10 steps

```bash
cd "${REALTIME_ROOT}"
export BENCHMARK_STEPS="50 25 20 10"
export FAST_BENCH_JOB_ID=$(sbatch --parsable --export=ALL scripts/slurm/benchmark.sh)
echo "FAST_BENCH_JOB_ID=${FAST_BENCH_JOB_ID}"
```

结束后：

```bash
sacct -j "${FAST_BENCH_JOB_ID}" --format=JobID,State,Elapsed,ExitCode,MaxRSS
export FAST_BENCHMARK_JSON=${EDGE_OUTPUT_DIR}/benchmark-${FAST_BENCH_JOB_ID}.json
python -m json.tool "${FAST_BENCHMARK_JSON}"
```

每个候选都要有 `deadline_candidate`。`false` 表示性能不达标，即使质量好也不能选。

### 8.6 对所有低 steps 候选做质量回归

先激活和 GPU 作业相同的环境：

```bash
source "${EDGE_ENV_INIT}"
conda activate "${EDGE_CONDA_ENV}"
cd "${REALTIME_ROOT}"
export BASELINE_STREAM=${EDGE_OUTPUT_DIR}/benchmark-${FAST_BENCH_JOB_ID}-steps-50/stream.ndjson
```

分别比较 25、20、10 steps：

```bash
for STEPS in 25 20 10; do
  python scripts/compare_quality.py \
    --fixture "${FIXTURE}" \
    --baseline-ndjson "${BASELINE_STREAM}" \
    --candidate-ndjson "${EDGE_OUTPUT_DIR}/benchmark-${FAST_BENCH_JOB_ID}-steps-${STEPS}/stream.ndjson" \
    --duet-edge-root "${DUET_EDGE_ROOT}" \
    --output "${EDGE_OUTPUT_DIR}/quality-${FAST_BENCH_JOB_ID}-steps-${STEPS}.json" \
    || echo "QUALITY FAIL: steps ${STEPS}"
done
```

脚本发现不合格候选会返回非零，并显示 `QUALITY FAIL`；上面的 `||` 会让循环继续检查其余候选。这不是工具故障，最终以每个 JSON 的 `passed` 为准。

查看结果：

```bash
for STEPS in 25 20 10; do
  echo "===== steps ${STEPS} ====="
  python -m json.tool "${EDGE_OUTPUT_DIR}/quality-${FAST_BENCH_JOB_ID}-steps-${STEPS}.json"
done
```

质量候选必须 `"passed": true`，它代表同时满足：

- 相对 50-step，LMA 绝对下降不超过 `0.02`；
- PFC 恶化不超过 `10%`；
- 75 帧拼接边界运动恶化不超过 `10%`；
- 没有 NaN/Inf。

### 8.7 选定最终 steps 和 delay

按以下固定顺序选择：

1. 从 benchmark JSON 中排除 `deadline_candidate=false` 的 steps。
2. 再排除质量 JSON 中 `passed=false` 的 25/20/10；50-step 作为质量基线不需要和自己比较。
3. 剩余候选中优先选择 steps 最大者，因为它最接近原模型采样质量。
4. 使用该候选的 `recommended_playout_delay_s`。

例如最终选择 25 steps、建议 delay 为 1.4 秒：

```bash
export SAMPLING_STEPS=25
export PLAYOUT_DELAY_S=1.4
```

不要照抄示例值，必须使用自己的 GPU JSON 结果。

---

## 9. 把 GPU 实测值回填配置

先确认当前 shell 中是实际选定值：

```bash
echo "SAMPLING_STEPS=${SAMPLING_STEPS}"
echo "PLAYOUT_DELAY_S=${PLAYOUT_DELAY_S}"
```

然后只更新配置中的两个字段：

```bash
cd "${REALTIME_ROOT}"
python scripts/update_runtime_config.py \
  --config configs/v1.cuda.json \
  --sampling-steps "${SAMPLING_STEPS}" \
  --playout-delay-s "${PLAYOUT_DELAY_S}"
git diff -- configs/v1.cuda.json compat/duet-edge.lock.json
```

核对：

- `model.sampling_steps` 等于最终选择值；
- `stream.playout_delay_s` 等于 benchmark 建议值；
- delay 必须小于 2.5 秒；
- 走基线路线时 lock 仍为原 commit；走优化路线时 lock 为优化 commit。

提交配置：

```bash
git add configs/v1.cuda.json compat/duet-edge.lock.json
git commit -m "Record V1 GPU-selected runtime settings"
```

若 lock 已在第 8 章提交，第二个文件不会重复产生变更，这是正常的。

---

## 10. 最终 GPU 作业：10 分钟真实墙钟验收

fixture 是 150 帧，即 5 秒。默认 `LOOP_COUNT=120` 正好产生 10 分钟输入。先确认：

```bash
python -c 'import os,numpy as np; d=np.load(os.environ["FIXTURE"]); print("frames per loop =",len(d["motion_151"])); print("120 loops seconds =",len(d["motion_151"])*120/30)'
```

第二行必须是 `600.0`。然后提交：

```bash
cd "${REALTIME_ROOT}"
mkdir -p logs
export LOOP_COUNT=120
export ACCEPT_JOB_ID=$(sbatch --parsable --export=ALL scripts/slurm/acceptance.sh)
echo "ACCEPT_JOB_ID=${ACCEPT_JOB_ID}"
```

该作业使用真实墙钟，所以至少运行约 10 分钟。监控：

```bash
squeue -j "${ACCEPT_JOB_ID}"
tail -f "logs/duet-v1-accept-${ACCEPT_JOB_ID}.out"
```

结束后执行最终机器验收：

```bash
sacct -j "${ACCEPT_JOB_ID}" --format=JobID,State,Elapsed,ExitCode,MaxRSS,MaxVMSize
export ACCEPT_DIR=${EDGE_OUTPUT_DIR}/acceptance-${ACCEPT_JOB_ID}
python scripts/check_run.py \
  --summary "${ACCEPT_DIR}/summary.json" \
  --ndjson "${ACCEPT_DIR}/stream.ndjson" \
  --duration-min 10 \
  --require-performance
python -m json.tool "${ACCEPT_DIR}/summary.json"
```

必须 `"passed": true`。工具会强制检查：

- 有完整 hello、连续 frame seq 和 eos；
- 输出为有限的 `[24,3]` joints；
- 时长至少 10 分钟；
- 推理 `p99 + 100ms < 2500ms`；
- playout delay 覆盖 `p99 + 100ms`；
- overload 和 underflow 都为 0；
- 输出 FPS 在 `29.7` 到 `30.3`；
- jitter p95 不超过 `20ms`。

### 10.1 可选：连接服务器上的实时 Viewer

先查作业运行在哪个计算节点：

```bash
squeue -j "${ACCEPT_JOB_ID}" -o '%.18i %.20N %.10T'
```

假设显示的计算节点是 `gpu-node-12`，在 Mac 新终端尝试：

```bash
ssh -N -J <你的用户名>@<登录节点地址> \
  -L 8765:127.0.0.1:8765 \
  <你的用户名>@gpu-node-12
```

然后在 Mac 实时仓库启动静态页面：

```bash
cd "/你的路径/duet-edge-realtime"
python3 -m http.server 8080 --directory web
```

浏览器打开 `http://127.0.0.1:8080`，连接 `ws://127.0.0.1:8765`。如果集群禁止 SSH 到计算节点，不把它判为核心失败；作业结束后把 NDJSON 下载到 Mac 回放：

```bash
scp <你的用户名>@<登录节点地址>:/data/<你的用户名>/duet-v1-runs/acceptance-<ACCEPT_JOB_ID>/stream.ndjson ./acceptance-stream.ndjson
```

Viewer 选择本地 `acceptance-stream.ndjson`，至少观察 3 个各 30 秒的片段，确认没有固定 2.5 秒跳变、冻结、回跳或坐标轴颠倒。

---

## 11. 最终需要交付的验收证据

在验收记录中填写并保存：

1. Mac `pytest`、WebSocket、30 分钟虚拟时钟、5 分钟墙钟结果。
2. realtime commit 和 `git status --short`；正式版本必须 clean。
3. duet-edge commit 和 Python 源码 clean 结果。
4. `compat/duet-edge.lock.json`。
5. checkpoint 完整路径、大小和 SHA-256。
6. 数据预处理 job id 和日志；如果复用现有数据，记录三个 val 目录的文件数。
7. 第一次和优化后（如有）的原模型 smoke job id、日志、summary。
8. realtime smoke job id、日志、summary、effective config 和 `real-fixture.npz`。
9. 原模型 50-step benchmark JSON。
10. 若进入优化路线：CFG 等价 JSON、50/25/20/10 benchmark JSON、每个质量 JSON、最终选择理由。
11. 最终 `sampling_steps`、`playout_delay_s` 及回填后的配置 commit。
12. 10 分钟作业日志、summary、effective config、NDJSON、`sacct` 输出和人工 Viewer 结论。

只有第 1 至 12 项中所有适用的强制项通过，才能标记 V1 验收完成。

---

## 12. 常见报错与处理顺序

### `engine commit ... does not match lock`

模型仓库 commit 与 lock 不一致。不要加 mismatch 参数绕过。执行：

```bash
cat "${REALTIME_ROOT}/compat/duet-edge.lock.json"
git -C "${DUET_EDGE_ROOT}" rev-parse HEAD
```

然后把模型仓库切到 lock commit；只有正式采用新模型提交时才能更新 lock。

### `engine Python worktree is dirty`

```bash
git -C "${DUET_EDGE_ROOT}" status --short -- '*.py'
```

先由开发者提交或处理这些修改。正式结果不能来自未提交源码。

### `root_scaled must be explicit` 或人物尺寸异常

原始 AIST `motions/*.pkl` 和 `prepare_aist_motion.py` 输出使用：

```bash
export ROOT_SCALED=false
```

模型仓库 `motions_sliced/*.pkl` 已经缩放，直接把这种文件作为输入时才使用 `true`。不能猜。

### `run directory already exists`

系统为防止覆盖证据会拒绝复用 run id。不要删除旧结果；重新提交 Slurm 会获得新 job id 和新目录。

### 原模型 smoke 找不到 slice 或 Jukebox feature

回到 4.2，确认 `val/motions_sliced`、`val/wavs_sliced`、`val/jukebox_feats` 都非空。如果为空，重新运行数据预处理并查看 `edge-v1-data-<jobid>.out`。

### CUDA 不可用

确认命令是在 `sbatch` 作业中执行，不是在登录节点直接运行，并查看日志中的 `CUDA_VISIBLE_DEVICES`。再执行：

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,AllocTRES
```

### `OVERLOAD`、underflow 或最终性能失败

先保存失败结果，不要删除。确认 benchmark 与最终验收使用同一 GPU 型号、同一引擎 commit、同一 checkpoint、同一 steps。如果基线 50-step 不达标，按第 8 章进入优化路线；如果已优化候选仍不达标，V1 当前硬件不通过，不能继续缩 steps 绕过质量门槛。
