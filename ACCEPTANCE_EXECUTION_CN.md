# Duet-EDGE Realtime V1 完整验收执行步骤

本文覆盖 Mac 非算力验收、外部引擎兼容校验、GPU smoke、性能/质量选型、10 分钟稳态和最终归档。配置中的 steps 与 playout delay 在 GPU 基准前只是占位值。

## 1. 两个独立仓库与版本记录

```bash
mkdir duet-v1-workspace && cd duet-v1-workspace
git clone https://github.com/just1bit/duet-edge.git
git clone https://github.com/just1bit/duet-edge-realtime.git

git -C duet-edge-realtime status --short
git -C duet-edge status --short -- '*.py'
git -C duet-edge rev-parse HEAD
cat duet-edge-realtime/compat/duet-edge.lock.json
shasum -a 256 /absolute/path/train-1800.pt
```

两个仓库必须并列独立，实时仓库中不得存在 `.gitmodules`、gitlink、`third_party/duet-edge` 或模型源码副本。正式运行的模型 commit 必须等于 lock，Python 源码 worktree clean。记录两个 commit、checkpoint 路径/大小/SHA-256、数据路径、执行人、日期、节点和全部 job id。

## 2. Mac 非算力自动验收

```bash
cd duet-edge-realtime
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
pytest
RUN_NETWORK_TESTS=1 python -m unittest tests.test_websocket_integration -v

python scripts/make_fake_fixture.py --frames 54000 --output /tmp/duet-v1-30min.npz
python -m duet_edge_realtime.service \
  --config configs/v1.fake.json --input /tmp/duet-v1-30min.npz \
  --output-dir /tmp/duet-v1-runs --run-id local-30min \
  --clock virtual --sink ndjson
python scripts/check_run.py \
  --summary /tmp/duet-v1-runs/local-30min/summary.json \
  --ndjson /tmp/duet-v1-runs/local-30min/stream.ndjson --duration-min 30
```

必须得到 54000 输入/输出帧、连续 seq、24×3 有限坐标、0 overload/error，窗口与统计样本内存有固定上限。再次使用同一 `run-id` 必须拒绝覆盖。

5 分钟真实墙钟：

```bash
python scripts/make_fake_fixture.py --frames 9000 --output /tmp/duet-v1-5min.npz
python -m duet_edge_realtime.service \
  --config configs/v1.fake.json --input /tmp/duet-v1-5min.npz \
  --output-dir /tmp/duet-v1-runs --run-id local-5min \
  --clock realtime --sink ndjson,websocket
```

检查输出 `30 ± 0.3 FPS`、jitter p95 `<=20ms`、underflow 0。Viewer 中途断开重连一次；慢 Viewer 可以增加展示丢帧，但 NDJSON 不得缺帧。

## 3. 路径优先级和配置产物

分别通过 JSON、环境变量和 CLI 指定不同的测试路径，确认最终选择为 CLI > 环境变量 > JSON。每个运行目录必须包含 `effective_config.json`，其中路径为最终绝对路径，`run_id` 与 NDJSON/summary 一致。产物目录必须在 Git 仓库外。

## 4. 服务器环境与 baseline smoke

复用能运行 Duet-EDGE 的现有 `edge` conda 环境，不另装 torch/PyTorch3D：

```bash
source /data/zliu753/init_env.sh
conda activate edge
cd /data/zliu753/duet-edge
<执行原仓库既有离线 smoke 命令>

cd /data/zliu753/duet-edge-realtime
python -m pip install -e .
export DUET_EDGE_ROOT=/data/zliu753/duet-edge
export EDGE_CHECKPOINT=/absolute/path/train-1800.pt
export EDGE_INPUT_MOTION=/absolute/path/aist/motions/example.pkl
export EDGE_OUTPUT_DIR=/absolute/path/realtime-runs
export ROOT_SCALED=false
sbatch --export=ALL scripts/slurm/smoke.sh
```

作业必须验证外部路径包含 `EDGE.py`、`model/diffusion.py`、`vis.py`；commit/dirty 与 lock 相符；checkpoint 包含 EMA 和 normalizer；CUDA 可用且不回退 CPU。检查 1 窗和至少 3 个连续窗口、真实 fixture、summary、effective config、checkpoint SHA-256 和 GPU 信息。

开发调试若使用 `--allow-engine-mismatch`，summary 必须出现 `non_reproducible=true`，该结果禁止进入 benchmark 和正式验收。

## 5. 性能选型

先只测 lock 中未修改引擎的 50-step/eta=1 baseline：

```bash
export FIXTURE=/absolute/path/normalized-fixture.npz
export BENCHMARK_STEPS=50
sbatch --export=ALL scripts/slurm/benchmark.sh
```

每候选 warm-up 3 窗后至少统计 30 窗，记录 wall/CUDA p50/p95/p99 和峰值显存。如果 `p99 + 0.1s < 2.5s`，保留 baseline。只有不达标时，才在独立 `duet-edge` 仓库提交可配置 DDIM/CFG 快路径，更新 lock，再运行原离线 smoke、固定输入 CFG 数值等价和 `50 25 20 10` benchmark。

减少 steps 时，相对 50-step 固定 seed 基线必须满足：3 个代表片段 LMA 绝对下降 `<=0.02`，PFC 和边界连续性恶化 `<=10%`，无 NaN/Inf。满足性能与质量门槛后优先 steps 较多者。最终：

```text
measured_p99 + 0.1s <= playout_delay < 2.5s
```

把最终 steps、playout delay 和引擎 commit 回填配置/lock并提交，保存 benchmark JSON、原始 summaries、质量表和选择理由。

## 6. 在线/离线连续性等价

用相同生成 chunks 比较实时 `OnlineContinuityProcessor` 与外部引擎 `_stitch_and_fk()` 的 `[T,24,3]`。最大绝对误差目标 `<=1e-5`；跨设备差异必须解释且不得高于 `1e-4`。人工观看至少 3 段、每段 30 秒，确认无固定 2.5 秒跳变、冻结、回跳或轴颠倒。

## 7. 10 分钟真实端到端

```bash
export FIXTURE=/absolute/path/long-normalized-input.npz
export SAMPLING_STEPS=<步骤5选定值>
export PLAYOUT_DELAY_S=<步骤5实测值>
export LOOP_COUNT=<保证至少10分钟>
sbatch --export=ALL scripts/slurm/acceptance.sh
```

若可端口转发，连接 Viewer 至少 5 分钟并断开重连；否则复制 NDJSON 回 Mac 使用相同 Viewer 回放。最终执行：

```bash
python scripts/check_run.py \
  --summary "$EDGE_OUTPUT_DIR/acceptance-<jobid>/summary.json" \
  --ndjson "$EDGE_OUTPUT_DIR/acceptance-<jobid>/stream.ndjson" \
  --duration-min 10 --require-performance
```

强制门槛：推理 `p99+0.1s<2.5s`；核心缺帧、underflow、overload、NaN/Inf 均为 0；输出 `30±0.3 FPS`；jitter p95 `<=20ms`；内存/显存无持续增长；EOF 帧数等于输入有效时长。

## 8. 最终归档

归档两个独立仓库 commit/clean 证据、lock、checkpoint SHA-256、三个 Slurm job 日志、effective configs、真实 fixture、summaries、benchmark/质量/连续性报告、最终 NDJSON、Viewer 记录和失败时最慢窗口列表。只有全部强制项 Pass 才标记 V1 完成；摄像头、实时音乐和更低延迟模型属于 V2。
