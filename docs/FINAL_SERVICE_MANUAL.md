# Duet-EDGE Realtime Final 服务操作手册

Final 服务入口覆盖首次准备、日常启动、状态查询、输入运行和有序停止。

## 1. 环境安装

进入项目目录并创建 Python 3.10 虚拟环境：

```bash
cd PROJECT_ROOT/duet-edge-realtime
python3.10 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
```

只运行本地开发和自动化检查时，安装基础依赖：

```bash
python3 -m pip install -e '.[local]'
```

运行 CUDA 推理时，在已激活的虚拟环境中安装 CUDA 12.8 版本的 PyTorch、GPU 依赖和
PyTorch3D：

```bash
python3 -m pip install 'torch==2.7.0' --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -e '.[gpu]'
python3 -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
```

需要 MediaPipe 摄像头输入时，完成上述 GPU 环境安装后再加入 camera 依赖：

```bash
python3 -m pip install -e '.[gpu,camera]'
```

运行环境还需要可用的 CUDA 驱动、模型代码、checkpoint 和基线输入。首次执行 `start` 时，
服务自动创建并准备运行目录。

已有运行目录通常包含：

```text
outputs/run-.../
├── config.json
├── config.sha256
├── calibration.json
├── logs/
└── evidence/
```

默认复用 `outputs/.final-run-current` 指向的运行。也可以通过 `--run outputs/run-...` 指定已有运行。

## 2. 服务操作

服务接口如下：

```bash
bash scripts/final_execution/service.sh start
bash scripts/final_execution/service.sh mode file|mediapipe
bash scripts/final_execution/service.sh status
bash scripts/final_execution/service.sh test /absolute/path/to/input.pkl
bash scripts/final_execution/service.sh stop
```

### start

```bash
bash scripts/final_execution/service.sh start \
  [--run outputs/run-... | --template /path/to/config.json] \
  [--mode file|mediapipe] [--full-check]
```

首次运行或指定 `--template` 时，`start` 自动完成：

1. 创建运行目录；
2. 执行运行时导入、默认输入结构和当前配置资产哈希的轻量检查；
3. 运行真实时钟基线并锁定配置；
4. 加载模型并执行 warmup；
5. 启动实时流组件；
6. 启动 Viewer 和 WebSocket 服务。

已有运行包含 `config.sha256` 时，`start` 复用校准结果并直接进入服务启动。成功后 Model、
Stream、Viewer 均为 `ready`，系统进入等待输入状态。

`--mode` 可以覆盖配置中的初始输入模式。`file` 模式等待 `test` 输入；`mediapipe` 模式启动
独立 ingest listener 并等待 producer 接入，启动成功不依赖 producer 是否在线。

首次发布、依赖或代码升级、故障排查时可启用完整检查：

```bash
bash scripts/final_execution/service.sh start --full-check
```

完整检查包括 pytest、三套测试资产哈希和 CUDA smoke。默认启动采用轻量检查路径。

### status

```bash
bash scripts/final_execution/service.sh status [--run outputs/run-...]
```

返回 Model、Stream、Viewer 和当前 session 状态。

### test

```bash
bash scripts/final_execution/service.sh test /absolute/path/to/input.pkl \
  [--root-scaled true|false] [--run outputs/run-...]
```

检查服务是否 Ready，随后校验并锁定 PKL 输入、注入服务并等待本次运行结束。省略路径时使用
`config.json` 中的 `paths.input_motion`。

`test` 只允许在 `file` 模式执行。输入模式可在模型常驻期间切换：

```bash
bash scripts/final_execution/service.sh mode file
bash scripts/final_execution/service.sh test /absolute/path/to/input.pkl

bash scripts/final_execution/service.sh mode mediapipe
bash scripts/final_execution/mediapipe.sh start
```

MediaPipe producer 使用独立的 `start / status / stop / debug / doctor` 生命周期。producer
停止或断开时 Service 保持运行并等待重新接入，详情见 `docs/MEDIAPIPE_INPUT.md`。

### stop

```bash
bash scripts/final_execution/service.sh stop [--run outputs/run-...]
```

请求 Runtime 有序停止并释放模型和 GPU 资源。

## 3. 推荐操作顺序

```bash
bash scripts/final_execution/service.sh start
bash scripts/final_execution/service.sh mode file
bash scripts/final_execution/service.sh status
bash scripts/final_execution/service.sh test /absolute/path/to/input.pkl
bash scripts/final_execution/service.sh stop
```

服务持续运行期间可以串行执行多次 `test`；每次测试 session 完成后继续等待下一次输入。

## 4. 输出与排障

每次 `test` 的主要结果写入所选运行目录：

```text
input-manifest.json
effective_config.json
summary.json
stream.ndjson
logs/
evidence/
```

启动或输入异常时，先执行 `status`，再查看运行目录中的 `logs/runtime.log` 和对应服务日志。
