# Duet-EDGE Realtime Final 服务操作手册

Final 服务入口覆盖首次准备、日常启动、状态查询、输入运行和有序停止。

## 1. 使用前提

运行环境包含 Python、CUDA、PyTorch、PyTorch3D、项目依赖、模型代码、checkpoint 和基线输入。
首次执行 `start` 时，服务自动创建并准备运行目录。

已有运行目录通常包含：

```text
outputs/run-.../
├── config.json
├── config.sha256
├── calibration.json
├── logs/
└── evidence/
```

默认复用 `outputs/.run-current` 指向的运行。也可以通过 `--run outputs/run-...` 指定已有运行。

## 2. 服务操作

服务接口如下：

```bash
bash scripts/final_execution/service.sh start
bash scripts/final_execution/service.sh status
bash scripts/final_execution/service.sh test /absolute/path/to/input.pkl
bash scripts/final_execution/service.sh stop
```

### start

```bash
bash scripts/final_execution/service.sh start \
  [--run outputs/run-... | --template /path/to/config.json] [--full-check]
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

输入控制与服务生命周期相互独立。MediaPipe 可通过输入适配器或对应输入参数接入
`start / status / test / stop` 服务接口。

### stop

```bash
bash scripts/final_execution/service.sh stop [--run outputs/run-...]
```

请求 Runtime 有序停止并释放模型和 GPU 资源。

## 3. 推荐操作顺序

```bash
bash scripts/final_execution/service.sh start
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
