# Duet-EDGE 近实时流式系统 V1：通用模块化单体方案

> 依据：`TASK_ANALYSIS_CN.md`（对应英文版：`TASK_ANALYSIS_EN.md`）  
> 实现仓库：[just1bit/duet-edge-realtime](https://github.com/just1bit/duet-edge-realtime.git)  
> 模型仓库：[just1bit/duet-edge](https://github.com/just1bit/duet-edge.git)  
> 开发环境：实验室电脑，覆盖协议、调度、展示、GPU 推理、基准和验收

## 1. 目标与系统定位

V1 建立从主舞动作输入到浏览器展示的持续近实时闭环：

```text
动作回放
  -> 规范化输入帧
  -> 150/75 滑动窗口
  -> Duet-EDGE 条件推理
  -> 在线对齐与重叠融合
  -> 连续时间线提交
  -> 播放缓冲
  -> NDJSON / WebSocket
  -> 浏览器可视化
```

系统采用固定启动缓冲：首个窗口形成后开始推理，结果按目标播放时间输出。V1 运行于单路输入、单个常驻模型实例和单 GPU 环境。模块化单体简化部署，接口支持扩展摄像头输入、音乐条件、独立推理进程和多会话服务。

30 FPS、150 帧窗口提供约 5 秒输入上下文，形成数秒级固定延迟。运行摘要和验收记录量化延迟、吞吐、播放稳定性和动作连续性。

## 2. 设计原则

1. **规范数据契约**：输入帧、推理窗口、提交批次和输出帧均使用显式 schema。
2. **统一时间语义**：区分源时间、接收时间、目标播放时间、单调时钟和墙上时钟。
3. **连续提交语义**：每个输出帧沿统一时间线提交一次，窗口重叠区经融合后提交。
4. **分通道服务质量**：记录通道保存完整流，Viewer 通道采用 latest-frame-wins 策略。
5. **显式背压与过载策略**：每条有界队列配置容量与满载行为，并记录等待和超时事件。
6. **可观察生命周期**：统一记录服务状态、推理延迟、队列水位、播放抖动和端到端延迟。
7. **轻量运行时耦合**：流式仓库通过 `DUET_EDGE_ROOT` 加载外部 Duet-EDGE 仓库，保持模型工程与服务工程独立维护。

## 3. 仓库与模型适配

### 3.1 并列仓库

```text
workspace/
├── duet-edge/               # 模型工程
└── duet-edge-realtime/      # 流式服务
```

运行时连接关系：

```text
duet-edge-realtime
  -> CudaDuetEdgeBackend
  -> DUET_EDGE_ROOT
  -> EDGE、normalizer、diffusion、SMPLSkeleton
```

真实后端启动时校验模型目录、核心模块、checkpoint 结构、CUDA 环境和数值输出。运行摘要记录模型仓库路径、remote、commit、工作区状态、checkpoint SHA256、PyTorch/CUDA 版本和 GPU 信息，用于复现环境和比较结果。

流式服务按运行时接口和结构校验加载模型。真实模型 smoke、协议测试、数值测试和质量验收验证模型版本变更。

### 3.2 后端接口

`FakeInferenceBackend` 与实验室电脑上的 `CudaDuetEdgeBackend` 共享以下生命周期：

```text
warmup()
infer(MotionWindow) -> GeneratedChunk
unnormalize(motion)
version_info()
close()
```

模型在服务启动阶段加载并预热，后续窗口复用同一实例。专用执行线程串行调用后端，使事件循环持续负责输入调度、播放、WebSocket 和指标采集。

## 4. 运行架构

```text
Input Adapters
  -> canonical MotionFrame
  -> Session / Timeline Coordinator
  -> SlidingWindowBuffer
  -> bounded InferenceQueue
  -> InferenceBackend
  -> OnlineContinuityProcessor
  -> TimelineCommitter
  -> bounded OutputQueue
  -> PlayoutClock
  -> FanOut
       ├── NDJSON recorder：完整流
       ├── WebSocket viewer：最新画面优先
       └── metrics / summary.json：指标与运行摘要
```

系统以单个 Python 服务进程运行，模块间通过数据对象和小型接口交互。`InferenceBackend` 可演进为独立进程或远程服务，并复用现有输入、提交、播放和 Viewer 协议。

### 4.1 输入与窗口

`MotionFrame` 包含：

| 字段 | 含义 |
|---|---|
| `seq` | 输入源内连续帧号 |
| `source_time_s` | 从源帧 0 开始的事件时间 |
| `ingest_monotonic_s` | 服务接收帧时的单调时钟 |
| `source_id` | 输入源标识 |
| `schema_version` | 数据契约版本 |
| `motion_151` | 规范化主舞动作向量 |

窗口缓冲保留最近 150 帧，每 75 帧生成一个 `MotionWindow`。窗口记录输入序列和源时间范围、触发时刻、有效尾帧数与确定性 seed。文件结束时使用末帧补齐上下文，`valid_frames` 界定最终提交范围。

### 4.2 在线连续性与提交

每个生成窗口依次完成：

1. 使用 checkpoint normalizer 还原动作参数。
2. 根据重叠区域对齐根节点位置。
3. 使用 raised-cosine 权重融合根节点轨迹。
4. 使用 quaternion slerp 融合关节旋转。
5. 通过 SMPL 前向运动学生成 `[N, 24, 3]` 三维关节坐标。
6. 保存窗口后半段作为下一窗口的融合状态。
7. 将稳定区或最终尾段交给 `TimelineCommitter`。

`TimelineCommitter` 维护 `next_frame_id`，校验每个批次从该位置连续提交。`CommittedBatch` 包含半开区间 `[start_frame_id, end_frame_id)`、来源窗口、提交类型和关节坐标。重复或跳跃区间触发结构化错误，保证帧连续且单次提交。

Duet-EDGE 离线长动作在 DDIM 去噪中共享相邻窗口状态；在线路径逐窗生成，再在参数空间后处理。两者共用动作表示、normalizer、融合原则和 FK，并对照边界连续性、数值结果和代表性片段。

### 4.3 播放时间线

协议使用以下时间字段：

| 字段 | 时间基准 |
|---|---|
| `source_time_s` | 输入/动作源从 0 开始的事件时间 |
| `target_playout_offset_s` | 从服务单调时钟起点开始的目标播放时间 |
| `emitted_monotonic_offset_s` | 从服务单调时钟起点开始的实际发送时间 |
| `emitted_wall_time_s` | Unix epoch 墙上时钟，用于跨进程日志关联 |

首帧预算为：

```text
first_frame_latency
  = (window_frames - 1) / fps
  + playout_delay_s
```

稳态计算预算为：

```text
inference_p99 + safety_margin < hop_frames / fps
playout_delay_s >= inference_p99 / 1000 + safety_margin
```

基准流程的初始安全余量为 100 ms，再根据 GPU 抖动、运行负载和验收结果校准。播放模块按绝对 deadline 调度，记录 jitter、underflow 和端到端延迟。

## 5. 背压、过载与通道策略

| 通道 | 有界策略 | 满载行为 | 记录指标 |
|---|---|---|---|
| 输入到推理 | `inference_queue_size` | `block` 等待容量；`fail` 结束运行并生成诊断 | 高水位、等待次数、等待时长、overload |
| 推理到播放 | `output_queue_size` | 等待播放消费，保持已提交批次完整 | 高水位、等待次数 |
| 播放到 NDJSON | 完整记录 | 调用完成代表记录已接收 | 帧数、连续序号、写入错误 |
| 播放到 Viewer | `viewer_queue_frames` | 丢弃该客户端最旧画面，保留最新帧与控制消息 | 每客户端丢帧汇总 |

文件回放默认使用 `block`：队列满时暂停读取输入。固定采集节奏的 session 使用 `fail`，由外部流程重启。策略、容量和实际等待写入有效配置与运行摘要。

推理耗时超过 `inference_slo_ms` 时，系统发送 `degraded` 事件并累计 deadline miss：

- `deadline_miss_policy=continue`：继续处理，underflow 指标反映时间线表现。
- `deadline_miss_policy=fail`：结束 session，保存当前记录、状态历史和错误摘要。

## 6. 生命周期与恢复语义

服务状态机如下：

```text
starting -> buffering -> playing -> draining -> finished
    |           |           |           |
    +-----------+-----------+-----------+-> failed
```

| 状态 | 含义 |
|---|---|
| `starting` | 启动 Sink，发布协议和运行标识 |
| `buffering` | 输入帧持续到达，系统形成首个窗口 |
| `playing` | 已提交批次按播放时钟输出 |
| `draining` | 输入结束，系统完成尾段与队列排空 |
| `finished` | 生成 EOS、摘要和完整记录 |
| `failed` | 保存已生成记录并发布结构化错误 |

状态转换同步写入 NDJSON、WebSocket 和 `summary.json`。Viewer 重连后接收 hello、当前状态和指标，恢复骨架定义与运行上下文。

## 7. 输出协议 V2

hello 消息定义：

- 协议与 schema 版本：`duet-edge-stream/v2`、`2.0.0`；
- 运行标识：`run_id`、`session_id`、`stream_id`；
- 动作定义：FPS、关节名称、父子关系和坐标系；
- 播放约定：时间基准、延迟预算和 delivery 策略。

frame 消息包含：

```text
schema_version
run_id / session_id / stream_id
frame_id / seq
source_time_s
target_playout_offset_s
emitted_monotonic_offset_s
emitted_wall_time_s
end_to_end_latency_ms
window_id
commit_start_frame_id / commit_end_frame_id / commit_kind
flags
joints[24][3]
```

`seq`、`motion_time_s` 和 `wall_time_s` 是 Viewer 与现有分析脚本的兼容字段。协议细节和消息示例见实时仓库的 `PROTOCOL.md`。

## 8. 配置

```json
{
  "backend": "cuda",
  "paths": {
    "duet_edge_root": "<model-repository-path>",
    "checkpoint": "<checkpoint-path>",
    "input_motion": "<input-motion-path>",
    "output_dir": "<run-output-path>"
  },
  "model": {
    "guidance_music": 0.0,
    "guidance_lead": 2.0,
    "sampling_steps": 50,
    "eta": 1.0,
    "seed": 1234
  },
  "stream": {
    "fps": 30,
    "window_frames": 150,
    "hop_frames": 75,
    "playout_delay_s": 2.0,
    "inference_queue_size": 1,
    "output_queue_size": 2,
    "viewer_queue_frames": 150,
    "inference_queue_policy": "block",
    "inference_slo_ms": 1900.0,
    "deadline_miss_policy": "continue",
    "jitter_slo_ms": 20.0
  },
  "server": {
    "bind_host": "127.0.0.1",
    "port": 8765
  }
}
```

配置按“命令行 > 环境变量 > JSON”覆盖。每次运行使用独立目录，写入 `effective_config.json`、`stream.ndjson` 和 `summary.json`。数据路径为绝对路径，WebSocket 和 Viewer 连接实验室电脑的本机地址。

## 9. 仓库结构

```text
duet-edge-realtime/
├── pyproject.toml
├── docs/
│   ├── README.md
│   ├── PROTOCOL.md
│   ├── TASK_ANALYSIS_CN.md
│   ├── TASK_ANALYSIS_EN.md
│   ├── VERSION1_STREAMING_PLAN.md
│   └── V1_ACCEPTANCE_EXECUTION_CN.md
├── configs/
│   ├── v1.fake.json
│   └── v1.cuda.json
├── src/duet_edge_realtime/
│   ├── service.py
│   ├── config.py
│   ├── schemas.py
│   ├── lifecycle.py
│   ├── timeline.py
│   ├── input_adapters.py
│   ├── window_buffer.py
│   ├── continuity.py
│   ├── playout.py
│   ├── sinks.py
│   ├── metrics.py
│   └── backends/
│       ├── base.py
│       ├── fake.py
│       └── duet_edge.py
├── web/
├── tests/
└── scripts/                  # smoke、基准汇总、配置更新和结果检查
```

## 10. 可观测性与运行记录

`summary.json` 汇总：

- 有效配置、服务仓库和模型运行时元数据；
- 模型加载、预热以及推理 wall/CUDA 延迟 p50/p95/p99；
- 输入帧数、观察帧率、序列错误和各窗口范围；
- inference SLO、hop 周期、headroom、队列水位、背压和 deadline miss；
- 输出帧数、提交批次、jitter、underflow、Viewer 丢帧和端到端延迟；
- 状态历史、错误、退出原因以及各项 SLO 结果。

指标历史采用有界采样，使长时间运行的内存占用保持稳定。

## 11. 测试与验收

### 11.1 本地自动验证

- 输入序列、窗口边界、尾段补齐和源时间；
- 连续性对齐、旋转融合、FK 和数值有效性；
- 时间线连续提交、重复区间和跳跃区间检测；
- v2 hello/frame/state/EOS/error 协议与生命周期；
- 推理 deadline 的 continue/fail 行为；
- Viewer latest-frame-wins 与控制消息保留；
- 长时间运行的指标、队列和内存边界；
- 虚拟时钟全链路与实时 WebSocket 集成。

### 11.2 实验室 GPU 电脑验证

- 模型目录、checkpoint、CUDA、预热和单窗口确定性；
- 连续窗口与尾段处理；
- 预热后推理 p50/p95/p99、CUDA 时间和显存峰值；
- 计算预算：`p99 + safety_margin < hop_period`；
- 播放预算：`playout_delay >= p99 + safety_margin`；
- 目标 30 FPS、jitter p95、underflow 和队列水位；
- 在线后处理与离线生成的边界连续性、自然度和坐标方向；
- 长时间运行的序列完整性和资源趋势。

## 12. 完成定义

V1 按以下条件判定完成：

1. 流式仓库可独立安装，本地模拟全链路运行通过。
2. 实验室电脑通过 `DUET_EDGE_ROOT`、checkpoint 和输入配置启动 GPU 后端。
3. 输入、窗口、提交和输出使用版本化数据契约；生命周期、背压、deadline 和播放行为写入有效配置与运行记录。
4. 自动测试、实时展示、NDJSON 回放和结构化验收通过。
5. 连续窗口、性能基准和动作质量的验收结果可归档与复现。
