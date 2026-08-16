# Duet-EDGE Stream Protocol V2

协议名称为 `duet-edge-stream/v2`，消息 schema 版本为 `2.0.0`。NDJSON 每行保存一个 JSON 消息；WebSocket 每个文本消息承载一个 JSON 对象。

## 标识与时间基准

| 字段 | 含义 |
|---|---|
| `run_id` | 运行目录与一次执行的标识 |
| `session_id` | 输入、推理和播放生命周期标识；V1 与 `run_id` 相同 |
| `stream_id` | 输出动作流标识，格式为 `<run_id>:companion-motion` |
| `source_time_s` | 从输入源帧 0 开始的事件时间 |
| `target_playout_offset_s` | 从服务单调时钟起点开始的目标播放时间 |
| `emitted_monotonic_offset_s` | 从服务单调时钟起点开始的实际发送时间 |
| `emitted_wall_time_s` | Unix epoch 秒，用于跨服务日志关联 |

## 消息顺序

正常文件运行形成以下序列：

```text
hello
state(starting)
state(buffering)
state(playing)
frame / metrics / degraded / backpressure ...
state(draining)
frame / metrics ...（已入队窗口和尾段继续播放）
state(finished)
eos
```

运行错误形成 `state(failed)` 和 `error`，已提交的 frame 保留在 NDJSON 中。

## hello

```json
{
  "type": "hello",
  "protocol": "duet-edge-stream/v2",
  "schema_version": "2.0.0",
  "run_id": "demo",
  "session_id": "demo",
  "stream_id": "demo:companion-motion",
  "fps": 30,
  "joint_count": 24,
  "joint_names": ["root"],
  "parents": [-1],
  "coordinate_system": {
    "handedness": "right",
    "x": "lateral",
    "y": "depth",
    "z": "up",
    "units": "model-space"
  },
  "axis": "x=lateral,y=depth,z=up",
  "timebases": {
    "source_time_s": "seconds from source frame 0",
    "target_playout_offset_s": "monotonic seconds from service start",
    "emitted_wall_time_s": "Unix epoch seconds",
    "emitted_monotonic_offset_s": "monotonic seconds from service start"
  },
  "fixed_latency_s": 6.9666666667,
  "latency_budget": {
    "window_fill_s": 4.9666666667,
    "playout_delay_s": 2.0,
    "hop_period_s": 2.5,
    "inference_slo_ms": 1900.0,
    "safety_margin_ms": 100.0,
    "jitter_slo_ms": 20.0
  },
  "delivery": {
    "timeline": "contiguous-exactly-once-commit",
    "recorder": "complete",
    "viewer": "latest-frame-wins",
    "inference_queue_policy": "block"
  }
}
```

实际 hello 的 `joint_names` 和 `parents` 均包含 24 项。

## frame

```json
{
  "type": "frame",
  "schema_version": "2.0.0",
  "run_id": "demo",
  "session_id": "demo",
  "stream_id": "demo:companion-motion",
  "frame_id": 75,
  "seq": 75,
  "source_time_s": 2.5,
  "motion_time_s": 2.5,
  "target_playout_offset_s": 9.4666666667,
  "emitted_monotonic_offset_s": 9.4668,
  "emitted_wall_time_s": 1786380000.0,
  "end_to_end_latency_ms": 6966.8,
  "window_id": 1,
  "commit_start_frame_id": 75,
  "commit_end_frame_id": 150,
  "commit_kind": "stable",
  "flags": ["generated", "stable"],
  "joints": [[0.0, 0.0, 0.0]]
}
```

`frame_id` 与 `seq` 连续递增。提交区间使用半开区间 `[commit_start_frame_id, commit_end_frame_id)`。`commit_kind=stable` 表示滑窗稳定区，`tail` 表示输入结束后的最终有效尾段。实际 `joints` 包含 24 个三维坐标。

## 状态与诊断

| `type` | 核心字段 | 用途 |
|---|---|---|
| `state` | `state`, `wall_time_s`, `monotonic_offset_s` | 生命周期转换 |
| `metrics` | p95、队列、总计/每客户端丢帧、underflow、deadline miss、backpressure waits | 实时运行状态 |
| `backpressure` | `window_id`, `policy`, `wait_ms` | 输入等待推理容量 |
| `overload` | `window_id`, `policy`, `reason` | fail 策略触发 |
| `degraded` | `window_id`, `observed_ms`, `slo_ms` | 推理 SLO miss |
| `eos` | `frames`, `reason` | 正常结束 |
| `error` | `error` | 结构化错误 |

Viewer 连接后先接收 hello，随后接收当前 state、最新 metrics、终态消息和实时帧。degraded、backpressure 与 overload 作为实时事件发送，完整历史由 NDJSON 保存。每个 Viewer mailbox 按消息类型合并待发送的 state、metrics、degraded、backpressure 与 overload，并在帧通道达到容量时保留最新帧。客户端可使用 `frame_id` 识别展示帧跨度，并使用 NDJSON 获得完整提交序列。
