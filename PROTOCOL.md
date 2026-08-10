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
  "timebases": {},
  "latency_budget": {},
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
| `metrics` | p95、队列、丢帧、underflow、deadline miss | 实时运行状态 |
| `backpressure` | `window_id`, `policy`, `wait_ms` | 输入等待推理容量 |
| `overload` | `window_id`, `policy`, `reason` | fail 策略触发 |
| `degraded` | `window_id`, `observed_ms`, `slo_ms` | 推理 SLO miss |
| `eos` | `frames`, `reason` | 正常结束 |
| `error` | `error` | 结构化错误 |

Viewer 连接后先接收 hello，随后接收各类型最新控制消息和实时帧。客户端可使用 `frame_id` 检测展示帧跳跃，并使用 NDJSON 获得完整提交序列。
