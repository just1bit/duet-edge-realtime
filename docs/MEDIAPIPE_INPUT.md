# MediaPipe 实时输入

MediaPipe 摄像头输入由独立 producer 和常驻 Service 协同完成。producer 负责摄像头、Pose
Landmarker 和 landmarks 传输；Service 完成 SMPL24 重定向、151 维编码、checkpoint
normalization、Duet-EDGE 推理和实时播放。

```text
mediapipe.sh
camera -> MediaPipe world landmarks -> TCP NDJSON ingest

service.sh
landmarks -> 30 FPS resampling -> SMPL24 retarget -> normalized motion_151
          -> 150/75 inference windows -> playout -> Viewer
```

producer 和 Service 使用 `duet-edge-mediapipe/v1` 本机协议。ingest 通过回环地址
`127.0.0.1` 传输，默认端口为 `8767`，可通过 `server.ingest_port` 配置。

## 环境自检

环境安装统一参见
[Final 服务操作手册的“环境安装”](FINAL_SERVICE_MANUAL.md#1-环境安装)。安装完成后执行自检：

```bash
bash scripts/final_execution/mediapipe.sh doctor
```

在运行目录的 `config.json` 中设置以下字段：

| 字段 | 作用 |
|---|---|
| `paths.mediapipe_model` | Pose Landmarker `.task` 模型路径 |
| `input.camera_index` | 摄像头编号 |
| `input.camera_width` / `camera_height` | 采集分辨率 |
| `input.maximum_missing_s` | 姿态时间间隔超过该值时重置重采样时间线 |
| `server.ingest_port` | producer 与 Service 的本机传输端口 |

首次运行可由示例模板创建独立运行目录：

```bash
bash scripts/final_execution/service.sh start \
  --template configs/mediapipe.example.json --mode mediapipe
```

## 模块化启动

Service 和 producer 各自管理生命周期。通常先启动 Service：

```bash
bash scripts/final_execution/service.sh start --mode mediapipe
bash scripts/final_execution/service.sh status
```

此时 `input.mode` 为 `mediapipe`，`input.ingest.state` 为 `waiting`。随后独立启动摄像头：

```bash
bash scripts/final_execution/mediapipe.sh start
bash scripts/final_execution/mediapipe.sh status
```

producer 接入后，Service 状态变为 `input.ingest.state=connected`；有效姿态到达后，Viewer
从等待状态进入首段输出准备。停止 producer 时模型继续常驻：

```bash
bash scripts/final_execution/mediapipe.sh stop
```

Service 回到等待接入状态，可以再次启动 producer。

## 前台调试和排障

前台运行会直接显示连接、摄像头和 tracking 错误：

```bash
bash scripts/final_execution/mediapipe.sh debug
bash scripts/final_execution/mediapipe.sh debug --max-observations 300
```

后台日志和状态位于当前运行目录：

```text
logs/mediapipe.log
evidence/mediapipe-status.json
mediapipe.pid
```

producer 可以先于 Service 启动，也可以在 Service 处于 `file` 模式时保持运行。此时摄像头
检测继续进行，连接状态为 `waiting_for_service`；Service 切换到 `mediapipe` 后自动接入。

常见状态含义：

| 状态 | 含义 |
|---|---|
| `input.ingest.state=waiting` | Service 已进入 MediaPipe 模式，正在等待 producer |
| `input.ingest.state=connected` | producer 已建立 ingest 连接 |
| `connection=waiting_for_service` | producer 正在采集并等待 Service 接入 |
| `pose_usable=false` | 当前姿态不足以生成新输入帧，Viewer 保留现场并显示暂停/等待提示 |

## 切回文件测试

```bash
bash scripts/final_execution/service.sh mode file
bash scripts/final_execution/service.sh test /absolute/path/to/input.pkl
```

切换到 `file` 会结束当前 MediaPipe session，producer 则保持自己的运行状态；继续采集时它会
进入 `waiting_for_service`。完成摄像头使用后可执行 `mediapipe.sh stop` 释放摄像头。

第一段推理输出在累计 150 个有效输入帧并经过 playout delay 后开始。实时重定向采用
body-centered 水平 root，根据脚部估计 root height，并利用前一帧稳定骨骼 twist。因此 Viewer
以身体动作为中心呈现姿态连续性和双人生成结果，水平位置保持在稳定的舞台中心。
