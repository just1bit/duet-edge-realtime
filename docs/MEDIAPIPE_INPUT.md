# MediaPipe 独立实时输入

MediaPipe 现在是独立于 Final Service 的摄像头 producer。Service 常驻加载模型，producer
只负责摄像头、Pose Landmarker 和 landmarks 传输；SMPL24 重定向、151 维编码和 checkpoint
normalization 在 Service 内完成。

```text
mediapipe.sh
camera -> MediaPipe world landmarks -> TCP NDJSON ingest

service.sh
landmarks -> 30 FPS resampling -> SMPL24 retarget -> normalized motion_151
          -> 150/75 inference windows -> playout -> Viewer
```

producer 和 Service 使用 `duet-edge-mediapipe/v1` 本机协议。ingest 固定绑定回环地址
`127.0.0.1`，默认端口为 `8767`，端口可通过 `server.ingest_port` 配置。

## 安装和检查

```bash
python -m pip install -e '.[gpu,camera]'
bash scripts/final_execution/mediapipe.sh doctor
```

在运行目录的 `config.json` 中设置 `paths.mediapipe_model`、camera index 和分辨率。可使用
`configs/mediapipe.example.json` 创建运行目录。

## 模块化启动

先启动 Service。启动完成不要求 MediaPipe 已经在线：

```bash
bash scripts/final_execution/service.sh start --mode mediapipe
bash scripts/final_execution/service.sh status
```

此时 `input.mode` 为 `mediapipe`，`input.ingest.state` 为 `waiting`。随后独立启动摄像头：

```bash
bash scripts/final_execution/mediapipe.sh start
bash scripts/final_execution/mediapipe.sh status
```

producer 接入后，Service 状态变为 `input.ingest.state=connected`。停止 producer 不会停止或
重启模型：

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

producer 可以在 Service 未启动或处于 `file` 模式时独立运行。它会继续执行摄像头检测并将
连接状态记录为 `waiting_for_service`；Service 切换到 `mediapipe` 后会在后续帧自动接入。

## 切回文件测试

```bash
bash scripts/final_execution/service.sh mode file
bash scripts/final_execution/service.sh test /absolute/path/to/input.pkl
```

切换到 `file` 会结束当前 MediaPipe session，但不会强制停止 producer；若 producer 仍在运行，
它会进入 `waiting_for_service`。建议不再使用摄像头时单独执行 `mediapipe.sh stop`。

第一段推理输出仍需等待 150 个有效输入帧加 playout delay。当前重定向使用固定水平 root、
根据脚部估计 root height，并通过前一帧约束骨骼 twist；准确全局移动仍需要后续相机/地面标定
和约束人体模型拟合。
