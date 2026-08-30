# MediaPipe live input

The MediaPipe input path is an optional CUDA profile. It leaves the existing
Duet-EDGE inference, 150/75 windowing, continuity, playout, and wire protocol
unchanged. The new source performs this conversion before `MotionFrame` ingest:

```text
camera -> MediaPipe world landmarks -> 30 FPS resampling -> SMPL24 retarget
       -> [4 contacts, 3 root position, 24 x 6D local rotation]
       -> checkpoint normalization -> MotionFrame.motion_151
```

The hot path does not create a pickle. NDJSON remains the output/evidence record.

## Installation

Install the project with its CUDA and camera extras in the environment used by
Duet-EDGE:

```bash
python -m pip install -e '.[gpu,camera]'
```

Download a MediaPipe Pose Landmarker `.task` model and set its path in
`configs/mediapipe.example.json`, or pass `--mediapipe-model`.

## Direct live run

From the repository root:

```bash
duet-edge-realtime \
  --config configs/mediapipe.example.json \
  --input-format mediapipe \
  --clock realtime \
  --sink ndjson,websocket,web \
  --output-dir outputs \
  --run-id camera-demo
```

Open `http://127.0.0.1:8080`. Keep the complete body, especially both hips,
knees, and ankles, visible. The first generated output appears after the 150-frame
input window fills plus the configured playout delay (about 5.72 seconds with the
example configuration).

CLI overrides are available for `--camera-index`, `--camera-width`, and
`--camera-height`. `MEDIAPIPE_POSE_MODEL` can supply the model path.

## Resident runtime

The resident runtime selects camera input when `input.mode` is `mediapipe` and
does not require an input manifest. Start the live session through `/run/start`
and stop it through `/run/stop`; the bundled runtime client exposes these as
`start-run` and `stop-run`.

Stop after at least five seconds of valid tracked input so the first 150-frame
window exists and the normal draining path can finish.

## Current retargeting boundary

This first implementation deliberately uses a fixed horizontal root and derives
root height from the observed feet. It also regularizes bone twist from the prior
frame because point landmarks cannot uniquely determine axial joint rotation.
This is sufficient to exercise the complete live pipeline, but camera/floor
calibration and a constrained body-model fitter remain the next quality step for
accurate global locomotion.
