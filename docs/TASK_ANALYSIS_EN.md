# Developing a Near-Real-Time Integrated System Based on Duet-EDGE

> Last updated: August 10, 2026

## 1. Task Background

The primary objective of this task is to: Develop a sustainable and coherent near-real-time integrated system based on the `duet-edge` project. The system should continuously generate accompanying dance movements online and stream stick-figure motion or skeletal coordinates for dynamic visualization.

The existing Duet-EDGE system can generate accompanying dance movements from a lead dancer's motion and music, and can stitch multiple short windows into a long video. However, its current workflow is offline: complete motion sequences, audio, and preprocessed features must all be available in advance, and the resulting video is produced only after generation, stitching, and rendering have finished.

Near-real-time integration requires a different operating model. The lead dancer's motion arrives continuously, the system generates the accompanying dancer's motion in rolling windows, and newly generated skeletal frames are emitted incrementally. “Near-real-time” allows a fixed amount of latency; the priority is continuous, coherent, and uninterrupted streaming after startup. During development, existing motion files can be replayed frame by frame to simulate live input. The system can later be connected to camera-based pose estimation or other motion sources.

> The main deliverables are a runnable end-to-end system, modular interfaces, stable continuous output, and observable performance -— not model training, metric comparisons, or algorithmic research.

## 2. Existing Duet-EDGE Implementation and Reusable Capabilities

Duet-EDGE already provides a complete offline pipeline from “lead dancer motion + music” to “accompanying dancer motion.” This task does not require retraining or redesigning the model. The main work is to reuse the existing checkpoint, motion representation, and skeletal conversion capabilities while adapting the inference entry point, window scheduler, and output mechanism for sustainable near-real-time operation.

The model operates on five-second windows at 30 FPS, with 150 frames per window:

| Data | Shape | Meaning |
|---|---|---|
| Lead dancer motion | `[150, 151]` | Motion condition |
| Jukebox music features | `[150, 4800]` | Music condition |
| Combined condition | `[150, 4951]` | Concatenated lead-motion and music features |
| Accompanying dancer motion | `[150, 151]` | Generated output |

Each 151-dimensional motion frame consists of four foot-contact values, a three-dimensional root position, and 6D rotations for 24 joints. The original AIST++ motion data contains root positions and axis-angle rotations at 60 FPS. `dataset/dance_dataset.py` downsamples it to 30 FPS, transforms the coordinate system and rotation representation, calculates foot contacts, and normalizes the data using statistics stored in the checkpoint.

The following core components of the existing code can be reused directly:

1. `model/model.py` and `model/diffusion.py` perform conditional diffusion inference for a single five-second window.
2. The normalizer stored in the checkpoint handles scaling for model inputs and outputs.
3. Generated results can be converted back into SMPL root positions and joint rotations.
4. `SMPLSkeleton.forward()` in `vis.py` converts the results into three-dimensional skeletal coordinates with shape `[number of frames, 24, 3]`.

Classifier-free guidance (CFG) weights can be used to select the input conditions:

| Mode | Conditions |
|---|---|
| `full` | Lead dancer motion and music |
| `lead` | Lead dancer motion only |
| `music` | Music only |
| `w0` | Both guidance paths disabled |

The first version of the pipeline should use `lead` mode and accept only lead-dancer motion. This avoids allowing real-time Jukebox feature extraction to block the main pipeline. Music conditioning can be added once the system is stable. This is an implementation sequence intended to reduce integration complexity; it does not change the model interface.

### 2.1 Why the Existing Long-Motion Pipeline Cannot Be Used Directly Online

Duet-EDGE generates only five seconds of motion per window. For long sequences, it uses a 2.5-second stride, producing an overlap of 75 frames between adjacent windows.

The `long_ddim_sample()` function in `model/diffusion.py` receives all windows for an entire sequence at once. At every DDIM denoising step, the first 75 frames of each later window are copied from the last 75 frames of the preceding window, causing adjacent windows to share their overlapping region.

After sampling, `render.py::_stitch_and_fk()` aligns root positions, blends overlapping positions using raised-cosine weights, blends overlapping joint rotations using quaternion spherical linear interpolation (slerp), and finally applies forward kinematics before rendering the video.

This implementation requires the complete input sequence before inference begins and processes all windows jointly during sampling. As the number of windows grows, GPU memory consumption and processing time also increase, so it cannot be used directly with an unbounded input stream. The near-real-time version must instead use a fixed-size sliding buffer, generate only one new window at a time, and retain only the overlapping frames and state required to preserve cross-window continuity.

## 3. Engineering Reference from the Existing Demo and Required Adaptations

The `ep1800_cfg_sweep/` demo verifies that checkpoint loading, condition configuration, long-window generation, skeletal conversion, and video rendering all work end to end. Code entry points directly relevant to this task include:

- `run_ep1800_cfg_sweep.sh`
- `eval/run_ep1800_cfg_sweep.py`
- `render.py`
- `model/diffusion.py`
- `vis.py`

The existing demo uses complete-file input, full-sequence sampling, and MP4 output. It is therefore suitable only as an offline baseline and cannot serve directly as a near-real-time application. Model loading, single-window inference, denormalization, and forward kinematics can be retained, while the remaining workflow should be replaced with the following engineering components:

1. **Input adapter:** Receives or replays lead-dancer frames at 30 FPS and converts them into the model's required 151-dimensional representation.
2. **Window buffer:** Maintains the most recent 150 frames and triggers inference at a configured stride.
3. **Incremental inference engine:** Keeps the model loaded in memory and processes only the current window, avoiding repeated initialization and unbounded accumulation of historical data.
4. **Continuity processor:** Retains the previous window's overlapping region and applies constraints, alignment, and blending to each new window.
5. **Output channel:** Writes newly generated three-dimensional skeletal frames to a queue or WebSocket. The visualization client consumes them independently so that rendering cannot block inference.

The first implementation should complete a closed loop consisting of “frame-by-frame motion-file replay → lead-only single-window inference → streamed skeletal coordinates → simple animated visualization.” Cross-window blending, performance optimization, camera input, and music conditioning can then be added in that order.

The implementation must address the following constraints:

- The time required to generate a single window must be shorter than the interval between window triggers. Otherwise, the stride, number of DDIM steps, or computational resources must be adjusted.
- Independent per-window inference no longer shares diffusion state across the full sequence. The overlap must therefore be retained and blended explicitly to prevent discontinuities at window boundaries.
- CFG should evaluate only the condition branches that are actually enabled, reducing unnecessary computation.
- Camera keypoints cannot be passed directly to the model. Future camera integration will require a separate pipeline for SMPL mapping, scale normalization, and rotation estimation.
- The runtime should record input frame rate, buffer backlog, per-window inference time, output latency, and error-recovery events so that pipeline bottlenecks can be identified.

## 4. Directory Overview

| Path | Contents and Purpose |
|---|---|
| `Real-Time Full-body Interaction with AI Dance Models, Responsiveness to Contemporary Dance.pdf` | Paper on real-time full-body interaction |
| `data+checkpoint/train-1800.pt` | Approximately 1.1 GB Duet-EDGE checkpoint containing model weights and the motion normalizer |
| `data+checkpoint/aist_plusplus_final.7z` | Archive containing AIST++ motion and audio data |
| `duet-edge/EDGE.py` | Assembles the network, diffusion module, and SMPL skeleton, and loads the checkpoint |
| `duet-edge/model/model.py` | Transformer and multi-condition CFG implementation |
| `duet-edge/model/diffusion.py` | DDIM, joint long-window sampling, and diffusion training logic |
| `duet-edge/dataset/dance_dataset.py` | Motion preprocessing, 151-dimensional representation, normalization, and duet pairing |
| `duet-edge/data/slice.py` | Slices complete motion and audio inputs into offline windows |
| `duet-edge/test.py` | Offline inference entry point for complete files and sliced-data directories |
| `duet-edge/render.py` | Long-window selection, stitching, and forward kinematics |
| `duet-edge/vis.py` | SMPL 24-joint skeleton and video rendering |
| `duet-edge/eval/` | PFC, LMA, CFG sweep, and related evaluation reports |
| `duet-edge/prepare_sfm_demo.py` | Prepares freestyle-motion demo data |
| `duet-edge/run_*.sh` | Training, evaluation, and rendering commands for server and Slurm environments |
| `duet-edge/SMPL-to-FBX/` | Optional SMPL-motion-to-FBX conversion tool |
| `ep1800_cfg_sweep/` | Offline videos and documentation generated with different CFG settings at epoch 1800 |

Several additional details about the directory contents are worth noting:

- The current checkpoint is located under `data+checkpoint/`, while some server scripts use `runs/train/exp9/weights/train-1800.pt` as their default path.
- Root positions in the original files under `motions/` retain AIST++ scale information; positions under `motions_sliced/` have already been rescaled.
- Existing sliced data generally uses a 0.5-second stride. Long-video stitching selects every fifth slice, resulting in a 2.5-second stride.
- Because of the original EDGE directory design, the training loop uses `data/test/` for progress monitoring, while standalone reports in the repository primarily use `data/val/`.
- Single-dancer EDGE uses a 4,800-dimensional music condition, whereas Duet-EDGE uses a 4,951-dimensional combined lead-motion and music condition. The checkpoint condition structures of the two models are different.
- `ep1800_cfg_sweep/` contains offline demonstration outputs; the sFM samples do not have ground-truth accompanying-dancer references.
