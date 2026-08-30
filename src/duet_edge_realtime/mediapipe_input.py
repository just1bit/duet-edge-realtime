from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from .schemas import MotionFrame
from .skeleton import OFFSETS, PARENTS


class PoseUnavailable(RuntimeError):
    pass


# MediaPipe Pose landmark indices used to construct the SMPL-style 24-joint rig.
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT, RIGHT_FOOT = 31, 32


@dataclass(frozen=True)
class PoseObservation:
    timestamp_s: float
    landmarks: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.landmarks, dtype=np.float32)
        if values.shape != (33, 4):
            raise ValueError(f"pose landmarks must be [33,4], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("pose landmarks contain NaN/Inf")
        object.__setattr__(self, "landmarks", values)


class PoseResampler:
    """Resample timestamped landmarks onto a contiguous fixed-rate timeline."""

    def __init__(self, fps: int = 30):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.period_s = 1.0 / fps
        self._previous: PoseObservation | None = None
        self._next_time_s: float | None = None

    def push(self, observation: PoseObservation) -> list[PoseObservation]:
        previous = self._previous
        if previous is not None and observation.timestamp_s <= previous.timestamp_s:
            return []
        if previous is None:
            self._previous = observation
            self._next_time_s = observation.timestamp_s
            return []

        assert self._next_time_s is not None
        output = []
        duration = observation.timestamp_s - previous.timestamp_s
        while self._next_time_s <= observation.timestamp_s + 1e-9:
            alpha = np.clip(
                (self._next_time_s - previous.timestamp_s) / duration, 0.0, 1.0
            )
            landmarks = (
                previous.landmarks * (1.0 - alpha)
                + observation.landmarks * alpha
            )
            output.append(PoseObservation(self._next_time_s, landmarks))
            self._next_time_s += self.period_s
        self._previous = observation
        return output

    def reset(self) -> None:
        self._previous = None
        self._next_time_s = None


class MediaPipeToMotion151:
    """Stateful MediaPipe-33 to normalized Duet-EDGE motion encoder.

    MediaPipe world landmarks describe positions, while Duet-EDGE expects a
    SMPL-style local-rotation representation.  This encoder performs a compact
    direction-based IK retarget with temporal smoothing and a fixed horizontal
    root.  The vertical root follows the observed floor distance.
    """

    def __init__(
        self,
        normalizer,
        *,
        fps: int = 30,
        smoothing: float = 0.35,
        minimum_visibility: float = 0.35,
        contact_velocity_m_s: float = 0.12,
        contact_height_m: float = 0.06,
    ):
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0,1]")
        self.normalizer = normalizer
        self.fps = fps
        self.smoothing = smoothing
        self.minimum_visibility = minimum_visibility
        self.contact_velocity_m_s = contact_velocity_m_s
        self.contact_height_m = contact_height_m
        self._smoothed: np.ndarray | None = None
        self._previous_feet: np.ndarray | None = None
        self._previous_local: np.ndarray | None = None
        self._root_height: float | None = None

    def encode(self, landmarks: np.ndarray) -> np.ndarray:
        values = np.asarray(landmarks, dtype=np.float64)
        if values.shape != (33, 4):
            raise ValueError(f"landmarks must have shape [33,4], got {values.shape}")
        required = np.asarray([
            LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
        ])
        if np.count_nonzero(values[required, 3] >= self.minimum_visibility) < 6:
            raise PoseUnavailable("too few reliable body landmarks")

        # MediaPipe world coordinates are x-right, y-down, z-depth.  Runtime
        # coordinates are x-lateral, y-depth, z-up.
        positions = np.stack(
            (values[:, 0], -values[:, 2], -values[:, 1]), axis=-1
        )
        if self._smoothed is None:
            self._smoothed = positions
        else:
            visible = values[:, 3:4] >= self.minimum_visibility
            updated = self._smoothed + self.smoothing * (positions - self._smoothed)
            self._smoothed = np.where(visible, updated, self._smoothed)

        joints = self._to_smpl24(self._smoothed)
        world_rotations = self._solve_world_rotations(joints)
        local_rotations = np.empty_like(world_rotations)
        for joint, parent in enumerate(PARENTS):
            local_rotations[joint] = (
                world_rotations[joint]
                if parent == -1
                else world_rotations[parent].T @ world_rotations[joint]
            )
        if self._previous_local is not None:
            local_rotations = self._stabilize_rotations(
                self._previous_local, local_rotations
            )
        self._previous_local = local_rotations

        feet = joints[[7, 8, 10, 11]]
        contacts = self._foot_contacts(feet)
        observed_height = max(0.0, -float(np.min(feet[:, 2])))
        self._root_height = (
            observed_height
            if self._root_height is None
            else self._root_height + self.smoothing * (observed_height - self._root_height)
        )
        root = np.asarray([0.0, 0.0, self._root_height], dtype=np.float64)
        rotations_6d = local_rotations[:, :2, :].reshape(-1)
        raw = np.concatenate((contacts, root, rotations_6d)).astype(np.float32)
        normalized = self._normalize(raw)
        if normalized.shape != (151,) or not np.isfinite(normalized).all():
            raise ValueError("normalizer returned an invalid 151D motion frame")
        return normalized

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        # Checkpoint normalizers operate on torch [B,T,C] tensors.  Accepting a
        # NumPy normalizer as well keeps the retargeter independently testable.
        try:
            import torch

            value = torch.from_numpy(raw[None, None])
            normalized = self.normalizer.normalize(value)
            if hasattr(normalized, "detach"):
                return normalized.detach().cpu().numpy()[0, 0].astype(np.float32)
        except (ImportError, TypeError, AttributeError):
            pass
        value = self.normalizer.normalize(raw[None, None])
        return np.asarray(value, dtype=np.float32)[0, 0]

    @staticmethod
    def _to_smpl24(p: np.ndarray) -> np.ndarray:
        pelvis = (p[LEFT_HIP] + p[RIGHT_HIP]) * 0.5
        shoulders = (p[LEFT_SHOULDER] + p[RIGHT_SHOULDER]) * 0.5
        neck = shoulders * 0.75 + ((p[LEFT_EAR] + p[RIGHT_EAR]) * 0.5) * 0.25
        head = (p[NOSE] + p[LEFT_EAR] + p[RIGHT_EAR]) / 3.0
        left_hand = (p[LEFT_WRIST] + p[LEFT_PINKY] + p[LEFT_INDEX] + p[LEFT_THUMB]) / 4.0
        right_hand = (p[RIGHT_WRIST] + p[RIGHT_PINKY] + p[RIGHT_INDEX] + p[RIGHT_THUMB]) / 4.0
        return np.asarray([
            pelvis,
            p[LEFT_HIP], p[RIGHT_HIP], pelvis * 0.67 + shoulders * 0.33,
            p[LEFT_KNEE], p[RIGHT_KNEE], pelvis * 0.33 + shoulders * 0.67,
            p[LEFT_ANKLE], p[RIGHT_ANKLE], shoulders,
            p[LEFT_FOOT], p[RIGHT_FOOT], neck,
            shoulders * 0.5 + p[LEFT_SHOULDER] * 0.5,
            shoulders * 0.5 + p[RIGHT_SHOULDER] * 0.5,
            head,
            p[LEFT_SHOULDER], p[RIGHT_SHOULDER],
            p[LEFT_ELBOW], p[RIGHT_ELBOW], p[LEFT_WRIST], p[RIGHT_WRIST],
            left_hand, right_hand,
        ], dtype=np.float64)

    def _solve_world_rotations(self, joints: np.ndarray) -> np.ndarray:
        children = [[] for _ in range(24)]
        for child, parent in enumerate(PARENTS):
            if parent >= 0:
                children[parent].append(child)
        rotations = np.empty((24, 3, 3), dtype=np.float64)
        identity = np.eye(3)
        for joint in range(24):
            usable = [
                child for child in children[joint]
                if np.linalg.norm(joints[child] - joints[joint]) > 1e-6
                and np.linalg.norm(OFFSETS[child]) > 1e-6
            ]
            if len(usable) >= 2:
                source = np.stack([self._unit(OFFSETS[c]) for c in usable])
                target = np.stack([
                    self._unit(joints[c] - joints[joint]) for c in usable
                ])
                rotations[joint] = self._kabsch(source, target)
            elif usable:
                child = usable[0]
                rotations[joint] = self._align_vectors(
                    OFFSETS[child], joints[child] - joints[joint]
                )
            else:
                parent = PARENTS[joint]
                rotations[joint] = identity if parent < 0 else rotations[parent]
        return rotations

    def _foot_contacts(self, feet: np.ndarray) -> np.ndarray:
        ground = float(np.min(feet[:, 2]))
        heights = feet[:, 2] - ground
        if self._previous_feet is None:
            speeds = np.zeros(4)
        else:
            speeds = np.linalg.norm(feet - self._previous_feet, axis=-1) * self.fps
        self._previous_feet = feet.copy()
        return (
            (heights <= self.contact_height_m)
            & (speeds <= self.contact_velocity_m_s)
        ).astype(np.float32)

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    @staticmethod
    def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        u, _, vt = np.linalg.svd(source.T @ target)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        return rotation

    @classmethod
    def _align_vectors(cls, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        a, b = cls._unit(source), cls._unit(target)
        cross = np.cross(a, b)
        cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
        sine = float(np.linalg.norm(cross))
        if sine < 1e-8:
            if cosine > 0:
                return np.eye(3)
            axis = cls._unit(np.cross(a, np.asarray([1.0, 0.0, 0.0])))
            if np.linalg.norm(axis) < 0.5:
                axis = cls._unit(np.cross(a, np.asarray([0.0, 1.0, 0.0])))
            return 2.0 * np.outer(axis, axis) - np.eye(3)
        skew = np.asarray([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ])
        return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))

    @staticmethod
    def _stabilize_rotations(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        # Project a linear temporal blend back onto SO(3).  This suppresses the
        # otherwise under-constrained per-frame bone twist without extra latency.
        blended = previous * 0.65 + current * 0.35
        output = np.empty_like(blended)
        for index, matrix in enumerate(blended):
            u, _, vt = np.linalg.svd(matrix)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1
                rotation = u @ vt
            output[index] = rotation
        return output


class MediaPipeCameraAdapter:
    """Async fixed-rate MotionFrame source backed by a local camera."""

    is_live = True

    def __init__(
        self,
        model_asset_path: str | Path,
        normalizer,
        *,
        camera_index: int = 0,
        fps: int = 30,
        width: int | None = None,
        height: int | None = None,
        maximum_missing_s: float = 0.5,
    ):
        self.model_asset_path = Path(model_asset_path).expanduser().resolve()
        if not self.model_asset_path.is_file():
            raise FileNotFoundError(self.model_asset_path)
        self.camera_index = camera_index
        self.fps = fps
        self.width = width
        self.height = height
        self.maximum_missing_s = maximum_missing_s
        self.codec = MediaPipeToMotion151(normalizer, fps=fps)
        self.resampler = PoseResampler(fps)
        self.identity = f"mediapipe-camera-{camera_index}"
        self.metadata = {
            "source": self.identity,
            "timeline_id": self.identity,
            "fps": fps,
            "live": True,
            "camera_index": camera_index,
        }
        self._capture = None
        self._landmarker = None
        self._last_landmarks: np.ndarray | None = None
        self._last_pose_time_s: float | None = None
        self._last_timestamp_ms = -1
        self._stopped = False

    async def frames_async(self) -> AsyncIterator[MotionFrame]:
        await asyncio.to_thread(self._open)
        seq = 0
        first_time_s = None
        try:
            while not self._stopped:
                try:
                    observation = await asyncio.to_thread(self._read_observation)
                except PoseUnavailable:
                    # Do not invent a timeline before the first pose, and pause
                    # the timeline when tracking has been lost for too long.
                    continue
                for sample in self.resampler.push(observation):
                    try:
                        motion = self.codec.encode(sample.landmarks)
                    except PoseUnavailable:
                        continue
                    if first_time_s is None:
                        first_time_s = sample.timestamp_s
                    yield MotionFrame(
                        seq=seq,
                        source_time_s=sample.timestamp_s - first_time_s,
                        motion_151=motion,
                        source_id=self.identity,
                    )
                    seq += 1
        finally:
            await asyncio.to_thread(self.close)

    def _open(self) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe camera input requires the 'camera' optional dependencies"
            ) from exc
        capture = cv2.VideoCapture(self.camera_index)
        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open camera index {self.camera_index}")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_asset_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._capture = capture
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _read_observation(self) -> PoseObservation:
        import cv2
        import mediapipe as mp

        ok, bgr = self._capture.read()
        if not ok:
            raise RuntimeError("camera frame read failed")
        timestamp_s = time.monotonic()
        timestamp_ms = max(self._last_timestamp_ms + 1, int(timestamp_s * 1000))
        self._last_timestamp_ms = timestamp_ms
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if result.pose_world_landmarks:
            reacquired_after_gap = (
                self._last_pose_time_s is not None
                and timestamp_s - self._last_pose_time_s > self.maximum_missing_s
            )
            self._last_landmarks = np.asarray([
                [item.x, item.y, item.z, getattr(item, "visibility", 1.0)]
                for item in result.pose_world_landmarks[0]
            ], dtype=np.float32)
            self._last_pose_time_s = timestamp_s
            if reacquired_after_gap:
                # Do not synthesize a burst of stale frames across a long
                # tracking or inference gap. Start a new 30 FPS sampling epoch
                # while preserving the model's contiguous sequence numbers.
                self.resampler.reset()
        if self._last_landmarks is None or self._last_pose_time_s is None:
            raise PoseUnavailable("no pose detected")
        if timestamp_s - self._last_pose_time_s > self.maximum_missing_s:
            raise PoseUnavailable("pose has been missing too long")
        return PoseObservation(timestamp_s, self._last_landmarks)

    def close(self) -> None:
        self._stopped = True
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def stop(self) -> None:
        self._stopped = True
