import numpy as np


def identity_motion(frames: int, root_step: float = 0.01) -> np.ndarray:
    motion = np.zeros((frames, 151), dtype=np.float32)
    motion[:, 4] = np.arange(frames, dtype=np.float32) * root_step
    identity = np.tile(np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32), 24)
    motion[:, 7:] = identity
    return motion


def standing_mediapipe_landmarks() -> np.ndarray:
    """Return a finite, fully visible MediaPipe-33 standing pose."""
    runtime = np.zeros((33, 3), dtype=np.float32)

    def set_point(index, xyz):
        runtime[index] = xyz

    set_point(0, [0.0, 0.0, 0.75])
    set_point(7, [0.08, 0.0, 0.68])
    set_point(8, [-0.08, 0.0, 0.68])
    set_point(11, [0.22, 0.0, 0.48])
    set_point(12, [-0.22, 0.0, 0.48])
    set_point(13, [0.48, 0.0, 0.28])
    set_point(14, [-0.48, 0.0, 0.28])
    set_point(15, [0.66, 0.0, 0.08])
    set_point(16, [-0.66, 0.0, 0.08])
    for index in (17, 19, 21):
        set_point(index, [0.70, 0.02, 0.06])
    for index in (18, 20, 22):
        set_point(index, [-0.70, 0.02, 0.06])
    set_point(23, [0.10, 0.0, 0.0])
    set_point(24, [-0.10, 0.0, 0.0])
    set_point(25, [0.10, 0.0, -0.48])
    set_point(26, [-0.10, 0.0, -0.48])
    set_point(27, [0.10, 0.0, -0.92])
    set_point(28, [-0.10, 0.0, -0.92])
    set_point(29, [0.10, -0.04, -0.96])
    set_point(30, [-0.10, -0.04, -0.96])
    set_point(31, [0.10, 0.16, -0.96])
    set_point(32, [-0.10, 0.16, -0.96])
    mediapipe = np.stack(
        (runtime[:, 0], -runtime[:, 2], -runtime[:, 1]), axis=-1
    )
    return np.concatenate(
        (mediapipe, np.ones((33, 1), dtype=np.float32)), axis=-1
    )
