import numpy as np


def identity_motion(frames: int, root_step: float = 0.01) -> np.ndarray:
    motion = np.zeros((frames, 151), dtype=np.float32)
    motion[:, 4] = np.arange(frames, dtype=np.float32) * root_step
    identity = np.tile(np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32), 24)
    motion[:, 7:] = identity
    return motion
