from __future__ import annotations

import numpy as np

JOINT_NAMES = [
    "root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle",
    "rankle", "chest", "ltoe", "rtoe", "neck", "lclavicle", "rclavicle",
    "head", "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist",
    "rwrist", "lhand", "rhand",
]

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
           16, 17, 18, 19, 20, 21]

OFFSETS = np.asarray([
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
], dtype=np.float64)


def rotation_6d_to_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    a1, a2 = values[..., :3], values[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_orth / np.maximum(np.linalg.norm(a2_orth, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Stable rotation matrix to wxyz quaternion conversion."""
    m = np.asarray(matrix, dtype=np.float64)
    q = np.empty(m.shape[:-2] + (4,), dtype=np.float64)
    trace = np.trace(m, axis1=-2, axis2=-1)
    q[..., 0] = np.sqrt(np.maximum(0.0, 1.0 + trace)) / 2.0
    q[..., 1] = np.copysign(
        np.sqrt(np.maximum(0.0, 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])) / 2.0,
        m[..., 2, 1] - m[..., 1, 2],
    )
    q[..., 2] = np.copysign(
        np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])) / 2.0,
        m[..., 0, 2] - m[..., 2, 0],
    )
    q[..., 3] = np.copysign(
        np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])) / 2.0,
        m[..., 1, 0] - m[..., 0, 1],
    )
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ), axis=-1)


def slerp(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64).copy()
    dot = np.sum(left * right, axis=-1, keepdims=True)
    right = np.where(dot < 0.0, -right, right)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    weight = np.broadcast_to(np.asarray(weight, dtype=np.float64), dot.shape)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    linear = sin_omega < 1e-6
    a = np.where(linear, 1.0 - weight, np.sin((1.0 - weight) * omega) / np.maximum(sin_omega, 1e-12))
    b = np.where(linear, weight, np.sin(weight * omega) / np.maximum(sin_omega, 1e-12))
    result = a * left + b * right
    return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)


def forward_kinematics(local_quaternions: np.ndarray, roots: np.ndarray) -> np.ndarray:
    local = np.asarray(local_quaternions, dtype=np.float64)
    roots = np.asarray(roots, dtype=np.float64)
    frames = local.shape[0]
    positions = np.empty((frames, 24, 3), dtype=np.float64)
    world_q = np.empty((frames, 24, 4), dtype=np.float64)
    for joint, parent in enumerate(PARENTS):
        if parent == -1:
            positions[:, joint] = roots
            world_q[:, joint] = local[:, joint]
        else:
            rotated = np.einsum("fij,j->fi", quaternion_to_matrix(world_q[:, parent]), OFFSETS[joint])
            positions[:, joint] = positions[:, parent] + rotated
            world_q[:, joint] = quat_multiply(world_q[:, parent], local[:, joint])
    return positions.astype(np.float32)
