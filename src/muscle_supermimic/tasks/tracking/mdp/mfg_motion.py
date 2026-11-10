"""Convert MFG's processed (52,T) recordings into SuperMimic's canonical format.

Coordinate conversion matches mfgenv/common_utils.py::convert_ref_traj_qpos:
p=(tx,-tz,ty+0.95), R=Rx(pi/2) Rz(tilt) Rx(list) Ry(rotation).
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from muscle_supermimic.assets.robots.model_info import ModelInfo, root_offset


def recording_qpos(path: str | Path, model) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        raw = np.asarray(data["qpos"], dtype=np.float64)
    # This format has exactly one coordinate per free-root Euler DoF and per
    # non-root joint; do not guess whether a different recording is transposed.
    if raw.ndim != 2 or raw.shape[0] != model.nv or raw.shape[1] < 3:
        raise ValueError(f"{path}: expected ({model.nv}, T>=3), got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: qpos contains non-finite values")
    raw = raw.T
    qpos = np.empty((len(raw), model.nq), dtype=np.float64)
    qpos[:, :3] = raw[:, [2, 0, 1]] * [1, -1, 1] + [0, 0, 0.95]
    orientation = Rotation.from_euler("x", np.pi / 2) * Rotation.from_euler("ZXY", raw[:, 3:6])
    qpos[:, 3:7] = orientation.as_quat()[:, [3, 0, 1, 2]]
    qpos[:, 7:] = raw[:, 6:]
    return qpos


def convert_recording(source: str | Path, destination: str | Path, model, fps: float = 100.0):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    qpos = recording_qpos(source, model)
    info = ModelInfo(model)
    root_R = Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_matrix()
    canonical_quat = Rotation.from_matrix(root_R @ root_offset(model)).as_quat()[:, [3, 0, 1, 2]]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joint_pos = qpos[:, model.jnt_qposadr[info.indep_ids]]
    # Make the reset pose and the reference target agree at the XML joint limits.
    # Raw recordings stay untouched and recording_qpos() preserves their values.
    joint_pos = np.clip(joint_pos, info.q_lo, info.q_hi)
    np.savez_compressed(destination, fps=fps, root_pos=qpos[:, :3], root_quat=canonical_quat,
                        joint_pos=joint_pos, joint_names=np.array(info.indep_names))
    return len(qpos)
