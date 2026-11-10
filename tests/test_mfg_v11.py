"""Local MFG integration: source coordinates, FK, muscles and hybrid action inputs."""

import importlib.util

import mujoco
import numpy as np
import pytest
import torch

from muscle_supermimic.assets.robots import mfg_v11
from muscle_supermimic.assets.robots.model_info import ModelInfo
from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
from muscle_supermimic.tasks.tracking.mdp.mfg_motion import convert_recording, recording_qpos
from muscle_supermimic.tasks.tracking.mdp.muscle_geometry import MuscleGeometry
from muscle_supermimic.utils.rotations import quat_to_mat

SOURCE_CLIP = mfg_v11.XML_PATH.parent / "train/one/Walk_Comfortable_005.npz"
pytestmark = pytest.mark.skipif(not mfg_v11.XML_PATH.is_file() or not SOURCE_CLIP.is_file(), reason="local MFG assets unavailable")


def test_source_model_keeps_muscles_and_upper_body_servos():
    original = mujoco.MjModel.from_xml_path(str(mfg_v11.XML_PATH))
    m = mfg_v11.compiled_model()
    info = ModelInfo(m)
    assert (m.nq, m.nv, m.nu, m.na) == (53, 52, 106, 80)
    assert len(info.indep_ids) == 40 and len(info.position_ids) == 26 and info.num_muscles == 80
    np.testing.assert_array_equal(m.body_mass, original.body_mass)
    np.testing.assert_array_equal(m.body_inertia, original.body_inertia)
    np.testing.assert_array_equal(m.jnt_range, original.jnt_range)
    np.testing.assert_array_equal(m.eq_data, original.eq_data)
    for a in range(m.nu):
        old = original.actuator(m.actuator(a).name).id
        np.testing.assert_array_equal(m.actuator_gainprm[a], original.actuator_gainprm[old])
        np.testing.assert_array_equal(m.actuator_dynprm[a], original.actuator_dynprm[old])


def test_recording_conversion_matches_source_environment(tmp_path):
    pytest.importorskip("quaternion")
    source = mfg_v11.XML_PATH.parent / "mfgenv/common_utils.py"
    spec = importlib.util.spec_from_file_location("mfg_reference_utils", source)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    m = mfg_v11.compiled_model()
    raw = np.load(SOURCE_CLIP)["qpos"]
    qpos = recording_qpos(SOURCE_CLIP, m)
    expected = np.array([utils.convert_ref_traj_qpos(raw[:, i]) for i in range(0, raw.shape[1], 127)])
    np.testing.assert_allclose(qpos[::127], expected, atol=1e-12, rtol=0)
    out = tmp_path / "canonical.npz"
    assert convert_recording(SOURCE_CLIP, out, m) == raw.shape[1]
    with np.load(out) as clip:
        info = ModelInfo(m)
        np.testing.assert_allclose(clip["joint_pos"], np.clip(qpos[:, m.jnt_qposadr[info.indep_ids]], info.q_lo, info.q_hi))
        kin = SkeletonKinematics(("pelvis",), "cpu", m, torch.float64)
        canonical_R = quat_to_mat(torch.tensor(clip["root_quat"]))
        original_R = quat_to_mat(torch.tensor(qpos[:, 3:7]))
        torch.testing.assert_close(canonical_R @ kin.R_off.T, original_R, atol=1e-12, rtol=0)


def test_recording_fk_and_muscle_force_match_mujoco():
    m = mfg_v11.compiled_model()
    info = ModelInfo(m)
    kin = SkeletonKinematics(mfg_v11.TRACKED_BODIES, "cpu", m, torch.float64)
    geometry = MuscleGeometry(m, "cpu", torch.float64)
    raw = recording_qpos(SOURCE_CLIP, m)[::401]
    q = torch.tensor(np.clip(raw[:, m.jnt_qposadr[info.indep_ids]], info.q_lo, info.q_hi))
    qd = torch.randn_like(q)
    p = torch.tensor(raw[:, :3])
    R = quat_to_mat(torch.tensor(raw[:, 3:7])) @ kin.R_off
    zero = torch.zeros_like(p)
    body_p, body_R, _, _ = kin.forward(p, R, zero, zero, q, qd)
    L, Ld = geometry.length_velocity(q, qd)
    act = torch.full_like(L, 0.2)
    force = geometry.force(L, Ld, act)
    qf, qdf = kin.full_joint_state(q, qd)
    d = mujoco.MjData(m)
    for i in range(len(q)):
        d.qpos[:7], d.qpos[7:] = raw[i, :7], qf[i].numpy()
        d.qvel[6:] = qdf[i].numpy()
        d.act[:] = act[i].numpy()
        mujoco.mj_forward(m, d)
        for j, name in enumerate(mfg_v11.TRACKED_BODIES):
            np.testing.assert_allclose(body_p[i, j], d.xpos[m.body(name).id], atol=1e-9)
            np.testing.assert_allclose(body_R[i, j], d.xmat[m.body(name).id].reshape(3, 3), atol=1e-9)
        np.testing.assert_allclose(L[i], d.actuator_length[info.muscle_ids], atol=1e-9, rtol=0)
        np.testing.assert_allclose(Ld[i], d.actuator_velocity[info.muscle_ids], atol=1e-8, rtol=0)
        np.testing.assert_allclose(force[i], d.actuator_force[info.muscle_ids], atol=1e-6, rtol=1e-8)
    differentiable_q = q.clone().requires_grad_()
    geometry.length_velocity(differentiable_q, qd)[0].sum().backward()
    assert torch.isfinite(differentiable_q.grad).all() and differentiable_q.grad.abs().sum() > 0


def test_world_model_receives_and_differentiates_position_controls():
    from muscle_supermimic.rl.supermimic.networks import WorldModel
    model = WorldModel(40, 80, 4, hidden=(32,), num_position_controls=26)
    S = (torch.zeros(2, 3), torch.eye(3).repeat(2, 1, 1), torch.zeros(2, 3),
         torch.zeros(2, 3), torch.zeros(2, 40), torch.zeros(2, 40))
    act = torch.zeros(2, 80)
    controls = torch.rand(2, 26, requires_grad=True)
    with torch.no_grad():
        model.net[-1].weight.normal_(std=0.01)
    body, touch = model.step(S, act, act, controls)
    sum(x.sum() for x in body).backward()
    assert torch.isfinite(controls.grad).all() and controls.grad.abs().sum() > 0
    assert model.inputs(S, act, act, controls).shape[-1] == 13 + 80 + 160 + 26
