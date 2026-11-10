"""Register the optional local MFG model when its XML is available."""

import mujoco
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import register_mjlab_task

from muscle_supermimic.assets.robots import mfg_v11
from muscle_supermimic.rl.supermimic import SuperMimicRunner, SuperMimicRunnerCfg
from muscle_supermimic.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from muscle_supermimic.tasks.tracking.mdp.observations import position_command


def mfg_v11_flat_tracking_env_cfg(play=False):
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": mfg_v11.get_mfg_v11_cfg()}
    m = mfg_v11.compiled_model()
    ids = [a for a in range(m.nu) if m.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_MUSCLE]
    names = [m.actuator(a).name for a in ids]
    # All policy outputs are normalized to [0,1]. Muscles occupy the first 80
    # channels; the final 26 channels map to the XML position ctrlranges.
    cfg.actions["positions"] = JointPositionActionCfg(
        entity_name="robot", actuator_names=tuple(names), preserve_order=True, use_default_offset=False,
        scale={name: float(m.actuator_ctrlrange[a, 1] - m.actuator_ctrlrange[a, 0]) for name, a in zip(names, ids)},
        offset={name: float(m.actuator_ctrlrange[a, 0]) for name, a in zip(names, ids)},
    )
    cfg.observations["policy"].terms["position_command"] = ObservationTermCfg(
        func=position_command, params={"command_name": "motion"})
    cmd = cfg.commands["motion"]
    cmd.body_names = mfg_v11.TRACKED_BODIES
    cmd.compile = False
    cfg.viewer.body_name = "pelvis"
    cfg.scene.sensors = (
        ContactSensorCfg(name=cmd.touch_sensor_name,
                         primary=ContactMatch(mode="body", pattern=mfg_v11.FOOT_BODIES, entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"), fields=("force",), reduce="netforce"),
    )
    # Source floor is z=-0.01. Grid environment origins remain at z=0.
    cfg.scene.terrain.init_state.pos = (0.0, 0.0, -0.01)
    cfg.sim.mujoco.timestep = 0.001
    cfg.sim.mujoco.integrator = "implicitfast"
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.impratio = 5.0
    cfg.sim.mujoco.iterations = 100
    cfg.sim.mujoco.ls_iterations = 50
    cfg.decimation = 10
    cfg.sim.nconmax, cfg.sim.njmax = 48, 320
    cfg.scene.num_envs = 512
    if play:
        cfg.scene.num_envs, cfg.auto_reset = 1, True
        cfg.episode_length_s = int(1e9)
        cmd.sampling_mode = "start"
        cfg.terminations.pop("time_out")
    return cfg


if mfg_v11.XML_PATH.is_file():
    register_mjlab_task(
        task_id="Mjlab-SuperMimic-Flat-MFG-V11",
        env_cfg=mfg_v11_flat_tracking_env_cfg(), play_env_cfg=mfg_v11_flat_tracking_env_cfg(play=True),
        rl_cfg=SuperMimicRunnerCfg(experiment_name="mfg_v11_tracking", wandb_project="muscle_supermimic",
                                 compile=False, pi_batch=512, buffer_steps=500),
        runner_cls=SuperMimicRunner,
    )
