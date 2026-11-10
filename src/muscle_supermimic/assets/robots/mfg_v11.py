"""Local MFG Raj2015 V11: original skeleton, 80 muscles and 26 upper-body position servos.

Matches the source environment's use_muscle=True actuator filtering (groups 0,2).
The external XML and mesh include are read in place and never modified.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from muscle_supermimic.assets.robots.mshuman700.mshuman700_cfg import _strip_scene

XML_PATH = Path(os.environ.get("MFG_V11_XML", r"E:\exo_terrain\Env\MFG_Musculoskelet_V11\MFG_Raj2015_V11.xml"))
TRACKED_BODIES = (
    "pelvis", "lumbar", "head", "humerus_l", "ulna_l", "hand_l",
    "humerus_r", "ulna_r", "hand_r", "femur_l", "tibia_l", "talus_l", "toes_l",
    "femur_r", "tibia_r", "talus_r", "toes_r",
)
FOOT_BODIES = ("calcn_r", "toes_r", "calcn_l", "toes_l")


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(XML_PATH.resolve(strict=True)))
    # Use mjlab's terrain, camera and contact sensors; retain every physical body,
    # joint, tendon, muscle, coupler and gain of the source model.
    _strip_scene(spec)
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for actuator in list(spec.actuators):
        if actuator.group not in (0, 2):
            spec.delete(actuator)
    return spec


@functools.lru_cache(maxsize=1)
def compiled_model() -> mujoco.MjModel:
    return get_spec().compile()


def get_mfg_v11_cfg() -> EntityCfg:
    m = compiled_model()
    position_names = tuple(m.joint(int(m.actuator_trnid[a, 0])).name for a in range(m.nu)
                           if m.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_MUSCLE)
    tendon_names = tuple(m.tendon(int(m.actuator_trnid[a, 0])).name for a in range(m.nu)
                         if m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_MUSCLE)
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(pos=tuple(m.qpos0[:3]), rot=tuple(m.qpos0[3:7]),
                                            joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(target_names_expr=tendon_names, transmission_type=TransmissionType.TENDON, command_field="effort"),
                XmlActuatorCfg(target_names_expr=position_names, transmission_type=TransmissionType.JOINT, command_field="position"),
            ),
            soft_joint_pos_limit_factor=1.0,
        ),
    )
