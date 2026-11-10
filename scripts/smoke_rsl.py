"""Run a tiny SuperMimic training job and verify policy updates and checkpoint loading.

Without --motion-file this uses a synthetic standing reference, solely to check
the installation. It does not measure motion tracking quality.
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Mjlab-SuperMimic-Flat-MyoFullBody")
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import mjlab.tasks  # Registers this project's tasks through its installed entry point.
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from muscle_supermimic.assets.robots import mshuman700, myofullbody, mfg_v11
    from muscle_supermimic.assets.robots.model_info import ModelInfo

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")
    torch.manual_seed(0)
    repo = Path(__file__).resolve().parents[1]
    label = args.task.removeprefix("Mjlab-SuperMimic-Flat-")
    log = repo / "logs/reproduction" / f"{label}_{args.device.replace(':', '_')}_{datetime.now():%Y%m%d_%H%M%S}"
    log.mkdir(parents=True)
    synthetic = args.motion_file is None
    if synthetic:
        if label == "MyoFullBody":
            model = myofullbody.compiled_model()
        elif label == "MFG-V11":
            model = mfg_v11.compiled_model()
        else:
            model = mshuman700.compiled_model(lock_hands="Locked-Hands" in label)
        info = ModelInfo(model)
        frames = 400
        clip = log / "synthetic_standing.npz"
        np.savez_compressed(
            clip, fps=100.0,
            root_pos=np.tile(model.qpos0[:3], (frames, 1)),
            root_quat=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
            joint_pos=np.tile(model.qpos0[model.jnt_qposadr[info.indep_ids]], (frames, 1)),
            joint_names=np.array(info.indep_names),
        )
    else:
        clip = args.motion_file.resolve(strict=True)

    cfg = load_env_cfg(args.task)
    cfg.commands["motion"].motion_file = str(clip)
    cfg.commands["motion"].sampling_mode = "start"
    cfg.commands["motion"].compile = False
    cfg.scene.num_envs = 2
    train_cfg = asdict(load_rl_cfg(args.task))
    train_cfg.update(
        wm_hidden=(64, 64), enc_hidden=(64,), dec_hidden=(64,), critic_hidden=(32,),
        buffer_steps=40, warmup_steps=20, wm_batch=16, wm_batches=2,
        pi_batch=8, horizon=4, wm_only_iters=2, log_interval=1,
        save_interval=1000, compile=False, logger="tensorboard",
    )
    (log / "config.json").write_text(json.dumps(train_cfg, indent=2), encoding="utf-8")
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device))
    try:
        runner = load_runner_cls(args.task)(env, train_cfg, log_dir=str(log), device=args.device)
        # Compare simulator and analytic paths on the same transition, including
        # hybrid actions. Reset before training so the check does not change it.
        for _ in range(3):
            S = runner.cmd.read_state()
            previous = env.unwrapped.action_manager.action.clone()
            action = torch.full((2, runner.na), 0.1, device=args.device)
            if runner.na > runner.nm:
                action[:, runner.nm:] = runner.cmd.position_default
            obs, reward, dones, _ = env.step(action)
            assert not dones.any(), "Unexpected termination during transition check"
            S1 = runner.cmd.read_state()
            torch.testing.assert_close(S1[6], runner.muscle.step(S[6], action[:, :runner.nm])[0], atol=1e-6, rtol=0)
            expected_reward, _ = runner._reward_term(S1, runner.cmd.f, action, previous)
            torch.testing.assert_close(expected_reward, reward, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(runner.obs_policy.assemble(None, runner.cmd.obs_ctx()), obs["policy"], atol=1e-5, rtol=1e-5)
        env.unwrapped.reset()
        before = {k: v.clone() for k, v in runner.pi.state_dict().items()}
        runner.learn(num_learning_iterations=4)
        after = {k: v.clone() for k, v in runner.pi.state_dict().items()}
        assert runner.it == 4 and runner.buffer.size > 0
        assert all(torch.isfinite(v).all() for v in after.values())
        assert any(not torch.equal(before[k], after[k]) for k in before), "Policy did not update"
        checkpoint = log / "model_000004.pt"
        assert checkpoint.is_file(), "Checkpoint was not saved"
        runner.load(str(checkpoint), load_cfg={"actor": True})
        for name, value in runner.pi.state_dict().items():
            torch.testing.assert_close(value, after[name])
        with torch.no_grad():
            action = runner.get_inference_policy()(env.get_observations())
        assert action.shape == (2, runner.na) and torch.isfinite(action).all()
        assert (action >= 0).all() and (action <= 1).all()
        result = dict(task=args.task, device=args.device, synthetic_reference=synthetic,
                      iterations=runner.it, policy_updated=True, checkpoint_roundtrip=True,
                      action_shape=list(action.shape), muscle_channels=runner.nm,
                      position_channels=runner.na - runner.nm, simulator_analytic_agreement=True,
                      checkpoint=str(checkpoint))
        (log / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
