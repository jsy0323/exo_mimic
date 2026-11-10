"""Check YAML precedence and native argument parsing without launching training."""

from unittest.mock import Mock

import pytest
import yaml

from scripts import train_rsl

CONFIG = train_rsl.REPO / "configs/train_mfg_v11.yaml"


def write_config(tmp_path, values):
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_yaml_native_parser_accepts_networks_booleans_and_hyperparameters():
    import mjlab
    import tyro
    from mjlab.scripts.train import TrainConfig

    cfg = train_rsl.load_config(CONFIG, {})
    command, env = train_rsl.build_launch(cfg)
    # Parse the actual generated command with mjlab's parser. Do not call
    # launch_training, construct a simulator, or initialize a W&B run.
    parsed = tyro.cli(TrainConfig, args=command[5:], default=TrainConfig.from_task(command[4]), config=mjlab.TYRO_FLAGS)
    assert parsed.env.scene.num_envs == 512
    assert parsed.env.commands["motion"].motion_file == str(train_rsl.REPO / "data/mfg_v11/train.txt")
    assert parsed.env.commands["motion"].compile is False and parsed.env.auto_reset is False
    assert parsed.env.sim.mujoco.timestep == 0.001 and parsed.env.decimation == 10
    assert parsed.agent.wm_hidden == (2048, 2048, 2048)
    assert parsed.agent.enc_hidden == (2048, 1024, 512, 256)
    assert parsed.agent.use_critic is True and parsed.agent.compile is False
    assert parsed.agent.max_iterations == 16000 and parsed.agent.wm_only_iters == 500
    assert parsed.agent.pi_batch == 512 and parsed.agent.horizon == 32
    assert parsed.agent.lr == 0.0002 and parsed.agent.logger == "wandb"
    assert parsed.agent.wandb_tags == ("MFG-V11", "rsl", "RTX5080")
    assert parsed.gpu_ids == [0]
    assert env["MFG_V11_XML"].endswith("MFG_Raj2015_V11.xml") and env["WANDB_MODE"] == "online"


def test_cli_overrides_yaml_without_losing_other_values():
    cfg = train_rsl.load_config(CONFIG, {"iterations": 1234, "num_envs": 256, "wandb_mode": "offline"})
    assert cfg["agent"]["max_iterations"] == 1234 and cfg["env"]["scene"]["num_envs"] == 256
    assert cfg["wandb"]["mode"] == "offline"
    assert cfg["agent"]["wm_hidden"] == [2048, 2048, 2048]
    assert cfg["env"]["sim"]["mujoco"]["timestep"] == 0.001
    assert train_rsl.load_config(CONFIG, {})["env"]["scene"]["num_envs"] == 512


def test_sparse_config_uses_legacy_defaults_and_new_agent_settings(tmp_path):
    path = write_config(tmp_path, {"agent": {"lr": 0.0001, "horizon": 16}})
    cfg = train_rsl.load_config(path, {})
    assert cfg["agent"]["lr"] == 0.0001 and cfg["agent"]["horizon"] == 16
    assert cfg["agent"]["max_iterations"] == 16000
    assert cfg["agent"]["logger"] == "wandb"
    assert cfg["env"]["scene"]["num_envs"] == 512


@pytest.mark.parametrize("values,message", [
    ({"iteratoins": 10}, "Unknown top-level"),
    ({"agent": {"max_iterations": -1}}, "positive integer"),
    ({"agent": {"wm_hidden": [64, 0]}}, "positive integers"),
    ({"env": {"scene": {"num_envs": True}}}, "positive integer"),
    ({"wandb": {"mode": "bad"}}, "wandb.mode"),
    ({"agent": []}, "agent must be"),
])
def test_invalid_config_fails_before_any_training(tmp_path, values, message):
    with pytest.raises(ValueError, match=message):
        train_rsl.load_config(write_config(tmp_path, values), {})


def test_relative_data_paths_use_project_root_even_from_another_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = train_rsl.load_config(CONFIG, {})
    command, _ = train_rsl.build_launch(cfg)
    motion = command[command.index("--env.commands.motion.motion-file") + 1]
    assert motion == str(train_rsl.REPO / "data/mfg_v11/train.txt")


def test_dry_run_never_launches_training_or_prints_inherited_secrets(monkeypatch, capsys):
    launch = Mock(side_effect=AssertionError("must not start a child process"))
    monkeypatch.setattr(train_rsl.subprocess, "call", launch)
    monkeypatch.setenv("WANDB_API_KEY", "do-not-print-this-test-key")
    assert train_rsl.main(["--config", str(CONFIG), "--num-envs", "256", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "num_envs: 256" in output and "Command (not executed)" in output
    assert "do-not-print-this-test-key" not in output
    launch.assert_not_called()


def test_no_yaml_keeps_legacy_cli_behavior():
    cfg = train_rsl.load_config(None, {"robot": "MyoFullBody", "iterations": 300})
    assert cfg["robot"] == "MyoFullBody" and cfg["motion_file"] is None
    assert cfg["model_xml"] is None and cfg["agent"]["max_iterations"] == 300
