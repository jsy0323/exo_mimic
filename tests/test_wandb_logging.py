"""Verify log delivery and cleanup without constructing an environment or training."""

import csv
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import train_rsl
from muscle_supermimic.rl.supermimic.config import SuperMimicRunnerCfg
from muscle_supermimic.rl.supermimic.runner import SuperMimicRunner


def logger_runner(directory):
    runner = object.__new__(SuperMimicRunner)
    runner.log_dir = str(directory)
    runner.cfg = SuperMimicRunnerCfg(logger="wandb", wandb_project="mfg-logging-check", run_name="logging-only")
    return runner


def test_launcher_passes_wandb_options_without_starting_training(monkeypatch, tmp_path):
    motion = tmp_path / "train.txt"
    motion.touch()
    monkeypatch.setattr(sys, "argv", ["train_rsl.py", "--motion-file", str(motion),
                                     "--wandb-project", "my-project", "--wandb-entity", "my-team",
                                     "--wandb-mode", "offline", "--run-name", "mfg-seed1", "--log-interval", "5"])
    launch = Mock(return_value=0)
    monkeypatch.setattr(train_rsl.subprocess, "call", launch)
    assert train_rsl.main() == 0
    args, kwargs = launch.call_args
    command, env = args[0], kwargs["env"]
    assert command[command.index("--agent.logger") + 1] == "wandb"
    assert command[command.index("--agent.wandb-project") + 1] == "my-project"
    assert command[command.index("--agent.run-name") + 1] == "mfg-seed1"
    assert command[command.index("--agent.log-interval") + 1] == "5"
    assert env["WANDB_MODE"] == "offline" and env["WANDB_ENTITY"] == "my-team"


def test_wandb_iteration_axis_csv_and_online_link(monkeypatch, tmp_path):
    import wandb
    run = Mock(settings=SimpleNamespace(mode="online"), url="https://wandb.ai/example/project/runs/test")
    init = Mock(return_value=run)
    monkeypatch.setattr(wandb, "init", init)
    runner = logger_runner(tmp_path)
    with runner._make_logger() as log:
        log({"iteration": 10, "sim_rew": 0.4, "wm_loss": 2.0})
    run.define_metric.assert_any_call("train/*", step_metric="train/iteration")
    run.log.assert_called_once_with({"train/iteration": 10, "train/sim_rew": 0.4, "train/wm_loss": 2.0}, step=10)
    run.finish.assert_called_once_with(exit_code=0)
    assert (tmp_path / "wandb_url.txt").read_text().strip() == run.url
    with (tmp_path / "train_log.csv").open() as fh:
        assert list(csv.DictReader(fh))[0]["iteration"] == "10"


def test_wandb_flushes_when_interrupted(monkeypatch, tmp_path):
    import wandb
    run = Mock(settings=SimpleNamespace(mode="offline"))
    monkeypatch.setattr(wandb, "init", Mock(return_value=run))
    runner = logger_runner(tmp_path)

    def interrupted_training(num_learning_iterations, init_at_random_ep_len, log):
        assert num_learning_iterations == 1000
        log({"iteration": 10, "sim_rew": 0.4})
        raise KeyboardInterrupt

    runner._learn = interrupted_training
    with pytest.raises(KeyboardInterrupt):
        runner.learn(1000)
    run.finish.assert_called_once_with(exit_code=1)
    # The file is released even if training is interrupted (important on Windows).
    (tmp_path / "train_log.csv").rename(tmp_path / "closed.csv")


def test_real_wandb_offline_writes_run_without_training(monkeypatch, tmp_path):
    import wandb
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_CONSOLE", "off")
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
    with logger_runner(tmp_path)._make_logger() as log:
        run = wandb.run
        assert run is not None and run.settings.mode == "offline"
        log({"iteration": 10, "sim_rew": 0.4, "wm_loss": 2.0})
        log({"iteration": 20, "sim_rew": 0.5, "wm_loss": 1.5})
    assert any(p.stat().st_size > 0 for p in tmp_path.rglob("*.wandb"))
    with (tmp_path / "train_log.csv").open() as fh:
        assert [row["iteration"] for row in csv.DictReader(fh)] == ["10", "20"]
