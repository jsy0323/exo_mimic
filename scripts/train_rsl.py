"""Launch SuperMimic from YAML, with optional command-line overrides.

Run with: python scripts/train_rsl.py --config configs/train_mfg_v11.yaml
"""

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ROBOTS = ("MFG-V11", "MyoFullBody", "MS-Human-700", "MS-Human-700-Locked-Hands")
DEFAULTS = {
    "robot": "MFG-V11", "model_xml": None, "motion_file": None,
    "gpu_ids": [0], "log_root": "logs/rsl_rl",
    "env": {"scene": {"num_envs": 512}, "commands": {"motion": {"compile": False}}},
    "agent": {
        "max_iterations": 16000, "pi_batch": 512, "buffer_steps": 500,
        "seed": 1, "compile": False, "logger": "wandb", "wandb_project": "muscle_supermimic",
        "run_name": "rsl_rtx5080", "log_interval": 10, "save_interval": 1000,
    },
    "wandb": {"entity": None, "mode": None},
}
CLI_PATHS = {
    "robot": ("robot",), "model_xml": ("model_xml",), "motion_file": ("motion_file",),
    "num_envs": ("env", "scene", "num_envs"), "iterations": ("agent", "max_iterations"),
    "policy_batch": ("agent", "pi_batch"), "buffer_steps": ("agent", "buffer_steps"),
    "seed": ("agent", "seed"), "logger": ("agent", "logger"),
    "wandb_project": ("agent", "wandb_project"), "wandb_entity": ("wandb", "entity"),
    "wandb_mode": ("wandb", "mode"), "run_name": ("agent", "run_name"),
    "log_interval": ("agent", "log_interval"),
}


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def merge_config(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()


def validate_config(cfg):
    if cfg["robot"] not in ROBOTS:
        raise ValueError(f"robot must be one of {ROBOTS}")
    for section in ("env", "agent", "wandb"):
        if not isinstance(cfg[section], dict):
            raise ValueError(f"{section} must be a YAML mapping")
    unknown = set(cfg["wandb"]) - {"entity", "mode"}
    if unknown:
        raise ValueError(f"Unknown wandb keys: {sorted(unknown)}; put wandb_project under agent")
    if cfg["wandb"]["mode"] not in (None, "online", "offline"):
        raise ValueError("wandb.mode must be online, offline, or null")
    if cfg["agent"]["logger"] not in ("wandb", "tensorboard"):
        raise ValueError("agent.logger must be wandb or tensorboard")
    for name in ("model_xml", "motion_file", "log_root"):
        value = cfg[name]
        if value is not None and (not isinstance(value, (str, Path)) or not str(value).strip()):
            raise ValueError(f"{name} must be a nonempty path or null")
    if cfg["log_root"] is None:
        raise ValueError("log_root must be a nonempty path")
    ids = cfg["gpu_ids"]
    if ids is not None and ids != "all":
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("gpu_ids must be a nonempty list of GPU indices, 'all', or null (CPU)")
    if cfg["model_xml"] and cfg["robot"] != "MFG-V11":
        raise ValueError("model_xml is only supported for the MFG-V11 task")
    try:
        num_envs = cfg["env"]["scene"]["num_envs"]
    except (KeyError, TypeError):
        raise ValueError("env.scene must be a mapping with num_envs") from None
    if type(num_envs) is not int or num_envs <= 0:
        raise ValueError("env.scene.num_envs must be a positive integer")
    positive_ints = ("max_iterations", "pi_batch", "buffer_steps", "log_interval", "save_interval",
                     "wm_window", "wm_batches", "wm_batch", "horizon", "latent_dim", "critic_epochs", "critic_minibatches")
    for key in positive_ints:
        if key in cfg["agent"] and (type(cfg["agent"][key]) is not int or cfg["agent"][key] <= 0):
            raise ValueError(f"agent.{key} must be a positive integer")
    for key in ("warmup_steps", "wm_only_iters"):
        if key in cfg["agent"] and (type(cfg["agent"][key]) is not int or cfg["agent"][key] < 0):
            raise ValueError(f"agent.{key} must be a nonnegative integer")
    for key in ("wm_hidden", "enc_hidden", "dec_hidden", "critic_hidden"):
        if key in cfg["agent"]:
            dims = cfg["agent"][key]
            if not isinstance(dims, list) or not dims or any(type(n) is not int or n <= 0 for n in dims):
                raise ValueError(f"agent.{key} must be a nonempty list of positive integers")


def load_config(path, overrides):
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with Path(path).expanduser().open(encoding="utf-8-sig") as fh:
            values = yaml.safe_load(fh)
        if not isinstance(values, dict):
            raise ValueError("Config must contain a YAML mapping")
        unknown = set(values) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown top-level config keys: {sorted(unknown, key=str)}")
        cfg = merge_config(cfg, values)
    for key, value in overrides.items():
        if key not in CLI_PATHS:
            continue
        node = cfg
        parts = CLI_PATHS[key]
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise ValueError(f"{part} must be a YAML mapping")
            node = node[part]
        node[parts[-1]] = value
    validate_config(cfg)
    return cfg


def python_literal(value):
    """mjlab's CLI uses Python tuple/dict syntax for structured fields."""
    if isinstance(value, list):
        return tuple(python_literal(v) for v in value)
    if isinstance(value, dict):
        return {k: python_literal(v) for k, v in value.items()}
    return value


def cli_value(value):
    return repr(python_literal(value)) if isinstance(value, (list, dict)) else str(value)


def env_arguments(values, prefix="env"):
    out = []
    for key, value in values.items():
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise ValueError(f"Invalid configuration key under {prefix}: {key!r}")
        name = f"{prefix}.{key.replace('_', '-')}"
        if isinstance(value, dict):
            out.extend(env_arguments(value, name))
        else:
            out.extend((f"--{name}", cli_value(value)))
    return out


def build_launch(cfg, smoke=False):
    task = "Mjlab-SuperMimic-Flat-" + cfg["robot"]
    env = dict(os.environ, PYTHONUTF8="1", WARP_CACHE_PATH=str(REPO / "logs/warp-cache"))
    if cfg["model_xml"]:
        xml = resolve_path(cfg["model_xml"])
        if not xml.is_file():
            raise ValueError(f"Model XML not found: {xml}")
        env["MFG_V11_XML"] = str(xml)
    motion = resolve_path(cfg["motion_file"]) if cfg["motion_file"] else None
    if smoke:
        command = [sys.executable, "-u", str(REPO / "scripts/smoke_rsl.py"), "--task", task, "--device", "cuda:0"]
        if motion is None and cfg["robot"] == "MFG-V11":
            motion = REPO / "data/mfg_v11/clips/Walk_Comfortable_005.npz"
        if motion:
            if not motion.is_file():
                raise ValueError(f"Motion file not found: {motion}")
            command += ["--motion-file", str(motion)]
        return command, env
    if motion is None:
        if cfg["robot"].startswith("MS-Human"):
            raise ValueError("MS-Human-700 needs its retargeted motion list; supply motion_file")
        motion = REPO / ("data/mfg_v11/train.txt" if cfg["robot"] == "MFG-V11"
                         else "data/local_musclemimic/kit_kinesis_train.txt")
    if not motion.is_file():
        raise ValueError(f"Motion file not found: {motion}; prepare the dataset first")
    environment = merge_config(cfg["env"], {"commands": {"motion": {"motion_file": str(motion)}}})
    command = [sys.executable, "-u", "-m", "mjlab.scripts.train", task]
    command += ["--gpu-ids", repr(cfg["gpu_ids"]) if isinstance(cfg["gpu_ids"], list) else str(cfg["gpu_ids"]),
                "--log-root", str(resolve_path(cfg["log_root"]))]
    command += env_arguments(environment)
    for key, value in cfg["agent"].items():
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise ValueError(f"Invalid agent configuration key: {key!r}")
        command += [f"--agent.{key.replace('_', '-')}", cli_value(value)]
    if cfg["agent"]["logger"] == "wandb":
        if cfg["wandb"]["entity"]:
            env["WANDB_ENTITY"] = cfg["wandb"]["entity"]
        if cfg["wandb"]["mode"]:
            env["WANDB_MODE"] = cfg["wandb"]["mode"]
    return command, env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help="YAML configuration; relative paths inside it use the project root")
    parser.add_argument("--dry-run", action="store_true", help="Print merged config and command; do not start training or W&B")
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--model-xml", type=Path, help="Override the MFG XML without editing the external source")
    parser.add_argument("--robot", choices=ROBOTS)
    parser.add_argument("--iterations", type=positive, help="Override agent.max_iterations")
    parser.add_argument("--num-envs", type=positive)
    parser.add_argument("--policy-batch", type=positive)
    parser.add_argument("--buffer-steps", type=positive)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--logger", choices=("wandb", "tensorboard"),
                        help="Training logger (CSV is always saved); smoke checks stay local")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity", help="W&B username/team; defaults to WANDB_ENTITY or the logged-in account")
    parser.add_argument("--wandb-mode", choices=("online", "offline"),
                        help="Override WANDB_MODE; otherwise use its value or W&B's online default")
    parser.add_argument("--run-name", help="Run label for W&B and the local log directory")
    parser.add_argument("--log-interval", type=positive, help="Write metrics every N learning iterations")
    parser.add_argument("--smoke", action="store_true", help="Tiny installation check; synthetic reference unless --motion-file is supplied")
    args = vars(parser.parse_args(argv))
    try:
        cfg = load_config(args.get("config"), args)
        command, env = build_launch(cfg, smoke=args.get("smoke", False))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if args.get("dry_run", False):
        # Only user configuration is printed; inherited environment variables
        # such as WANDB_API_KEY are never included.
        printable = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
        print(yaml.safe_dump(printable, sort_keys=False, allow_unicode=True))
        print("Command (not executed):")
        print(subprocess.list2cmdline(command))
        return 0
    return subprocess.call(command, cwd=REPO, env=env)


if __name__ == "__main__":
    sys.exit(main())
