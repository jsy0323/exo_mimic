"""Connect an existing MuscleMimic KIT download without copying the dataset."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Dataset root, MyoFullBody, gmr, or KIT directory")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Directory does not exist: {root}")
    repo = Path(__file__).resolve().parents[1]
    prefixes = ["MyoFullBody/gmr/KIT", "gmr/KIT", "KIT", ""]
    kit = next((root / p for p in prefixes if (root / p / "314/walking_medium08_poses.npz").is_file()), None)
    if kit is None:
        parser.error("Cannot find KIT/314/walking_medium08_poses.npz. Supply the retargeted MyoFullBody KIT data.")
    output = repo / "data/local_musclemimic"
    planned = {}
    for split in ("train", "test"):
        source = repo / f"data/musclemimic/kit_kinesis_{split}.txt"
        entries = [line.split("#", 1)[0].strip() for line in source.read_text().splitlines()]
        paths = [(kit / entry.removeprefix("MyoFullBody/gmr/KIT/")).resolve() for entry in entries if entry]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            parser.error(f"{split}: {len(missing)}/{len(paths)} clips missing; examples: {missing[:3]}")
        planned[split] = paths
    output.mkdir(parents=True, exist_ok=True)
    for split, paths in planned.items():
        (output / f"kit_kinesis_{split}.txt").write_text("\n".join(p.as_posix() for p in paths) + "\n", encoding="utf-8")
    clip = kit / "314/walking_medium08_poses.npz"
    target = repo / "data/myofullbody/walk.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() != clip.resolve():
        if target.exists() and target.read_bytes() != clip.read_bytes():
            parser.error(f"Refusing to replace a different existing test clip: {target}")
        shutil.copy2(clip, target)
    report = {"kit_directory": str(kit), "train_clips": len(planned["train"]), "test_clips": len(planned["test"])}
    (output / "source.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Prepared lists: {output}")


if __name__ == "__main__":
    main()
