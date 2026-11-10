"""Convert local MFG V11 recordings; originals stay in place."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"E:\exo_terrain\Env\MFG_Musculoskelet_V11\train\1"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/mfg_v11")
    parser.add_argument("--fps", type=float, default=100.0)
    args = parser.parse_args()
    from muscle_supermimic.assets.robots.mfg_v11 import XML_PATH, compiled_model
    from muscle_supermimic.tasks.tracking.mdp.mfg_motion import convert_recording
    files = sorted(args.source.resolve(strict=True).glob("*.npz"))
    if not files:
        parser.error(f"No NPZ recordings found under {args.source}")
    model = compiled_model()
    out = args.output.resolve()
    if out == args.source.resolve():
        parser.error("Output must differ from the original data directory")
    groups = {}
    for source in files:
        groups.setdefault(source.stem.rsplit("_", 1)[0], []).append(source)
    # Deterministic per-category file split. These are repeated/preprocessed
    # recordings, so this is a local validation split, not the paper's KIT split.
    train, test, counts = [], [], {}
    for category, sources in sorted(groups.items()):
        ranked = sorted(sources, key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())
        held_out = set(ranked[:max(1, len(ranked) // 10)]) if len(ranked) > 1 else set()
        for source in sources:
            relative = Path("clips") / source.name
            counts[source.name] = convert_recording(source, out / relative, model, args.fps)
            (test if source in held_out else train).append(relative.as_posix())
    for name, entries in (("train", train), ("test", test)):
        (out / f"{name}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    report = dict(xml=str(XML_PATH), source=str(args.source.resolve()), fps=args.fps,
                  train_clips=len(train), test_clips=len(test), frames=counts,
                  velocity_source="finite differences after resampling, as in upstream SuperMimic",
                  joint_limits="canonical joint positions clipped to XML limits; source recordings unchanged",
                  split="deterministic per-category file holdout; not KIT or subject-disjoint")
    (out / "source.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Converted {len(files)} clips ({sum(counts.values())} frames): {len(train)} train, {len(test)} validation")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
