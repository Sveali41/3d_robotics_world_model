"""Download the official CI houses and only the two clutter objects in episode 0."""

import argparse
import json
from pathlib import Path


SNAPSHOTS = [
    ("ai-habitat/hssd-partnr-ci", "d319456261d356466d2db70cb7f14e3112efad5b",
     "hssd-partnr-ci", None),
    ("ai-habitat/partnr_episodes", "c6cffbdd82691c2f4240b75aa5289cd9efd0a32b",
     "datasets/partnr_episodes", ["v0_0/ci.json.gz"]),
    ("ai-habitat/OVMM_objects", "e69d754e7256f6f906a5d553f79ad38edc37e462",
     "objects_ovmm", [
         "train_val/ai2thorhab/configs/objects/pillow_9.object_config.json",
         "train_val/ai2thorhab/assets/objects/pillow_9.glb",
         "train_val/ai2thorhab/assets/collision_meshes/pillow_9.glb",
         "train_val/google_scanned/configs/CREATIVE_BLOCKS_35_MM.object_config.json",
         "train_val/google_scanned/assets/objects/CREATIVE_BLOCKS_35_MM/*",
         "train_val/google_scanned/assets/collision_meshes/CREATIVE_BLOCKS_35_MM/*",
     ]),
    ("ai-habitat/hab_spot_arm", "0267f9c3eaea0081a788716e87ee1088afa3cb50",
     "robots/hab_spot_arm", None),
    ("ai-habitat/habitat_humanoids", "79cd237f3cd009f1119ebedf0e7f0eff5f6581d7",
     "humanoids/humanoid_data", ["female_0/*"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partnr-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Actually download; default only prints the plan")
    parser.add_argument("--robot", choices=("spot", "fetch"), default="spot")
    args = parser.parse_args()
    root = args.partnr_root.resolve()
    snapshots = SNAPSHOTS
    if args.robot == "fetch":
        snapshots = [item for item in SNAPSHOTS if item[0] != "ai-habitat/hab_spot_arm"]
        snapshots.append(("ai-habitat/hab_fetch", "d3f7c41602e6eea1a46f8f9dfc4d8fb2778c6773",
                          "robots/hab_fetch", None))
    plan = {"root": str(root), "robot": args.robot, "snapshots": snapshots,
            "scope": "CI household episode index 0; other episodes may need more clutter assets"}
    print(json.dumps(plan, indent=2))
    if not args.download:
        return
    if not (root / "habitat_llm").is_dir():
        parser.error("PARTNR source is absent; run scripts/setup_household.sh runtime first")
    from huggingface_hub import snapshot_download

    for repo, revision, destination, patterns in snapshots:
        snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                          local_dir=str(root / "data" / destination),
                          allow_patterns=patterns)
    (root / "data" / "household_asset_manifest.json").write_text(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
