"""Launch a kitchen or a generated ProcTHOR house."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .env import PROJECT_ROOT, ThorEnv


def main(argv=None):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".runtime" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".runtime" / "thor-cache"))
    os.environ.setdefault("XDG_CONFIG_HOME", str(PROJECT_ROOT / ".runtime" / "thor-config"))
    runtime = PROJECT_ROOT / ".runtime" / "thor" / "bin" / "python"
    if runtime.is_file() and (Path(sys.prefix).resolve() != runtime.parent.parent.resolve()
                              or not sys.flags.no_user_site):
        os.execv(str(runtime), [str(runtime), "-s", str(PROJECT_ROOT / "main.py"),
                               "thor", *(argv if argv is not None else sys.argv[1:])])
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scene", default="FloorPlan1", help="Built-in iTHOR scene")
    mode.add_argument("--generate", choices=("kitchen", "living-room", "bedroom", "bathroom",
                                            "kitchen-living-room", "4-room"))
    mode.add_argument("--house", type=Path, help="Load a previously saved ProcTHOR house.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-room", choices=("Kitchen", "LivingRoom", "Bedroom", "Bathroom"),
                        help="Extract this room from --house; preserve its objects and close connections to removed rooms")
    parser.add_argument("--start-near", help="Setup only: initialize at an interactable pose for one object type or ID")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Save frames/state and exit without a GUI")
    args = parser.parse_args(argv)
    if args.only_room and not args.house:
        parser.error("--only-room requires --house so the source layout is saved separately")
    print(f"Starting {'ProcTHOR ' + args.generate if args.generate else args.house or args.scene}…", flush=True)
    with ThorEnv(scene=args.scene, generate=args.generate, house=args.house,
                 seed=args.seed, output_dir=args.output_dir, start_near=args.start_near,
                 only_room=args.only_room) as env:
        print(f"Output: {env.output_dir}", flush=True)
        if args.smoke:
            from PIL import Image

            event = env.controller.last_event
            Image.fromarray(event.frame).save(env.output_dir / "ego.png")
            Image.fromarray(event.third_party_camera_frames[0]).save(env.output_dir / "topdown.png")
            print(json.dumps({"objects": len(event.metadata["objects"]),
                              "openable": [o["objectId"] for o in event.metadata["objects"] if o["openable"]],
                              "pickupable": [o["objectId"] for o in event.metadata["objects"] if o["pickupable"]]},
                             indent=2), flush=True)
            return 0
        from .gui import ThorWindow

        ThorWindow(env).run()
    return 0
