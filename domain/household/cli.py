"""Interactive and scripted household skill runner."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from .partnr import DEFAULT_PARTNR_ROOT, PROJECT_ROOT, HouseholdEnv, check_installation, skill_argument


HELP = """Commands (use actual names printed by entities):
  entities                         list rooms, furniture, objects and robot
  Navigate <entity>
  Pick <object>
  Place <object> <destination> [on|within]
  Open <furniture>
  Close <furniture>
  state                            save current privileged graph and simulator state
  topdown                          save a full-house overview image
  reset                            reset to the selected episode
  help
  exit
"""


def parse_action(command):
    parts = shlex.split(command)
    if not parts:
        raise ValueError("Empty command")
    skill = parts[0]
    if skill == "Place" and len(parts) in (3, 4):
        action = {"skill": skill, "target": parts[1], "destination": parts[2],
                  "relation": parts[3] if len(parts) == 4 else "on"}
    elif skill != "Place" and len(parts) == 2:
        action = {"skill": skill, "target": parts[1]}
    else:
        raise ValueError("Invalid command.\n" + HELP)
    skill_argument(**action)
    return action


def print_entities(state):
    print(f"Scene: {state['scene_id']}  Episode: {state['episode_id']}")
    for node in state["nodes"]:
        properties = node["properties"]
        print(f"  {node['kind']:12s} {node['name']:30s} type={properties.get('type', '')}")
    print("Relations:")
    for edge in state["edges"]:
        print(f"  {edge['source']} --{edge['relation']}--> {edge['target']}")


def ci_demo_actions(state):
    """Official test_skills.py sequence, resolving scene handles at runtime."""
    names = {node["sim_handle"]: node["name"] for node in state["nodes"]}
    try:
        drawer = names["2d93837fd0fb80b2f2e9b0d3d55eb6506b22d4c1_:0000"]
        cushion = names["pillow_9_:0000"]
        chair = names["7d8617011c72509329256bd40de9ec5547688286_:0000"]
    except KeyError as error:
        raise ValueError("This episode lacks the official CI demo entities; use interactive commands") from error
    return [{"skill": "Navigate", "target": drawer},
            {"skill": "Open", "target": drawer},
            {"skill": "Close", "target": drawer},
            {"skill": "Navigate", "target": cushion},
            {"skill": "Pick", "target": cushion},
            {"skill": "Navigate", "target": chair},
            {"skill": "Place", "target": cushion, "destination": chair}]


def main(argv=None):
    runtime_python = PROJECT_ROOT / ".runtime/partnr/bin/python"
    if runtime_python.is_file() and (Path(sys.executable).resolve() != runtime_python.resolve()
                                     or not sys.flags.no_user_site):
        os.execv(str(runtime_python), [str(runtime_python), "-s", str(PROJECT_ROOT / "main.py"),
                                      "household", *(argv if argv is not None else sys.argv[1:])])
    parser = argparse.ArgumentParser(description="Full PARTNR household with oracle semantic skills")
    parser.add_argument("--partnr-root", type=Path, default=DEFAULT_PARTNR_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Default: a new timestamped outputs/household directory")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot", choices=("spot", "fetch"), default="spot")
    parser.add_argument("--max-skill-steps", type=int, default=1500)
    parser.add_argument("--video", action="store_true", help="Save an MP4 per skill")
    parser.add_argument("--render", action="store_true", help="Live robot view with a skill-command input window")
    parser.add_argument("--check", action="store_true", help="Check prerequisites without loading the simulator")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--demo", action="store_true", help="Run the official CI household skill sequence")
    modes.add_argument("--commands", type=Path, help="JSON list of skill command strings")
    args = parser.parse_args(argv)
    if args.check:
        report = check_installation(args.partnr_root, args.robot)
        print(json.dumps(report, indent=2))
        return 0 if report["ready_for_runtime_attempt"] else 1
    # Validate the entire script before any simulator action or working-directory change.
    commands = None
    if args.commands:
        commands = json.loads(args.commands.read_text())
        if not isinstance(commands, list) or not all(isinstance(item, str) for item in commands):
            parser.error("--commands must contain a JSON list of skill command strings")
        commands = [parse_action(command) for command in commands]
    try:
        with HouseholdEnv(args.partnr_root, args.output_dir, args.episode_index,
                          args.seed, args.video, args.max_skill_steps, robot=args.robot) as env:
            if args.demo:
                commands = ci_demo_actions(env.state())
            if args.render:
                from .gui import HouseholdWindow

                return HouseholdWindow(env).run(commands)
            print_entities(env.state())
            print(f"Household overview: {env.save_topdown()}")
            print(f"Output: {env.output_dir}")
            if commands is not None:
                for action in commands:
                    result = env.execute_skill(**action)
                    print(f"{action}: {result['response']} ({result['control_steps']} control steps)")
                    if not result["success"]:
                        return 2
                return 0
            print(HELP)
            while True:
                try:
                    command = input("household> ").strip()
                except EOFError:
                    break
                if command in {"exit", "quit"}:
                    break
                try:
                    if not command:
                        continue
                    if command == "help":
                        print(HELP)
                    elif command == "entities":
                        print_entities(env.state())
                    elif command == "state":
                        path = env.output_dir / "current_state.json"
                        path.write_text(json.dumps(env.state(), indent=2))
                        print(path)
                    elif command == "topdown":
                        print(env.save_topdown())
                    elif command == "reset":
                        print_entities(env.reset())
                    else:
                        result = env.execute_skill(**parse_action(command))
                        print(json.dumps({key: value for key, value in result.items()
                                          if key not in {"state_before", "state_after"}}, indent=2))
                except ValueError as error:
                    print(error)
    except (RuntimeError, ImportError, OSError, ValueError) as error:
        print(f"Household startup/execution failed: {error}")
        return 1
    return 0
