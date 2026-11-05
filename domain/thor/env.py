"""Pinned upstream THOR runtime, house generation, and action recording."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNITY_ROOT = PROJECT_ROOT / ".runtime" / "thor-unity"
THOR_COMMIT = "f0825767cd50d69f666c7f282e54abfe58f1e917"
PROCTHOR_COMMIT = "53d5bd4c8c96a699e6a615dc390abb670cc9d353"


def create_controller():
    from ai2thor.controller import Controller
    from ai2thor.platform import Linux64

    # Upstream hardcodes ~/.ai2thor; keep builds and logs in this workspace.
    class ProjectController(Controller):
        @property
        def base_dir(self):
            return str(UNITY_ROOT)

    executable = UNITY_ROOT / "releases" / f"thor-Linux64-{THOR_COMMIT}" / f"thor-Linux64-{THOR_COMMIT}"
    if not executable.is_file():
        raise RuntimeError("Unity runtime is absent. Download it explicitly with "
                           "'bash scripts/setup_thor.sh unity' (about 807 MB).")
    return ProjectController(
        commit_id=THOR_COMMIT, platform=Linux64, scene="Procedural",
        width=640, height=480, quality="Low", gridSize=0.25,
        rotateStepDegrees=90, visibilityDistance=1.5, snapToGrid=True,
        renderInstanceSegmentation=True, server_start_timeout=120,
    )


def require_success(event, operation):
    if not event.metadata["lastActionSuccess"]:
        raise RuntimeError(f"{operation}: {event.metadata['errorMessage']}")
    return event


def snapshot(event):
    return {key: event.metadata.get(key) for key in
            ("sceneName", "sceneBounds", "agent", "objects", "inventoryObjects",
             "lastAction", "lastActionSuccess", "errorMessage")}


class ThorEnv:
    def __init__(self, *, scene="FloorPlan1", generate=None, house=None, seed=42,
                 output_dir=None, start_near=None, only_room=None):
        self.output_dir = Path(output_dir or PROJECT_ROOT / "outputs" / "thor" /
                               datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / "run.json").exists():
            raise FileExistsError(f"Choose a new output directory: {self.output_dir}")
        self.controller = None
        self.house = None
        self.scene = scene
        self.seed = seed
        self.decision = 0
        self.resets = 0
        self.start_pose = None
        try:
            self.controller = create_controller()
            if generate:
                from procthor.generation import HouseGenerator

                generator = HouseGenerator(split="train", seed=seed, room_spec=generate,
                                           controller=self.controller)
                generated, _ = generator.sample()
                warnings = generated.validate(self.controller)
                if warnings:
                    raise RuntimeError(f"Generated house did not pass upstream validation: {warnings}")
                self.house = generated.data
            elif house:
                self.house = json.loads(Path(house).read_text())
            self.reset()
            if only_room:
                from .rooms import extract_room

                if self.house is None:
                    raise ValueError("Room extraction requires a ProcTHOR house JSON")
                positions = require_success(self.controller.step(action="GetReachablePositions"),
                                            "Source reachable positions").metadata["actionReturn"]
                self.house = extract_room(self.house, only_room, positions)
                self.reset()
            if start_near:
                candidates = [obj for obj in self.controller.last_event.metadata["objects"]
                              if obj["objectId"] == start_near or obj["objectType"] == start_near]
                if len(candidates) != 1:
                    raise ValueError(f"--start-near requires one matching object; found {len(candidates)}. "
                                     "Use an exact objectId if the type is ambiguous.")
                poses = require_success(self.controller.step(
                    action="GetInteractablePoses", objectId=candidates[0]["objectId"],
                    rotations=[0, 90, 180, 270], horizons=[0, 30], standings=[True]),
                    "Find initial interaction pose").metadata["actionReturn"]
                if not poses:
                    raise RuntimeError(f"No initial interaction pose for {start_near}")
                self.start_pose = poses[0]
                if self.house is not None:
                    self.house["metadata"]["agent"] = self.start_pose
                self.reset()
            if self.house is not None:
                (self.output_dir / "house.json").write_text(json.dumps(self.house, indent=2))
            (self.output_dir / "run.json").write_text(json.dumps({
                "ai2thor_commit": THOR_COMMIT, "procthor_commit": PROCTHOR_COMMIT,
                "seed": seed, "room_spec": generate, "scene": scene if self.house is None else "Procedural",
                "initialization_start_near": start_near, "initialization_pose": self.start_pose,
                "extracted_room": only_room,
                "grid_size_metres": 0.25, "rotation_degrees": 90,
                "visibility_distance_metres": 1.5, "force_action": False,
                "pickup": "upstream iTHOR abstract hand; no robot arm controller",
                "put_place_stationary": True, "privileged_state": True,
            }, indent=2))
        except BaseException:
            self.close()
            raise

    def reset(self):
        if self.house is None:
            self.controller.reset(scene=self.scene)
            if self.start_pose is not None:
                require_success(self.controller.step(action="TeleportFull", **self.start_pose), "Initial agent pose")
        else:
            self.controller.reset(scene="Procedural")
            require_success(self.controller.step(action="CreateHouse", house=self.house), "CreateHouse")
            require_success(self.controller.step(action="TeleportFull", **self.house["metadata"]["agent"]),
                            "Initial agent pose")
        require_success(self.controller.step(action="SetRandomSeed", seed=self.seed), "Random seed")
        camera = require_success(self.controller.step(action="GetMapViewCameraProperties"), "Map camera")
        camera_parameters = {**camera.metadata["actionReturn"], "skyboxColor": "black"}
        if camera_parameters.get("orthographic"):
            camera_parameters["orthographicSize"] *= 1.1
        require_success(self.controller.step(action="AddThirdPartyCamera", **camera_parameters),
                        "Add map camera")
        self.resets += 1
        (self.output_dir / f"reset_{self.resets}.json").write_text(
            json.dumps(snapshot(self.controller.last_event), indent=2))
        return self.controller.last_event

    def step(self, action, **parameters):
        before = snapshot(self.controller.last_event)
        event = self.controller.step(action=action, **parameters)
        self.decision += 1
        record = {"decision": self.decision, "reset": self.resets, "action": action, "parameters": parameters,
                  "state_before": before, "state_after": snapshot(event)}
        with (self.output_dir / "transitions.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return event

    def close(self):
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
