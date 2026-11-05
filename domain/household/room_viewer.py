"""Launch the upstream Habitat-Sim mouse viewer in a small configurable bedroom."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=750)
    args = parser.parse_args(argv)
    if args.width < 1 or args.height < 1:
        parser.error("Window dimensions must be positive")
    runtime_python = ROOT / ".runtime/partnr/bin/python"
    if Path(sys.executable).resolve() != runtime_python.resolve() or not sys.flags.no_user_site:
        if not runtime_python.is_file():
            parser.error("Install the viewer runtime first: bash scripts/setup_room_viewer.sh")
        # The user's Python 3.9 site-packages contains NumPy 2; Habitat requires 1.x.
        os.execv(str(runtime_python), [str(runtime_python), "-s", str(ROOT / "main.py"),
                                    "room", *(argv if argv is not None else sys.argv[1:])])
    source = ROOT / "third_party/habitat-sim/examples/viewer.py"
    room = ROOT / "data/habitat_demo/van-gogh-room.glb"
    if not source.is_file() or not room.is_file():
        parser.error("Viewer source or room asset missing: bash scripts/setup_room_viewer.sh")
    # The upstream viewer sets RTLD_GLOBAL before importing Magnum/Habitat so
    # the window and simulator share one OpenGL context. Preserve that order.
    spec = importlib.util.spec_from_file_location("habitat_official_viewer", source)
    viewer_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = viewer_module
    spec.loader.exec_module(viewer_module)
    import habitat_sim
    import numpy as np
    from habitat_sim.utils.common import quat_from_angle_axis
    from habitat_sim.utils.settings import default_sim_settings

    settings = default_sim_settings.copy()
    settings.update({
        "scene": "bedroom",
        "scene_dataset_config_file": str(ROOT / "configs/room_demo/room.scene_dataset_config.json"),
        "enable_physics": True, "use_default_lighting": True,
        "enable_batch_renderer": False, "num_environments": 1,
        "composite_files": None, "window_width": args.width, "window_height": args.height,
        "default_agent_navmesh": False, "enable_hbao": False,
    })
    viewer = viewer_module.HabitatSimInteractiveViewer(settings)
    state = habitat_sim.AgentState()
    state.position = viewer_module.mn.Vector3(1.8, 0.18, 0.0)
    state.rotation = quat_from_angle_axis(3 * math.pi / 4, np.array([0, 1, 0]))
    viewer.sim.initialize_agent(viewer.agent_id, state)
    print("Small bedroom ready. WASD: move; left-drag: look; M: grab mode; Esc: quit.", flush=True)
    viewer.exec()
    return 0
