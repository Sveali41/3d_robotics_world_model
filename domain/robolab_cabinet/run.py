"""Render target/mini cabinet previews using the installed RoboLab stack."""
import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2  # Must precede Isaac Lab imports.
import yaml
from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/robolab_cabinet.yaml")
    parser.add_argument("--variant", choices=["target", "mini", "both"], default="both")
    parser.add_argument("--num-steps", type=int, default=40)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_steps < 1:
        parser.error("--num-steps must be positive")
    args.enable_cameras = True
    app = AppLauncher(args).app
    try:
        import torch
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env, end_episode
        from robolab.core.utils.isaaclab_compat import as_torch
        from robolab.core.utils.video_utils import VideoWriter
        from domain.robolab_cabinet.environment import register_cabinet

        config = yaml.safe_load(args.config.read_text())
        variants = ["target", "mini"] if args.variant == "both" else [args.variant]
        for variant in variants:
            output = Path("/opt/robolab/output") / f"cabinet_{variant}"
            output.mkdir(parents=True, exist_ok=True)
            set_output_dir(str(output))
            name, spec = register_cabinet(config, variant, output)
            env, env_cfg = create_env(name, device=args.device, num_envs=1)
            writer = VideoWriter(str(output / "preview_env0.mp4"), fps=config["control_freq"])
            try:
                env.reset()
                tracked = [report_item["name"] for report_item in spec["items"]] + ["door"]
                initial_poses = {name: as_torch(env.scene[name].data.root_pose_w).clone()
                                 for name in tracked}
                # Hold the authored reset posture; this preview is not an exploration policy.
                robot = env.scene["robot"]
                action = torch.zeros((1, 8), device=env.device)
                action[:, :7] = as_torch(robot.data.joint_pos)[:, :7]
                for step in range(args.num_steps):
                    obs, _, _, _, _ = env.step(action)
                    frame = obs["image_obs"]["cabinet_camera"][0].cpu().numpy()
                    writer.write(frame)
                    if step == args.num_steps - 1:
                        cv2.imwrite(str(output / "preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                # Check that the same scene can reset after stepping.
                # RoboLab freezes finished eval environments until this public reset hook is called.
                env.reset_eval_state()
                env.reset()
                for name, pose in initial_poses.items():
                    torch.testing.assert_close(as_torch(env.scene[name].data.root_pose_w), pose,
                                               atol=1e-5, rtol=1e-5)
                report = {
                    "variant": variant, "num_items": len(spec["items"]),
                    "item_names": [item["name"] for item in spec["items"]],
                    "steps": args.num_steps, "reset_after_steps": True,
                    "action": "7 absolute arm joint angles (rad), gripper: 0=open, 1=close",
                    "task_success": None,
                    "note": "Reward-free exploration scene; no task success metric defined.",
                    "source": spec["source"], "source_sha256": spec["source_sha256"],
                    "config": config,
                }
                (output / "preview_report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"CABINET_PREVIEW_OK {variant}: {len(spec['items'])} items; {output}", flush=True)
                end_episode(env)
            finally:
                writer.release()
                env.close()
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        app.close()


if __name__ == "__main__":
    main()
