"""Deterministic primitive sequence used to calibrate cabinet grasping."""

from __future__ import annotations

from typing import Callable

import numpy as np
from omegaconf import DictConfig

from domain.robocasa.primitives import PrimitiveAction


# This deliberately uses only the public 20D RAPS interface. It is a
# calibration baseline, not an object-aware controller used by PPO.
SCRIPTED_MUG_SEQUENCE = (
    ("open_gripper", {}),
    # The reset pose is aligned with cabinet_item_16 at seed 0-4:
    # x needs about 10 cm of insertion and z only about 3 cm of correction.
    ("move_delta_ee_pose", {"dx": 0.0, "dy": 0.0, "dz": 0.03}),
    # Horizontal side grasp: stop on simultaneous two-finger contact and
    # close while holding the TCP position.
    ("top_grasp", {"x_forward": 0.10, "y_offset": 0.0, "z_offset": 0.0}),
    ("lift", {"z_dist": 0.10}),
)

SCRIPTED_GRASP_SEQUENCES = {
    "cabinet_item_16": SCRIPTED_MUG_SEQUENCE,
    "cabinet_item_17": (
        ("open_gripper", {}),
        # Grasp the resized bowl above its center to clear the shelf.
        ("top_grasp", {"x_forward": 0.10, "y_offset": 0.109, "z_offset": 0.02}),
        ("lift", {"z_dist": 0.10}),
    ),
}


def run_scripted_grasp(cfg: DictConfig, env_factory: Callable[[DictConfig], object]) -> int:
    """Run a fixed one-object grasp attempt for each configured seed."""
    target_object = str(cfg.scripted.target_object)
    sequence = SCRIPTED_GRASP_SEQUENCES[target_object]
    env = env_factory(cfg)
    successes = 0
    try:
        for seed in cfg.scripted.seeds:
            env.reset(seed=int(seed))
            target_grasped = False
            target_lifted = False
            contact = False
            timeout = False
            final_tcp = None
            for primitive_index, (name, parameters) in enumerate(
                sequence, start=1
            ):
                action = PrimitiveAction(name, parameters).to_vector()
                _, _, terminated, truncated, info = env.step(action)
                report = info.get("exploration_reward", {})
                raw = report.get("raw", {})
                target_grasped |= target_object in report.get(
                    "ever_grasped_object_ids", []
                )
                target_lifted |= target_object in report.get(
                    "lifted_object_ids", []
                )
                contact |= target_object in report.get("contacted_object_ids", [])
                timeout |= bool(raw.get("primitive_timeout", False))
                record = info.get("primitive_record")
                if record is not None:
                    final_tcp = record.outcome_metrics.get("final_tcp_position")
                print(
                    f"seed={int(seed):02d} primitive={primitive_index:02d} "
                    f"type={name} status={getattr(record, 'status', 'unknown')} "
                    f"grasped={report.get('ever_grasped_object_ids', [])} "
                    f"lifted={report.get('lifted_object_ids', [])} "
                    f"forces={record.outcome_metrics.get('max_finger_contact_force', {})} "
                    f"moved={record.outcome_metrics.get('object_translations', {})}"
                )
                if terminated or truncated:
                    break
            successes += int(target_lifted)
            print(
                f"scripted seed={int(seed):02d} target={target_object} "
                f"grasped={target_grasped} lifted={target_lifted} "
                f"success={target_lifted} contact={contact} timeout={timeout} "
                f"final_tcp={np.asarray(final_tcp).tolist() if final_tcp is not None else None}"
            )
    finally:
        env.close()

    total = len(cfg.scripted.seeds)
    print(f"Scripted grasp success: {successes}/{total} target={target_object}")
    return 0 if successes == total else 1
