"""State-based RAPS/GNN imagination actor; an explicit alternative to iCEM.

Reuses the vendored RAPS conditional actor, distributions, MLP and lambda-return.
This is a research adaptation, not the original image/RSSM Plan2Explore trainer.
Each model transition is one completed skill. All physical ticks are also kept.
"""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import sys
import time
import tempfile
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch

from domain.robocasa.primitives import RAPS_ACTION_SCHEMA, RAPS_COMPACT_ACTION_SCHEMA, RAPS_SPLIT_ACTION_SCHEMA
from domain.robocasa.grid_pick_place import GridPickPlaceActionSchema
from domain.robocasa.grasp_orientations import GRASP_ORIENTATION_YAW_PITCH_DEG
from domain.robocasa.skill_execution import RAPSSkillExecution, PlannedSkillExecution
from domain.robocasa.pick_place import (
    PICK_PLACE_ACTION_SCHEMA, TCP_BOUNDS, PickPlaceActionSchema,
    PlannedPickPlaceSkillExecution, pick_place_contract,
)
from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.cee_us.cee_us_cabinet import (
    ROOT, make_env, prepare_output, replay_arrays, restore_rng, rng_state,
)
from exploration.cee_us.cee_us_metrics import interaction_summary, prediction_metrics, screen
from exploration.raps.cpu_sampling import CpuRoundCollector, sampling_contract
from exploration.diagnostics.diagnose_cabinet import jsonable, snapshot, write_json


SCHEMA = "cabinet-raps-gnn-actor-v1"
RAPS_REFERENCE_COMMIT = "1c56b0fe367cc32d3ee548f407ac3a9af9e3ce2e"
DEFAULTS = dict(
    reconfigure_on_reset=False,
    pick_place_primitives=False, grasp_retention="friction",
    skill_set="legacy10", hidden_size=128, actor_layers=2, critic_layers=3, init_std=1.0,
    imagination_horizon=5, imagination_batch_size=64, updates_per_round=100,
    discount=0.8, lambda_return=0.95, actor_lr=8e-5, critic_lr=8e-5,
    adam_eps=1e-5, entropy_scale=1e-4, gradient_clip=100.0,
    target_update_period=100, hold_closed_during_motion=False,
    latched_gripper=False, stop_drop_on_support=False,
    grasp_stop_on_contact=True,
    controlled_placement=False,
    disagreement_weight=1.0,
    motion_execution="ee_delta", motion_plan_max_steps=400,
)


def reference_components(device):
    """Import the checked-in reference, never an unrelated installed rlkit."""
    source = ROOT / "exploration/raps/official/rlkit"
    if not (source / "rlkit/torch/model_based/dreamer/actor_models.py").is_file():
        raise FileNotFoundError(f"Official RAPS source is required at {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import rlkit
    if Path(rlkit.__file__).resolve().parent != source / "rlkit":
        raise ImportError("Another rlkit is already imported; use a fresh process for RAPS")
    from rlkit.torch import pytorch_util as ptu
    from rlkit.torch.model_based.dreamer.actor_models import ConditionalActorModel, TanhBijector
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.utils import lambda_return
    # PyTorch 2 requires transform metadata omitted by the torch-1.7 source.
    # Keep the reference tanh, inverse clamp and Jacobian calculations intact.
    TanhBijector.domain = torch.distributions.constraints.real
    TanhBijector.codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    ptu.set_gpu_mode(torch.device(device).type == "cuda", torch.device(device).index or 0)
    return ConditionalActorModel, Mlp, lambda_return


def skill_schema(settings):
    from domain.robocasa.cabinet_grid import resolve_grid_layout
    grid_layout = settings.get("grid_layout", "legacy_3x4")
    resolve_grid_layout(grid_layout)
    if grid_layout != "legacy_3x4" and not settings.get("pick_place_grid", False):
        raise ValueError("grid_layout requires pick_place_grid")
    if not isinstance(settings.get("grasp_orientation_candidates", False), bool):
        raise ValueError("raps_actor.grasp_orientation_candidates must be boolean")
    if settings.get("grasp_orientation_candidates", False) and not (
            settings.get("pick_place_grid", False) and settings.get("pick_place_primitives", False)
            and settings.get("motion_execution") == "planner"):
        raise ValueError("grasp_orientation_candidates requires pick_place_grid, pick_place_primitives and planner motion_execution")
    if not isinstance(settings.get("pick_place_grid", False), bool):
        raise ValueError("raps_actor.pick_place_grid must be boolean")
    if settings.get("pick_place_grid", False) and not settings.get("pick_place_primitives", False):
        raise ValueError("pick_place_grid requires pick_place_primitives")
    if settings.get("grasp_only", False) and not settings.get("pick_place_primitives", False):
        raise ValueError("grasp_only requires pick_place_primitives")
    if "grasp_z_bounds_m" in settings and not settings.get("pick_place_primitives", False):
        raise ValueError("grasp_z_bounds_m requires pick_place_primitives")
    if settings.get("pick_place_primitives", False):
        specs = ((PICK_PLACE_ACTION_SCHEMA.spec_by_name["grasp"],)
                 if settings.get("grasp_only", False) else PICK_PLACE_ACTION_SCHEMA.specs)
        if "grasp_z_bounds_m" not in settings:
            if settings.get("pick_place_grid", False):
                return GridPickPlaceActionSchema(grasp_only=settings.get("grasp_only", False),
                    grasp_orientation_candidates=settings.get("grasp_orientation_candidates", False),
                    grid_layout=grid_layout)
            return PickPlaceActionSchema(specs) if settings.get("grasp_only", False) else PICK_PLACE_ACTION_SCHEMA
        bounds = np.asarray(settings["grasp_z_bounds_m"], dtype=float)
        original = TCP_BOUNDS["grasp"][2]
        if (bounds.shape != (2,) or not np.isfinite(bounds).all()
                or not original[0] <= bounds[0] < bounds[1] <= original[1]):
            raise ValueError(f"grasp_z_bounds_m must be an ordered pair within {original}, in root meters")
        spec = PICK_PLACE_ACTION_SCHEMA.spec_by_name["grasp"]
        z = replace(spec.parameters[2], scale=float((bounds[1] - bounds[0]) / 2),
                    center=float((bounds[1] + bounds[0]) / 2))
        if settings.get("pick_place_grid", False):
            return GridPickPlaceActionSchema(z, grasp_only=settings.get("grasp_only", False),
                grasp_orientation_candidates=settings.get("grasp_orientation_candidates", False),
                grid_layout=grid_layout)
        grasp = replace(spec, parameters=spec.parameters[:2] + (z,))
        return PickPlaceActionSchema(tuple(grasp if item.name == "grasp" else item
                                          for item in specs))
    name = settings.get("skill_set", "legacy10")
    if name == "legacy10":
        return RAPS_ACTION_SCHEMA
    if name == "compact6":
        return RAPS_COMPACT_ACTION_SCHEMA
    if name == "split7":
        return RAPS_SPLIT_ACTION_SCHEMA
    raise ValueError(f"Unknown raps_actor.skill_set: {name!r}; use legacy10, compact6 or split7")


def action_contract(settings):
    schema = skill_schema(settings)
    contract = dict(
        action_dim=schema.action_dim, primitive_names=list(schema.primitive_names),
        parameter_indices=schema.primitive_name_to_action_idx,
        parameters={spec.name: {p.name: p.scale for p in spec.parameters} for spec in schema.specs},
        model_transition="completed_skill_endpoint", velocity_dt_seconds=0.05,
        controller="existing_cabinet_side_grasp_on_ee_delta_reset_orientation",
        hold_closed_during_motion=settings["hold_closed_during_motion"],
        inactive_model_arguments="zero", policy_gradient="REINFORCE_all_reference_arguments",
        disagreement="sample_std_dynamic_state_sum_at_shared_state_action",
        discount="fixed_per_skill_no_learned_termination", state_propagation="ensemble_mean",
    )
    if settings.get("skill_set") == "split7" and not settings.get("pick_place_primitives", False):
        contract["controller"] = "split_yz_x_on_ee_delta_reset_orientation_v1"
        contract["split_motion"] = dict(
            frame="controller_root", gripper="preserve_latched_command",
            align_yz=dict(tolerance_m=.015, normalized_speed=.30, max_control_steps=100),
            approach_x=dict(direction="positive_x_abs_distance", tolerance_m=.012,
                            normalized_speed=.08, max_control_steps=150,
                            contact_threshold_N=.05, contact_stop="both_fingers_same_object"),
            automatic_gripper_closure=False, fixed_skill_order=False)
    if settings["latched_gripper"] or settings["stop_drop_on_support"]:
        contract.update(latched_gripper=settings["latched_gripper"],
                        stop_drop_on_support=settings["stop_drop_on_support"],
                        ground_support_threshold_N=0.05,
                        controller_state_in_observation=settings["latched_gripper"],
                        command_memory_excluded_from_curiosity=True)
    if not settings["grasp_stop_on_contact"]:
        contract["grasp_stop_on_contact"] = False
    if settings["controlled_placement"]:
        contract["controlled_placement"] = dict(
            version=1, slowdown_distance_m=0.08, min_normalized_speed=0.025,
            drop_goal_tolerance_m=0.002, hold_position_tolerance_m=0.003,
            stable_speed_m_s=0.01, stable_contact_checks=4,
            settling_budget_steps=20, gripper_position_hold=True,
            support_and_gripper_orientation="actual_skill_pose",
            upward_retreat_actual_orientation_below_m=0.15)
    if settings.get("disagreement_weight", 1.0) != 1.0:
        contract["disagreement_weight"] = settings["disagreement_weight"]
    if settings.get("reconfigure_on_reset", False):
        contract["physics_reset"] = "fresh_scene_on_episode_v1"
    if settings.get("grasp_target_heuristic", False):
        contract["grasp_target_heuristic"] = dict(
            version=2, formula="approach_weight * exp(-min_eligible_object_distance_m / distance_scale_m)",
            goal="schema_decoded_grasp_XYZ", object_centers="current_imagined_state_root_meters",
            object_weights="positive_values_select_eligible_objects_only",
            selected_grasp="replace_endpoint_TCP_progress_heuristic",
            selected_place="zero_heuristic_curiosity_unchanged", disabled="object_heuristic_is_None")
    if settings.get("grasp_preflight", False):
        contract["grasp_preflight"] = dict(version=1,
            route="open_hand_retreat_align_approach_in_current_scene",
            proposal="same_policy_rejection_sampling_no_random_fallback",
            max_candidates=settings.get("grasp_preflight_max_candidates", 32),
            invalid_preconditions="reject_before_physics", rejected_transitions="audit_only",
            imagination="unchanged_unfiltered", checked_plans="discard_and_replan_before_execution")
    if settings.get("motion_execution", "ee_delta") == "planner":
        contract["controller"] = "cabinet_shared_mplib_joint_plan_v1"
        contract["motion_execution"] = dict(
            planner="CabinetMotionPlanner", algorithm="RRTConnect", control_mode="pd_joint_pos",
            low_level_action="seven_joint_targets_rad_plus_normalized_gripper",
            max_control_steps_per_skill=settings.get("motion_plan_max_steps", 400),
            target_orientation="episode_reset_or_explicit_actual_pose_hold",
            allowed_contact_objects="all_current_cabinet_items_finger_links_only",
            endpoint_position_tolerance_m=.002, endpoint_angle_tolerance_deg=.5,
            settling_steps=15, plan_failure="record_without_fallback",
            planning_seed="episode_local_segment_index", intermediate_axes="planner_free",
            bounded_ik_solver="absolute_central_jacobian_1e-4rad_lsmr_v2",
            legacy_composite_opening="stationary_before_alignment")
        if "split_motion" in contract:
            contract["split_motion"] = dict(frame="controller_root", gripper="preserve_latched_command",
                align_yz="relative_yz_target_preserve_target_x",
                approach_x="relative_positive_x_target_preserve_target_yz",
                automatic_gripper_closure=False, fixed_skill_order=False,
                ee_delta_speed_caps_used=False)
        if settings["controlled_placement"]:
            contract["controlled_placement"] = dict(version="joint_plan_v1", ground_support_threshold_N=.05,
                stable_speed_m_s=.01, stable_contact_checks=4, settling_budget_steps=20,
                hold="current_joint_position", ee_delta_slowdown_used=False)
    if settings.get("pick_place_primitives", False):
        contract["controller"] = "bounded_fixed_orientation_planned_grasp_place_v1"
        contract.pop("ground_support_threshold_N", None)
        contract["pick_place"] = pick_place_contract(settings.get("grasp_retention", "friction"),
            grasp_orientation_candidates=settings.get("grasp_orientation_candidates", False))
        if settings.get("grasp_only", False):
            contract["grasp_only"] = dict(version=2, primitive_names=list(schema.primitive_names),
                action_dim=schema.action_dim, reset_after_skill="grasp_success",
                failure="reset_if_still_holding_else_continue_current_scene",
                endpoint="before_reset", terminal_flags="environment_only",
                preparation=dict(version="planned_default_joints_before_each_grasp_v2",
                    route="open_then_fixed_orientation_exterior_retreat_then_default_joints",
                    exterior_retreat_tolerance_m=.001,
                    observation="after_home_before_action_selection", object_reset=False,
                    controls="counted_nontrainable_return_default_records",
                    joint_tolerance_rad=.001, joint_speed_tolerance_rad_s=.01,
                    failure="stop_collection_without_grasp_or_reset_fallback"))
            contract["pick_place"]["action_dim"] = schema.action_dim
        if settings.get("pick_place_grid", False):
            contract["pick_place_grid"] = dict(
                version=1, xy_root_m=[list(xy) for xy in schema.grid_xy_root_m],
                branches=list(schema.primitive_names), ordering="row_major_x_depth_then_y_column",
                grasp_z_root_m=[schema.grasp_z.center - schema.grasp_z.scale,
                                schema.grasp_z.center + schema.grasp_z.scale],
                place_z="automatic_existing_geometry_and_lower_shelf_support",
                action_dim=schema.action_dim, categorical_dim=schema.num_primitives, continuous_dim=1)
            contract["pick_place"]["action_dim"] = schema.action_dim
            if schema.grid_layout == "trapezoid_4_3_2_1":
                parameters = schema.physical_action_schema().spec_by_name["grasp"].parameters
                contract["pick_place"]["tcp_bounds_root_m"]["grasp"] = [
                    [parameter.center - parameter.scale, parameter.center + parameter.scale]
                    for parameter in parameters]
                contract["pick_place_grid"]["physical_validation"] = "grid_target_envelope; exact_cell_action_encoding"
                contract["pick_place_grid"]["cup_only_reset_contacts"] = "invalid_until_first_physics_substep; active_binding_checked_separately"
            if schema.grid_layout != "legacy_3x4":
                contract["pick_place_grid"]["layout"] = schema.grid_layout
                contract["pick_place_grid"]["grasp_xy_reference"] = "tcp_root"
                contract["pick_place_grid"]["place_xy_reference"] = "collision_AABB_center_at_reset_orientation"
        if settings.get("grasp_orientation_candidates", False):
            contract["controller"] = "bounded_selected_orientation_planned_grasp_place_v1"
            contract["grasp_orientation_candidates"] = dict(
                version=1, yaw_pitch_degrees=[list(GRASP_ORIENTATION_YAW_PITCH_DEG[index])
                                             for index in schema.grasp_orientation_ids],
                rotation="Rz(root_yaw) @ Ry(root_pitch) @ R_reset",
                grasp_argument="orientation_id", ordering="grasp_cell_major_then_orientation_id_then_place_cell",
                branch_cell_indices=list(schema.branch_cell_indices),
                branch_orientation_indices=list(schema.branch_orientation_indices),
                num_grasp_branches=schema.num_grasp_branches,
                alignment="rotate_to_selected_orientation_only_at_exterior_alignment",
                approach="fixed_selected_orientation", place="exterior_preplace_to_reset_orientation")
            if len(schema.grasp_orientation_ids) != len(GRASP_ORIENTATION_YAW_PITCH_DEG):
                contract["grasp_orientation_candidates"]["orientation_ids"] = list(schema.grasp_orientation_ids)
            contract["pick_place_grid"]["ordering"] = "grasp_cell_major_then_orientation_id_then_place_cell"
            contract["pick_place_grid"]["version"] = 2
        if "grasp_z_bounds_m" in settings:
            z = schema.grasp_z if settings.get("pick_place_grid", False) else schema.spec_by_name["grasp"].parameters[2]
            contract["pick_place"]["tcp_bounds_root_m"]["grasp"][2] = [z.center - z.scale, z.center + z.scale]
        if "motion_execution" in contract:
            motion = contract["motion_execution"]
            motion["max_control_steps_per_segment"] = motion.pop("max_control_steps_per_skill")
            motion["target_orientation"] = "shared_reset_orientation_path_and_feedback_audited"
            if settings.get("grasp_orientation_candidates", False):
                motion["target_orientation"] = "exterior_alignment_to_selected_grasp_orientation_then_fixed_approach_exterior_preplace_reset"
            motion["intermediate_axes"] = "shared_macro_waypoints_with_checked_joint_paths"
            motion.pop("legacy_composite_opening", None)
    return contract


def action_contract_matches(contract, settings):
    """Accept the name-only top_grasp -> approach migration, retaining all checks."""
    expected = action_contract(settings)
    if contract == expected:
        return True
    if "approach" not in expected["primitive_names"]:
        return False
    legacy = deepcopy(expected)
    legacy["primitive_names"] = ["top_grasp" if name == "approach" else name
                                 for name in legacy["primitive_names"]]
    for field in ("parameter_indices", "parameters"):
        legacy[field]["top_grasp"] = legacy[field].pop("approach")
    return contract == legacy


def canonical_action(actions, schema=RAPS_ACTION_SCHEMA):
    """The endpoint model sees only arguments that the real controller consumes."""
    if actions.shape[-1] != schema.action_dim:
        raise ValueError(f"RAPS actions must have {schema.action_dim} entries")
    mask = actions.new_zeros((schema.num_primitives, schema.num_parameters))
    for index, name in enumerate(schema.primitive_names):
        mask[index, schema.primitive_name_to_action_idx[name]] = 1
    indices = actions[..., :schema.num_primitives].argmax(-1)
    onehot = torch.nn.functional.one_hot(indices, schema.num_primitives).to(actions)
    return torch.cat((onehot, actions[..., schema.num_primitives:] * mask[indices]), -1)


class RAPSImaginationActor:
    def __init__(self, backend, settings=None):
        self.backend = backend
        self.settings = {**DEFAULTS, **(settings or {})}
        unknown = set(self.settings) - set(DEFAULTS) - {"grasp_preflight", "grasp_preflight_max_candidates", "grasp_target_heuristic", "grasp_z_bounds_m", "grasp_only", "pick_place_grid", "grasp_orientation_candidates", "grid_layout"}
        if unknown:
            raise ValueError(f"Unknown raps_actor settings: {sorted(unknown)}")
        if not isinstance(self.settings.get("grasp_only", False), bool):
            raise ValueError("raps_actor.grasp_only must be boolean")
        if self.settings.get("grasp_only", False) and not self.settings["pick_place_primitives"]:
            raise ValueError("grasp_only requires pick_place_primitives")
        if not isinstance(self.settings.get("grasp_target_heuristic", False), bool):
            raise ValueError("raps_actor.grasp_target_heuristic must be boolean")
        if self.settings.get("grasp_target_heuristic", False) and not self.settings["pick_place_primitives"]:
            raise ValueError("grasp_target_heuristic requires pick_place_primitives")
        if not isinstance(self.settings.get("grasp_preflight", False), bool):
            raise ValueError("raps_actor.grasp_preflight must be boolean")
        candidates = self.settings.get("grasp_preflight_max_candidates", 32)
        if isinstance(candidates, bool) or int(candidates) != candidates or candidates < 1:
            raise ValueError("raps_actor.grasp_preflight_max_candidates must be a positive integer")
        if self.settings.get("grasp_preflight", False) and not self.settings["pick_place_primitives"]:
            raise ValueError("grasp_preflight requires planned pick_place_primitives")
        for name in ("hidden_size", "actor_layers", "critic_layers", "imagination_horizon",
                     "imagination_batch_size", "updates_per_round", "target_update_period"):
            if int(self.settings[name]) != self.settings[name] or self.settings[name] < 1:
                raise ValueError(f"raps_actor.{name} must be a positive integer")
        for name in ("actor_lr", "critic_lr", "adam_eps", "gradient_clip", "init_std"):
            if not np.isfinite(self.settings[name]) or self.settings[name] <= 0:
                raise ValueError(f"raps_actor.{name} must be finite and positive")
        for name in ("discount", "lambda_return"):
            if not np.isfinite(self.settings[name]) or not 0 <= self.settings[name] <= 1:
                raise ValueError(f"raps_actor.{name} must be in [0, 1]")
        if not np.isfinite(self.settings["entropy_scale"]) or self.settings["entropy_scale"] < 0:
            raise ValueError("raps_actor.entropy_scale must be finite and nonnegative")
        if not np.isfinite(self.settings["disagreement_weight"]) or self.settings["disagreement_weight"] < 0:
            raise ValueError("raps_actor.disagreement_weight must be finite and nonnegative")
        for name in ("hold_closed_during_motion", "latched_gripper", "stop_drop_on_support", "grasp_stop_on_contact", "controlled_placement", "pick_place_primitives", "reconfigure_on_reset"):
            if not isinstance(self.settings[name], bool):
                raise ValueError(f"{name} must be boolean")
        if self.settings["grasp_retention"] not in {"friction", "constraint"}:
            raise ValueError("grasp_retention must be friction or constraint")
        if self.settings["grasp_retention"] == "constraint" and not self.settings["pick_place_primitives"]:
            raise ValueError("constraint retention requires pick_place_primitives")
        if self.settings["pick_place_primitives"]:
            if self.settings["motion_execution"] != "planner" or not self.settings["latched_gripper"]:
                raise ValueError("pick_place_primitives requires planner and latched_gripper: true")
            if any(self.settings[key] for key in ("hold_closed_during_motion", "stop_drop_on_support",
                                                  "grasp_stop_on_contact", "controlled_placement")):
                raise ValueError("pick_place_primitives uses its own grasp/support gates; disable legacy flags")
        if self.settings["latched_gripper"] and self.settings["hold_closed_during_motion"]:
            raise ValueError("Select latched_gripper or legacy hold_closed_during_motion, not both")
        if self.settings["controlled_placement"] and not (
                self.settings["latched_gripper"] and self.settings["stop_drop_on_support"]):
            raise ValueError("controlled_placement requires latched_gripper and stop_drop_on_support")
        self.physical_agent_dim = backend.env.agent_dim - int(self.settings["latched_gripper"])
        if self.physical_agent_dim not in (30, 32):
            raise ValueError("GNN agent state must include the enabled gripper-command memory")
        if self.settings["motion_execution"] not in {"ee_delta", "planner"}:
            raise ValueError("motion_execution must be ee_delta or planner")
        if self.settings["motion_execution"] == "planner" and not self.settings["latched_gripper"]:
            raise ValueError("Planned primitives require latched_gripper: true")
        budget = self.settings["motion_plan_max_steps"]
        if isinstance(budget, bool) or int(budget) != budget or budget < 1:
            raise ValueError("motion_plan_max_steps must be a positive integer")
        self.schema = skill_schema(self.settings)
        if self.settings["skill_set"] == "split7" and not self.settings["latched_gripper"]:
            raise ValueError("split7 requires latched_gripper: true to preserve gripper commands")
        if backend.action_dim != self.schema.action_dim:
            raise ValueError(f"Imagination actor requires a fresh {self.schema.action_dim}D skill GNN")
        Actor, Mlp, self.lambda_return = reference_components(backend.device)
        env = SimpleNamespace(num_primitives=self.schema.num_primitives,
                              primitive_name_to_action_idx=self.schema.primitive_name_to_action_idx)
        self.actor = Actor(
            hidden_size=self.settings["hidden_size"], obs_dim=backend.state_dim, env=env,
            num_layers=self.settings["actor_layers"], discrete_action_dim=env.num_primitives,
            continuous_action_dim=self.schema.num_parameters,
            init_std=self.settings["init_std"], min_std=0.1,
        ).to(backend.device)
        self.critic = Mlp([self.settings["hidden_size"]] * self.settings["critic_layers"],
                          input_size=backend.state_dim, output_size=1).to(backend.device)
        # deepcopy of the reference ScriptModule can produce non-leaf tensors
        # under modern PyTorch. Construct a separate module and copy weights.
        self.target_critic = Mlp([self.settings["hidden_size"]] * self.settings["critic_layers"],
                                 input_size=backend.state_dim, output_size=1).to(backend.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.settings["actor_lr"],
                                                eps=self.settings["adam_eps"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.settings["critic_lr"],
                                                 eps=self.settings["adam_eps"])
        self.updates = 0

    def features(self, states):
        model = self.backend.model
        agent, dyn, static = model._obs_preprocessing(states.unsqueeze(0))
        agent = model.input_normalizer_agent.normalize(agent.reshape(-1, model.agent_dim))
        dyn = model.input_normalizer_obj_dyn.normalize(dyn.reshape(-1, 12)).reshape(-1, 12 * self.backend.env.nObj)
        static = model.input_normalizer_obj_stat.normalize(static.reshape(-1, 5)).reshape(-1, 5 * self.backend.env.nObj)
        result = torch.cat((agent, dyn, static), -1).float()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite normalized actor state")
        return result

    @torch.no_grad()
    def action(self, state):
        state = self.backend._array(state, (self.backend.state_dim,), "actor state")
        features = self.features(torch.as_tensor(state, device=self.backend.device)[None])
        action = self.actor(features).rsample()[0]
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite RAPS actor action")
        return action.cpu().numpy()

    @torch.no_grad()
    def imagine(self, starts):
        """No env access or model gradients: REINFORCE uses detached imagined data."""
        states, actions, rewards, components, heuristics = [], [], [], [], []
        state = starts
        for _ in range(self.settings["imagination_horizon"]):
            states.append(state)
            raw = self.actor(self.features(state)).rsample()
            action = canonical_action(raw, self.schema)
            obs = state[None].expand(self.backend.ensemble_size, -1, -1)
            predicted = self.backend.model.predict(
                observations=obs, actions=action[None].expand(self.backend.ensemble_size, -1, -1),
                states=None)[0]
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("Nonfinite skill-model imagination; no random fallback")
            # Query all members at the SAME state/action: accumulated trajectory
            # divergence is not substituted for one-step epistemic disagreement.
            std = predicted[..., :self.backend.dynamic_dim].std(dim=0, unbiased=True)
            next_state = predicted.mean(0)
            object_start = self.backend.env.agent_dim
            # The extra command is controller memory, not physical novelty.
            intrinsic = std[:, :self.physical_agent_dim].sum(-1) + std[:, object_start:].sum(-1)
            heuristic = torch.zeros_like(intrinsic)
            if self.backend.object_heuristic is not None:
                if self.settings.get("grasp_target_heuristic", False):
                    if self.settings.get("pick_place_grid", False):
                        selected_grasp = action[..., :self.schema.num_grasp_branches].sum(-1) == 1
                    else:
                        selected_grasp = action[..., self.schema.primitive_names.index("grasp")] == 1
                    heuristic = torch.where(selected_grasp,
                                            self.grasp_target_heuristic(state, action), heuristic)
                else:
                    heuristic, _ = self.backend._object_heuristic_scores(state, next_state)
            actions.append(raw)
            reward = self.imagination_reward(state, action, next_state, starts,
                                             intrinsic, heuristic)
            if reward.shape != intrinsic.shape or not torch.isfinite(reward).all():
                raise FloatingPointError("Invalid imagination reward")
            rewards.append(reward.unsqueeze(-1))
            components.append(torch.stack((std[:, :18].sum(-1), std[:, 18:30].sum(-1),
                                           std[:, 30:self.physical_agent_dim].sum(-1), std[:, object_start:].sum(-1)), -1))
            heuristics.append(heuristic)
            state = next_state
        states.append(state)
        return dict(states=torch.stack(states), actions=torch.stack(actions),
                    rewards=torch.stack(rewards), components=torch.stack(components),
                    heuristics=torch.stack(heuristics))

    def grasp_target_heuristic(self, state, action):
        setting = self.backend.object_heuristic
        if setting is None:
            return state.new_zeros(state.shape[:-1])
        if self.settings.get("pick_place_grid", False):
            branches = action[..., :self.schema.num_primitives].argmax(-1)
            cells = torch.tensor(self.schema.branch_cell_indices, device=action.device)[branches]
            xy = action.new_tensor(self.schema.grid_xy_root_m)[cells]
            z = self.schema.grasp_z.center + self.schema.grasp_z.scale * action[..., -1:].clamp(-1., 1.)
            goal = torch.cat((xy, z), -1)
        else:
            parameters = self.schema.spec_by_name["grasp"].parameters
            center = action.new_tensor([parameter.center for parameter in parameters])
            scale = action.new_tensor([parameter.scale for parameter in parameters])
            goal = center + scale * action[..., self.schema.num_primitives:].clamp(-1., 1.)
        centers = torch.stack([
            state[..., self.backend.env.agent_dim + index * self.backend.env.object_dyn_dim:
                  self.backend.env.agent_dim + index * self.backend.env.object_dyn_dim + 3]
            for index, name in enumerate(self.backend.object_names)
            if setting["object_weights"].get(name, 0) > 0], dim=-2)
        distance = torch.linalg.vector_norm(goal.unsqueeze(-2) - centers, dim=-1).amin(dim=-1)
        return setting["approach_weight"] * torch.exp(-distance / setting["distance_scale_m"])

    def imagination_reward(self, state, action, next_state, starts, intrinsic, heuristic):
        return self.settings["disagreement_weight"] * intrinsic + heuristic

    def _step(self, module, optimizer, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite imagination actor/critic loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(module.parameters())
        if self.backend.runtime_optimizations:
            checks = [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
            invalid = bool(checks) and not bool(torch.stack(checks).all())
        else:
            invalid = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters)
        if invalid:
            raise FloatingPointError("Nonfinite imagination gradient; optimizer update stopped")
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.settings["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        return norm.detach() if self.backend.runtime_optimizations else float(norm)

    def train(self, replay, updates=None, *, start_groups=None):
        eligible = np.flatnonzero(~np.asarray(replay["terminated"], dtype=bool))
        if not len(eligible):
            raise ValueError("No nonterminal real replay states for imagination")
        source = torch.as_tensor(np.asarray(replay["states"])[eligible], device=self.backend.device)
        groups = None
        if start_groups is not None:
            labels = np.asarray(start_groups)
            if labels.shape != (len(replay["states"]),) or not np.isin(labels, [0, 1]).all():
                raise ValueError("start_groups must identify old=0/new=1 for every replay state")
            groups = [torch.as_tensor(np.flatnonzero(labels[eligible] == group), device=source.device)
                      for group in (0, 1)]
            if any(len(group) == 0 for group in groups) or self.settings["imagination_batch_size"] % 2:
                raise ValueError("Balanced imagination requires both nonterminal groups and an even batch size")
        metrics = []
        count = self.settings["updates_per_round"] if updates is None else int(updates)
        if count < 1:
            raise ValueError("Actor updates must be positive")
        for _ in range(count):
            if groups is None:
                indices = torch.randint(len(source), (self.settings["imagination_batch_size"],), device=source.device)
            else:
                half = self.settings["imagination_batch_size"] // 2
                indices = torch.cat([group[torch.randint(len(group), (half,), device=source.device)]
                                     for group in groups])
            trajectory = self.imagine(source[indices])
            state = trajectory["states"]
            horizon, batch = trajectory["actions"].shape[:2]
            features = self.features(state.reshape(-1, self.backend.state_dim)).detach()
            with torch.no_grad():
                values = self.target_critic(features).reshape(horizon + 1, batch, 1)
                discount = torch.full_like(trajectory["rewards"], self.settings["discount"])
                returns = self.lambda_return(trajectory["rewards"], values[:-1], discount,
                                             values[-1], lambda_=self.settings["lambda_return"])
                baseline = self.critic(features).reshape(horizon + 1, batch, 1)[:-1]
                advantages = (returns - baseline).reshape(-1)
                weights = torch.cumprod(torch.cat((torch.ones_like(discount[:1]), discount[:-1]), 0), 0)
            actor_features = features.reshape(horizon + 1, batch, -1)[:-1].reshape(horizon * batch, -1)
            distribution = self.actor(actor_features)
            log_prob = distribution.log_prob(trajectory["actions"].reshape(horizon * batch, -1).detach())
            entropy = distribution.entropy()
            actor_loss = ((-log_prob * advantages - self.settings["entropy_scale"] * entropy)
                          * weights.reshape(-1)).mean()
            actor_norm = self._step(self.actor, self.actor_optimizer, actor_loss)
            value = self.critic(actor_features).reshape(horizon, batch, 1)
            # Unit-variance Normal log likelihood, as in the reference value head.
            critic_loss = (-torch.distributions.Normal(value, 1).log_prob(returns.detach()) * weights).mean()
            critic_norm = self._step(self.critic, self.critic_optimizer, critic_loss)
            self.updates += 1
            if self.updates % self.settings["target_update_period"] == 0:
                self.target_critic.load_state_dict(self.critic.state_dict())
            values = dict(actor_loss=actor_loss.detach(), critic_loss=critic_loss.detach(),
                                actor_gradient_norm=actor_norm, critic_gradient_norm=critic_norm,
                                entropy=entropy.detach().mean(), imagined_reward=trajectory["rewards"].mean(),
                                disagreement=trajectory["components"].sum(-1).mean(),
                                heuristic=trajectory["heuristics"].mean(),
                                joint_disagreement=trajectory["components"][..., 0].mean(),
                                tcp_disagreement=trajectory["components"][..., 1].mean(),
                                door_disagreement=trajectory["components"][..., 2].mean(),
                                object_disagreement=trajectory["components"][..., 3].mean())
            metrics.append(values if self.backend.runtime_optimizations else
                           {key: float(value) for key, value in values.items()})
        if self.backend.runtime_optimizations:
            rows = torch.stack([torch.stack(list(m.values())) for m in metrics]).cpu().numpy()
            return {**dict(zip(metrics[0], rows.astype(np.float64).mean(0).tolist())),
                    "updates": count, "total_updates": self.updates}
        return {**{key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]},
                "updates": count, "total_updates": self.updates}

    def state_dict(self):
        return dict(settings=self.settings, updates=self.updates,
                    actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    target_critic=self.target_critic.state_dict(),
                    actor_optimizer=self.actor_optimizer.state_dict(),
                    critic_optimizer=self.critic_optimizer.state_dict())

    def load_state_dict(self, payload):
        if {**DEFAULTS, **payload["settings"]} != self.settings:
            raise ValueError("Checkpoint actor settings differ; use its saved config")
        for name in ("actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        self.updates = payload["updates"]


class SkillCollector:
    """Collect completed skill endpoints and every real control tick."""
    def __init__(self, env, encoder, actor, directory, *, collection_optimizations=False):
        self.env, self.encoder, self.actor = env, encoder, actor
        self.schema = skill_schema(actor.settings)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.total_steps = self.episode_id = self.skill_id = 0
        self.state = None
        if not isinstance(collection_optimizations, bool):
            raise ValueError("collection_optimizations must be boolean")
        self.collection_optimizations = collection_optimizations
        self.profile_timings = {}
        self._context_cache = None
        self.planned_motion = actor.settings.get("motion_execution", "ee_delta") == "planner"
        execution_kwargs = {}
        executor = RAPSSkillExecution
        if self.planned_motion:
            if self.env.env.action_mode != "joint_position":
                raise ValueError("Planned primitive collection requires action_mode: joint_position")
            executor = (PlannedPickPlaceSkillExecution if actor.settings.get("pick_place_primitives", False)
                        else PlannedSkillExecution)
            execution_kwargs = dict(orientation_target=lambda: np.roll(
                self.env.env._ee_orientation_target.as_quat(), 1),
                max_control_steps=actor.settings.get("motion_plan_max_steps", 400))
        if actor.settings.get("pick_place_primitives", False):
            execution_kwargs["grasp_retention"] = actor.settings.get("grasp_retention", "friction")
            execution_kwargs["grasp_orientation_candidates"] = actor.settings.get("grasp_orientation_candidates", False)
            if actor.settings.get("grid_layout") == "trapezoid_4_3_2_1":
                execution_kwargs["physical_action_schema"] = self.schema.physical_action_schema()
        self.execution = executor(
            **execution_kwargs,
            latched_gripper=actor.settings.get("latched_gripper", False),
            stop_drop_on_support=actor.settings.get("stop_drop_on_support", False),
            grasp_stop_on_contact=actor.settings.get("grasp_stop_on_contact", True),
            controlled_placement=actor.settings.get("controlled_placement", False),
            hold_closed_during_motion=actor.settings["hold_closed_during_motion"])

    def reset(self, seed):
        started = time.perf_counter()
        self._context_cache = None
        if self.actor.settings.get("reconfigure_on_reset", False):
            # Ordinary pose/velocity reset retains PhysX scene history and can
            # change the outcome of identical observed state/action pairs.
            self.env.reset(seed=int(seed), options={"reconfigure": True})
            self.encoder.rebind_after_reset()
        else:
            self.env.reset(seed=int(seed))
        raw = getattr(self.env, "unwrapped", self.env.env)
        raw.collection_optimizations = self.collection_optimizations
        raw._collection_profile = self.profile_timings
        self.episode_id += 1
        self.execution.reset()
        self.encoder.gripper_command = self.execution.gripper_command
        self.state = self.encoder.encode()
        if self.actor.settings.get("grasp_only", False):
            self._capture_home()
        self._add_time("reset_s", started)

    def _capture_home(self):
        from domain.robocasa.motion_planning import _array
        raw = self.env.unwrapped
        self._home_qpos = _array(raw.agent.robot.get_qpos()).reshape(9).copy()
        tcp = raw.agent.robot.pose.inv() * raw.agent.tcp.pose
        self._home_position = _array(tcp.p).reshape(3).copy()
        self._home_quaternion = _array(tcp.q).reshape(4).copy()

    def _home_actions(self):
        """Physically restore reset joints; keep the current object scene."""
        from domain.robocasa.motion_planning import MotionPlanExecution, MotionPlanningError, _array
        raw = self.env.unwrapped
        def ready():
            q = _array(raw.agent.robot.get_qpos()).reshape(9)
            v = _array(raw.agent.robot.get_qvel()).reshape(9)
            return np.max(np.abs(q-self._home_qpos)) <= .001 and np.max(np.abs(v)) <= .01
        if ready():
            self.execution.gripper_command = 1.
            return
        if any(self._context().is_grasping(key) for key in self._context().objects):
            raise MotionPlanningError("home_still_holding", "End the held-object episode before homing")
        # Open at the measured arm pose before planning with fixed finger width.
        for _ in range(9):
            yield np.r_[_array(raw.agent.robot.get_qpos()).reshape(9)[:7], 1.].astype(np.float32)
        def execute(plan):
            cursor = MotionPlanExecution(raw, plan, gripper_command=1.,
                max_steps=self.actor.settings.get("motion_plan_max_steps", 400))
            while cursor.status == "running":
                yield cursor.next_action()
                cursor.record_step()
            if cursor.status != "success":
                raise MotionPlanningError("home_tracking_failed", {"status": cursor.status})
        # Rotate only after a straight withdrawal clears the cup/cabinet interior.
        from domain.robocasa.pick_place import OUTSIDE_X
        tcp = raw.agent.robot.pose.inv() * raw.agent.tcp.pose
        position = _array(tcp.p).reshape(3).copy()
        if position[0] > OUTSIDE_X + .001:
            position[0] = OUTSIDE_X
            retreat = raw.plan_to_pose(position, _array(tcp.q).reshape(4),
                method="bounded_ik", maintain_orientation=True)
            yield from execute(retreat)
        target = self._home_qpos.copy()
        target[7:] = _array(raw.agent.robot.get_qpos()).reshape(9)[7:]
        plan = raw.plan_to_pose(self._home_position, self._home_quaternion,
            target_qpos=target, densify_rrt_path=True)
        yield from execute(plan)
        if not ready():
            raise MotionPlanningError("home_tracking_failed", dict(
                message="Default joints/velocity not reached; no grasp was started"))

    def _return_default(self, ticks, skills, count, policy, stage, low_log, skill_log, on_step):
        initial, before, start = self.state.copy(), self._snapshot(), self.total_steps
        complete = False
        for low in self._home_actions():
            current, low_before = self.state.copy(), self._snapshot()
            self._context_cache = None
            _, _, terminated, truncated, info = self.env.step(low)
            self.total_steps += 1
            self.execution.gripper_command = 1.
            self.encoder.gripper_command = 1.
            self.state = self.encoder.encode()
            if not np.isfinite(self.state).all():
                raise FloatingPointError("Nonfinite return_default state")
            record = info["primitive_record"]
            row = jsonable(dict(step=self.total_steps, stage=stage, policy=policy,
                episode_id=self.episode_id, skill_id=self.skill_id, skill_type="return_default",
                state=current, next_state=self.state, requested_action=low,
                action=record.low_actions[0], full_controller_action=record.full_controller_actions[0],
                primitive_type="joint_position", duration_low_steps=1, status=record.status,
                terminated=bool(terminated), truncated=bool(truncated),
                run_budget_exhausted=len(ticks)+1 == count, parameters_physical={},
                contact_events=record.contact_events, outcome_metrics=record.outcome_metrics,
                before=low_before, after=self._snapshot(), reward=info["exploration_reward"], planner={}))
            ticks.append(row)
            low_log.write(json.dumps(row, allow_nan=False)+"\n")
            if on_step is not None:
                on_step(row)
            if terminated or truncated:
                raise RuntimeError("Episode ended during return_default; no grasp started")
            if len(ticks) == count:
                break
        else:
            complete = True
        duration = self.total_steps-start
        if duration:
            skill = jsonable(dict(skill_id=self.skill_id, episode_id=self.episode_id,
                stage=stage, policy=policy, primitive_type="return_default",
                state=initial, next_state=self.state, duration_low_steps=duration,
                start_step=start, end_step=self.total_steps, trainable=False, completed=complete,
                terminated=False, truncated=False, status="success" if complete else "budget_truncated",
                before=before, after=self._snapshot()))
            skills.append(skill)
            skill_log.write(json.dumps(skill, allow_nan=False)+"\n")
            skill_log.flush()
            self.skill_id += 1
            self.execution.reset()
        # Read measured feedback even when no movement was required.
        self.encoder.gripper_command = self.execution.gripper_command
        self.state = self.encoder.encode()
        return complete

    def _add_time(self, name, started):
        self.profile_timings[name] = self.profile_timings.get(name, 0.) + time.perf_counter() - started

    def _context(self):
        started = time.perf_counter()
        try:
            if not self.collection_optimizations:
                return self.env.env._context()
            if self._context_cache is None:
                self._context_cache = self.env.env._context()
                if self._context_cache is not None:
                    self._context_cache._contact_force_cache = {}
            return self._context_cache
        finally:
            self._add_time("context_read_s", started)

    def _snapshot(self):
        started = time.perf_counter()
        try:
            if self.collection_optimizations:
                return snapshot(self.env, context=self._context(), convert=False)
            return snapshot(self.env)
        finally:
            self._add_time("snapshot_s", started)

    def _propose_action(self, policy):
        started = time.perf_counter()
        try:
            if policy == "random":
                index = np.random.randint(self.schema.num_primitives)
                return np.r_[np.eye(self.schema.num_primitives)[index],
                             np.random.uniform(-1, 1, self.schema.num_parameters)].astype(np.float32)
            return self.actor.action(self.state)
        finally:
            # Some diagnostic fixtures use only the proposal method.
            if hasattr(self, "profile_timings"):
                self._add_time("action_proposal_s", started)

    def _select_preflight_action(self, policy, stage):
        maximum = int(self.actor.settings.get("grasp_preflight_max_candidates", 32))
        context = self._context()
        held = any(context.is_grasping(key) for key in context.objects)
        with (self.directory / "preflight.jsonl").open("a") as audit:
            for attempt in range(1, maximum + 1):
                requested = self._propose_action(policy)
                actual = canonical_action(torch.as_tensor(requested), self.schema).numpy()
                decoded = self.schema.decode(actual)
                started = time.perf_counter()
                if decoded.primitive_type == "grasp":
                    result = (dict(accepted=False, status="already_holding") if held else
                              self.execution.preflight_grasp(decoded, context))
                else:
                    valid = held and self.execution.gripper_command < 0
                    result = dict(accepted=valid, status="holding_precondition_met" if valid else "missing_grasp")
                self._add_time("preflight_s", started)
                seconds = time.perf_counter() - started
                conversion_started = time.perf_counter()
                row = jsonable(dict(
                    stage=stage, policy=policy, episode_id=self.episode_id, skill_id=self.skill_id,
                    at_real_step=self.total_steps, candidate=attempt, requested_action=requested,
                    action=actual, primitive_type=decoded.primitive_type,
                    seconds=seconds, result=result))
                self._add_time("log_conversion_s", conversion_started)
                serialization_started = time.perf_counter()
                encoded = json.dumps(row, allow_nan=False) + "\n"
                self._add_time("log_serialization_s", serialization_started)
                write_started = time.perf_counter()
                audit.write(encoded)
                audit.flush()
                self._add_time("log_write_s", write_started)
                if result["accepted"]:
                    return requested, actual, decoded
        raise RuntimeError(f"No feasible action among {maximum} {policy} proposals; "
                           "no physics tick executed for rejected candidates. Inspect preflight.jsonl "
                           "and revise the grasp workspace/planner or policy; no random fallback was used.")

    def collect(self, count, policy, *, seed, stage, on_step=None,
                on_skill_start=None, on_skill_end=None, stop_when=None):
        if count < 1 or policy not in {"random", "curiosity", "task"}:
            raise ValueError("Positive real-step budget and random/curiosity/task policy required")
        self.profile_timings = {}
        self.reset(seed)
        ticks, skills = [], []
        empty_attempts = 0
        with (self.directory / "transitions.jsonl").open("a") as low_log, \
                (self.directory / "skills.jsonl").open("a") as skill_log:
            while len(ticks) < count:
                self.skill_id += 1
                if self.actor.settings.get("grasp_only", False):
                    if not self._return_default(ticks, skills, count, policy, stage,
                                                low_log, skill_log, on_step):
                        break
                if self.actor.settings.get("grasp_preflight", False):
                    requested, actual, decoded = self._select_preflight_action(policy, stage)
                else:
                    requested = self._propose_action(policy)
                    actual = canonical_action(torch.as_tensor(requested), self.schema).numpy()
                    decoded = self.schema.decode(actual)
                context = self._context()
                self.execution.start(decoded, context)
                primitive = self.execution.primitive
                initial, before = self.state.copy(), self._snapshot()
                if on_skill_start is not None:
                    on_skill_start(self.episode_id, self.skill_id, decoded, actual, initial)
                start = len(ticks)
                terminated = truncated = False
                complete = False
                while len(ticks) < count:
                    context = self._context()
                    action_started = time.perf_counter()
                    planning_before = self.profile_timings.get("planner_total_s", 0.)
                    low = self.execution.next_action(context)
                    self._add_time("execution_action_s", action_started)
                    self.profile_timings["actual_planning_s"] = self.profile_timings.get("actual_planning_s", 0.) + (
                        self.profile_timings.get("planner_total_s", 0.) - planning_before)
                    if low is None:
                        complete = True
                        break
                    current, low_before = self.state.copy(), self._snapshot()
                    if self.actor.settings.get("controlled_placement", False):
                        self.env.env.set_ee_orientation_override(self.execution.orientation_hold)
                    self._context_cache = None
                    step_started = time.perf_counter()
                    try:
                        _, _, terminated, truncated, info = self.env.step(low)
                    finally:
                        self._add_time("env_step_s", step_started)
                    record = info["primitive_record"]
                    if record.duration_low_steps != 1:
                        raise RuntimeError("Skill collector requires exactly one recorded control tick")
                    self.total_steps += 1
                    feedback_started = time.perf_counter()
                    if self.planned_motion:
                        self.execution.record_action(record.low_actions[0], terminated=terminated, truncated=truncated)
                    else:
                        self.execution.record_action(record.low_actions[0])
                    if self.actor.settings.get("pick_place_primitives", False):
                        self.execution.check_retention(self._context())
                    self._add_time("feedback_s", feedback_started)
                    encoding_started = time.perf_counter()
                    self.encoder.gripper_command = self.execution.gripper_command
                    self.state = self.encoder.encode()
                    self._add_time("state_encoding_s", encoding_started)
                    if not np.isfinite(self.state).all():
                        raise FloatingPointError("Nonfinite real skill state")
                    # Match compact primitive execution's every-two-step contact feedback.
                    if (len(ticks) - start + 1) % 2 == 0 or terminated or truncated:
                        observe_started = time.perf_counter()
                        self.execution.observe(self._context())
                        self._add_time("observation_s", observe_started)
                    after_snapshot = self._snapshot()
                    conversion_started = time.perf_counter()
                    row = jsonable(dict(
                        step=self.total_steps, stage=stage, policy=policy, episode_id=self.episode_id,
                        skill_id=self.skill_id, skill_type=decoded.primitive_type,
                        skill_action_requested=requested, skill_action=actual,
                        state=current, next_state=self.state, requested_action=low,
                        action=record.low_actions[0], full_controller_action=record.full_controller_actions[0],
                        terminated=bool(terminated), truncated=bool(truncated),
                        run_budget_exhausted=len(ticks) + 1 == count,
                        primitive_type="joint_position" if self.planned_motion else "ee_delta", duration_low_steps=1,
                        parameters_physical=dict(zip(
                            [f"joint_{i}" for i in range(7)] + ["gripper"] if self.planned_motion
                            else ("dx", "dy", "dz", "gripper"), record.low_actions[0])),
                        action_units=("seven_joint_targets_rad_plus_normalized_gripper" if self.planned_motion
                                      else "normalized_controller"), status=record.status,
                        contact_events=record.contact_events, outcome_metrics=record.outcome_metrics,
                        before=low_before, after=after_snapshot, reward=info["exploration_reward"], planner={},
                    ))
                    if self.actor.settings.get("pick_place_primitives", False):
                        row["grasp_retention"] = self.actor.settings.get("grasp_retention", "friction")
                        row["macro_phase"] = primitive.phase
                        row["macro_status"] = primitive.status
                    if self.planned_motion:
                        row["motion_execution"] = self.execution.last_motion_tick
                    self._add_time("log_conversion_s", conversion_started)
                    serialization_started = time.perf_counter()
                    encoded = json.dumps(row, allow_nan=False) + "\n"
                    self._add_time("log_serialization_s", serialization_started)
                    write_started = time.perf_counter()
                    low_log.write(encoded)
                    self._add_time("log_write_s", write_started)
                    ticks.append(row)
                    if on_step is not None:
                        on_step(row)
                    if len(ticks) % 32 == 0 or terminated or truncated:
                        flush_started = time.perf_counter()
                        low_log.flush()
                        self._add_time("log_write_s", flush_started)
                    if self.actor.settings.get("pick_place_primitives", False) and primitive.done:
                        complete = True
                        break
                    if terminated or truncated:
                        break
                duration = len(ticks) - start
                if duration == 0:
                    empty_attempts += 1
                    if empty_attempts >= 100:
                        raise RuntimeError("100 consecutive zero-duration skills; check actor/primitive parameters")
                else:
                    empty_attempts = 0
                after_snapshot = self._snapshot()
                conversion_started = time.perf_counter()
                skill = jsonable(dict(
                    skill_id=self.skill_id, stage=stage, policy=policy, episode_id=self.episode_id,
                    state=initial, next_state=self.state, requested_action=requested, action=actual,
                    primitive_type=decoded.primitive_type, parameters_physical=decoded.parameters,
                    duration_low_steps=duration, terminated=bool(terminated), truncated=bool(truncated),
                    completed=complete, trainable=complete and duration > 0,
                    status=primitive.status if complete else (
                        "terminated" if terminated else "episode_truncated" if truncated else "budget_truncated"),
                    support_stopped_object_ids=self.execution.support_stopped_object_ids,
                    start_step=self.total_steps - duration, end_step=self.total_steps,
                    before=before, after=after_snapshot,
                    object_contact=(_object_contact(before.get("contact_snapshot", {}))
                                    or _object_contact(after_snapshot.get("contact_snapshot", {}))
                                    or any(_object_contact(event) for tick in ticks[start:]
                                           for event in tick["contact_events"])),
                ))
                if self.actor.settings.get("pick_place_primitives", False):
                    skill["grasp_retention"] = self.actor.settings.get("grasp_retention", "friction")
                    skill["macro_phase"] = primitive.phase
                if self.planned_motion:
                    if not complete:
                        self.execution.finish_motion(skill["status"])
                    skill["motion_planning"] = self.execution.motion_attempts
                reset_after_grasp = (self.actor.settings.get("grasp_only", False)
                                     and complete and primitive.status == "success"
                                     and decoded.primitive_type == "grasp")
                reset_after_held_failure = (
                    self.actor.settings.get("grasp_only", False)
                    and complete and primitive.status != "success"
                    and decoded.primitive_type == "grasp"
                    and any(obj["grasped"] for obj in
                            after_snapshot.get("contact_snapshot", {}).get("objects", {}).values()))
                if reset_after_grasp:
                    skill["reset_after_skill"] = "grasp_success"
                elif reset_after_held_failure:
                    skill["reset_after_skill"] = "grasp_failure_still_holding"
                self._add_time("log_conversion_s", conversion_started)
                serialization_started = time.perf_counter()
                encoded = json.dumps(skill, allow_nan=False) + "\n"
                self._add_time("log_serialization_s", serialization_started)
                write_started = time.perf_counter()
                skill_log.write(encoded)
                skill_log.flush()
                self._add_time("log_write_s", write_started)
                skills.append(skill)
                if on_skill_end is not None:
                    on_skill_end(skill)
                if stop_when is not None and stop_when():
                    break
                if (terminated or truncated or reset_after_grasp or reset_after_held_failure) and len(ticks) < count:
                    self.reset(seed + self.episode_id)
            close_started = time.perf_counter()
        self._add_time("log_write_s", close_started)
        return ticks, skills


def _object_contact(snapshot):
    return any(obj.get("grasped", False) or obj.get("left_force", 0) > 0 or obj.get("right_force", 0) > 0
               for obj in snapshot.get("objects", {}).values())


def skill_arrays(skills):
    valid = [row for row in skills if row["trainable"]]
    if len(valid) < 2:
        raise ValueError("Need at least two completed nonzero-duration skills; inspect skills.jsonl for truncations")
    arrays = replay_arrays(valid)
    arrays["duration_low_steps"] = np.asarray([row["duration_low_steps"] for row in valid], dtype=np.int64)
    return arrays


def _timed(backend, operation):
    if backend.device.type == "cuda":
        torch.cuda.synchronize(backend.device)
    started = time.perf_counter()
    result = operation()
    if backend.device.type == "cuda":
        torch.cuda.synchronize(backend.device)
    return result, time.perf_counter() - started


def _write_checkpoint(path, payload, arrays, low_arrays, timings):
    started = time.perf_counter()
    targets = (path.with_suffix(".replay.npz"), path.with_suffix(".low_level.npz"), path)
    temporary = []
    try:
        for target, value in zip(targets, (arrays, low_arrays, payload)):
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + target.name,
                                             suffix=".tmp", delete=False) as handle:
                temporary.append(Path(handle.name))
                if target == path:
                    torch.save(value, handle)
                else:
                    np.savez_compressed(handle, **value)
        # Publish the checkpoint marker only after both replay files are ready.
        path.unlink(missing_ok=True)
        for source, target in zip(temporary, targets):
            source.replace(target)
    finally:
        for source in temporary:
            source.unlink(missing_ok=True)
    return dict(path=str(path), **timings, checkpoint_write_s=time.perf_counter() - started)


def save(path, backend, actor, encoder, config, ticks, skills, *, executor=None):
    path = Path(path)
    started = time.perf_counter()
    arrays = skill_arrays(skills)
    # Full low-level data remain usable for subsequent control-step WMs.
    low_arrays = replay_arrays(ticks)
    replay_seconds = time.perf_counter() - started
    payload, snapshot_seconds = _timed(backend, lambda: dict(
                    schema=SCHEMA, backend=backend.state_dict(), actor=actor.state_dict(),
                    rng=rng_state(), state_contract=encoder.contract(),
                    action_contract=action_contract(actor.settings),
                    config=OmegaConf.to_container(config, resolve=True),
                    sampling_contract=sampling_contract(config),
                    real_steps=len(ticks), completed_skills=len(arrays["states"]),
                    environment_state_restored=False))
    if executor is not None:
        # state_dict() aliases live actor/optimizer storage; the writer must own CPU copies.
        started = time.perf_counter()
        payload = backend._move_tensors(payload, "cpu")
        snapshot_seconds += time.perf_counter() - started
    timings = dict(checkpoint_replay_s=replay_seconds, checkpoint_snapshot_s=snapshot_seconds)
    if executor is not None:
        return executor.submit(_write_checkpoint, path, payload, arrays, low_arrays, timings)
    return _write_checkpoint(path, payload, arrays, low_arrays, timings)


def load(path, backend, actor, encoder, *, sampling=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Requires a RAPS/GNN actor checkpoint; 4D CEE-US checkpoints are incompatible")
    if payload["state_contract"] != encoder.contract() or not action_contract_matches(payload["action_contract"], actor.settings):
        raise ValueError("Checkpoint state/action/controller contract differs")
    if payload.get("sampling_contract") != sampling:
        raise ValueError("Checkpoint sampling contract differs; use its original collection configuration")
    backend.load_state_dict(payload["backend"])
    actor.load_state_dict(payload["actor"])
    restore_rng(payload["rng"])
    return payload


def resume_history(path, config):
    """Resume only at a saved round boundary, where collection resets physics."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy = str(config.get("collection_policy", "curiosity"))
    if payload.get("schema") != SCHEMA or path.parent.name != policy:
        raise ValueError(f"train --checkpoint requires a {policy} RAPS/GNN round checkpoint")
    old = deepcopy(payload["config"])
    new = OmegaConf.to_container(config, resolve=True)
    for value in (old, new):
        value.pop("rounds", None)
        value.pop("evaluation_steps", None)
        value.setdefault("collection_policy", "curiosity")
        value["raps_actor"] = {**DEFAULTS, **value["raps_actor"]}
    if old != new:
        raise ValueError("Resume may change only rounds and evaluation_steps; retain saved settings")
    steps = payload["real_steps"]
    collected = steps - int(config.bootstrap_steps)
    if collected < 0 or collected % int(config.steps_per_round):
        raise ValueError("Checkpoint is not at the configured collection-round boundary")
    next_round = collected // int(config.steps_per_round) + 1
    if next_round > int(config.rounds):
        raise ValueError("Resume requires a larger total rounds budget")
    ticks, skills = [], []
    source = path.parent.parent
    for directory in (source / "bootstrap", source / policy):
        with (directory / "transitions.jsonl").open() as handle:
            ticks.extend(row for line in handle if (row := json.loads(line))["stage"] != "evaluation"
                         and row["step"] <= steps)
        with (directory / "skills.jsonl").open() as handle:
            skills.extend(row for line in handle if (row := json.loads(line))["stage"] != "evaluation"
                          and row["end_step"] <= steps)
    if len(ticks) != steps or ticks[-1]["step"] != steps:
        raise ValueError("Source JSONL does not contain the complete saved training prefix")
    with np.load(path.with_suffix(".replay.npz")) as saved:
        for key, value in skill_arrays(skills).items():
            np.testing.assert_array_equal(value, saved[key], err_msg=f"Resume replay mismatch: {key}")
    return ticks, skills, next_round


def fit(backend, actor, skills, config, directory, label, *, smoke=False):
    arrays_started = time.perf_counter()
    arrays = skill_arrays(skills)
    timings = dict(replay_preparation_s=time.perf_counter() - arrays_started)
    started = time.perf_counter()
    previous_profile = backend.profile_timings
    backend.profile_timings = {}
    try:
        replay_metrics = None
        train_options = dict(epochs=1 if smoke else int(config.epochs))
        if config.get("wm_replay") is not None:
            from exploration.raps.protected_replay import protected_replay_indices, bounded_dynamic_replay_indices
            options = _protected_replay_config(config)
            mode = options.pop("mode", "protected")
            sampling = dict(
                agent_dim=backend.env.agent_dim, object_dyn_dim=backend.env.object_dyn_dim,
                num_objects=backend.env.nObj,
                batch_size=backend.model.batch_size, **{**options, "updates": 2 if smoke else options["updates"]})
            if mode == "bounded_dynamic":
                indices, replay_metrics = bounded_dynamic_replay_indices(
                    arrays["states"], arrays["next_states"], **sampling)
            else:
                valid = [row for row in skills if row["trainable"]]
                if any("object_contact" not in row for row in valid):
                    raise ValueError("Protected replay requires object_contact on every macro; "
                                     "use newly collected records with intermediate contact summaries")
                indices, replay_metrics = protected_replay_indices(
                    arrays["states"], arrays["actions"], arrays["next_states"],
                    new_mask=np.asarray([row["stage"] == label for row in valid], dtype=bool),
                    contact_mask=np.asarray([row["object_contact"] for row in valid], dtype=bool), **sampling)
            train_options = dict(epochs=1, training_indices=indices)
        model_metrics, timings["world_model_fit_s"] = _timed(
            backend, lambda: backend.train(arrays, **train_options))
        if replay_metrics is not None:
            if model_metrics["updates"] != (2 if smoke else options["updates"]):
                raise RuntimeError("WM replay update budget was not honored")
            model_metrics["replay"] = replay_metrics
        timings.update(backend.profile_timings)
        actor_metrics, timings["actor_critic_s"] = _timed(
            backend, lambda: actor.train(arrays, updates=2 if smoke else None))
    finally:
        backend.profile_timings = previous_profile
    result = dict(label=label, real_steps=sum(row["duration_low_steps"] for row in skills),
                  completed_skills=len(arrays["states"]), model=model_metrics, actor=actor_metrics,
                  seconds=time.perf_counter() - started, timings=timings)
    with (directory / "training.jsonl").open("a") as handle:
        handle.write(json.dumps(result, allow_nan=False) + "\n")
    return result


def _protected_replay_config(config):
    options = dict(config["wm_replay"])
    mode = options.pop("mode", "protected")
    if mode not in ("protected", "bounded_dynamic"):
        raise ValueError("wm_replay.mode must be protected or bounded_dynamic")
    required = ({"updates", "movement_threshold_m", "z_bin_m", "joint_bin_degrees"}
                if mode == "protected" else {"updates", "movement_threshold_m"})
    if set(options) != required:
        raise ValueError(f"wm_replay {mode} requires exactly {sorted(required)}")
    if isinstance(options["updates"], bool) or not isinstance(options["updates"], int) or options["updates"] < 1:
        raise ValueError("wm_replay.updates must be a positive integer")
    for key in required - {"updates"}:
        value = options[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"wm_replay.{key} must be finite and positive")
    settings = config["raps_actor"]
    if not (settings.get("grasp_only") and settings.get("pick_place_grid")
            and settings.get("grid_layout") == "trapezoid_4_3_2_1"
            and settings.get("grasp_orientation_candidates")
            and list(settings.get("grasp_z_bounds_m", [])) == [.692, .772]):
        raise ValueError("Protected replay requires the trapezoid grasp cell/pose/Z action contract")
    if mode == "protected":
        options["joint_bin_rad"] = np.deg2rad(options.pop("joint_bin_degrees"))
    else:
        options["mode"] = mode
    return options


def summarize(ticks, skills, schema=RAPS_ACTION_SCHEMA):
    maintenance = [row for row in skills if row["primitive_type"] == "return_default"]
    return dict(interactions=interaction_summary(ticks), skill_decisions=len(skills)-len(maintenance),
                homing_controls=sum(row["duration_low_steps"] for row in maintenance),
                completed_skills=sum(row["trainable"] for row in skills),
                incomplete_skills=sum(not row["completed"] for row in skills),
                per_skill={name: dict(selected=sum(row["primitive_type"] in labels for row in skills),
                                     completed=sum(row["primitive_type"] in labels and row["trainable"] for row in skills))
                           for name in getattr(schema, "logical_primitive_names", schema.primitive_names)
                           for labels in [(name, "top_grasp") if name == "approach" else (name,)]})


def run(args, config):
    wall_started = time.perf_counter()
    config.agent = "raps_gnn_actor"
    if args.stage in {"train", "smoke"}:
        config.collection_policy = getattr(args, "policy", "curiosity")
    policies = ("random", "curiosity") if args.stage == "compare" else (str(config.get("collection_policy", "curiosity")),)
    config.raps_actor = OmegaConf.merge(DEFAULTS, config.get("raps_actor", {}))
    if config.get("wm_replay") is not None:
        _protected_replay_config(config)
    if not isinstance(config.get("collection_optimizations", False), bool):
        raise ValueError("collection_optimizations must be boolean")
    collector_options = ({"collection_optimizations": True}
                         if config.get("collection_optimizations", False) else {})
    collection_contract = sampling_contract(config)
    if config.raps_actor.pick_place_primitives:
        config.raps_actor.motion_execution = "planner"
        config.action_mode = "joint_position"
        config.raps_actor.latched_gripper = True
        for key in ("hold_closed_during_motion", "stop_drop_on_support",
                    "grasp_stop_on_contact", "controlled_placement"):
            config.raps_actor[key] = False
    if args.device:
        config.device = args.device
    if getattr(args, "evaluation_steps", None) is not None:
        config.evaluation_steps = args.evaluation_steps
    output = args.output or ROOT / "logs" / time.strftime("raps_gnn_actor_%Y%m%dT%H%M%S")
    if args.stage == "report":
        if not output.is_dir():
            raise ValueError("report requires an existing --output")
        reports = {path.parent.name: json.loads(path.read_text()) for path in output.glob("*/evaluation.json")}
        verdict = (screen(reports["random"]["interactions"], reports["curiosity"]["interactions"])
                   if {"random", "curiosity"} <= reports.keys() else {"status": "no_matched_random_comparison"})
        write_json(output / "report.json", dict(evaluations=reports, screening=verdict))
        print(json.dumps(verdict, ensure_ascii=False))
        return 0
    execution = str(config.get("execution", "reference"))
    if execution not in {"reference", "cuda_graph"}:
        raise ValueError("execution must be reference or cuda_graph")
    if execution == "cuda_graph" and config.device != "cuda":
        raise ValueError("cuda_graph execution requires device=cuda")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if config.get("project_goals_to_region", False):
        raise ValueError("Region projection is not supported by the skill endpoint model")
    expected_mode = "joint_position" if config.raps_actor.motion_execution == "planner" else "ee_delta"
    if (config.action_mode != expected_mode or config.exploration_mode != "baseline"
            or config.sim_freq != 100 or config.control_freq != 20):
        raise ValueError(f"RAPS actor requires baseline 100/20 Hz {expected_mode} execution")
    for key in ("bootstrap_steps", "rounds", "steps_per_round", "epochs", "cpu_threads"):
        if int(config.get(key, 1)) < 1:
            raise ValueError(f"{key} must be positive")
    if int(config.evaluation_steps) < 0:
        raise ValueError("evaluation_steps must be nonnegative")
    if args.stage == "evaluate" and int(config.evaluation_steps) == 0:
        raise ValueError("evaluate requires positive evaluation_steps")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    if args.stage == "evaluate" and args.checkpoint is None:
        raise ValueError("evaluate requires --checkpoint")
    torch.set_num_threads(int(config.get("cpu_threads", 1)))
    random.seed(int(config.seed)); np.random.seed(int(config.seed)); torch.manual_seed(int(config.seed))
    item_layout = config.get("cabinet_item_layout", "two_upper_second_row")
    if item_layout not in {"single_upper_second_row", "two_upper_second_row"}:
        raise ValueError(f"Unsupported cabinet_item_layout={item_layout!r}")
    object_names = (("cabinet_item_16",) if item_layout == "single_upper_second_row"
                    else ("cabinet_item_16", "cabinet_item_17"))
    from exploration.cee_us.cee_us_state import cabinet_agent_dim
    backend = CEEUSBackend(device=str(config.device), seed=int(config.seed),
                           action_dim=skill_schema(config.raps_actor).action_dim,
                           agent_dim=cabinet_agent_dim(config.get("cabinet_door", False),
                                                       latched_gripper=config.raps_actor.latched_gripper),
                           num_objects=len(object_names), object_names=object_names,
                           execution=execution,
                           object_heuristic=OmegaConf.to_container(config.object_heuristic, resolve=True)
                           if config.get("object_heuristic") is not None else None,
                           **dict(config.get("wm_training") or {}),
                           runtime_optimizations=bool(config.get("runtime_optimizations", False)))
    actor = RAPSImaginationActor(backend, OmegaConf.to_container(config.raps_actor, resolve=True))
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    env = None
    prepared = False
    writer = None
    pending_checkpoint = None

    def timing(row):
        with (output / "timings.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    def drain_checkpoint():
        nonlocal pending_checkpoint
        if pending_checkpoint is not None:
            timing(dict(stage="checkpoint", **pending_checkpoint.result()))
            pending_checkpoint = None

    def checkpoint(path, ticks, skills):
        nonlocal pending_checkpoint
        # Bound both the task count and the number of owned full replay snapshots.
        drain_checkpoint()
        result = save(path, backend, actor, encoder, config, ticks, skills, executor=writer)
        if writer is None:
            timing(dict(stage="checkpoint", **result))
        else:
            pending_checkpoint = result

    def collect(collector, *arguments, **keywords):
        if pending_checkpoint is not None and pending_checkpoint.done():
            drain_checkpoint()
        result, seconds = _timed(backend, lambda: collector.collect(*arguments, **keywords))
        timing(dict(stage="collection", label=keywords["stage"], collection_s=seconds,
                    collection_components=dict(getattr(collector, "profile_timings", {})),
                    nested_components=["context_read_s", "actual_planning_s", "planner_total_s",
                                       "contact_force_read_s", "planner_geometry_s", "planner_ik_s",
                                       "planner_path_search_s", "planner_topp_s"]))
        return result

    try:
        env, env_config = make_env(
            episode_steps=config.get("episode_steps", 100),
            centered_initial_tcp=config.get("centered_initial_tcp", False),
            cabinet_door=config.get("cabinet_door", False),
            cabinet_item_layout=item_layout,
            cabinet_grid=config.raps_actor.get("pick_place_grid", False),
            cabinet_grid_layout=config.raps_actor.get("grid_layout", "legacy_3x4"),
            object_shift_xy=getattr(args, "object_shift_xy", (0.0, 0.0)),
            action_mode=expected_mode,
            low_level_episode_steps=(int(config.get("episode_steps", 100))
                                     if config.raps_actor.pick_place_primitives else None),
        )
        env.reset(seed=int(config.seed))
        encoder = CabinetSkillStateEncoder(env, latched_gripper=actor.settings["latched_gripper"])
        if encoder.contract()["dimension"] != backend.state_dim or tuple(encoder.object_names) != backend.object_names:
            raise ValueError("Cabinet state encoder and world model object/state dimensions differ")
        def new_collector(directory):
            if collection_contract is not None:
                return CpuRoundCollector(config, actor, directory,
                    object_shift_xy=getattr(args, "object_shift_xy", (0., 0.)))
            return SkillCollector(env, encoder, actor, directory, **collector_options)
        prepare_output(output, config, env_config)
        prepared = True
        write_json(output / "action_contract.json", action_contract(actor.settings))
        write_json(output / "state_contract.json", encoder.contract())
        manifest = json.loads((output / "manifest.json").read_text())
        manifest.update(agent=config.agent, checkpoint_schema=SCHEMA, raps_reference_commit=RAPS_REFERENCE_COMMIT,
                        sampling_contract=collection_contract,
                        collection_optimizations=bool(config.get("collection_optimizations", False)),
                        collection_policy=config.get("collection_policy", None),
                        collection_policies=list(policies) if args.stage != "evaluate" else [],
                        episode_steps=env_config.max_ee_steps_per_episode,
                        prediction_horizon_seconds=None, imagination_horizon_skills=actor.settings["imagination_horizon"],
                        log_flush_interval_steps=32,
                        adaptations=manifest["adaptations"] + [
                            "RAPS reference conditional actor/MLP/lambda-return; state GNN replaces RSSM",
                            "GNN endpoint sample std summed over dynamic dimensions, not latent P2E std mean",
                            "Fixed per-skill discount; no learned continuation or duration model",
                            "Shared-state ensemble disagreement, ensemble-mean imagined state propagation",
                            "Inactive model arguments zeroed; reference actor logprob/entropy retains all arguments",
                            "Skill-controller latch and ground-support stop are explicit flag branches",
                            "Enabled command latch is observed as agent state, excluded from curiosity",
                            "Incomplete skills excluded from endpoint replay; all real ticks preserved",
                        ])
        if config.raps_actor.pick_place_primitives:
            manifest.update(
                action_semantics=(f"{actor.schema.action_dim}D: {actor.schema.num_primitives} row-major grid "
                                  f"{'/'.join(actor.schema.logical_primitive_names)} branches "
                                  f"({actor.schema.num_grasp_branches} grasp, cell-major then orientation ID when enabled) "
                                  "plus shared grasp-only Z; completed macro endpoints"
                                  if config.raps_actor.get("pick_place_grid", False) else
                                  "4D constant grasp choice plus bounded XYZ; completed grasp endpoints before lifecycle reset"
                                  if config.raps_actor.get("grasp_only", False) else
                                  "5D grasp/place choice plus schema-defined bounded position arguments; completed macro endpoints"),
                action_dim=actor.schema.action_dim,
                grasp_retention=actor.settings["grasp_retention"],
                collection_reset=("Success/held failure resets; unheld failure retains objects. Planned default joints before every grasp; observe/select after homing. Homing controls count separately; endpoints retained before homing/reset"
                                  if config.raps_actor.get("grasp_only", False) else
                                  "Every collection call resets physics; unfinished macros are excluded from WM replay"))
        if collection_contract is not None:
            manifest.update(collection_reset="Fresh CPU scene per stream per collection call; success/held reset, unheld failure retains objects; planned default joints before every grasp",
                            collection_merge="Stream ID then local order; exact total real-control quota",
                            collection_updates="All streams finish and merge before WM fit then actor/critic fit",
                            proposal_device="cpu")
        if actor.settings["reconfigure_on_reset"]:
            manifest["physics_reset"] = "fresh_scene_on_episode_v1; rebuild scene and rebind encoder only at existing episode resets"
        write_json(output / "manifest.json", manifest)
        if args.stage == "evaluate":
            load(args.checkpoint, backend, actor, encoder, sampling=collection_contract)
            collector = new_collector(output / "evaluation")
            ticks, skills = collect(collector, int(config.evaluation_steps), args.policy,
                                              seed=int(config.seed) + 100000, stage="evaluation")
            result = summarize(ticks, skills, actor.schema)
            valid = [row for row in skills if row["trainable"]]
            result["prediction"] = (prediction_metrics(backend, skill_arrays(valid),
                horizons=(1,) if actor.settings.get("grasp_only", False) else (1, 3, 5))
                                    if len(valid) >= 2 else {"status": "insufficient_completed_skills"})
            result.update(policy=args.policy, collection_policy=config.get("collection_policy", None),
                          model_updated=False, actor_updated=False, normalizers_updated=False,
                          source_checkpoint=str(args.checkpoint.resolve()),
                          object_shift_xy_m=list(getattr(args, "object_shift_xy", (0.0, 0.0))))
            write_json(collector.directory / "evaluation.json", result)
            return 0
        if backend.runtime_optimizations:
            writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="raps-checkpoint")
        smoke = args.stage == "smoke"
        resuming = args.stage == "train" and args.checkpoint is not None
        start_round = 1
        if resuming:
            ticks, skills, start_round = resume_history(args.checkpoint, config)
            load(args.checkpoint, backend, actor, encoder, sampling=collection_contract)
            initial_episode = ticks[-1]["episode_id"]
            initial_skill = max(row["skill_id"] for row in skills)
            # Keep independent complete JSONL logs; do not modify the source run.
            for name, rows in (("transitions.jsonl", ticks), ("skills.jsonl", skills)):
                for directory in ("bootstrap", config.collection_policy):
                    target = output / directory
                    target.mkdir(exist_ok=True)
                    with (target / name).open("w") as handle:
                        for row in rows:
                            if (row["stage"] == "bootstrap") == (directory == "bootstrap"):
                                handle.write(json.dumps(row, allow_nan=False) + "\n")
            shutil.copy2(args.checkpoint.parent / "training.jsonl", output / config.collection_policy / "training.jsonl")
            manifest.update(resume_checkpoint=str(args.checkpoint.resolve()),
                            imported_training_steps=len(ticks), next_round=start_round,
                            simulation_resume="Physics resets at the next collection round, as in baseline")
            write_json(output / "manifest.json", manifest)
        else:
            bootstrap = new_collector(output / "bootstrap")
            ticks, skills = collect(bootstrap, 192 if smoke else int(config.bootstrap_steps), "random",
                                              seed=int(config.seed), stage="bootstrap")
            fit(backend, actor, skills, config, bootstrap.directory, "bootstrap", smoke=smoke)
            checkpoint(bootstrap.directory / "checkpoint.pt", ticks, skills)
            initial_episode, initial_skill = bootstrap.episode_id, bootstrap.skill_id
        initial_model, initial_actor, initial_rng = deepcopy(backend.state_dict()), deepcopy(actor.state_dict()), rng_state()
        for policy in policies:
            backend.load_state_dict(deepcopy(initial_model))
            actor.load_state_dict(deepcopy(initial_actor))
            restore_rng(initial_rng)
            history, endpoints = list(ticks), list(skills)
            collector = new_collector(output / policy)
            collector.episode_id, collector.skill_id = initial_episode, initial_skill
            collector.total_steps = len(ticks)
            rounds = 1 if smoke else int(config.rounds)
            for index in range(start_round, rounds + 1):
                write_json(output / "progress.json", dict(stage="collecting", policy=policy, round=index,
                                                          arm_training_steps=len(history)))
                latest, macros = collect(collector, 32 if smoke else int(config.steps_per_round), policy,
                                                     seed=int(config.seed) + 10000 * index, stage=f"round_{index}")
                history.extend(latest); endpoints.extend(macros)
                write_json(output / "progress.json", dict(stage="training", policy=policy, round=index,
                                                          arm_training_steps=len(history)))
                training = fit(backend, actor, endpoints, config, collector.directory, f"round_{index}", smoke=smoke)
                path = collector.directory / f"round_{index:03d}.pt"
                checkpoint(path, history, endpoints)
                write_json(path.with_suffix(".json"), dict(training=training, collection=summarize(latest, macros, actor.schema)))
                print(json.dumps(dict(policy=policy, round=index, real_training_steps=len(history),
                                      completed_skills=training["completed_skills"], actor_loss=training["actor"]["actor_loss"])), flush=True)
            if smoke:
                replay = skill_arrays(endpoints)
                expected = backend.predict(replay["states"][:1], replay["actions"][:1])
                random_before = rng_state()
                expected_action = actor.action(replay["states"][0])
                # Checkpoint stores RNG before the sampled action above.
                restore_rng(random_before)
                checkpoint(path, history, endpoints)
                drain_checkpoint()
                load(path, backend, actor, encoder, sampling=collection_contract)
                np.testing.assert_array_equal(expected, backend.predict(replay["states"][:1], replay["actions"][:1]))
                np.testing.assert_array_equal(expected_action, actor.action(replay["states"][0]))
            if not smoke and int(config.evaluation_steps) == 0:
                write_json(collector.directory / "evaluation.json", dict(status="not_run", real_steps=0, policy=policy))
                continue
            evaluation, eval_skills = collect(collector, 32 if smoke else int(config.evaluation_steps), "curiosity" if policy == "curiosity" else "random",
                                                         seed=int(config.seed) + 100000, stage="evaluation")
            result = summarize(evaluation, eval_skills, actor.schema)
            result["policy"] = policy
            # Short smoke/evaluation may contain no complete skill; don't claim
            # that fitting static endpoints validates interaction dynamics.
            valid = [row for row in eval_skills if row["trainable"]]
            if len(valid) >= 2:
                result["prediction"] = prediction_metrics(backend, skill_arrays(valid),
                    horizons=(1,) if actor.settings.get("grasp_only", False) else (1, 3, 5))
            else:
                result["prediction"] = {"status": "insufficient_completed_evaluation_skills"}
            write_json(collector.directory / "evaluation.json", result)
        real_steps = len(ticks) + len(policies) * ((rounds - start_round + 1) * (32 if smoke else int(config.steps_per_round))
                                                  + (32 if smoke else int(config.evaluation_steps)))
        if smoke and real_steps > 300:
            raise AssertionError("Physical smoke budget exceeded")
        drain_checkpoint()
        write_json(output / "result.json", dict(status="completed", real_steps=real_steps,
                                               imported_training_steps=len(ticks) if resuming else 0,
                                               new_real_steps=real_steps - (len(ticks) if resuming else 0),
                                               smoke=smoke, checkpoint_restore_checked=smoke,
                                               matched_random_baseline=args.stage == "compare",
                                               collection_policy=config.get("collection_policy", None),
                                               collection_policies=list(policies),
                                               wall_seconds=time.perf_counter() - wall_started))
        write_json(output / "progress.json", dict(stage="completed", real_steps=real_steps))
    except BaseException as exc:
        if prepared:
            write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc),
                                                     status="stopped_without_random_fallback"))
        raise
    finally:
        try:
            drain_checkpoint()
        except BaseException as exc:
            if prepared:
                write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc),
                                                         status="stopped_without_random_fallback"))
            raise
        finally:
            if writer is not None:
                writer.shutdown(wait=True)
            if env is not None:
                env.close()
    return 0
