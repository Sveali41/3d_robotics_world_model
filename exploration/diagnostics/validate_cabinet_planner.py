"""Bounded planner validation with independent PhysX substep contact auditing.

Run in the documented MPLib/NumPy 1.26 robotics runtime. No training, primitive
changes, physical welding, or hidden goal adjustment is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from domain.robocasa.motion_planning import MotionPlanningError, _array, _pose_vector, _plane_min_distance
from exploration.cee_us.cee_us_cabinet import make_env

ROOT = Path(__file__).resolve().parents[2]
FINGERS = {"panda_leftfinger", "panda_rightfinger"}


def contact_category(first, second, *, payload=None, allowed_items=(),
                     allow_support=False, position=None, normal=None, shelf_z=.672):
    """Classify contact intent independently of the planner's collision query."""
    if first[0] == "robot" or second[0] == "robot":
        robot, other = (first, second) if first[0] == "robot" else (second, first)
        if other[0] == "robot":
            return "forbidden_robot_self"
        if robot[1] == "panda_link0" and other[0] == "ground":
            return "allowed_fixed_base_ground"
        if (robot[1] in FINGERS and other[0] == "object"
                and (other[1] in allowed_items or other[1] == payload)):
            return "allowed_finger_object"
        return "forbidden_robot_environment"
    if payload is not None and (first == ("object", payload) or second == ("object", payload)):
        other = second if first == ("object", payload) else first
        if (allow_support and other == ("cabinet", "frame") and position is not None
                and abs(position[2]-shelf_z) <= .002 and abs(normal[2]) >= .99):
            return "allowed_initial_support"
        return "forbidden_payload_environment"
    return None


def point_is_active(separation, impulse):
    # Preserve the raw separation/impulse too: speculative positive-distance
    # zero-impulse manifolds are not physical impacts.
    return ((separation <= 0 and np.linalg.norm(impulse) > 1e-7)
            or separation < -1e-5)


class PhysicalCollision(RuntimeError):
    pass


class ContactAudit:
    def __init__(self, raw, path, max_substeps):
        self.raw, self.max_substeps = raw, max_substeps
        self.stream = path.open("w")
        self.substeps = 0
        self.stage = "initial"
        self.payload = None
        self.allowed_items = ()
        self.allow_support = False
        self.support_cleared = False
        self.stop_on_violation = True
        self.records = []
        self.stage_stats = {}
        self.names = {}
        for link in raw.agent.robot.get_links():
            self.names[link._objs[0].name] = ("robot", link.name)
        for link in raw.cabinet.get_links():
            self.names[link._objs[0].name] = ("cabinet", link.name)
        for kind, actors in (("object", raw.cabinet_items), ("ground", [raw.ground]), ("wall", [raw.wall])):
            for actor in actors:
                self.names[actor._bodies[0].name] = (kind, actor.name)
        original = self._original_after_simulation_step = raw._after_simulation_step

        def observe():
            original()
            self.observe()
        self._installed_after_simulation_step = observe
        raw._after_simulation_step = observe

    def set_stage(self, stage, *, allowed_items=(), support=False):
        self.stage, self.allowed_items = stage, tuple(allowed_items)
        self.allow_support, self.support_cleared = support, False

    def observe(self):
        self.substeps += 1
        if self.substeps > self.max_substeps:
            raise RuntimeError("Declared physics-step budget exceeded")
        stats = self.stage_stats.setdefault(self.stage, dict(substeps=0, forbidden_substeps=0,
            allowed_contact_substeps=0, maximum_forbidden_force_N=0., maximum_forbidden_penetration_m=0.,
            maximum_allowed_force_N=0., maximum_allowed_penetration_m=0.))
        stats["substeps"] += 1
        shelf_z = float(np.asarray(self.raw.CABINET_POSE.p)[2]+.012)
        if self.payload and self.allow_support and not self.support_cleared:
            planner = self.raw.motion_planner
            actor = next(actor for actor in self.raw.cabinet_items if actor.name == self.payload)
            minimum = min(_plane_min_distance(shape,
                planner._shape_pose(shape, planner._pose(actor.pose)*shape.local_pose),
                planner.sapien.Pose(), planner._kind(shape))
                for shape in actor._bodies[0].get_collision_shapes())
            self.support_cleared = minimum-shelf_z > .002
        pairs = []
        forbidden = allowed = False
        for contact in self.raw.scene.get_contacts():
            names = [body.name for body in contact.bodies]
            if any(name not in self.names for name in names):
                raise RuntimeError(f"Unmapped physical contact body: {names}")
            first, second = [self.names[name] for name in names]
            points = []
            for point in contact.points:
                position, normal, impulse = map(lambda v: np.asarray(v, dtype=float),
                                                (point.position, point.normal, point.impulse))
                category = contact_category(first, second, payload=self.payload,
                    allowed_items=self.allowed_items,
                    allow_support=self.allow_support and not self.support_cleared,
                    position=position, normal=normal, shelf_z=shelf_z)
                if category is None:
                    continue
                active = bool(point_is_active(point.separation, impulse))
                force = float(np.linalg.norm(impulse)/self.raw.sim_timestep)
                penetration = max(0., -float(point.separation))
                points.append(dict(category=category, active=active, separation_m=float(point.separation),
                    impulse_Ns=impulse.tolist(), force_magnitude_N=force,
                    position_world_m=position.tolist(), normal=normal.tolist()))
                if active:
                    prefix = "forbidden" if category.startswith("forbidden") else "allowed"
                    stats[f"maximum_{prefix}_force_N"] = max(stats[f"maximum_{prefix}_force_N"], force)
                    stats[f"maximum_{prefix}_penetration_m"] = max(stats[f"maximum_{prefix}_penetration_m"], penetration)
                    if prefix == "forbidden":
                        forbidden = True
                    else:
                        allowed = True
            if points:
                pairs.append(dict(first=first, second=second, points=points))
        stats["forbidden_substeps"] += int(forbidden)
        stats["allowed_contact_substeps"] += int(allowed)
        row = dict(substep=self.substeps, time_s=self.substeps*self.raw.sim_timestep,
                   stage=self.stage, forbidden=forbidden, support_cleared=self.support_cleared, pairs=pairs)
        self.stream.write(json.dumps(row, allow_nan=False)+"\n")
        if forbidden:
            self.records.append(row)
            self.stream.flush()
            if self.stop_on_violation:
                raise PhysicalCollision(f"Forbidden physical contact at substep {self.substeps}, stage {self.stage}")

    def close(self):
        try:
            if self.raw._after_simulation_step is self._installed_after_simulation_step:
                self.raw._after_simulation_step = self._original_after_simulation_step
        finally:
            self.stream.close()


def rotated(quaternion, axis, degrees):
    rotation = Rotation.from_euler(axis, degrees, degrees=True)*Rotation.from_quat(np.roll(quaternion, -1))
    return np.roll(rotation.as_quat(), 1)


def run_episode(output, layout, mode, repeat, *, limit=420):
    name = f"{layout}_{mode}_{repeat}"
    directory = output/name
    directory.mkdir()
    shift = .03 if layout == "shifted" else 0.
    env, _ = make_env(episode_steps=1000, centered_initial_tcp=True, object_shift_xy=(shift, 0.))
    report = dict(name=name, layout=layout, mode=mode, repeat=repeat, seed=3100000+repeat,
                  attempts=[], negative_tests=[], status="running")
    audit = None
    trace = (directory/"control_ticks.jsonl").open("w")
    def save():
        if audit is not None:
            report.update(physics_substeps=audit.substeps, stage_stats=audit.stage_stats,
                          first_forbidden_contact=audit.records[0] if audit.records else None)
        (directory/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    try:
        env.reset(seed=report["seed"])
        raw = env.unwrapped
        planner = raw.motion_planner
        audit = ContactAudit(raw, directory/"physics_contacts.jsonl", limit*5)

        def direct(action, stage):
            raw.step(np.asarray(action,dtype=np.float32)[None,:])
            trace.write(json.dumps(dict(stage=stage, action=np.asarray(action).tolist(),
                qpos=_array(raw.agent.robot.get_qpos()).reshape(9).tolist()))+"\n")

        def execute(position, quaternion, stage, *, allowed=(), support=False, main=False):
            request = dict(stage=stage, target_position_root=np.asarray(position).tolist(),
                           target_quaternion_wxyz=np.asarray(quaternion).tolist(),
                           start_qpos=_array(raw.agent.robot.get_qpos()).reshape(9).tolist(),
                           start_scene=planner.snapshot_scene(), main_trial=main)
            report["attempts"].append(request)
            audit.set_stage(stage,allowed_items=allowed,support=support)
            before = audit.substeps
            try:
                plan = raw.plan_to_pose(position, quaternion,seed=20261103+repeat,
                                        allowed_contact_objects=allowed)
                request["planning"] = plan.metadata
                np.savez(directory/(stage+"_plan.npz"),position=plan.position,time=plan.time)
                result=raw.execute_motion_plan(plan,max_steps=180,
                    on_step=lambda row:trace.write(json.dumps(dict(trial=stage,**row))+"\n"))
                request.update(status=result["status"],control_steps=result["control_steps"],
                    endpoint_position_error_m=result["endpoint_position_error_m"],
                    endpoint_angle_error_deg=result["endpoint_angle_error_deg"])
            except MotionPlanningError as exc:
                request.update(status=exc.status,detail=exc.detail)
                plan=None
            except PhysicalCollision as exc:
                request.update(status="physical_collision",detail=str(exc))
                raise
            finally:
                request["physics_substeps"] = audit.substeps-before
                save()
            return plan,request["status"]

        audit.set_stage("setup_open")
        for _ in range(9):
            direct([0,0,0,0,0,0,1],"setup_open")
        if mode != "empty":
            source=json.loads((ROOT/"logs/grasp_complete_diagnostic_20261103/complete_summary.json").read_text())["verified"]["shifted_"+mode]
            position=np.asarray(source["goal_root_m"])+[shift-.03,0,0]
            target="cabinet_item_16" if mode=="cup" else "cabinet_item_17"
            plan,status=execute(position,source["quaternion_wxyz"],"setup_approach",
                                allowed=(target,) if mode=="cup" else ())
            if status!="success":
                report["status"]="setup_planning_failed"
                return report
            audit.set_stage("setup_close",allowed_items=(target,))
            raw.agent.set_control_mode("pd_joint_pos");raw.agent.controller.reset()
            for _ in range(35):
                direct([*plan.position[-1,:7],-1.],"setup_close")
            raw.agent.set_control_mode("pd_ee_delta_pose");raw.agent.controller.reset()
            raw.agent.controller.controllers["gripper"].set_action(torch.tensor([[-1.]],device=raw.device))
            attachment=planner.attachment_state()
            if attachment is None or attachment["name"]!=target:
                report["status"]="setup_grasp_failed"
                return report
            report["initial_attachment"]=attachment
            audit.payload=target
        if repeat==1:
            tcp=raw.agent.robot.pose.inv()*raw.agent.tcp.pose
            delta=[-.01,-.015,.01] if mode=="empty" else [-.01,0,.02]
            _,status=execute(_array(tcp.p).reshape(3)+delta,_array(tcp.q).reshape(4),
                             "setup_preposition",support=mode!="empty")
            if status!="success":
                report["status"]="setup_preposition_failed"
                return report
        tcp=raw.agent.robot.pose.inv()*raw.agent.tcp.pose
        origin=_array(tcp.p).reshape(3)
        quaternion=_array(tcp.q).reshape(4)
        sign=1 if repeat==0 else -1
        if mode=="empty":
            goals=[(origin+[-.02,sign*.03,.04],rotated(quaternion,"z",sign*5)),
                   (origin+[.015,-sign*.025,.06],rotated(quaternion,"y",-sign*5)),
                   (origin+[-.025,sign*.015,.02],quaternion)]
        else:
            goals=[(origin+[0,0,.045],quaternion),
                   (origin+[-.045,sign*.012,.055],rotated(quaternion,"z",sign*5)),
                   (origin+[-.07,-sign*.015,.035],rotated(quaternion,"y",sign*5))]
        report["declared_goals"]=[dict(position=p.tolist(),quaternion=q.tolist()) for p,q in goals]
        for index,(position,goal_quaternion) in enumerate(goals):
            _,status=execute(position,goal_quaternion,f"trial_{index}",support=mode!="empty" and index==0,main=True)
            # A rejected request is retained. Stop this chain after an execution
            # failure; subsequent requests would have a different start state.
            if status!="success":
                report["status"]="trajectory_failed_or_rejected"
                break
        else:
            report["status"]="completed"
        for label,position in (("inside_shelf",[.50,0.,.660]),("inside_wall",[.80,0.,.740])):
            before=audit.substeps
            test=dict(name=label,position_root=position)
            report["negative_tests"].append(test)
            try:
                raw.plan_to_pose(position,quaternion,seed=20261103+repeat)
                test.update(status="unexpected_plan_accepted",passed=False)
            except MotionPlanningError as exc:
                test.update(status=exc.status,detail=exc.detail,
                            passed=exc.status in ("no_collision_free_ik","path_collision"))
            test["physics_substeps"]=audit.substeps-before
            assert test["physics_substeps"]==0
        if mode=="empty" and repeat==0:
            tcp=raw.agent.robot.pose.inv()*raw.agent.tcp.pose
            before=audit.substeps
            test=dict(name="scene_changed_after_planning")
            report["negative_tests"].append(test)
            try:
                plan=raw.plan_to_pose(_array(tcp.p).reshape(3)+[0,0,.01],_array(tcp.q).reshape(4),seed=20261103)
                item=raw.cabinet_items[0]
                import sapien
                pose=_pose_vector(item.pose)
                item.set_pose(sapien.Pose(pose[:3]+[.005,0,0],pose[3:]))
                raw.execute_motion_plan(plan)
                test.update(status="unexpected_stale_plan_execution",passed=False)
            except MotionPlanningError as exc:
                test.update(status=exc.status,passed=exc.status=="stale_plan")
            test["physics_substeps"]=audit.substeps-before
            assert test["physics_substeps"]==0
    except PhysicalCollision as exc:
        report.update(status="physical_collision",failure=str(exc))
    except Exception as exc:
        report.update(status="error",failure=repr(exc))
    finally:
        save()
        if audit is not None:
            audit.close()
        trace.close()
        env.close()
    return report


def old_collision_control(output,obj):
    """Execute the archived failing low-level actions as detector control."""
    directory=output/("old_no_planner_"+obj);directory.mkdir()
    env,_=make_env(episode_steps=1000,centered_initial_tcp=True,object_shift_xy=(.03,0.))
    audit=None
    report=dict(name="old_no_planner_"+obj,object=obj,expected_collision=True)
    try:
        env.reset(seed=2000003 if obj=="bowl" else 2000002)
        raw=env.unwrapped
        audit=ContactAudit(raw,directory/"physics_contacts.jsonl",350)
        audit.stop_on_violation=False
        audit.set_stage("old_direct_execution")
        path=ROOT/f"logs/grasp_path_diagnostic_20261103/baseline/shifted_{obj}/transitions.jsonl"
        rows=[json.loads(line) for line in path.read_text().splitlines()][:70]
        for row in rows:
            raw.step(np.asarray(row["full_controller_action"],dtype=np.float32).reshape(1,7))
        report.update(status="detector_positive_control_passed" if audit.records else "detector_missed_collision",
                      physics_substeps=audit.substeps,stage_stats=audit.stage_stats,
                      first_forbidden_contact=audit.records[0] if audit.records else None)
    except Exception as exc:
        report.update(status="error",failure=repr(exc))
    finally:
        if audit is not None:audit.close()
        (directory/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False))
        env.close()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    sources=("domain/robocasa/motion_planning.py","domain/wall_cabinet_env.py",str(Path(__file__).relative_to(ROOT)))
    manifest=dict(layouts=["original","shifted"],modes=["empty","cup","bowl"],repeats=[0,1],
        declared_main_trajectory_requests=36,declared_episode_count=12,control_freq_hz=20,physics_freq_hz=100,
        max_control_ticks_per_episode=420,max_total_control_ticks_including_controls=5180,
        expected_negative_goal_tests=24,expected_stale_plan_tests=2,old_collision_controls=2,
        actual_contact_definition="separation<=0 and impulse norm>1e-7 Ns, OR separation<-1e-5 m",
        support_exception="Only initial shelf departure, upward/downward normal |nz|>=.99 and contact z within2mm of shelf; expires after payload bottom clears2mm",
        unknown_contact_body_policy="fail",training=False,planner_code_frozen=True,
        seed_note="Reset is deterministic; repeat1 adds a planned preposition, then changes target offsets/rotations. MPLib RRT seed is unavailable.",
        goals_note="Calibration fixtures set up held objects. Main targets are declared relative offsets and orientations, not learned policy actions.",
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    reports=[];start=time.monotonic()
    for layout in manifest["layouts"]:
        for mode in manifest["modes"]:
            for repeat in manifest["repeats"]:
                report=run_episode(output,layout,mode,repeat)
                reports.append(report)
                (output/"progress.json").write_text(json.dumps(dict(elapsed_s=time.monotonic()-start,reports=reports),indent=2))
                print(json.dumps(dict(episode=report["name"],status=report["status"],physics_substeps=report.get("physics_substeps"),
                    main_trials=[r["status"] for r in report["attempts"] if r["main_trial"]])),flush=True)
    controls=[old_collision_control(output,obj) for obj in ("cup","bowl")]
    result=dict(manifest=manifest,elapsed_s=time.monotonic()-start,episodes=reports,controls=controls)
    (output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(done=True,elapsed_s=result["elapsed_s"])),flush=True)


if __name__=="__main__":main()
