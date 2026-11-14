"""Bounded TCP geometry and explicit grasp-endpoint diagnostics; no grasp execution."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from domain.robocasa.motion_planning import (
    MotionPlanningError, _angle, _array, _pose_vector, _plane_min_distance, _native_modules,
)
from exploration.cee_us.cee_us_cabinet import make_env
from exploration.cee_us.cee_us_state import OBJECT_NAMES
from exploration.diagnostics.collect_wm_validation import SCENES
from exploration.diagnostics.diagnose_cabinet import source_evidence, write_json
from exploration.diagnostics.validate_cabinet_planner import ContactAudit


def candidate_goals(position, quaternion, top_z, jitter):
    """Declare seventeen requests in robot-root coordinates, without adaptive search."""
    position, jitter = np.asarray(position), np.asarray(jitter)
    baseline = position + jitter
    rotation = Rotation.from_quat(np.roll(quaternion, -1))
    orientations = [('held', rotation),
        ('root_y_plus30', Rotation.from_euler('y', 30, degrees=True) * rotation),
        ('root_y_minus30', Rotation.from_euler('y', -30, degrees=True) * rotation),
        ('tool_z_plus45', rotation * Rotation.from_euler('z', 45, degrees=True)),
        ('tool_z_minus45', rotation * Rotation.from_euler('z', -45, degrees=True))]
    points = [('legacy_grasp', baseline, orientations[:1]),
              ('legacy_standoff', baseline + [-.08, 0., 0.], orientations[:1]),
              ('front_raised', baseline + [-.02, 0., .02], orientations),
              ('upper_front', np.array([baseline[0]-.02, baseline[1], top_z-.01]), orientations),
              ('inward_raised', baseline + [-.02, .025, .02], orientations)]
    return [dict(name=point_name+'__'+orientation_name,
                 position_root=np.asarray(point).tolist(),
                 offset_from_object_root=(point-position).tolist(),
                 orientation=orientation_name,
                 quaternion_wxyz=np.roll(orientation.as_quat(), 1).tolist())
            for point_name, point, variants in points for orientation_name, orientation in variants]


def _pose(value):
    import sapien
    value = _pose_vector(value)
    return sapien.Pose(value[:3], value[3:])


def distal_pad_boxes(boxes):
    """The Panda rubber pad is the distal-most box along TCP local +Z."""
    return [max((box for box in boxes if box['link'] == name),
                key=lambda box: box['pose_in_tcp'][2])
            for name in ('panda_leftfinger', 'panda_rightfinger')]


def frame_audit(raw):
    planner = raw.motion_planner
    links = {link.name: link for link in raw.agent.robot.get_links()}
    tcp, hand, root = map(_pose, (raw.agent.tcp.pose, links['panda_hand'].pose, planner.root_link.pose))
    boxes = []
    for name in ('panda_leftfinger', 'panda_rightfinger'):
        for shape in links[name]._objs[0].get_collision_shapes():
            if planner._kind(shape) == 'Box':
                relative = tcp.inv() * _pose(links[name].pose) * shape.local_pose
                boxes.append(dict(link=name, pose_in_tcp=_pose_vector(relative).tolist(),
                                 half_size_m=np.asarray(shape.half_size).tolist()))
    pads = distal_pad_boxes(boxes)
    qpos = _array(raw.agent.robot.get_qpos()).reshape(9)
    planner.kin.pmodel.compute_forward_kinematics(qpos[planner.pindices])
    fk = root * planner.kin.pmodel.get_link_pose(planner.kin.end_link_idx)
    error_m, error_deg = float(np.linalg.norm(fk.p-tcp.p)), _angle(fk.q, tcp.q)
    if error_m > 1e-5 or error_deg > .01:
        raise RuntimeError(f'Controller FK and live TCP disagree: {error_m}m, {error_deg}deg')
    centers = np.asarray([p['pose_in_tcp'][:3] for p in pads])
    return dict(tcp_link=raw.agent.tcp.name, planner_end_link='panda_hand_tcp',
        hand_to_tcp=_pose_vector(hand.inv()*tcp).tolist(), pads=pads, finger_collision_boxes=boxes,
        pad_selection='distal-most collision box center along TCP local +Z, per finger',
        pad_center_midpoint_in_tcp_m=centers.mean(axis=0).tolist(),
        open_pad_center_distance_m=float(np.linalg.norm(centers[0]-centers[1])),
        fk_position_error_m=error_m, fk_orientation_error_deg=error_deg,
        tcp_in_root=_pose_vector(root.inv()*tcp).tolist())


def object_audit(raw, actor):
    planner = raw.motion_planner
    planner.mplib, planner.fcl, planner.sapien = _native_modules()
    root = _pose(planner.root_link.pose)
    vertices = actor.get_first_collision_mesh(to_world_frame=True).vertices
    vertices = Rotation.from_quat(np.roll(root.q, -1)).inv().apply(vertices-root.p)
    bounds = [shape_bounds(s, planner._shape_pose(s, root.inv()*_pose(actor.pose)*s.local_pose), planner._kind(s))
              for s in actor._bodies[0].get_collision_shapes()]
    bounds = np.stack([np.min([b[0] for b in bounds], axis=0), np.max([b[1] for b in bounds], axis=0)])
    return dict(name=actor.name, pose_in_root=_pose_vector(root.inv()*_pose(actor.pose)).tolist(),
        collision_bounds_root_m=bounds.tolist(), collision_dimensions_root_m=np.diff(bounds, axis=0)[0].tolist(),
        bounds_method='exact support of native collision shapes, using planner axis conversion',
        mesh_helper_bounds_root_m=np.stack([vertices.min(axis=0), vertices.max(axis=0)]).tolist(),
        mesh_helper_note='Installed mesh helper rotates a Z-aligned trimesh cylinder with a PhysX X-axis pose; not used for goals',
        shapes=[dict(kind=raw.motion_planner._kind(s), local_pose=_pose_vector(s.local_pose).tolist())
                for s in actor._bodies[0].get_collision_shapes()])


def shape_bounds(shape, pose, kind):
    """Reuse the planner's exact plane support on the six root-axis normals."""
    import sapien
    h = 2**-.5
    positive = ([h, 0, h, 0], [h, -h, 0, 0], [1, 0, 0, 0])
    negative = ([h, 0, -h, 0], [h, h, 0, 0], [0, 1, 0, 0])
    lo = [_plane_min_distance(shape, pose, sapien.Pose(q=q), kind) for q in positive]
    hi = [-_plane_min_distance(shape, pose, sapien.Pose(q=q), kind) for q in negative]
    return np.array([lo, hi])


def check_goal(raw, goal, seed, planning_time_s):
    planner = raw.motion_planner
    before = planner.snapshot_scene()
    qpos = _array(raw.agent.robot.get_qpos()).copy()
    row = dict(goal, ik_seed=seed)
    try:
        plan = raw.plan_to_pose(goal['position_root'], goal['quaternion_wxyz'], seed=seed,
            planning_time_s=planning_time_s, allowed_contact_objects=tuple(OBJECT_NAMES))
        row.update(path_status='success', ik_diagnostics=plan.metadata['ik_diagnostics'],
                   endpoint_joint_positions_rad=plan.position[-1].tolist())
    except MotionPlanningError as exc:
        detail = exc.detail
        row.update(path_status=exc.status, detail=detail,
                   ik_diagnostics=detail.get('ik_diagnostics') if isinstance(detail, dict) else None)
    if planner.snapshot_scene() != before or not np.array_equal(_array(raw.agent.robot.get_qpos()), qpos):
        raise RuntimeError('Endpoint diagnostic changed physical state')
    diagnostics = row['ik_diagnostics']
    row['endpoint_ik_verified'] = diagnostics is not None and diagnostics['counts']['accepted'] > 0
    row['grasp_success_verified'] = False
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sampling-plan', type=Path, required=True)
    parser.add_argument('--ik-seed', type=int, default=0)
    parser.add_argument('--planning-time-s', type=float, default=.2)
    args = parser.parse_args(argv)
    if args.ik_seed < 0 or not np.isfinite(args.planning_time_s) or not 0 < args.planning_time_s <= .5:
        raise ValueError('Use a nonnegative IK seed and a path-planning budget in (0, 0.5] seconds')
    plans = [p for p in json.loads(args.sampling_plan.read_text()) if p['intent'] == 'lift']
    if len(plans) != 2 or {p['object'] for p in plans} != set(OBJECT_NAMES):
        raise ValueError('Sampling plan must contain exactly one lift plan for each target object')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    write_json(args.output/'manifest.json', dict(source_evidence=source_evidence(args.output),
        sampling_plan=str(args.sampling_plan), sampling_plan_sha256=hashlib.sha256(args.sampling_plan.read_bytes()).hexdigest(),
        plans=plans, ik_seed=args.ik_seed, path_planning_time_s_per_candidate=args.planning_time_s,
        candidate_count_per_case=17, setup_open_control_ticks_per_case=9,
        endpoint_candidate_execution=False, physical_grasp_test=False,
        coordinates='robot root; quaternion wxyz; root rotations pre-multiply, tool rotations post-multiply',
        dependencies={n:importlib.metadata.version(n) for n in ('numpy','mplib','sapien','torch')}))
    results = []
    for scene, shift in SCENES.items():
        env = None
        try:
            env, _ = make_env(episode_steps=800, centered_initial_tcp=True,
                object_shift_xy=shift, action_mode='joint_position')
            for plan in plans:
                name = scene+'_'+('cup' if plan['object'] == OBJECT_NAMES[0] else 'bowl')
                directory = args.output/name
                directory.mkdir()
                env.reset(seed=plan['seed'])
                raw = env.unwrapped
                planner = raw.motion_planner
                audit = ContactAudit(raw, directory/'setup_physics_contacts.jsonl', 45)
                audit.set_stage('stationary_open', allowed_items=OBJECT_NAMES)
                try:
                    qpos = _array(raw.agent.robot.get_qpos()).reshape(9)
                    for _ in range(9):
                        env.step(np.r_[qpos[:7], 1.].astype(np.float32))
                    frames = frame_audit(raw)
                    actor = next(a for a in raw.cabinet_items if a.name == plan['object'])
                    geometry = object_audit(raw, actor)
                    goals = candidate_goals(geometry['pose_in_root'][:3], frames['tcp_in_root'][3:],
                        geometry['collision_bounds_root_m'][1][2], plan['jitter'])
                    write_json(directory/'geometry.json', dict(frames=frames, object=geometry,
                        setup_qpos=_array(raw.agent.robot.get_qpos()).tolist(), scene=planner.snapshot_scene()))
                    rows = []
                    with (directory/'candidates.jsonl').open('w') as stream:
                        for goal in goals:
                            row = check_goal(raw, goal, args.ik_seed, args.planning_time_s)
                            stream.write(json.dumps(row, allow_nan=False)+'\n')
                            stream.flush()
                            rows.append(row)
                    assert audit.substeps == 45
                    result = dict(case=name, object=actor.name, seed=plan['seed'],
                        endpoint_feasible_candidates=[r['name'] for r in rows if r['endpoint_ik_verified']],
                        planned_candidates=[r['name'] for r in rows if r['path_status'] == 'success'],
                        candidates=rows, setup_physics_substeps=audit.substeps,
                        forbidden_contact_substeps=len(audit.records), frames=frames, object_geometry=geometry)
                    write_json(directory/'summary.json', result)
                    results.append(result)
                    print(json.dumps({k:v for k,v in result.items() if k not in {'candidates','frames','object_geometry'}}), flush=True)
                finally:
                    audit.close()
        finally:
            if env is not None:
                env.close()
    write_json(args.output/'summary.json', results)


if __name__ == '__main__':
    main()
