"""Finite local pose probes and gated physical grasp/lift tests, without training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import torch

from domain.robocasa.motion_planning import MotionPlanExecution, MotionPlanningError, _array, _pose_vector
from exploration.cee_us.cee_us_cabinet import make_env
from exploration.cee_us.cee_us_state import OBJECT_NAMES
from exploration.diagnostics.collect_wm_validation import SCENES
from exploration.diagnostics.diagnose_cabinet import jsonable, snapshot, source_evidence, write_json
from exploration.diagnostics.diagnose_grasp_targets import check_goal, frame_audit
from exploration.diagnostics.validate_cabinet_planner import ContactAudit, PhysicalCollision
from exploration.raps.raps_replay_compare import events


POSITION_RADIUS_M = np.array([.015, .025, .020])
ORIENTATION_RADIUS_DEG = np.array([15., 15., 20.])


def local_goals(anchor, seed):
    """Nine height-line points plus sixteen continuous stratified 6D samples."""
    line = np.zeros((9, 6))
    line[:, 2] = np.linspace(-POSITION_RADIUS_M[2], POSITION_RADIUS_M[2], 9)
    radius = np.r_[POSITION_RADIUS_M, ORIENTATION_RADIUS_DEG]
    neighborhood = (2*qmc.LatinHypercube(d=6, seed=seed).random(16)-1)*radius
    rotation = Rotation.from_quat(np.roll(anchor['quaternion_wxyz'], -1))
    result = []
    for i, delta in enumerate(np.vstack([line, neighborhood])):
        orientation = Rotation.from_euler('xyz', delta[3:], degrees=True)*rotation
        result.append(dict(name=f'height_{i:02d}' if i < 9 else f'local_{i-9:02d}',
            anchor_name=anchor['name'], sampling_family='height_line' if i < 9 else 'local_6d',
            position_root=(np.asarray(anchor['position_root'])+delta[:3]).tolist(),
            quaternion_wxyz=np.roll(orientation.as_quat(), 1).tolist(),
            position_delta_root_m=delta[:3].tolist(), root_xyz_euler_delta_deg=delta[3:].tolist()))
    return result


def execution_candidates(rows):
    """Predeclared selection: anchor, then lowest/highest planned goals, at most three."""
    feasible = [row for row in rows if row['path_status'] == 'success']
    ordered = [row for row in feasible if row['name'] == 'height_04']
    if feasible:
        ordered += [min(feasible, key=lambda r: (r['position_root'][2], r['name'])),
                    max(feasible, key=lambda r: (r['position_root'][2], r['name']))]
    return list({row['name']: row for row in ordered}.values())


def confirmed_grasp(rows, target, attachment):
    return (attachment is not None and attachment['name'] == target and len(rows) >= 4
        and all(row['parameters_physical']['gripper'] < 0
                and row['after']['contact_snapshot']['objects'][target]['grasped'] for row in rows[-4:]))


def physical_trial(env, directory, goal, target, seed, ik_seed, planning_time_s, limit=300):
    directory.mkdir()
    env.reset(seed=seed)
    raw = env.unwrapped
    planner = raw.motion_planner  # Initialize before any closing ticks.
    audit = ContactAudit(raw, directory/'physics_contacts.jsonl', limit*5)
    rows, actions = [], 0
    report = dict(goal=goal, seed=seed, target_object=target, status='running', motions=[],
                  control_tick_budget=limit, physical_grasp_verified=False, physical_lift_verified=False)
    trace = (directory/'control_ticks.jsonl').open('w')
    requests = (directory/'requested_actions.jsonl').open('w')

    def step(action, phase):
        nonlocal actions
        if actions >= limit:
            raise MotionPlanningError('budget_exhausted', 'Physical diagnostic control budget exhausted')
        before = snapshot(env)
        actions += 1
        requests.write(json.dumps(dict(step=actions, phase=phase, action=np.asarray(action).tolist()))+'\n')
        requests.flush()  # Retain the action even if the substep audit stops physics.
        _, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        record = info['primitive_record']
        row = jsonable(dict(step=actions, stage='local_grasp_probe', episode_id=0,
            execution_phase=phase, primitive_type='joint_position', action=record.low_actions[0],
            parameters_physical=dict(gripper=float(action[-1])),
            before=before, after=snapshot(env), contact_events=record.contact_events,
            reward=info['exploration_reward'], qpos=_array(raw.agent.robot.get_qpos()).reshape(9),
            terminated=bool(terminated), truncated=bool(truncated)))
        rows.append(row)
        trace.write(json.dumps(row, allow_nan=False)+'\n')
        if terminated or truncated:
            raise MotionPlanningError('terminated' if terminated else 'truncated', phase)
        return terminated, truncated

    def execute(position, quaternion, phase, gripper):
        if actions >= limit:
            raise MotionPlanningError('budget_exhausted', 'No controls remain for '+phase)
        audit.set_stage(phase, allowed_items=OBJECT_NAMES, support=phase == 'lift')
        request = dict(phase=phase, position_root=np.asarray(position).tolist(),
                       quaternion_wxyz=np.asarray(quaternion).tolist(), status='planning')
        report['motions'].append(request)
        try:
            plan = raw.plan_to_pose(position, quaternion, seed=ik_seed,
                planning_time_s=planning_time_s, allowed_contact_objects=tuple(OBJECT_NAMES))
            request['planning'] = plan.metadata
            np.savez(directory/(phase+'_plan.npz'), position=plan.position, time=plan.time)
            cursor = MotionPlanExecution(raw, plan, gripper_command=gripper, max_steps=limit-actions)
            while cursor.status == 'running':
                terminated, truncated = step(cursor.next_action(), phase)
                cursor.record_step(terminated=terminated, truncated=truncated)
            write_json(directory/(phase+'_execution.json'), cursor.result())
            request.update(status=cursor.status, control_steps=len(cursor.records),
                endpoint_position_error_m=cursor.endpoint_position_error_m,
                endpoint_angle_error_deg=cursor.endpoint_angle_error_deg)
            if cursor.status != 'success':
                raise MotionPlanningError(cursor.status, cursor.records[-1].get('attachment_error') or phase)
        except MotionPlanningError as exc:
            request.update(status=exc.status, detail=exc.detail)
            raise

    try:
        audit.set_stage('stationary_open', allowed_items=OBJECT_NAMES)
        joints = _array(raw.agent.robot.get_qpos()).reshape(9)[:7]
        for _ in range(9):
            step(np.r_[joints, 1.], 'stationary_open')
        report['initial_snapshot'] = snapshot(env)
        execute(goal['position_root'], goal['quaternion_wxyz'], 'approach', 1.)
        audit.set_stage('close', allowed_items=OBJECT_NAMES)
        joints = _array(raw.agent.robot.get_qpos()).reshape(9)[:7]
        close_rows = []
        for _ in range(35):
            step(np.r_[joints, -1.], 'close')
            close_rows.append(rows[-1])
        attachment = planner.attachment_state()
        report['post_close_attachment'] = attachment
        if not confirmed_grasp(close_rows, target, attachment):
            report['status'] = 'grasp_not_confirmed'
            return report
        report['physical_grasp_verified'] = True
        report['post_close_snapshot'] = snapshot(env)
        audit.payload = target
        tcp = _pose_vector(raw.agent.robot.pose.inv()*raw.agent.tcp.pose)
        execute(tcp[:3]+[0., 0., .04], tcp[3:], 'lift', -1.)
        final = snapshot(env)
        rise = final['objects'][target]['position_root'][2]-report['post_close_snapshot']['objects'][target]['position_root'][2]
        stable = confirmed_grasp(rows, target, planner.attachment_state())
        report.update(post_close_object_rise_m=rise,
            physical_lift_verified=bool(stable and rise >= .01),
            status='lift_verified' if stable and rise >= .01 else 'lift_not_verified')
    except MotionPlanningError as exc:
        report.update(status=exc.status, detail=exc.detail)
    except PhysicalCollision as exc:
        report.update(status='physical_collision', detail=str(exc))
    except Exception as exc:
        report.update(status='error', detail=repr(exc))
        raise
    finally:
        report.update(attempted_control_ticks=actions, completed_control_ticks=len(rows),
            physics_substeps=audit.substeps, forbidden_contact_substeps=len(audit.records),
            first_forbidden_contact=audit.records[0] if audit.records else None,
            stage_stats=audit.stage_stats, event_metrics=events(rows, []), final_snapshot=snapshot(env),
            final_scene=planner.snapshot_scene())
        write_json(directory/'summary.json', report)
        audit.close()
        trace.close()
        requests.close()
    return report


def plot_results(output, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, case in zip(axes.flat, results):
        for status, color, label in [('rejected', 'tab:red', 'No accepted endpoint'),
                                      ('endpoint', 'tab:orange', 'Endpoint only'),
                                      ('path', 'tab:green', 'Endpoint + path')]:
            rows = [r for r in case['candidates'] if
                ('path' if r['path_status'] == 'success' else 'endpoint' if r['endpoint_ik_verified'] else 'rejected') == status]
            axis.scatter([1000*r['position_delta_root_m'][1] for r in rows],
                [1000*r['position_delta_root_m'][2] for r in rows], c=color, label=label, s=40)
        for trial in case['physical_trials']:
            delta = trial['goal']['position_delta_root_m']
            axis.scatter(1000*delta[1], 1000*delta[2], facecolors='none', edgecolors='black', s=160)
            axis.annotate(trial['status'], (1000*delta[1], 1000*delta[2]), fontsize=7, xytext=(5, 5), textcoords='offset points')
        axis.set(title=case['case'], xlabel='Root Y offset from anchor (mm)', ylabel='Root Z offset from anchor (mm)')
        axis.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle('Finite local grasp probes: YZ projection; X and orientation also vary\nBlack rings: physically tested; green points alone do not establish grasp success')
    figure.savefig(output/'local_pose_probes.png', dpi=180)
    figure.savefig(output/'local_pose_probes.svg')
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Completed diagnose_grasp_targets directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    source_manifest = json.loads((args.source/'manifest.json').read_text())
    anchors = json.loads((args.source/'selected_candidates.json').read_text())
    source_cases = json.loads((args.source/'summary.json').read_text())
    if len(anchors) != 4 or {case['case'] for case in source_cases} != {'original_cup', 'original_bowl', 'shifted_cup', 'shifted_bowl'}:
        raise ValueError('Source must contain the four original/shifted cup/bowl cases')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol='local_grasp_region_v1', source=str(args.source), source_evidence=source_evidence(args.output),
        source_manifest_sha256=hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
        position_radius_root_m=POSITION_RADIUS_M.tolist(), root_xyz_euler_radius_deg=ORIENTATION_RADIUS_DEG.tolist(),
        orientation_convention='extrinsic root xyz Euler delta pre-multiplies anchor; quaternion wxyz',
        candidates_per_case=25, sampling_seed=20261105, ik_seed=source_manifest['ik_seed'],
        planning_time_s=source_manifest['path_planning_time_s_per_candidate'], physical_trials_per_case_at_most=3,
        physical_trial_control_budget=300, setup_open_ticks=9, close_ticks=35, lift_request_root_m=[0.,0.,.04],
        grasp_gate='final four closed-grasp control ticks on intended object plus matching measured attachment',
        physical_lift_gate='planned lift execution succeeds, final four held ticks, object rises >= .01 m from post-close',
        candidate_selection='anchor if planned, then min/max height among planned candidates; unique, at most three',
        limitation='Finite local diagnostic samples, not certified continuous regions or an unbiased WM validation set')
    write_json(args.output/'manifest.json', manifest)
    results = []
    for source_case, anchor in zip(source_cases, anchors):
        case_name, target = source_case['case'], source_case['object']
        if anchor.get('case') != case_name or anchor['path_status'] != 'success':
            raise ValueError(f'Expected a path-accepted anchor explicitly labeled {case_name}')
        scene = 'shifted' if case_name.startswith('shifted') else 'original'
        directory = args.output/case_name
        directory.mkdir()
        env, _ = make_env(episode_steps=800, centered_initial_tcp=True,
                          object_shift_xy=SCENES[scene], action_mode='joint_position')
        try:
            env.reset(seed=source_case['seed'])
            raw = env.unwrapped
            planner = raw.motion_planner
            audit = ContactAudit(raw, directory/'setup_physics_contacts.jsonl', 45)
            try:
                audit.set_stage('stationary_open', allowed_items=OBJECT_NAMES)
                joints = _array(raw.agent.robot.get_qpos()).reshape(9)[:7]
                for _ in range(9):
                    env.step(np.r_[joints, 1.].astype(np.float32))
                previous_position = np.asarray(source_case['object_geometry']['pose_in_root'][:3])
                current_position = np.asarray(snapshot(env)['objects'][target]['position_root'])
                if np.linalg.norm(current_position-previous_position) > .001:
                    raise RuntimeError('Reset object differs from source by more than 1 mm; regenerate anchors')
                write_json(directory/'frames.json', frame_audit(raw))
                candidates = []
                with (directory/'candidates.jsonl').open('w') as stream:
                    for goal in local_goals(anchor, manifest['sampling_seed']):
                        row = check_goal(raw, goal, manifest['ik_seed'], manifest['planning_time_s'])
                        candidates.append(row)
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                        stream.flush()
                result = dict(case=case_name, object=target, seed=source_case['seed'], candidates=candidates,
                    setup_physics_substeps=audit.substeps, setup_forbidden_contact_substeps=len(audit.records), physical_trials=[])
            finally:
                audit.close()
            selected = execution_candidates(candidates)
            write_json(directory/'selected_for_execution.json', selected)
            print(json.dumps(dict(case=case_name, phase='local_probes',
                endpoints=sum(r['endpoint_ik_verified'] for r in candidates),
                paths=sum(r['path_status']=='success' for r in candidates))), flush=True)
            for index, goal in enumerate(selected):
                trial = physical_trial(env, directory/f'physical_{index:02d}_{goal["name"]}', goal,
                    target, source_case['seed'], manifest['ik_seed'], manifest['planning_time_s'])
                result['physical_trials'].append(trial)
                write_json(directory/'summary.json', result)
                print(json.dumps(dict(case=case_name, goal=goal['name'], status=trial['status'],
                    controls=trial['attempted_control_ticks'])), flush=True)
            results.append(result)
            write_json(args.output/'summary.json', results)
        finally:
            env.close()
    plot_results(args.output, results)


if __name__ == '__main__':
    main()
