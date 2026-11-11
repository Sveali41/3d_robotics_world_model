"""Compare recorded cabinet grasp paths with bounded planner execution; no training."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from exploration.cee_us.cee_us_cabinet import make_env
from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
from exploration.diagnostics.calibrate_wm_validation import CalibrationProvider, checkpoint_contract
from exploration.diagnostics.collect_cabinet_grasps import Finished, read_rows, sha256, stop_callback
from exploration.diagnostics.diagnose_cabinet import write_json
from exploration.raps.raps_gnn_actor import SkillCollector

ROOT = Path(__file__).resolve().parents[2]


def validate_baseline(directory, checkpoint=None):
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest['stage'] != 'smoke' or manifest['split'] != 'validation_only' or
            manifest['case_budget'] != 800 or manifest['physics_hz'] != 100 or
            manifest['control_hz'] != 20 or manifest['object_spacing_m'] != .12 or
            [case['case_id'] for case in manifest['cases']] != [27, 81]):
        raise ValueError('Baseline must be the two-case original-layout smoke collection')
    checkpoint = Path(checkpoint or manifest['checkpoint'])
    if sha256(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs from baseline')
    for name, expected in manifest['source_sha256'].items():
        if sha256(ROOT/name) != expected or sha256(directory/'sources'/name) != expected:
            raise ValueError(f'Baseline source hash mismatch: {name}')
    payload, config, settings = checkpoint_contract(checkpoint)
    if (payload['state_contract'] != manifest['state_contract'] or
            payload['action_contract'] != manifest['action_contract'] or
            OmegaConf.to_container(config, resolve=True) != manifest['config'] or settings != manifest['settings']):
        raise ValueError('Baseline checkpoint configuration or contracts differ')
    return manifest, checkpoint, payload, settings


def trajectories(rows, initial):
    states = np.asarray([initial] + [row['next_state'] for row in rows], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 67 or not np.isfinite(states).all():
        raise ValueError('Expected finite state67 trajectory')
    return dict(tcp=states[:, 18:21], cup=states[:, 33:36], bowl=states[:, 45:48])


def path_length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def verify_initial(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != (67,) or actual.shape != (67,) or not np.isfinite(actual).all():
        raise ValueError('Invalid initial state67')
    error = float(np.max(np.abs(expected-actual)))
    if error > 1e-6:
        raise ValueError(f'Initial state mismatch: maximum absolute error {error:.9g} > 1e-6')
    return error


def monitor_row(row):
    if len(row['action']) != 8:
        raise ValueError('Planner monitoring expects eight controller action values')
    return dict(row, action=[0., 0., 0., row['action'][-1]])


class MatchedProvider(CalibrationProvider):
    def __init__(self, settings, case, args, commands):
        super().__init__(settings, case, args)
        self.baseline_commands = {command['phase']: command for command in commands if command['phase'] != 'park'}
        self.matches = []

    def issue(self, phase, name, **parameters):
        baseline = self.baseline_commands.get(phase)
        if baseline is not None:
            if baseline['primitive'] != name:
                raise ValueError(f'Baseline primitive mismatch at phase {phase}')
            parameters = baseline['parameters']
        action = super().issue(phase, name, **parameters)
        match = None if baseline is None else bool(np.array_equal(action, np.asarray(baseline['requested_action'], dtype=np.float32)))
        if match is False:
            raise ValueError(f'Baseline requested action mismatch at phase {phase}')
        self.matches.append(dict(phase=phase, matched_baseline=baseline is not None,
                                 requested_action_equal=match, parameters=parameters))
        return action


def run_planner(case, baseline_summary, baseline_first, payload, settings, directory):
    random.seed(case['seed']); np.random.seed(case['seed']); torch.manual_seed(case['seed'])
    env = None
    start_time = time.perf_counter()
    try:
        env, env_config = make_env(episode_steps=800, centered_initial_tcp=True,
                                   project_goals_to_region=False, object_shift_xy=(0., 0.), action_mode='joint_position')
        env.reset(seed=case['seed'])
        encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
        if encoder.contract() != payload['state_contract']:
            raise ValueError('Planner encoder contract differs from baseline')
        x, y, z = case['offsets']
        args = argparse.Namespace(x_offset=x, cup_y_offset=y, bowl_y_offset=y, cup_z_offset=z,
                                  bowl_z_offset=z, lift_distance=case['lift_distance'])
        planner_settings = dict(settings, motion_execution='planner', motion_plan_max_steps=400)
        provider = MatchedProvider(planner_settings, 'original_'+case['object'], args, baseline_summary['report']['commands'])
        collector = SkillCollector(env, encoder, provider, directory)
        initial = {}
        def on_start(episode, skill_id, decoded, action, state):
            if skill_id == 1:
                initial['max_abs_error'] = verify_initial(baseline_first, state)
                initial['state'] = state.tolist()
            # Associate zero-duration rejected plans with their requested phase.
            provider.commands[-1]['skill_id'] = skill_id
        try:
            collector.collect(800, 'curiosity', seed=case['seed'], stage='planner_diagnostic',
                              on_step=lambda row: provider.on_step(monitor_row(row)),
                              on_skill_start=on_start, on_skill_end=stop_callback(provider, collector))
        except Finished:
            pass
        elapsed = time.perf_counter()-start_time
        skills = read_rows(directory/'skills.jsonl')
        rows = read_rows(directory/'transitions.jsonl')
        report = provider.report(skills)
        write_json(directory/'phase_trace.json', provider.trace)
        result = dict(case=case, control_ticks=len(rows), wall_seconds=elapsed,
                      initial_max_abs_error=initial['max_abs_error'], initial_state=initial['state'],
                      initial_objects=dict(cup=initial['state'][33:36], bowl=initial['state'][45:48]),
                      environment=OmegaConf.to_container(env_config, resolve=True), calibration=report,
                      phase_action_matches=provider.matches,
                      skill_statuses=[dict(skill_id=s['skill_id'], primitive=s['primitive_type'], status=s['status'],
                                           completed=s['completed'], duration_low_steps=s['duration_low_steps'],
                                           motion_planning=s.get('motion_planning', [])) for s in skills])
        write_json(directory/'summary.json', result)
        return result, rows, skills
    finally:
        if env is not None:
            env.close()


def rejected_targets(skills):
    result = []
    for skill in skills:
        for plan in skill.get('motion_planning', []):
            if plan['status'] not in ('success', 'running'):
                result.append(dict(status=plan['status'], stage=plan['stage'],
                                   target=plan['target_position_root'], start=skill['state'][18:21]))
    return result


def draw(output, entries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    for i, entry in enumerate(entries):
        rejected = entry['rejected_targets']
        for j, projection in enumerate(('3d', 'xy', 'xz')):
            ax = fig.add_subplot(2, 3, 3*i+j+1, projection='3d' if projection == '3d' else None)
            def plot(points, **kw):
                return ax.plot(points[:, 0], points[:, 1], points[:, 2], **kw) if projection == '3d' else ax.plot(points[:, 0], points[:, 1 if projection == 'xy' else 2], **kw)
            for name, color in (('baseline', '#1764ad'), ('planner', '#e88216')):
                path = entry[name]['tcp']
                plot(path, color=color, lw=2)
                plot(entry[name][entry['case']['object']], color=color, lw=1.5, linestyle='--')
                plot(path[[0]], marker='o', color=color, linestyle='None', ms=5)
                plot(path[[-1]], marker='s', color=color, linestyle='None', ms=5)
            for target in rejected:
                points = np.asarray([target['start'], target['target']])
                plot(points, color='#9b2477', linestyle='--', lw=1)
                plot(points[[-1]], color='#9b2477', marker='x', linestyle='None', ms=7)
            for name, marker in (('cup', '^'), ('bowl', 'v')):
                plot(entry['baseline'][name][[0]], marker=marker, color='#303030', linestyle='None', ms=6)
            # Shelf top root-frame bounds: world cabinet center x .125 relative to robot x -.40.
            shelf = np.array([[.29, -.15, .672], [.76, -.15, .672], [.76, .39, .672], [.29, .39, .672], [.29, -.15, .672]])
            shelf_style = dict(color='#9b8c7a', linestyle=':', lw=1)
            if projection == '3d': shelf_style['axlim_clip'] = True
            plot(shelf, **shelf_style)
            ax.set_xlabel('Root X (m)')
            ax.set_ylabel('Root Y (m)' if projection != 'xz' else 'Root Z (m)')
            if projection == '3d': ax.set_zlabel('Root Z (m)')
            result = entry['planner_result']
            verdict = 'success' if result['calibration']['calibration_success'] else result['calibration']['failure']
            detail = rejected[0]['status'] if rejected else verdict
            ax.set_title(f"{entry['case']['object'].title()} · {projection.upper()}\nPlanner: {detail}", fontsize=10)
            extent = [entry[v][name] for v in ('baseline', 'planner') for name in ('tcp', entry['case']['object'])]
            extent += [entry['baseline'][name][[0]] for name in ('cup', 'bowl')]
            extent += [np.asarray([target['target']]) for target in rejected]
            points = np.concatenate(extent, axis=0)
            lower, upper = points.min(axis=0), points.max(axis=0)
            lower[2], upper[2] = min(lower[2], .672), max(upper[2], .672)
            lower -= .02
            upper += .02
            ax.set_xlim(lower[0], upper[0])
            if projection == '3d':
                ax.set_ylim(lower[1], upper[1])
                ax.set_zlim(lower[2], upper[2])
                ax.set_box_aspect(upper-lower)
            elif projection == 'xy':
                ax.set_ylim(lower[1], upper[1])
                ax.set_aspect('equal', adjustable='box')
            else:
                ax.set_ylim(lower[2], upper[2])
                ax.set_aspect('equal', adjustable='box')
    legend = [Line2D([0], [0], color='#1764ad', label='Baseline TCP'), Line2D([0], [0], color='#e88216', label='Planner TCP'),
              Line2D([0], [0], color='#555555', ls='--', label='Target object path (same colors)'),
              Line2D([0], [0], color='#9b2477', marker='x', ls='--', label='Rejected target (link illustrative)'),
              Line2D([0], [0], color='black', marker='o', ls='', label='Start'), Line2D([0], [0], color='black', marker='s', ls='', label='End'),
              Line2D([0], [0], color='#303030', marker='^', ls='', label='Initial cup / bowl (▲ / ▼)'),
              Line2D([0], [0], color='#9b8c7a', ls=':', label='Shelf top')]
    fig.legend(handles=legend, loc='outside lower center', ncol=4)
    fig.suptitle('Cabinet grasp paths: planner and baseline\nTCP paths do not certify whole-arm collision clearance.', fontsize=13)
    fig.savefig(output/'paths.png', dpi=200)
    fig.savefig(output/'paths.svg')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, checkpoint, payload, settings = validate_baseline(args.baseline, args.checkpoint)
    args.output.mkdir(parents=True)
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    entries, summaries, arrays = [], [], {}
    for case in manifest['cases']:
        base_dir = args.baseline/'cases'/f"case_{case['case_id']:03d}"
        baseline_summary = json.loads((base_dir/'summary.json').read_text())
        if baseline_summary['case'] != case: raise ValueError('Baseline case summary mismatch')
        attempt_dir = base_dir/f"attempt_{baseline_summary['attempt']:06d}"
        baseline_rows = read_rows(attempt_dir/'transitions.jsonl')
        baseline_skills = read_rows(attempt_dir/'skills.jsonl')
        if not baseline_skills: raise ValueError('Baseline has no initial skill')
        first = baseline_skills[0]['state']
        baseline_paths = trajectories(baseline_rows, first)
        directory = args.output/f"case_{case['case_id']:03d}"
        result, rows, skills = run_planner(case, baseline_summary, first, payload, settings, directory)
        planner_paths = trajectories(rows, result['initial_state'])
        rejected = rejected_targets(skills)
        for variant, paths in (('baseline', baseline_paths), ('planner', planner_paths)):
            for name, path in paths.items(): arrays[f"case_{case['case_id']:03d}_{variant}_{name}"] = path
        summary = dict(case=case, baseline=dict(control_ticks=len(baseline_rows), tcp_path_length_m=path_length(baseline_paths['tcp']),
                       calibration=baseline_summary['report'], wall_seconds=None,
                       skill_statuses=[dict(primitive=s['primitive_type'], status=s['status']) for s in baseline_skills]),
                       planner=result, planner_tcp_path_length_m=path_length(planner_paths['tcp']), rejected_targets=rejected,
                       planner_stationary=bool(path_length(planner_paths['tcp']) < 1e-6))
        summaries.append(summary)
        entries.append(dict(case=case, baseline=baseline_paths, planner=planner_paths, planner_result=result, rejected_targets=rejected))
    np.savez_compressed(args.output/'paths.npz', **arrays)
    draw(args.output, entries)
    write_json(args.output/'report.json', dict(split='diagnostic_only', automatic_training=False, baseline=str(args.baseline.resolve()),
               checkpoint=str(checkpoint.resolve()), checkpoint_sha256=manifest['checkpoint_sha256'],
               baseline_source_sha256=manifest['source_sha256'], diagnostic_source_sha256=sha256(__file__),
               planner_source_sha256=sha256(ROOT/'domain/robocasa/motion_planning.py'),
               source_action_contract=payload['action_contract'], source_settings=settings,
               planner_execution_settings=dict(settings, motion_execution='planner', motion_plan_max_steps=400),
               limits='Same requested actions for shared phases; actual planned targets depend on actual state. TCP paths do not certify whole-arm collision clearance. Baseline wall time was not recorded.',
               cases=summaries))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
