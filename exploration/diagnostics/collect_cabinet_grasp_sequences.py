"""Collect paired axis-order and grasp insertion validation sequences."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
from omegaconf import OmegaConf
import torch

from exploration.cee_us.cee_us_cabinet import make_env
from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder, OBJECT_NAMES
from exploration.diagnostics.calibrate_wm_validation import CalibrationProvider, checkpoint_contract
from exploration.diagnostics.collect_cabinet_grasps import (
    BUDGET, ROOT, SOURCES as BASE_SOURCES, Finished, cases as base_cases, export,
    identity as base_identity, read_rows, results as base_results, sha256, stop_callback,
)
from exploration.diagnostics.diagnose_cabinet import write_json
from exploration.raps.raps_gnn_actor import SkillCollector, action_contract, skill_schema

TEMPLATES = ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', 'post_grasp', 'mid_lift', 'pre_carry')
INSERTION_M = .03
SOURCES = BASE_SOURCES + ('exploration/diagnostics/collect_cabinet_grasp_sequences.py',)
GPU_SOURCES = ('domain/robocasa/batched_cabinet.py', 'domain/robocasa/batched_skills.py',
               'exploration/diagnostics/collect_gpu_split7.py',
               'exploration/diagnostics/trajectory_store.py')


def cases(seed=1000000):
    result = []
    for base in base_cases(seed):
        for index, template in enumerate(TEMPLATES):
            result.append(dict(base, base_case_id=base['case_id'], case_id=base['case_id']*9+index,
                               template=template))
    return result


def selected_cases(stage, seed):
    all_cases = cases(seed)
    if stage == 'smoke':
        return [c for c in all_cases if c['base_case_id'] in (27, 81) and
                c['template'] in ('yzx', 'post_grasp', 'mid_lift', 'pre_carry')]
    return all_cases


class SequenceProvider(CalibrationProvider):
    def __init__(self, settings, case, args):
        super().__init__(settings, 'original_'+case['object'], args)
        self.template = case['template']
        self.waypoints = []
        self.next_waypoint = 0
        self.current_tcp = None
        self.insertion_anchor = None
        self.insertion_lost = False
        self.pending = None
        self.full_lift_z = None

    def movement(self, phase, goal):
        goal = np.asarray(goal, dtype=float)
        delta = goal-self.current_tcp
        values = dict(zip(('dx', 'dy', 'dz'), delta.tolist()))
        specs = self.schema.spec_by_name['move_delta_ee_pose'].parameters
        if any(not np.isfinite(values[p.name]) or abs(values[p.name]) > p.scale for p in specs):
            return self.fail('waypoint_parameter_out_of_range')
        action = super().issue(phase, 'move_delta_ee_pose', **values)
        self.commands[-1].update(requested_goal_root=goal.tolist(), start_tcp_root=self.current_tcp.tolist())
        return action

    def lift(self, phase, distance):
        scale = self.schema.spec_by_name['lift'].parameters[0].scale
        if not np.isfinite(distance) or not 0 <= distance <= scale:
            return self.fail('remaining_lift_out_of_range')
        action = super().issue(phase, 'lift', z_dist=float(distance))
        goal = self.current_tcp.copy()
        goal[2] += distance
        self.commands[-1].update(requested_goal_root=goal.tolist(), start_tcp_root=self.current_tcp.tolist())
        return action

    def insert(self, pending):
        self.pending = pending
        self.insertion_anchor = self.current_tcp.copy()
        self.insertion_lost = False
        return self.movement('insert_out', self.insertion_anchor + [0., INSERTION_M, 0.])

    def issue(self, phase, name, **parameters):
        if phase == 'lift' and self.template == 'post_grasp':
            return self.insert(('lift', parameters['z_dist']))
        if phase == 'lift' and self.template == 'mid_lift':
            self.full_lift_z = float(self.current_tcp[2] + parameters['z_dist'])
            return self.lift('lift_half', parameters['z_dist']/2)
        if phase == 'carry' and self.template == 'pre_carry':
            return self.insert(('carry', parameters))
        if phase == 'lift':
            return self.lift('lift', parameters['z_dist'])
        action = super().issue(phase, name, **parameters)
        if phase == 'grasp' and name == 'approach':
            self.commands[-1].update(requested_goal_root=self.initial['grasp_goal_root'],
                                     start_tcp_root=self.current_tcp.tolist())
        return action

    def action(self, state):
        self.current_tcp = np.asarray(state[18:21], dtype=float).copy()
        previous = self.phase
        if self.failure:
            return super().action(state)
        if self.template in TEMPLATES[:6] and previous == 'open':
            index = OBJECT_NAMES.index(self.target)
            obj = np.asarray(state[33+12*index:36+12*index], dtype=float)
            bowl = self.case.endswith('bowl')
            offsets = np.array([self.args.x_offset, self.args.bowl_y_offset if bowl else self.args.cup_y_offset,
                                self.args.bowl_z_offset if bowl else self.args.cup_z_offset])
            goal = obj + offsets
            self.initial.update(grasp_object_root=obj.tolist(), grasp_tcp=self.current_tcp.tolist(),
                                grasp_goal_root=goal.tolist(), offsets=offsets.tolist())
            waypoint = self.current_tcp.copy()
            for axis in self.template:
                waypoint = waypoint.copy()
                waypoint['xyz'.index(axis)] = goal['xyz'.index(axis)]
                self.waypoints.append(waypoint)
            self.next_waypoint = 1
            return self.movement('waypoint_'+self.template[0], self.waypoints[0])
        if previous and previous.startswith('waypoint_'):
            if self.next_waypoint < 3:
                axis = self.template[self.next_waypoint]
                waypoint = self.waypoints[self.next_waypoint]
                self.next_waypoint += 1
                return self.movement('waypoint_'+axis, waypoint)
            return super().issue('grasp', 'close_gripper')
        if previous == 'lift_half':
            if self.lift_lost_grasp or self.streak < 4:
                return self.fail('lost_grasp_during_lift')
            return self.insert(('lift_remaining', None))
        if previous == 'insert_out':
            if self.insertion_lost or self.streak < 4:
                return self.fail('lost_grasp_during_insertion')
            return self.movement('insert_return', self.insertion_anchor)
        if previous == 'insert_return':
            if self.insertion_lost or self.streak < 4:
                return self.fail('lost_grasp_during_insertion')
            phase, value = self.pending
            if phase == 'lift_remaining':
                action = self.lift('lift_remaining', self.args.lift_distance/2)
                self.commands[-1]['full_lift_goal_z_root'] = self.full_lift_z
                return action
            if phase == 'lift':
                return self.lift('lift', value)
            return super().issue('carry', 'move_delta_ee_pose', **value)
        if previous == 'lift_remaining':
            self.phase = 'lift'
        return super().action(state)

    def on_step(self, row):
        phase = self.phase
        if phase in ('lift_half', 'lift_remaining'):
            self.phase = 'lift'
        super().on_step(row)
        self.phase = phase
        self.trace[-1]['phase'] = phase
        if phase in ('insert_out', 'insert_return') and not self.trace[-1]['target_grasped']:
            self.insertion_lost = True
        if phase in ('insert_out', 'insert_return') and self.insertion_lost:
            self.trace[-1]['insertion_grasp_lost'] = True

    def on_end(self, skill):
        command = self.commands[-1]
        command['skill_id'] = skill['skill_id']
        if 'requested_goal_root' in command:
            achieved = np.asarray(skill['next_state'][18:21], dtype=float)
            goal = np.asarray(command['requested_goal_root'])
            start = np.asarray(command['start_tcp_root'])
            command.update(achieved_tcp_root=achieved.tolist(), goal_error_root=(achieved-goal).tolist(),
                           goal_error_m=float(np.linalg.norm(achieved-goal)),
                           goal_within_primitive_tolerance=bool(np.linalg.norm(achieved-goal) <= (.015 if command['primitive'] == 'align_yz' else .012)),
                           actual_displacement_root=(achieved-start).tolist())
            if 'full_lift_goal_z_root' in command:
                command['full_lift_goal_z_error_m'] = float(achieved[2]-command['full_lift_goal_z_root'])


class SplitSequenceProvider(SequenceProvider):
    def split_move(self, phase, goal, axis=None):
        goal = np.asarray(goal, dtype=float)
        start = self.current_tcp
        if axis == 'x' or phase == 'split_approach':
            distance = float(goal[0]-start[0])
            scale = self.schema.spec_by_name['approach_x'].parameters[0].scale
            if not np.isfinite(distance) or distance <= 0 or distance > scale:
                return self.fail('approach_x_nonpositive_or_out_of_range')
            requested_goal = start.copy()
            requested_goal[0] = goal[0]
            action = super(SequenceProvider, self).issue(phase, 'approach_x', x_forward=distance)
        else:
            dy = float(goal[1]-start[1]) if axis in (None, 'y') else 0.
            dz = float(goal[2]-start[2]) if axis in (None, 'z') else 0.
            params = dict(y_offset=dy, z_offset=dz)
            if any(not np.isfinite(params[p.name]) or abs(params[p.name]) > p.scale
                   for p in self.schema.spec_by_name['align_yz'].parameters):
                return self.fail('align_yz_parameter_out_of_range')
            requested_goal = start.copy()
            requested_goal[1:] += [dy, dz]
            action = super(SequenceProvider, self).issue(phase, 'align_yz', **params)
        self.commands[-1].update(requested_goal_root=requested_goal.tolist(), start_tcp_root=start.tolist(),
                                 frozen_grasp_goal_root=self.initial['grasp_goal_root'])
        return action

    def movement(self, phase, goal):
        if phase.startswith('waypoint_'):
            return self.split_move(phase, goal, phase[-1])
        return super().movement(phase, goal)

    def issue(self, phase, name, **parameters):
        if phase == 'grasp' and name == 'approach':
            return self.split_move('split_align', self.initial['grasp_goal_root'])
        return super().issue(phase, name, **parameters)

    def action(self, state):
        if not self.failure and self.phase in ('split_align', 'split_approach'):
            self.current_tcp = np.asarray(state[18:21], dtype=float).copy()
            if self.phase == 'split_align':
                return self.split_move('split_approach', self.initial['grasp_goal_root'])
            return super(SequenceProvider, self).issue('grasp', 'close_gripper')
        return super().action(state)


def identity(args, payload, config, settings):
    manifest = base_identity(args, payload, config, settings)
    selected_skill_set = getattr(args, 'skill_set', 'compact6')
    effective_settings = dict(settings, skill_set=selected_skill_set)
    effective_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    effective_config.raps_actor.skill_set = selected_skill_set
    manifest.update(source_checkpoint_config=OmegaConf.to_container(config, resolve=True),
                    source_checkpoint_action_contract=payload['action_contract'],
                    config=OmegaConf.to_container(effective_config, resolve=True), settings=effective_settings,
                    action_contract=action_contract(effective_settings), skill_set=selected_skill_set,
                    checkpoint_weights_used=False)
    manifest.update(protocol='paired_axis_order_and_grasp_insertion_v1', cases=selected_cases(args.stage, args.seed),
                    templates=list(TEMPLATES), insertion=dict(frame='controller_root', out_delta_m=[0., INSERTION_M, 0.],
                    return_target='pre_insertion_tcp', position_tolerance_m=.012),
                    source_sha256={name: sha256(ROOT/name) for name in SOURCES})
    if getattr(args, 'collector', 'scalar') == 'gpu_batch':
        import importlib.metadata
        import hashlib
        import subprocess
        import mani_skill
        import sapien
        from exploration.diagnostics.trajectory_store import FORMAT
        manifest.update(collector='gpu_batch', sim_backend='gpu', num_envs=args.num_envs,
                        trajectory_format=FORMAT, export_every_cases=args.export_every_cases,
                        native_sapien_sha256=hashlib.sha256(Path(sapien.pysapien.__file__).read_bytes()).hexdigest(),
                        dependencies={name: importlib.metadata.version(name)
                                      for name in ('numpy', 'torch', 'sapien', 'mani_skill', 'gymnasium')})
        upstream = Path(mani_skill.__file__).resolve().parents[1]
        try:
            commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
            patch = subprocess.check_output(['git', '-C', str(upstream), 'diff', '--binary', 'HEAD'])
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'GPU source provenance requires the pinned ManiSkill checkout at {upstream}') from error
        manifest['mani_skill_source'] = dict(commit=commit, tracked_patch_sha256=hashlib.sha256(patch).hexdigest())
        native_library = Path(sapien.pysapien.__file__).parent.parent/'sapien.libs'/'libsapien.so'
        manifest['native_sapien_library_sha256'] = sha256(native_library)
        patch_manifest = os.environ.get('SAPIEN_SPARSE_PATCH_MANIFEST')
        if patch_manifest:
            patch_manifest = Path(patch_manifest)
            manifest['native_sparse_patch'] = dict(
                manifest_sha256=sha256(patch_manifest),
                build=json.loads(patch_manifest.read_text()))
        manifest['source_sha256'].update({name: sha256(ROOT/name) for name in GPU_SOURCES})
    return manifest


def results(output, planned, *, summaries=None):
    result = base_results(output, planned, summaries=summaries)
    result['per_template'] = {}
    for template in TEMPLATES:
        rows = [s for s in result['cases'] if s['case']['template'] == template]
        successes = sum(bool(s['success']) for s in rows)
        failures = {}
        for row in rows:
            if not row['success']:
                reason = row['report']['failure'] or row['status']
                failures[reason] = failures.get(reason, 0) + 1
        result['per_template'][template] = dict(closed=len(rows), successes=successes,
            success_rate=successes/len(rows) if rows else None, failure_reasons=failures)
    return result


def run_case(output, case, payload, settings):
    case_dir = output/'cases'/f"case_{case['case_id']:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    attempts = [int(p.name.split('_')[1]) for p in case_dir.glob('attempt_*') if p.is_dir()]
    attempt = max(attempts, default=0)+1
    directory = case_dir/f'attempt_{attempt:06d}'
    directory.mkdir(exist_ok=False)
    random.seed(case['seed']); np.random.seed(case['seed']); torch.manual_seed(case['seed'])
    env = None
    try:
        env, env_config = make_env(episode_steps=BUDGET, centered_initial_tcp=True,
                                   project_goals_to_region=False, object_shift_xy=(0., 0.))
        env.reset(seed=case['seed'])
        encoder = CabinetSkillStateEncoder(env, latched_gripper=settings['latched_gripper'])
        if encoder.contract() != payload['state_contract']:
            raise ValueError('Encoder state contract differs from checkpoint')
        x, y, z = case['offsets']
        args = argparse.Namespace(x_offset=x, cup_y_offset=y, bowl_y_offset=y,
                                  cup_z_offset=z, bowl_z_offset=z, lift_distance=case['lift_distance'])
        provider_cls = SplitSequenceProvider if settings['skill_set'] == 'split7' else SequenceProvider
        provider = provider_cls(settings, case, args)
        collector = SkillCollector(env, encoder, provider, directory)
        write_json(directory/'attempt.json', dict(case=case, attempt=attempt,
                   environment=OmegaConf.to_container(env_config, resolve=True)))
        def end(skill):
            provider.on_end(skill)
            stop_callback(provider, collector)(skill)
        try:
            collector.collect(BUDGET, 'curiosity', seed=case['seed'], stage='validation_only',
                              on_step=provider.on_step, on_skill_end=end)
        except Finished:
            pass
        skills = read_rows(directory/'skills.jsonl')
        report = provider.report(skills)
        success = report['calibration_success']
        summary = dict(case=case, attempt=attempt, status='success' if success else
                       ('failure' if provider.failure else 'timeout'), success=success,
                       control_ticks=collector.total_steps, report=report)
        write_json(directory/'phase_trace.json', provider.trace)
        write_json(directory/'summary.json', summary)
        closure = case_dir/'summary.tmp'
        write_json(closure, summary)
        os.replace(closure, case_dir/'summary.json')
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('plan', 'smoke', 'collect'), default='plan')
    parser.add_argument('--seed', type=int, default=1000000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skill-set', choices=('compact6', 'split7'), default='compact6')
    parser.add_argument('--collector', choices=('scalar', 'gpu_batch'), default='scalar')
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--export-every-cases', type=int, default=64)
    args = parser.parse_args(argv)
    if args.export_every_cases < 1:
        parser.error('--export-every-cases must be positive')
    if args.collector == 'gpu_batch':
        if args.skill_set != 'split7':
            parser.error('gpu_batch requires --skill-set split7')
        if args.num_envs is None or args.num_envs < 1:
            parser.error('gpu_batch requires an explicit positive --num-envs')
    elif args.num_envs not in (None, 1):
        parser.error('scalar supports only one environment')
    if not 0 <= args.seed <= 2**32 - 1 - 107:
        raise ValueError('Seed must leave all 108 base case seeds in the NumPy uint32 range')
    if args.stage == 'plan':
        if args.resume: raise ValueError('Plan cannot be resumed')
        plan = dict(split='validation_only', cases=cases(args.seed), cases_count=972,
                    templates=list(TEMPLATES), insertion_m=INSERTION_M, skill_set=args.skill_set,
                    action_dim=13 if args.skill_set == 'split7' else 12,
                    primitive_names=list(skill_schema({'skill_set': args.skill_set}).primitive_names), max_control_ticks_per_case=BUDGET,
                    max_control_ticks=972*BUDGET, physics_hz=100, control_hz=20)
        if args.collector == 'gpu_batch':
            plan.update(collector=args.collector, num_envs=args.num_envs, sim_backend='gpu')
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output/'plan.json', plan)
        print(json.dumps(plan))
        return 0
    payload, config, settings = checkpoint_contract(args.checkpoint)
    manifest = identity(args, payload, config, settings)
    settings = dict(settings, skill_set=args.skill_set)
    payload = dict(payload, action_contract=action_contract(settings))
    if args.resume:
        if json.loads((args.output/'manifest.json').read_text()) != manifest:
            raise ValueError('Resume manifest differs from checkpoint, configuration, source or cases')
        for name, digest in manifest['source_sha256'].items():
            if sha256(args.output/'sources'/name) != digest:
                raise ValueError(f'Resume source snapshot mismatch: {name}')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        for name in manifest['source_sha256']:
            destination = args.output/'sources'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, destination)
        write_json(args.output/'manifest.json', manifest)
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    if args.collector == 'gpu_batch':
        from exploration.diagnostics.collect_gpu_split7 import run_gpu
        run_gpu(args.output, manifest, payload, settings, args.num_envs, args.export_every_cases)
        return 0
    for case in manifest['cases']:
        if not (args.output/'cases'/f"case_{case['case_id']:03d}"/'summary.json').exists():
            run_case(args.output, case, payload, settings)
        omitted = []
        export(args.output, omitted, action_dim=13 if args.skill_set == 'split7' else 12)
        result = results(args.output, manifest['cases'])
        result['incomplete_jsonl_tail_omitted'] = omitted
        write_json(args.output/'result.json', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
