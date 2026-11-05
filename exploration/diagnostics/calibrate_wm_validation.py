"""Bounded manual calibration and CPU process benchmark; never formal validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from domain.robocasa.primitives import PrimitiveAction
from exploration.cee_us.cee_us_cabinet import make_env
from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder, OBJECT_NAMES
from exploration.diagnostics.diagnose_cabinet import source_evidence, write_json
from exploration.raps.raps_gnn_actor import DEFAULTS, SCHEMA, SkillCollector, action_contract, action_contract_matches, skill_schema
from exploration.raps.raps_replay_compare import events

CASES = ('original_cup', 'original_bowl', 'shifted_cup', 'shifted_bowl')


def instrument_ik(kinematics, calls):
    """Observe the exact None fallback without changing solver inputs or outputs."""
    original = kinematics.compute_ik

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result is None)
        return result

    kinematics.compute_ik = observed
    return original


def contact_diagnostic(context, provider, stream):
    from mani_skill.utils.structs import Articulation

    arm, raw = context.arm_controller, context.env
    calls = []
    original = instrument_ik(arm.kinematics, calls)
    targets = [('ground', raw.ground)]
    cabinet = raw.cabinet
    if isinstance(cabinet, Articulation):
        targets.extend((f'cabinet/{link.name}', link) for link in cabinet.get_links())
    else:
        targets.append(('cabinet', cabinet))
    targets.extend((name, obj.actor) for name, obj in context.objects.items())
    links = raw.agent.robot.get_links()
    summary = dict(ik_calls=0, ik_none=0, phases={}, max_force_by_link_target_phase={},
                   max_qpos_tracking_error=0., missing_fields=[], self_collision='not queried')

    def array(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=float)

    def on_step(row):
        phase = provider.phase
        fields, missing = {}, []
        for name in ('_target_qpos', '_start_qpos', 'qpos', 'joint_indices'):
            value = getattr(arm, name, None)
            if value is None:
                missing.append(name)
            else:
                fields[name] = array(value).tolist()
        target_pose = getattr(arm, '_target_pose', None)
        if target_pose is None:
            missing.append('_target_pose')
        else:
            fields['target_pose_position'] = array(target_pose.p).tolist()
            fields['target_pose_quaternion'] = array(target_pose.q).tolist()
        fields['actual_tcp_root'] = row['after']['ee_position']
        if 'joint_indices' in fields:
            indices = np.asarray(fields['joint_indices'], dtype=int)
            fields['robot_qpos_arm_indices'] = array(raw.agent.robot.get_qpos())[..., indices].tolist()
        if '_target_qpos' in fields and 'qpos' in fields:
            error = float(np.max(np.abs(array(fields['_target_qpos'])-array(fields['qpos']))))
            fields['qpos_tracking_error_max_abs'] = error
            summary['max_qpos_tracking_error'] = max(summary['max_qpos_tracking_error'], error)
        stats = summary['phases'].setdefault(phase, dict(steps=0, ik_calls=0, ik_none=0, max_qpos_tracking_error=0.))
        stats['steps'] += 1
        stats['ik_calls'] += len(calls)
        stats['ik_none'] += sum(calls)
        stats['max_qpos_tracking_error'] = max(stats['max_qpos_tracking_error'], fields.get('qpos_tracking_error_max_abs', 0.))
        summary['ik_calls'] += len(calls)
        summary['ik_none'] += sum(calls)
        forces = {}
        for link in links:
            for name, target in targets:
                key = f'{link.name}|{name}'
                try:
                    force = float(np.linalg.norm(array(raw.scene.get_pairwise_contact_forces(link, target))))
                except Exception as exc:
                    raise RuntimeError(f'Contact diagnostic failed: {key}') from exc
                forces[key] = force
                full_key = f'{phase}|{key}'
                summary['max_force_by_link_target_phase'][full_key] = max(summary['max_force_by_link_target_phase'].get(full_key, 0.), force)
        summary['missing_fields'] = sorted(set(summary['missing_fields']) | set(missing))
        stream.write(json.dumps(dict(step=row['step'], phase=phase, ik_returned_none=list(calls),
                                     controller=fields, missing_fields=missing, pairwise_force_norms=forces), allow_nan=False)+'\n')
        calls.clear()

    def restore():
        arm.kinematics.compute_ik = original

    return on_step, summary, restore


class CalibrationProvider:
    """Manual goals with gates based only on actual executed control ticks."""
    def __init__(self, settings, case, args):
        self.settings = dict(settings)
        self.schema = skill_schema(self.settings)
        self.case, self.args = case, args
        self.target = OBJECT_NAMES[int(case.endswith('bowl'))]
        self.phase = None
        self.commands, self.trace, self.positions = [], [], {}
        self.streak = self.max_streak = 0
        self.ever_grasped = self.lift_valid = self.released = False
        self.max_rise_held = self.max_event_rise = 0.
        self.grasp_z = None
        self.lift_lost_grasp = False
        self.failure = None
        self.carry_displacement = 0.
        self.initial = self.last = None

    def encode(self, name, **parameters):
        for parameter in self.schema.spec_by_name[name].parameters:
            value = parameters[parameter.name]
            if not np.isfinite(value) or abs(value) > parameter.scale:
                raise ValueError(f'{name}.{parameter.name}={value} exceeds scale {parameter.scale}')
        if name == 'approach' and parameters['x_forward'] <= 0:
            raise ValueError('approach requires strictly positive x_forward from reset TCP')
        return self.schema.encode(PrimitiveAction(name, parameters))

    def issue(self, phase, name, **parameters):
        action = self.encode(name, **parameters)
        self.phase = phase
        self.commands.append(dict(phase=phase, primitive=name, parameters=parameters,
                                  status='requested', start=None, end=None))
        return action

    def fail(self, reason):
        self.failure = self.failure or reason
        return self.issue('park', 'open_gripper')

    def action(self, state):
        tcp = np.asarray(state[18:21], dtype=float)
        index = OBJECT_NAMES.index(self.target)
        obj = np.asarray(state[33+12*index:36+12*index], dtype=float)
        if self.initial is None:
            self.initial = dict(tcp=tcp.tolist(), object_root=obj.tolist())
            self.positions['initial'] = obj.tolist()
            return self.issue('open', 'open_gripper')
        previous = self.phase
        if self.commands[-1]['status'] == 'executing':
            self.commands[-1]['status'] = 'finished'
        if self.failure or previous in ('retreat', 'park'):
            return self.issue('park', 'open_gripper')
        if previous == 'open':
            bowl = self.case.endswith('bowl')
            offset = np.array([self.args.x_offset,
                               self.args.bowl_y_offset if bowl else self.args.cup_y_offset,
                               self.args.bowl_z_offset if bowl else self.args.cup_z_offset])
            goal = obj + offset
            self.initial.update(grasp_object_root=obj.tolist(), grasp_tcp=tcp.tolist(),
                                grasp_goal_root=goal.tolist(), offsets=offset.tolist())
            return self.issue('grasp', 'approach', **dict(zip(
                ('x_forward', 'y_offset', 'z_offset'), (goal-tcp).tolist())))
        if previous in ('grasp', 'closure'):
            if self.streak < 4:
                if previous == 'grasp':
                    return self.issue('closure', 'close_gripper')
                return self.fail('grasp_not_confirmed')
            self.positions['grasp'] = obj.tolist()
            return self.issue('lift', 'lift', z_dist=self.args.lift_distance)
        if previous == 'lift':
            self.positions['lift'] = obj.tolist()
            self.lift_valid = bool(not self.lift_lost_grasp and self.streak >= 4 and self.max_rise_held >= .01 and self.max_event_rise >= .01)
            if not self.lift_valid:
                return self.fail('lost_grasp_during_lift' if self.lift_lost_grasp or self.streak < 4 else 'lift_not_achieved')
            return self.issue('carry', 'move_delta_ee_pose', dx=-.03, dy=.03, dz=0.)
        if previous == 'carry':
            self.positions['carry'] = obj.tolist()
            if self.streak < 4:
                return self.fail('lost_grasp_during_carry')
            if self.carry_displacement < .015:
                return self.fail('carry_not_achieved')
            return self.issue('drop', 'drop', z_dist=self.args.lift_distance)
        if previous == 'drop':
            self.positions['drop'] = obj.tolist()
            return self.issue('release', 'open_gripper')
        if previous == 'release':
            self.positions['release'] = obj.tolist()
            if not self.released:
                return self.fail('release_not_confirmed')
            return self.issue('retreat', 'move_delta_ee_pose', dx=-.05, dy=0., dz=0.)
        raise AssertionError(previous)

    def on_step(self, row):
        obj = row['after']['objects'][self.target]
        position = np.asarray(obj['position_root'])
        held = bool(row['action'][3] < 0 and row['contact_events'][-1]['objects'][self.target]['grasped'])
        if held:
            if self.streak == 0:
                self.grasp_z = float(position[2])
            self.streak += 1
        else:
            self.streak = 0
        if self.phase == 'lift' and not held:
            self.lift_lost_grasp = True
        self.max_streak = max(self.max_streak, self.streak)
        self.ever_grasped |= self.streak >= 4
        if self.streak >= 4:
            self.max_rise_held = max(self.max_rise_held, float(position[2]-self.initial['object_root'][2]))
            self.max_event_rise = max(self.max_event_rise, float(position[2]-self.grasp_z))
        command = self.commands[-1]
        if command['start'] is None:
            command.update(start=row['step'], before=row['before'], skill_id=row['skill_id'])
        command.update(end=row['step'], after=row['after'], status='executing')
        if self.phase == 'carry':
            start = np.asarray(command['before']['objects'][self.target]['position_root'])
            self.carry_displacement = float(np.linalg.norm(position-start))
        if self.phase == 'release':
            self.released = bool(row['action'][3] > 0 and not row['contact_events'][-1]['objects'][self.target]['grasped'])
        if row['terminated'] or row['truncated']:
            self.failure = self.failure or 'episode_ended'
        self.trace.append(dict(step=row['step'], skill_id=row['skill_id'], phase=self.phase,
                               closed_command=bool(row['action'][3] < 0), target_grasped=held,
                               stable_grasp_ticks=self.streak, object_root=position.tolist()))
        self.last = row

    def report(self, skills):
        by_id = {s['skill_id']: s for s in skills}
        for command in self.commands:
            skill = by_id.get(command.get('skill_id'))
            if skill:
                command.update(status=skill['status'], completed=skill['completed'],
                               requested_action=skill['requested_action'], executed_action=skill['action'])
        unfinished = [c['phase'] for c in self.commands if not c.get('completed', False) and c['phase'] != 'park']
        success = bool(not self.failure and self.ever_grasped and self.lift_valid and
                       self.carry_displacement >= .015 and self.released and self.phase == 'park')
        return dict(calibration_success=success, failure=self.failure or (None if success else 'path_budget_exhausted'),
                    path_status='complete' if success else 'failed', unfinished_phases=unfinished,
                    target=self.target, initial_goals=self.initial, object_positions=self.positions,
                    ever_confirmed_grasp=self.ever_grasped, max_stable_grasp_ticks=self.max_streak,
                    max_root_rise_while_confirmed_held_m=self.max_rise_held,
                    max_continuous_grasp_rise_m=self.max_event_rise, held_lift_verified=self.lift_valid,
                    lost_grasp_during_lift=self.lift_lost_grasp,
                    carry_displacement_m=self.carry_displacement, released_after_drop=self.released,
                    final_failed_object_ids=self.last['after']['failed_object_ids'] if self.last else [],
                    commands=self.commands)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('serial', 'parallel'), required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--seed', type=int, default=2000000)
    p.add_argument('--case', choices=CASES)
    p.add_argument('--diagnose-contacts', action='store_true', help='Record optional IK/controller and pairwise contact evidence')
    for flag, default in (('cup-z-offset', .006), ('bowl-z-offset', .028),
                          ('x-offset', -.005), ('bowl-y-offset', -.011),
                          ('cup-y-offset', 0.), ('lift-distance', .08)):
        p.add_argument('--'+flag, type=float, default=default)
    return p


def case_jobs(args):
    return [dict(vars(args), case=case, seed=args.seed+i,
                 output=str(args.output/case), checkpoint=str(args.checkpoint.resolve()))
            for i, case in enumerate(CASES) if args.case is None or case == args.case]


def checkpoint_contract(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    config = OmegaConf.create(payload['config'])
    config.raps_actor = OmegaConf.merge(DEFAULTS, config.raps_actor)
    settings = OmegaConf.to_container(config.raps_actor, resolve=True)
    if (payload.get('schema') != SCHEMA or settings['skill_set'] != 'compact6' or
            not settings['latched_gripper'] or settings['grasp_stop_on_contact'] or
            not action_contract_matches(payload['action_contract'], settings)):
        raise ValueError('Requires source compact6/12D latched checkpoint; grasp_stop_on_contact must be false')
    if (config.episode_steps != 800 or not config.centered_initial_tcp or
            config.execution != 'reference' or payload['state_contract']['dimension'] != 67):
        raise ValueError('Requires state67, episode800, centered reset, reference execution')
    return payload, config, settings


def run_case(job):
    began = time.perf_counter()
    args = argparse.Namespace(**job)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    payload, config, settings = checkpoint_contract(args.checkpoint)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    env = diagnostic_stream = restore_diagnostic = None
    try:
        shift = (.03, 0.) if args.case.startswith('shifted') else (0., 0.)
        env, env_config = make_env(episode_steps=800, centered_initial_tcp=True,
                                  project_goals_to_region=False, object_shift_xy=shift)
        env.reset(seed=args.seed)
        encoder = CabinetSkillStateEncoder(env, latched_gripper=settings['latched_gripper'])
        if encoder.contract() != payload['state_contract']:
            raise ValueError('Encoder state contract differs from source checkpoint')
        provider = CalibrationProvider(settings, args.case, args)
        collector = SkillCollector(env, encoder, provider, directory)
        on_step = provider.on_step
        diagnostic_summary = None
        if args.diagnose_contacts:
            diagnostic_stream = (directory/'contact_diagnostics.jsonl').open('w')
            diagnose, diagnostic_summary, restore_diagnostic = contact_diagnostic(env.env._context(), provider, diagnostic_stream)

            def on_step(row):
                provider.on_step(row)
                diagnose(row)

        collect_start = time.perf_counter()
        ticks, skills = collector.collect(args.steps, 'curiosity', seed=args.seed,
                                           stage='calibration_only', on_step=on_step)
        elapsed = time.perf_counter()-collect_start
        summary = dict(split='calibration_only', case=args.case, seed=args.seed, real_steps=len(ticks),
                       requested_steps=args.steps, init_seconds=collect_start-began, collect_seconds=elapsed,
                       settings=settings, state_contract=payload['state_contract'],
                       action_contract=payload['action_contract'], environment=OmegaConf.to_container(env_config, resolve=True),
                       events=events(ticks, skills), **provider.report(skills))
        if diagnostic_summary is not None:
            summary['contact_diagnostics'] = diagnostic_summary
        write_json(directory/'phase_trace.json', provider.trace)
        write_json(directory/'summary.json', summary)
        print(json.dumps({k: summary[k] for k in ('case', 'real_steps', 'calibration_success', 'failure',
                                                 'max_stable_grasp_ticks', 'max_root_rise_while_confirmed_held_m',
                                                 'carry_displacement_m')}), flush=True)
        return summary
    finally:
        if restore_diagnostic is not None:
            restore_diagnostic()
        if diagnostic_stream is not None:
            diagnostic_stream.close()
        if env is not None:
            env.close()


def compare_raw(left, right):
    """Compare saved float32 transitions, independently of timing or success."""
    result = {}
    for filename, fields in (('transitions.jsonl', ('state', 'action', 'next_state')),
                              ('skills.jsonl', ('state', 'action', 'next_state'))):
        a, b = ([json.loads(line) for line in (Path(path)/filename).read_text().splitlines()]
                for path in (left, right))
        report = dict(left_rows=len(a), right_rows=len(b), fields={})
        for field in fields:
            x, y = (np.asarray([row[field] for row in rows], dtype=np.float32) for rows in (a, b))
            report['fields'][field] = dict(exact=bool(np.array_equal(x, y)),
                max_abs_diff=float(np.max(np.abs(x-y))) if x.shape == y.shape and x.size else (0. if x.shape == y.shape else None))
        result[filename] = report
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.steps <= 400 or args.workers < 1 or args.seed < 0:
        raise ValueError('Require steps in [1,400], positive workers and nonnegative seed')
    if not np.isfinite([args.x_offset, args.cup_y_offset, args.bowl_y_offset,
                        args.cup_z_offset, args.bowl_z_offset, args.lift_distance]).all() or not 0 < args.lift_distance <= .16:
        raise ValueError('Require finite goals and lift distance in (0,.16]')
    if args.output.exists():
        raise FileExistsError(f'Refusing existing output: {args.output}')
    payload, config, settings = checkpoint_contract(args.checkpoint)
    jobs = case_jobs(args)
    args.output.mkdir(parents=True)
    evidence = source_evidence(args.output)
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    manifest = dict(split='calibration_only', automatic_training=False, config=OmegaConf.to_container(config, resolve=True),
                    settings=settings, state_contract=payload['state_contract'], action_contract=payload['action_contract'],
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), source_evidence=evidence,
                    args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    cases=jobs, physics_hz=100, control_hz=20, episode_steps=800,
                    threads={v:os.environ[v] for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS')})
    write_json(args.output/'manifest.json', manifest)
    began = time.perf_counter()
    if args.mode == 'serial':
        summaries = [run_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            summaries = list(pool.map(run_case, jobs))
    elapsed = time.perf_counter()-began
    steps = sum(s['real_steps'] for s in summaries)
    result = dict(split='calibration_only', mode=args.mode, wall_seconds=elapsed, real_steps=steps,
                  control_steps_per_second=steps/elapsed, cases=summaries)
    write_json(args.output/'result.json', result)
    print(json.dumps(dict(mode=args.mode, real_steps=steps, wall_seconds=elapsed,
                          control_steps_per_second=steps/elapsed)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
