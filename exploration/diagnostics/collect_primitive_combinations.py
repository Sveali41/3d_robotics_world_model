"""Object-relative, budgeted split7 collection through the real skill collector."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch

from domain.robocasa.primitives import PrimitiveAction
from domain.robocasa.motion_planning import _native_modules
from exploration.cee_us.cee_us_cabinet import make_env, replay_arrays
from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder, OBJECT_NAMES
from exploration.diagnostics.collect_wm_validation import (
    SCENES, annotate, endpoint_hash, replay_exclusions, select_balanced, spatial_audit, windows_index,
)
from exploration.diagnostics.diagnose_cabinet import source_evidence, write_json
from exploration.diagnostics.diagnose_grasp_targets import object_audit
from exploration.diagnostics.validate_cabinet_planner import ContactAudit
from exploration.raps.raps_gnn_actor import DEFAULTS, SkillCollector, action_contract, skill_schema
from exploration.raps.raps_replay_compare import events


RECIPES = ('align_approach', 'align_approach_refine', 'align_high_approach_drop',
           'approach_align_refine', 'approach_align_close')
PROTOCOL = 'object_relative_split7_v1'


def commands_for(recipe, goal):
    """Each recipe's nominal closure position is the same sampled root-space goal."""
    goal = np.asarray(goal, dtype=float)
    if recipe not in RECIPES:
        raise ValueError(f'Unknown combination: {recipe}')
    closure = goal.copy()
    if recipe == 'approach_align_close':
        goal = goal + [.02, 0., 0.]
    high = .02 if recipe == 'align_high_approach_drop' else 0.
    short = .02 if recipe == 'align_approach_refine' else 0.
    commands = [dict(primitive='open_gripper', parameters={}),
        dict(primitive='align_yz', target_root=(goal+[0, 0, high]).tolist()),
        dict(primitive='approach_x', target_root=(goal+[-short, 0, high]).tolist())]
    if recipe in {'approach_align_refine', 'approach_align_close'}:
        commands = [dict(primitive='open_gripper', parameters={}),
            dict(primitive='approach_x', target_root=(goal+[-.06, 0, 0]).tolist()),
            dict(primitive='align_yz', target_root=goal.tolist()),
            dict(primitive='approach_x', target_root=(goal+[-.02, 0, 0]).tolist())]
        if recipe == 'approach_align_refine':
            commands.append(dict(primitive='move_delta_ee_pose', target_root=goal.tolist()))
    if short:
        commands.append(dict(primitive='move_delta_ee_pose', target_root=goal.tolist()))
    if high:
        commands.append(dict(primitive='drop', target_root=closure.tolist()))
    # Four ordinary closing primitives give 36 ticks, rather than hiding extra
    # controller steps outside the action/transition record.
    commands += [dict(primitive='close_gripper', parameters={}) for _ in range(4)]
    commands.append(dict(primitive='lift', parameters=dict(z_dist=.04)))
    return commands


def sampling_bounds(bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError('Native collision bounds must be a finite (2,3) array')
    result = bounds + [[-.02, -.02, .01], [.02, .02, -.01]]
    if np.any(result[1] <= result[0]):
        raise ValueError('Object bounds leave no sampling volume after the 1 cm Z inset')
    return result


def generate_sampling_plan(geometry, recipes=RECIPES, grid=(2, 2, 2), seed=4100000):
    grid = np.asarray(grid, dtype=int)
    if grid.shape != (3,) or np.any(grid < 1) or seed < 0:
        raise ValueError('Use three positive grid dimensions and a nonnegative seed')
    plans = []
    cells = list(np.ndindex(*grid))
    for object_index, target in enumerate(OBJECT_NAMES):
        rng = np.random.default_rng(np.random.SeedSequence([seed, object_index]))
        for cell_index, cell in enumerate(cells):
            fraction = rng.random(3)
            normalized = (np.asarray(cell)+fraction)/grid
            for recipe in recipes:
                pair_id = (object_index*len(cells)+cell_index)*len(RECIPES)+RECIPES.index(recipe)
                for scene_index, scene in enumerate(SCENES):
                    if scene not in geometry or target not in geometry[scene]:
                        continue
                    item = geometry[scene][target]
                    bounds = sampling_bounds(item['collision_bounds_root_m'])
                    position = bounds[0]+normalized*(bounds[1]-bounds[0])
                    plans.append(dict(trial_id=scene_index*len(OBJECT_NAMES)*len(cells)*len(RECIPES)+pair_id,
                        pair_id=pair_id, scene=scene, object=target, recipe=recipe, intent='lift',
                        cell=list(cell), cell_fraction=fraction.tolist(), normalized_position=normalized.tolist(),
                        position_root=position.tolist(), sampling_bounds_root_m=bounds.tolist(),
                        object_position_root=item['pose_in_root'][:3], seed=int(seed+pair_id+1)))
    # Interleave cases so a global budget cannot be consumed by only one layout/object.
    return sorted(plans, key=lambda p: (p['pair_id'] % (len(cells)*len(RECIPES)),
                                       OBJECT_NAMES.index(p['object']), list(SCENES).index(p['scene'])))


def select_pilot(plans):
    selected = []
    for scene in SCENES:
        for target in OBJECT_NAMES:
            group = [p for p in plans if p['scene'] == scene and p['object'] == target]
            if not group:
                continue
            ordered = sorted(group, key=lambda p: (p['cell'], p['trial_id']))
            first = next((p for p in ordered if p['recipe'] == 'align_approach'), ordered[0])
            last = next((p for p in reversed(ordered) if p['recipe'] == 'approach_align_refine'), ordered[-1])
            selected.append(first)
            if last['trial_id'] != first['trial_id']:
                selected.append(last)
    return selected


class InvalidCombinationParameters(ValueError):
    pass


class CombinationProvider:
    def __init__(self, settings, commands, target):
        self.settings, self.commands, self.target = settings, commands, target
        self.schema = skill_schema(settings)
        self.index, self.current = 0, None
        self.audit = None
        self.outcome = None
        self.grasp_streak = self.max_grasp_streak = 0
        self.lift_start_z = None
        self.lift_lost_grasp = False
        self.max_held_lift_m = 0.
        self.issued = []
        self.rejected_command = None

    def action(self, state):
        if self.outcome is not None or self.index >= len(self.commands):
            raise RuntimeError('Combination attempt has ended')
        command = self.current = self.commands[self.index]
        self.index += 1
        name = command['primitive']
        tcp = np.asarray(state[18:21], dtype=float)
        parameters = dict(command.get('parameters', {}))
        rejection = None
        if 'target_root' in command:
            delta = np.asarray(command['target_root'])-tcp
            if name == 'align_yz':
                parameters = dict(y_offset=float(delta[1]), z_offset=float(delta[2]))
            elif name == 'approach_x':
                if delta[0] < 0:
                    rejection = 'Manual approach target is behind the TCP'
                parameters = dict(x_forward=float(delta[0]))
            elif name == 'move_delta_ee_pose':
                parameters = dict(zip(('dx', 'dy', 'dz'), delta.tolist()))
            elif name in {'drop', 'lift'}:
                distance = float(delta[2] if name == 'lift' else -delta[2])
                parameters = dict(z_dist=distance)
                if distance < 0:
                    rejection = f'{name} target has the wrong vertical direction'
        for spec in self.schema.spec_by_name[name].parameters:
            value = parameters[spec.name]
            if not np.isfinite(value) or abs(value) > spec.scale:
                rejection = f'{name}.{spec.name}={value} exceeds {spec.scale}; no clipping permitted'
        if rejection is not None:
            self.rejected_command = dict(command_index=self.index, primitive=name,
                parameters=parameters, tcp_before=tcp.tolist(), reason=rejection)
            raise InvalidCombinationParameters(rejection)
        if name == 'lift':
            i = OBJECT_NAMES.index(self.target)
            self.lift_start_z = float(state[35+12*i])
        if self.audit is not None:
            attachment = self.audit.raw.motion_planner.attachment_state()
            self.audit.payload = attachment['name'] if attachment else None
            # Closing can confirm attachment while the object still rests on
            # its shelf. Preserve the same shallow support intent as initial lift.
            self.audit.set_stage(f'{self.index:02d}_{name}', allowed_items=OBJECT_NAMES,
                                 support=name in {'close_gripper', 'lift'})
        self.issued.append(dict(command_index=self.index, primitive=name, parameters=parameters, tcp_before=tcp.tolist()))
        return self.schema.encode(PrimitiveAction(name, parameters))

    def on_step(self, row):
        held = (row['parameters_physical']['gripper'] < 0
                and row['after']['contact_snapshot']['objects'][self.target]['grasped'])
        self.grasp_streak = self.grasp_streak+1 if held else 0
        self.max_grasp_streak = max(self.max_grasp_streak, self.grasp_streak)
        if self.current['primitive'] == 'lift':
            self.lift_lost_grasp |= not held
            z = row['after']['objects'][self.target]['position_root'][2]
            if held:
                self.max_held_lift_m = max(self.max_held_lift_m, z-self.lift_start_z)

    def on_skill_end(self, skill):
        failure = None
        if self.audit is not None and self.audit.records:
            failure = 'forbidden_physical_contact'
        elif not skill['completed'] or skill['status'] != 'success':
            failure = 'primitive_failed'
        elif (skill['primitive_type'] == 'close_gripper'
              and self.commands[self.index]['primitive'] == 'lift' and self.grasp_streak < 4):
            failure = 'unconfirmed_grasp'
        elif skill['primitive_type'] == 'lift':
            if self.lift_lost_grasp or self.grasp_streak < 4 or self.max_held_lift_m < .01:
                failure = 'held_lift_not_verified'
            else:
                self.outcome = dict(status='grasp_and_lift_verified')
        if failure is not None:
            self.outcome = dict(status='failed', reason=failure, primitive=skill['primitive_type'],
                                primitive_status=skill['status'], skill_id=skill['skill_id'])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'pilot', 'collect'), default='plan')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/raps_gnn_actor_cabinet_split7_planner.yaml'))
    parser.add_argument('--trial-controls', type=int, default=400)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--quota', type=int, default=20)
    parser.add_argument('--grid', type=int, nargs=3, default=[2, 2, 2])
    parser.add_argument('--exclude-replay', type=Path, action='append', default=[])
    parser.add_argument('--calibration-data', type=Path, action='append', default=[])
    parser.add_argument('--recipes', nargs='+', choices=RECIPES, default=list(RECIPES))
    parser.add_argument('--scene', choices=tuple(SCENES))
    parser.add_argument('--object', choices=OBJECT_NAMES)
    args = parser.parse_args(argv)
    if not 1 <= args.trial_controls <= 400 or args.quota < 1 or min(args.grid) < 1:
        raise ValueError('Use 1-400 controls per trial, positive grid dimensions and a positive quota')
    if len(set(args.recipes)) != len(args.recipes):
        raise ValueError('Recipe names must not be repeated')
    args.seed = args.seed if args.seed is not None else (5100000 if args.stage == 'collect' else 4100000)
    if args.seed < 0:
        raise ValueError('Sampling seed must be nonnegative')
    if args.stage == 'collect' and args.max_steps is None:
        raise ValueError('collect requires an explicit --max-steps global control budget')
    if args.max_steps is None:
        args.max_steps = 3200 if args.stage == 'pilot' else 0
    if args.max_steps < 0 or (args.stage != 'plan' and args.max_steps < 1):
        raise ValueError('Execution requires a positive global control budget')
    if args.stage == 'pilot' and args.max_steps > 3200:
        raise ValueError('pilot is capped at 3200 total controls')
    # Preserve the existing annotation thresholds and exploration-region audit.
    args.radius, args.displacement_threshold = .10, .015
    return args


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def completed_endpoints(skills):
    valid = []
    for i, row in enumerate(skills):
        if not row['completed'] or row['duration_low_steps'] <= 0:
            continue
        if (np.asarray(row['state']).shape != (67,) or np.asarray(row['next_state']).shape != (67,)
                or np.asarray(row['action']).shape != (13,)):
            raise ValueError('Expected the existing 67D state / 13D split7 action contract')
        if not np.isfinite(np.r_[row['state'], row['action'], row['next_state']]).all():
            raise ValueError('Nonfinite completed skill endpoint')
        valid.append(dict(row, raw_reference=i,
                          endpoint_hash=endpoint_hash(row['state'], row['action'], row['next_state'])))
    return valid


def calibration_exclusions(paths, contract, seed):
    hashes, evidence = set(), []
    for directory in dict.fromkeys(p.resolve() for p in paths):
        manifest_path = directory/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('action_contract') != contract:
            raise ValueError(f'Calibration action contract differs: {directory}')
        if manifest.get('sampling_seed') == seed:
            raise ValueError('Collection and calibration must use independent sampling seeds')
        files = sorted(directory.rglob('skills.jsonl'))
        if not files:
            raise ValueError(f'No raw calibration skills found: {directory}')
        digests = []
        for path in files:
            hashes.update(r['endpoint_hash'] for r in completed_endpoints(read_jsonl(path)))
            digests.append(dict(path=str(path.relative_to(directory)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        evidence.append(dict(path=str(directory), files=digests,
                             manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()))
    return hashes, evidence


def build_indices(trials, quota, seed, training_hashes=(), calibration_hashes=()):
    """Exclude overlapping whole attempts; never splice balanced rows into rollouts."""
    training_hashes, calibration_hashes = set(training_hashes), set(calibration_hashes)
    rows, windows, excluded = [], {str(h): [] for h in (1, 3, 5)}, []
    seen_windows = {h: set() for h in windows}
    for trial in trials:
        valid = completed_endpoints(trial['skills'])
        hashes = {r['endpoint_hash'] for r in valid}
        training_overlap = bool(hashes & training_hashes)
        calibration_overlap = bool(hashes & calibration_hashes)
        if training_overlap or calibration_overlap:
            excluded.append(dict(trial_id=trial['plan']['trial_id'], raw_path=trial['raw_path'],
                                 training_overlap=training_overlap, calibration_overlap=calibration_overlap))
        for row in trial['annotations']:
            rows.append(dict(row, scene=trial['plan']['scene'], training_overlap=training_overlap,
                             calibration_overlap=calibration_overlap))
        if training_overlap or calibration_overlap:
            continue
        by_reference = {r['raw_reference']: r for r in valid}
        for h, sequences in windows_index(valid).items():
            for seq in sequences:
                key = tuple(by_reference[i]['endpoint_hash'] for i in seq)
                if key in seen_windows[h]:
                    continue
                seen_windows[h].add(key)
                windows[h].append(dict(raw_path=trial['raw_path'], trial_id=trial['plan']['trial_id'],
                    scene=trial['plan']['scene'], intended_object=trial['plan']['object'],
                    raw_references=seq, endpoint_hashes=list(key)))
    balanced = select_balanced([r for r in rows if not r['calibration_overlap']], quota, seed)
    full_state, seen = [], set()
    for row in balanced['samples']:
        if row['endpoint_hash'] not in seen:
            seen.add(row['endpoint_hash'])
            full_state.append({k: row[k] for k in ('raw_path', 'trial_id', 'skill_id', 'scene', 'endpoint_hash')})
    window_coverage = [dict(scene=scene, object=target,
        counts={h: sum(w['scene'] == scene and w['intended_object'] == target for w in values)
                for h, values in windows.items()}) for scene in SCENES for target in OBJECT_NAMES]
    return dict(balanced=balanced, windows=windows, full_state=full_state, annotations=rows,
                excluded_attempts=excluded, window_coverage=window_coverage,
                windows_complete=all(all(c['counts'][h] > 0 for h in windows) for c in window_coverage))


def coverage_report(plans, scheduled, trials):
    scheduled_ids = {p['trial_id'] for p in scheduled}
    attempted = {t['plan']['trial_id']: t for t in trials}
    report = []
    for scene in SCENES:
        for target in OBJECT_NAMES:
            candidates = [p for p in plans if p['scene'] == scene and p['object'] == target]
            if not candidates:
                continue
            cells = []
            for cell in sorted({tuple(p['cell']) for p in candidates}):
                ids = [p['trial_id'] for p in candidates if tuple(p['cell']) == cell]
                cells.append(dict(cell=list(cell), planned=len(ids), scheduled=sum(i in scheduled_ids for i in ids),
                                  attempted=sum(i in attempted for i in ids)))
            recipes = [dict(recipe=recipe,
                planned=sum(p['recipe'] == recipe for p in candidates),
                attempted=sum(p['recipe'] == recipe and p['trial_id'] in attempted for p in candidates))
                for recipe in dict.fromkeys(p['recipe'] for p in candidates)]
            report.append(dict(scene=scene, object=target, candidate_cells=cells, recipes=recipes))
    return report


def collect_trial(env, encoder, settings, plan, directory, budget, args):
    commands = commands_for(plan['recipe'], plan['position_root'])
    provider = CombinationProvider(settings, commands, plan['object'])
    collector = SkillCollector(env, encoder, provider, directory)
    audit = ContactAudit(env.unwrapped, directory/'physics_contacts.jsonl', budget*5)
    audit.stop_on_violation = False
    audit.set_stage('initial', allowed_items=OBJECT_NAMES)
    provider.audit = audit
    write_json(directory/'declared_commands.json', commands)

    def observe(row):
        provider.on_step(row)
        if audit.records:
            collector.execution.finish_motion('physical_collision')
            collector.execution.primitive.status = 'physical_collision'

    try:
        try:
            ticks, skills = collector.collect(budget, 'curiosity', seed=plan['seed'],
                stage='object_relative_'+args.stage, on_step=observe, on_skill_end=provider.on_skill_end,
                stop_when=lambda: provider.outcome is not None)
        except InvalidCombinationParameters as exc:
            # The collector flushes existing logs on exit. The invalid proposal
            # has no encoded action and must not create a fictitious transition.
            ticks, skills = read_jsonl(directory/'transitions.jsonl'), read_jsonl(directory/'skills.jsonl')
            provider.outcome = dict(status='failed', reason='invalid_parameters', detail=str(exc))
            write_json(directory/'rejected_command.json', provider.rejected_command)
        valid = completed_endpoints(skills)
        arrays = replay_arrays(valid) if valid else dict(states=np.empty((0, 67), np.float32),
            actions=np.empty((0, 13), np.float32), next_states=np.empty((0, 67), np.float32),
            episode_ids=np.empty(0, np.int64), terminated=np.empty(0, bool), truncated=np.empty(0, bool))
        np.savez_compressed(directory/'skill_endpoints.npz', **arrays,
            raw_reference=np.asarray([r['raw_reference'] for r in valid], dtype=np.int64),
            status=np.asarray([r['status'] for r in valid], dtype=str),
            duration_low_steps=np.asarray([r['duration_low_steps'] for r in valid], dtype=np.int64))
        write_json(directory/'windows.json', windows_index(valid))
        region = encoder.action_projection_contract()
        annotations = [r for skill in valid for r in annotate(
            skill, ticks, plan, str(directory/'skills.jsonl'), region, args)]
        summary = dict(scene=plan['scene'], object=plan['object'], recipe=plan['recipe'],
            trial_id=plan['trial_id'], seed=plan['seed'], goal_root=plan['position_root'],
            outcome=provider.outcome or dict(status='budget_exhausted'), controls=len(ticks),
            requested_control_budget=budget, skills=len(skills), completed_nonzero_skills=len(valid),
            completed_successful_skills=sum(r['status'] == 'success' for r in valid),
            max_grasp_streak=provider.max_grasp_streak, max_held_lift_m=provider.max_held_lift_m,
            issued=provider.issued, event_metrics=events(ticks, skills), physics_substeps=audit.substeps,
            forbidden_contact_substeps=len(audit.records),
            first_forbidden_contact=audit.records[0] if audit.records else None)
        if len(ticks) > budget or audit.substeps > budget*5:
            raise RuntimeError('Physical control budget exceeded')
        write_json(directory/'summary.json', summary)
        return dict(plan=plan, raw_path=str(directory/'skills.jsonl'), skills=skills,
                    annotations=annotations, summary=summary)
    except Exception as exc:
        write_json(directory/'failure.json', dict(status='error', error=repr(exc), plan=plan,
            physics_substeps=audit.substeps, control_budget=budget))
        raise
    finally:
        audit.close()


def main(argv=None):
    args = parse_args(argv)
    cfg = OmegaConf.load(args.config)
    settings = {**DEFAULTS, **OmegaConf.to_container(cfg.raps_actor, resolve=True)}
    if settings['skill_set'] != 'split7' or settings['motion_execution'] != 'planner':
        raise ValueError('Use the existing split7/planner action contract')
    _native_modules()  # Reject incompatible runtimes before creating a partial dataset.
    torch.set_num_threads(1)
    contract = action_contract(settings)
    training_hashes, training_evidence = replay_exclusions(
        SimpleNamespace(exclude_replay=args.exclude_replay, checkpoint=None), {'config': {}}, action_dim=13)
    calibration_hashes, calibration_evidence = calibration_exclusions(args.calibration_data, contract, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    scenes = [args.scene] if args.scene else list(SCENES)
    targets = [args.object] if args.object else list(OBJECT_NAMES)
    manifest = dict(protocol=PROTOCOL, stage=args.stage, status='initializing',
        dataset_role='validation_candidate' if args.stage == 'collect' else 'calibration_pilot',
        source_evidence=source_evidence(args.output), action_contract=contract,
        config=OmegaConf.to_container(cfg, resolve=True), sampling_seed=args.seed, grid=args.grid,
        orientation='Unchanged episode-reset target; no explicit orientation overrides',
        coordinates='Native collision AABBs in robot root coordinates; X/Y margin .02 m, Z inset .01 m',
        closure_target='Every recipe has the same sampled nominal closure position',
        recipes=args.recipes, scenes=scenes, target_objects=targets, control_budget_per_trial=args.trial_controls,
        maximum_total_controls=args.max_steps, close_primitive_repetitions=4, numpy_version=np.__version__,
        physics_audit='Every 100 Hz substep; finish and retain current 20 Hz tick on violation, then stop',
        policy='Stratified object-relative targets and fixed primitive recipes; no learned policy or WM',
        training_overlap_checked=bool(training_evidence), calibration_overlap_checked=bool(calibration_evidence),
        training_replays=training_evidence, calibration_data=calibration_evidence,
        quota_per_scene_object_event=args.quota, event_units='object-level completed nonzero skill endpoints',
        validation_ready=False)
    write_json(args.output/'manifest.json', manifest)
    geometry, state_contract = {}, None
    for scene in scenes:
        env, _ = make_env(episode_steps=800, centered_initial_tcp=True,
                          object_shift_xy=SCENES[scene], action_mode='joint_position')
        try:
            env.reset(seed=args.seed)
            encoder = CabinetSkillStateEncoder(env, latched_gripper=settings['latched_gripper'])
            if state_contract is not None and state_contract != encoder.contract():
                raise ValueError('State contracts differ between layouts')
            state_contract = encoder.contract()
            geometry[scene] = {actor.name: object_audit(env.unwrapped, actor)
                               for actor in env.unwrapped.cabinet_items if actor.name in targets}
        finally:
            env.close()
    plans = generate_sampling_plan(geometry, args.recipes, args.grid, args.seed)
    scheduled = select_pilot(plans) if args.stage == 'pilot' else plans
    plan_path = args.output/'sampling_plan.json'
    write_json(plan_path, plans)
    write_json(args.output/'scheduled_trials.json', [p['trial_id'] for p in scheduled])
    manifest.update(state_contract=state_contract, geometry=geometry, status='planned',
        candidate_attempts=len(plans), scheduled_attempts=len(scheduled),
        sampling_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
    write_json(args.output/'manifest.json', manifest)
    if args.stage == 'plan':
        print(json.dumps(dict(status='planned', candidate_attempts=len(plans), primitive_control_ticks=0)), flush=True)
        return 0
    trials, controls, env, active_scene = [], 0, None, None
    try:
        for plan in scheduled:
            remaining = args.max_steps-controls
            if remaining <= 0:
                break
            if active_scene != plan['scene']:
                if env is not None:
                    env.close()
                env, _ = make_env(episode_steps=800, centered_initial_tcp=True,
                    object_shift_xy=SCENES[plan['scene']], action_mode='joint_position')
                env.reset(seed=plan['seed'])
                encoder = CabinetSkillStateEncoder(env, latched_gripper=settings['latched_gripper'])
                if encoder.contract() != state_contract:
                    raise ValueError('Execution state contract differs from the sampling contract')
                active_scene = plan['scene']
            directory = args.output/plan['scene']/f'trial_{plan["trial_id"]:06d}'
            trial = collect_trial(env, encoder, settings, plan, directory,
                                  min(args.trial_controls, remaining), args)
            trials.append(trial)
            controls += trial['summary']['controls']
            write_json(args.output/'summary.json', [t['summary'] for t in trials])
            print(json.dumps({k:trial['summary'][k] for k in (
                'scene', 'object', 'recipe', 'trial_id', 'outcome', 'controls', 'max_held_lift_m')}), flush=True)
    finally:
        if env is not None:
            env.close()
    indices = build_indices(trials, args.quota, args.seed, training_hashes, calibration_hashes)
    write_json(args.output/'balanced_index.json', indices['balanced'])
    write_json(args.output/'full_state_index.json', indices['full_state'])
    write_json(args.output/'windows_index.json', indices['windows'])
    with (args.output/'annotations.jsonl').open('w') as stream:
        for row in indices['annotations']:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    coverage = coverage_report(plans, scheduled, trials)
    write_json(args.output/'coverage_report.json', dict(candidate_coverage=coverage,
        physical_endpoint_coverage=spatial_audit(indices['annotations'], [], args.grid),
        event_strata=indices['balanced']['strata'], window_coverage=indices['window_coverage'],
        duplicate_object_endpoints=indices['balanced']['duplicate_object_endpoints'],
        training_overlap_object_endpoints=indices['balanced']['training_overlap_object_endpoints'],
        excluded_attempts=indices['excluded_attempts']))
    checks = dict(event_quotas=indices['balanced']['status'] == 'complete',
        contiguous_windows=indices['windows_complete'], training_independence=bool(training_evidence),
        calibration_independence=bool(calibration_evidence), all_scheduled_attempts=len(trials) == len(scheduled))
    ready = args.stage == 'collect' and all(checks.values())
    status = 'complete' if ready else 'quota_unmet' if not checks['event_quotas'] else 'validation_incomplete'
    summary = dict(status=status, stage=args.stage, validation_ready=ready, readiness_checks=checks,
        physical_control_ticks=controls, physics_substeps=sum(t['summary']['physics_substeps'] for t in trials),
        attempted_trials=len(trials), scheduled_trials=len(scheduled), budget_exhausted=controls >= args.max_steps,
        balanced_object_samples=len(indices['balanced']['samples']),
        unique_full_state_samples=len(indices['full_state']),
        window_counts={h:len(v) for h,v in indices['windows'].items()},
        forbidden_contact_substeps=sum(t['summary']['forbidden_contact_substeps'] for t in trials),
        excluded_attempts=indices['excluded_attempts'])
    write_json(args.output/'collection_summary.json', summary)
    manifest.update(status=status, validation_ready=ready, physical_control_ticks=controls,
                    attempted_trials=len(trials), readiness_checks=checks)
    write_json(args.output/'manifest.json', manifest)
    print(json.dumps(summary), flush=True)
    return 0 if args.stage == 'pilot' or ready else 2


if __name__ == '__main__':
    raise SystemExit(main())
