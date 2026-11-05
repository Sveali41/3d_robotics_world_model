"""Independent, scripted validation-only real skill endpoints; never trains a model."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import traceback
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from domain.robocasa.primitives import PrimitiveAction
from exploration.cee_us.cee_us_cabinet import make_env, replay_arrays
from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder, OBJECT_NAMES
from exploration.diagnostics.diagnose_cabinet import source_evidence, write_json
from exploration.raps.raps_gnn_actor import DEFAULTS, SCHEMA, SkillCollector, action_contract, action_contract_matches, skill_schema
from exploration.raps.raps_replay_compare import events

CLASSES = ('lift', 'displacement', 'contact_only', 'free_motion', 'static')
SCENES = {'original': (0., 0.), 'shifted': (.03, 0.)}


def endpoint_hash(state, action, target):
    return hashlib.sha256(b''.join(np.asarray(x, dtype=np.float32).tobytes() for x in (state, action, target))).hexdigest()


def generate_plans(args, count):
    rng = np.random.default_rng(args.seed)
    plans = []
    while len(plans) < count:
        # Each block balances requested object/intent; these are not observed labels.
        pairs = [(obj, intent) for obj in OBJECT_NAMES for intent in CLASSES]
        rng.shuffle(pairs)
        for obj, intent in pairs:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            plans.append(dict(trial_id=len(plans), seed=args.seed + len(plans), object=obj,
                              intent=intent, jitter=rng.uniform(-args.jitter, args.jitter, 3).tolist(),
                              lift=float(rng.uniform(*args.lift_range)),
                              neighborhood_offset=(direction * args.radius * rng.random()**(1/3)).tolist(),
                              cell=[int(rng.integers(n)) for n in args.grid],
                              cell_fraction=rng.random(3).tolist()))
            if len(plans) == count:
                break
    return plans


def local_to_root(local, region):
    return np.asarray(region['root_to_cabinet_rotation']).T @ (np.asarray(local) - region['root_to_cabinet_translation'])


def region_info(tcp, objects, region, grid, radius):
    local = np.asarray(region['root_to_cabinet_rotation']) @ tcp + region['root_to_cabinet_translation']
    lo, hi = np.asarray(region['lower']), np.asarray(region['upper'])
    inside = bool(np.all(local >= lo) and np.all(local <= hi))
    cell = np.minimum(np.floor((local-lo)/(hi-lo)*grid).astype(int), np.asarray(grid)-1).tolist() if inside else 'outside'
    return dict(local_position=local.tolist(), spatial_bin=cell,
                object_neighborhoods={n: bool(np.linalg.norm(np.asarray(tcp)-o['position_root']) <= radius) for n, o in objects.items()})


class ScriptedProvider:
    """Decides goals only; SkillCollector executes every actual controller tick."""
    def __init__(self, settings, plan, region, args):
        self.settings, self.plan, self.region, self.args = settings, plan, region, args
        self.schema = skill_schema(settings)
        self.phase = 'open'
        self.goal = None
        self.approach_attempts = 0
        self.free_attempts = 0
        self.audit = None

    def encode(self, name, **parameters):
        for p in self.schema.spec_by_name[name].parameters:
            if not np.isfinite(parameters[p.name]) or abs(parameters[p.name]) > p.scale:
                raise ValueError(f'{name}.{p.name} outside primitive scale {p.scale}: {parameters}')
        if self.audit is not None:
            attachment = self.audit.raw.motion_planner.attachment_state()
            self.audit.payload = attachment['name'] if attachment else None
            self.audit.set_stage(name, allowed_items=OBJECT_NAMES, support=name == 'lift')
        return self.schema.encode(PrimitiveAction(name, parameters))

    def move(self, delta):
        # Split distant goals into bounded segments, without changing the goal.
        scales = np.array([p.scale for p in self.schema.spec_by_name['move_delta_ee_pose'].parameters])
        delta = delta / max(1., float(np.max(np.abs(delta)/scales)))
        return self.encode('move_delta_ee_pose', **dict(zip(('dx', 'dy', 'dz'), delta)))

    def action(self, state):
        tcp = np.asarray(state[18:21], dtype=float)
        obj_index = OBJECT_NAMES.index(self.plan['object'])
        obj = np.asarray(state[33+12*obj_index:36+12*obj_index], dtype=float)
        intent = self.plan['intent']
        if self.phase == 'open':
            self.phase = 'approach'
            return self.encode('open_gripper')
        if intent == 'static':
            return self.encode('open_gripper')
        if intent == 'free_motion':
            if self.goal is None:
                lo, hi = np.asarray(self.region['lower']), np.asarray(self.region['upper'])
                local = lo + (np.asarray(self.plan['cell']) + self.plan['cell_fraction']) / self.args.grid * (hi-lo)
                self.goal = local_to_root(local, self.region)
            if (np.linalg.norm(self.goal-tcp) < .015 or
                    (self.settings.get('motion_execution') == 'planner' and self.free_attempts >= 12)):
                return self.encode('open_gripper')
            self.free_attempts += 1
            return self.move(self.goal-tcp)
        if self.phase == 'approach':
            target = obj + self.plan['jitter']
            target[0] -= self.args.standoff
            if np.linalg.norm(target-tcp) > .015 and self.approach_attempts < 12:
                self.approach_attempts += 1
                return self.move(target-tcp)
            self.phase = 'interact'
        if self.phase == 'interact':
            self.phase = 'lift' if intent == 'lift' else 'probe'
            delta = obj + self.plan['jitter'] - tcp
            if intent == 'contact_only':
                return self.move(delta)  # open-gripper push; observed contact still decides label
            if delta[0] <= 0 or abs(delta[0]) > .35 or abs(delta[1]) > .20 or abs(delta[2]) > .12:
                self.phase = 'approach'
                return self.move(delta - np.array([self.args.standoff, 0, 0]))
            if self.settings['skill_set'] == 'split7':
                self.phase = 'approach_x'
                return self.encode('align_yz', y_offset=float(delta[1]), z_offset=float(delta[2]))
            return self.encode('approach', x_forward=float(delta[0]), y_offset=float(delta[1]), z_offset=float(delta[2]))
        if self.phase == 'approach_x':
            self.phase = 'close'
            distance = float(obj[0] + self.plan['jitter'][0] - tcp[0])
            if not 0 <= distance <= .35:
                self.phase = 'approach'
                return self.move(obj + self.plan['jitter'] - tcp - np.array([self.args.standoff, 0, 0]))
            return self.encode('approach_x', x_forward=distance)
        if self.phase == 'close':
            self.phase = 'lift' if intent == 'lift' else 'probe'
            return self.encode('close_gripper')
        if self.phase == 'lift':
            self.phase = 'probe'
            return self.encode('lift', z_dist=self.plan['lift'])
        if self.phase == 'probe':
            self.phase = 'drop'
            return self.move(obj + self.plan['neighborhood_offset'] - tcp)
        if self.phase == 'drop':
            self.phase = 'release'
            return self.encode('drop', z_dist=self.plan['lift'])
        if self.phase == 'release':
            self.phase = 'approach'
            self.approach_attempts = 0
            return self.encode('open_gripper')
        raise AssertionError(self.phase)


def annotate(skill, ticks, plan, raw_path, region, args):
    if not skill['completed'] or skill['duration_low_steps'] <= 0:
        return []
    selected = [r for r in ticks if r['skill_id'] == skill['skill_id']]
    observed = events(selected, [skill])['per_object']
    any_contact = any(max(o['left_force'], o['right_force']) > r['reward']['weights']['contact_force_threshold']
                      for r in selected for snap in r['contact_events'] for o in snap['objects'].values())
    tcp_delta = float(np.linalg.norm(np.asarray(skill['after']['ee_position'])-skill['before']['ee_position']))
    digest = endpoint_hash(skill['state'], skill['action'], skill['next_state'])
    rows = []
    for name in OBJECT_NAMES:
        start = np.asarray(skill['before']['objects'][name]['position_root'])
        positions = np.asarray([r['after']['objects'][name]['position_root'] for r in selected])
        world_z = [r['after']['objects'][name]['position_world'][2] for r in selected]
        endpoint = float(np.linalg.norm(positions[-1]-start))
        force = max(max(s['objects'][name]['left_force'], s['objects'][name]['right_force']) for r in selected for s in r['contact_events'])
        touch = any(max(s['objects'][name]['left_force'], s['objects'][name]['right_force']) > r['reward']['weights']['contact_force_threshold'] for r in selected for s in r['contact_events'])
        flags = dict(lift=observed[name]['lift_events'] > 0, displacement=endpoint >= args.displacement_threshold,
                     contact=touch, free_motion=tcp_delta >= .015 and not any_contact)
        flags['static'] = not any(flags.values())
        primary = next((c for c in CLASSES[:-1] if flags['contact' if c == 'contact_only' else c]), 'static')
        rows.append(dict(raw_path=raw_path, skill_id=skill['skill_id'], episode_id=skill['episode_id'],
                         trial_id=plan['trial_id'], start_step=skill['start_step'], end_step=skill['end_step'],
                         object=name, intended_object=plan['object'], intended_event=plan['intent'],
                         primary_event=primary, event_booleans=flags, grasp=observed[name]['grasp_events'] > 0,
                         endpoint_hash=digest, object_hash=digest+':'+name, max_finger_force=force,
                         endpoint_displacement_m=endpoint, max_displacement_m=float(np.linalg.norm(positions-start, axis=1).max()),
                         max_root_rise_m=float(max(0., (positions[:,2]-start[2]).max())),
                         max_world_rise_m=float(max(0., max(world_z)-skill['before']['objects'][name]['position_world'][2])),
                         failed_object_ids=sorted({o for r in selected for o in r['after']['failed_object_ids']}),
                         terminated=any(r['terminated'] for r in selected), truncated=any(r['truncated'] for r in selected),
                         status=skill['status'],
                         tick_flags=[dict(step=r['step'],failed_object_ids=r['after']['failed_object_ids'],
                                          terminated=r['terminated'],truncated=r['truncated'],status=r.get('status')) for r in selected],
                         region={k: region_info(skill[k]['ee_position'], skill[k]['objects'], region, args.grid, args.radius) for k in ('before', 'after')}))
    return rows


def windows_index(skills):
    result = {str(h): [] for h in (1, 3, 5)}
    for h in (1, 3, 5):
        for i in range(len(skills)-h+1):
            chunk = skills[i:i+h]
            if all(a['episode_id'] == b['episode_id'] and a['skill_id']+1 == b['skill_id'] and a['end_step'] == b['start_step'] and not a['terminated'] and not a['truncated'] for a, b in zip(chunk, chunk[1:])):
                result[str(h)].append([s['raw_reference'] for s in chunk])
    return result


def select_balanced(rows, quota, seed):
    rng = np.random.default_rng(seed)
    pools = {(scene, obj, cls): [] for scene in SCENES for obj in OBJECT_NAMES for cls in CLASSES}
    seen, counts, duplicates, overlaps = set(), dict.fromkeys(pools, 0), 0, 0
    for r in rows:
        if r['training_overlap']:
            overlaps += 1
            continue
        key = (r['scene'], r['object_hash'])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        stratum = (r['scene'], r['object'], r['primary_event'])
        counts[stratum] += 1
        pool = pools[stratum]
        if len(pool) < quota:
            pool.append(r)
        else:
            j = int(rng.integers(counts[stratum]))
            if j < quota:
                pool[j] = r
    report = [dict(scene=k[0], object=k[1], event=k[2], requested=quota, achieved=len(v), missing=max(0,quota-len(v)), observed_unique=counts[k]) for k,v in pools.items()]
    return dict(samples=[r for v in pools.values() for r in v], strata=report,
                duplicate_object_endpoints=duplicates, training_overlap_object_endpoints=overlaps,
                status='complete' if all(len(v)==quota for v in pools.values()) else 'quota_unmet')


def spatial_audit(rows, plans, grid):
    """Count physical skill endpoints once, including all zero-visit bins."""
    result = {}
    for scene in SCENES:
        cells = {str(list(cell)): dict(requested_free_motion_trials=0, start_endpoints=0, end_endpoints=0)
                 for cell in np.ndindex(*grid)}
        cells['outside'] = dict(requested_free_motion_trials=0, start_endpoints=0, end_endpoints=0)
        for plan in plans:
            if plan['intent'] == 'free_motion':
                cells[str(plan['cell'])]['requested_free_motion_trials'] += 1
        neighborhoods = {name: dict(start_endpoints=0, end_endpoints=0) for name in OBJECT_NAMES}
        seen = set()
        for row in rows:
            key = (row['raw_path'], row['skill_id'])
            if row['scene'] != scene or key in seen:
                continue
            seen.add(key)
            for position, label in (('before', 'start_endpoints'), ('after', 'end_endpoints')):
                info = row['region'][position]
                cell = info['spatial_bin']
                cells[cell if isinstance(cell, str) else str(cell)][label] += 1
                for name, near in info['object_neighborhoods'].items():
                    neighborhoods[name][label] += int(near)
        result[scene] = dict(metric='completed physical skill endpoints; not dense ticks',
                             physical_skills=len(seen), bins=cells, object_neighborhoods=neighborhoods,
                             unvisited_bins=[cell for cell, count in cells.items()
                                             if cell != 'outside' and not count['start_endpoints'] and not count['end_endpoints']])
    return result


def replay_exclusions(args, payload, action_dim=12):
    paths = list(args.exclude_replay)
    if args.checkpoint is not None:
        companion = args.checkpoint.with_suffix('.replay.npz')
        if companion.exists():
            paths.append(companion)
    source = payload['config'].get('replay_experiment', {}).get('source')
    if source:
        paths.append(Path(source).with_suffix('.replay.npz'))
    hashes, evidence = set(), []
    for path in dict.fromkeys(p.resolve() for p in paths):
        if not path.is_file():
            raise FileNotFoundError(f'Required training replay missing: {path}; supply the correct --exclude-replay paths')
        with np.load(path) as z:
            s,a,y = (z[k] for k in ('states','actions','next_states'))
            if s.shape != y.shape or s.shape != (len(s),67) or a.shape != (len(s),action_dim):
                raise ValueError(f'Incompatible training replay: {path}')
            hashes.update(endpoint_hash(*triple) for triple in zip(s,a,y))
        evidence.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return hashes, evidence


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('plan','smoke','collect'), default='plan')
    p.add_argument('--resume', action='store_true', help='Resume an interrupted matching collection, counting every saved tick')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--config', type=Path, help='Collect new split7/planner data without loading an old WM')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--trial-steps', type=int, default=800)
    p.add_argument('--quota', type=int, default=20)
    p.add_argument('--seed', type=int, default=1000000)
    p.add_argument('--grid', type=int, nargs=3, default=[2,2,2])
    p.add_argument('--radius', type=float, default=.10)
    p.add_argument('--standoff', type=float, default=.08)
    p.add_argument('--jitter', type=float, default=.01)
    p.add_argument('--lift-range', type=float, nargs=2, default=[.02,.10])
    p.add_argument('--displacement-threshold', type=float, default=.015)
    p.add_argument('--exclude-replay', type=Path, action='append', default=[])
    return p


def validate_args(args):
    if args.stage == 'collect' and args.max_steps is None:
        raise ValueError('collect requires explicit --max-steps global budget')
    if args.max_steps is None:
        args.max_steps = 200 if args.stage == 'smoke' else 1600
    if args.max_steps < 2 or args.max_steps % 2 or not 1 <= args.trial_steps <= 800:
        raise ValueError('Budget must be positive, even, >=2; trial steps must be in [1,800]')
    if args.stage == 'smoke' and args.max_steps > 200:
        raise ValueError('smoke is capped at 200 total control ticks across both scenes')
    values = [args.radius,args.standoff,args.jitter,*args.lift_range,args.displacement_threshold]
    if not np.isfinite(values).all() or args.radius <= 0 or not 0 < args.standoff <= .35 or args.jitter < 0 or not 0 < args.lift_range[0] <= args.lift_range[1] <= .16 or args.displacement_threshold <= 0 or min(args.grid) < 1 or args.quota < 1 or args.seed < 0:
        raise ValueError('Invalid finite sampling ranges, grid, quota or seed; lift must fit primitive 0..0.16m')


def raw_attempts(directory, plan):
    root = directory / f"trial_{plan['trial_id']:06d}"
    paths = ([root] if root.exists() else []) + sorted(root.glob('attempt_*'))
    attempts = []
    for path in paths:
        def read(name):
            file = path / name
            return [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []
        attempts.append((path, read('transitions.jsonl'), read('skills.jsonl')))
    return attempts


def resume_inventory(args, plans, candidate):
    manifest = json.loads((args.output/'manifest.json').read_text())
    if args.stage != 'collect' or manifest.get('stage') != 'collect' or manifest.get('status') in ('complete', 'quota_unmet'):
        raise ValueError('Resume requires an interrupted collect output, not a finished or unrelated output')
    if manifest.get('split') != 'validation_only' or json.loads((args.output/'sampling_plan.json').read_text()) != plans:
        raise ValueError('Resume validation split or sampling plan differs')
    for key in ('checkpoint', 'checkpoint_sha256', 'state_contract', 'action_contract', 'training_replays', 'args', 'source_config', 'source_config_sha256'):
        if manifest.get(key) != candidate[key]:
            raise ValueError(f'Resume {key} differs; preserve original checkpoint, replay exclusions and sampling arguments')
    counts = {}
    partial = []
    for scene in SCENES:
        counts[scene] = 0
        for plan in plans:
            attempts = raw_attempts(args.output/scene, plan)
            saved = sum(len(ticks) for _, ticks, _ in attempts)
            target = min(args.trial_steps, args.max_steps//2-plan['trial_id']*args.trial_steps)
            if saved > target:
                raise ValueError(f'Saved ticks exceed planned budget: {scene} trial {plan["trial_id"]}')
            counts[scene] += saved
            if attempts and saved < target:
                partial.append(dict(scene=scene, trial_id=plan['trial_id'], saved_ticks=saved,
                                    remaining_ticks=target-saved, reset_required=True,
                                    reason='interrupted native attempt; see failure.json; physics reset before continuation'
                                    if (attempts[-1][0]/'failure.json').exists() else
                                    'interrupted attempt; original failure: 100 consecutive zero-duration skills at reached free-motion goal'
                                    if plan['intent']=='free_motion' and len(attempts[-1][2]) >= 99 and all(x['duration_low_steps']==0 for x in attempts[-1][2][-99:])
                                    else 'interrupted collection; physics state cannot be restored'))
    manifest.setdefault('recoveries', []).append(dict(source_evidence=candidate['source_evidence'],
        script_sha256=candidate['script_sha256'], dependencies=candidate['dependencies'],
        saved_control_ticks=counts, remaining_control_ticks={scene:args.max_steps//2-n for scene,n in counts.items()},
        partial_trials=partial, change=('explicit path-timing failure status; raw data, goals and event definitions unchanged'
            if candidate['config']['raps_actor'].get('motion_execution') == 'planner' else
            'free-motion reached-goal open-gripper dwell; goals and event definitions unchanged'),
        continuity='every new attempt resets physics with its original plan seed; windows never cross attempts'))
    return manifest, sum(counts.values())


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f'Refusing existing output: {args.output}')
    if args.resume and not args.output.is_dir():
        raise FileNotFoundError('Resume requires an existing interrupted output')
    payload = None if args.config is not None else torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = OmegaConf.load(args.config) if payload is None else OmegaConf.create(payload['config'])
    config.raps_actor = OmegaConf.merge(DEFAULTS, config.raps_actor)
    settings = OmegaConf.to_container(config.raps_actor, resolve=True)
    planned = settings['motion_execution'] == 'planner'
    if payload is None:
        if settings['skill_set'] != 'split7' or not planned or config.action_mode != 'joint_position' or not settings['latched_gripper']:
            raise ValueError('--config requires split7/planner, joint_position and latched gripper')
        state_contract = (json.loads((args.output/'manifest.json').read_text())['state_contract']
                          if args.resume else None)
        source_action_contract = action_contract(settings)
    else:
        if payload.get('schema') != SCHEMA or settings['skill_set'] != 'compact6' or not settings['latched_gripper'] or not action_contract_matches(payload['action_contract'], settings):
            raise ValueError('Requires matching compact6/12D checkpoint action contract with latched gripper')
        state_contract = payload['state_contract']
        source_action_contract = payload['action_contract']
    if (config.episode_steps != 800 or not config.centered_initial_tcp or config.execution != 'reference'
            or (state_contract is not None and state_contract['dimension'] != 67)):
        raise ValueError('Requires state67 centered reset reference execution episode800 checkpoint/config')
    dimension = skill_schema(settings).action_dim
    excluded, replay_evidence = replay_exclusions(args, payload if payload is not None else
        dict(config=OmegaConf.to_container(config, resolve=True)), dimension)
    plans = generate_plans(args, math.ceil((args.max_steps//2)/args.trial_steps))
    evidence = source_evidence(args.output)
    manifest = dict(split='validation_only', immutable=True, designed_diagnostic_benchmark=True,
                    distribution='scripted diagnostic; not natural on-policy', stage=args.stage,
                    checkpoint=str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint is not None else None,
                    source_config=str(args.config.resolve()) if args.config is not None else None,
                    source_config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest() if args.config is not None else None,
                    config=OmegaConf.to_container(config, resolve=True), state_contract=state_contract,
                    action_contract=source_action_contract, args={k:str(v) if isinstance(v,Path) else [str(x) for x in v] if k=='exclude_replay' else v for k,v in vars(args).items() if k != 'resume' and not (k == 'config' and v is None)},
                    training_replays=replay_evidence, source_evidence=evidence,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    dependencies={n: importlib.metadata.version(n) for n in ('torch','numpy','sapien','gymnasium')},
                    physics_hz=100, control_hz=20, episode_steps=800, project_goals_to_region=False,
                    scenes=SCENES, contact_metric=('fingertip events plus independent 100Hz physical contact audit' if planned
                        else 'fingertip-object forces only; no generic cabinet/world collision label'),
                    lift_metric='existing events: closed+grasp >=4 ticks and rise >=0.01m',
                    free_motion_threshold_m=.015, sampling='seeded reservoir within observed scene/object/primary event; spatial coverage audited separately',
                    correlation='two object endpoints can share one physical skill',
                    windows='raw consecutive skills only; never concatenate balanced/gapped samples', automatic_training=False)
    if planned:
        manifest['free_motion_attempt_limit'] = 12
        manifest['free_motion_after_limit'] = 'open-gripper dwell; original goal and every failed attempt retained'
    physical = 0
    if args.resume:
        manifest, physical = resume_inventory(args,plans,manifest)
    else:
        args.output.mkdir(parents=True)
        write_json(args.output/'sampling_plan.json', plans)
    write_json(args.output/'manifest.json', manifest)
    if args.stage == 'plan':
        return 0
    torch.set_num_threads(1)
    all_rows, excluded_skills, incomplete = [], 0, 0
    collision_trials = []
    scene_counts = {}
    for scene, shift in SCENES.items():
        env = None
        clean = []
        scene_ticks = 0
        episode_ids = {}
        try:
            env, env_config = make_env(episode_steps=800, centered_initial_tcp=True, project_goals_to_region=False, object_shift_xy=shift,
                                       action_mode='joint_position' if planned else 'ee_delta')
            env.reset(seed=args.seed)
            encoder = CabinetSkillStateEncoder(env, latched_gripper=settings['latched_gripper'])
            if manifest['state_contract'] is None:
                manifest['state_contract'] = encoder.contract()
                write_json(args.output/'manifest.json', manifest)
            elif encoder.contract() != manifest['state_contract']:
                raise ValueError('Scene encoder differs from source state contract')
            region = encoder.action_projection_contract()
            manifest.setdefault('regions', {})[scene] = region
            manifest.setdefault('environment', {})[scene] = OmegaConf.to_container(env_config, resolve=True)
            directory = args.output/scene
            directory.mkdir(exist_ok=args.resume)
            with (directory/'annotations.jsonl').open('w') as annotation_file:
                for plan in plans:
                    attempts = raw_attempts(directory,plan)
                    saved = sum(len(ticks) for _, ticks, _ in attempts)
                    target = min(args.trial_steps,args.max_steps//2-plan['trial_id']*args.trial_steps)
                    budget = target-saved
                    if budget > 0:
                        root = directory/f"trial_{plan['trial_id']:06d}"
                        raw = root if not root.exists() else root/f'attempt_{len(attempts):06d}'
                        if raw.exists():
                            raise FileExistsError(f'Refusing to append raw attempt: {raw}')
                        provider = ScriptedProvider(settings, plan, region, args)
                        collector = SkillCollector(env,encoder,provider,raw)
                        audit = None
                        try:
                            if planned:
                                from exploration.diagnostics.validate_cabinet_planner import ContactAudit
                                audit = ContactAudit(env.unwrapped, raw/'physics_contacts.jsonl', budget*5)
                                audit.stop_on_violation = False
                                audit.set_stage('initial', allowed_items=OBJECT_NAMES)
                                provider.audit = audit
                            ticks, skills = collector.collect(budget,'curiosity',seed=plan['seed'],stage='validation_only')
                        except Exception as exc:
                            write_json(raw/'failure.json', dict(error_type=type(exc).__name__, message=str(exc),
                                traceback=traceback.format_exc(), trial_plan=plan))
                            raise
                        finally:
                            if audit is not None:
                                report = dict(scene=scene, trial_id=plan['trial_id'], physics_substeps=audit.substeps,
                                              stage_stats=audit.stage_stats,
                                              forbidden_contact_substeps=len(audit.records),
                                              first_forbidden_contact=audit.records[0] if audit.records else None)
                                write_json(raw/'collision_summary.json', report)
                                collision_trials.append(report)
                                audit.close()
                                provider.audit = None
                        physical += len(ticks)
                        if physical > args.max_steps:
                            raise AssertionError('Global physical budget exceeded')
                        attempts.append((raw,ticks,skills))
                        if planned:
                            print(json.dumps(dict(scene=scene, trial=plan['trial_id'], intent=plan['intent'],
                                object=plan['object'], control_ticks=len(ticks),
                                forbidden_contact_substeps=collision_trials[-1]['forbidden_contact_substeps'],
                                skill_statuses=[s['status'] for s in skills])), flush=True)
                    for raw, ticks, skills in attempts:
                        scene_ticks += len(ticks)
                        for skill in skills:
                            rows = annotate(skill,ticks,plan,str(raw.relative_to(args.output)/'skills.jsonl'),region,args)
                            if not rows:
                                incomplete += 1
                                continue
                            overlap = rows[0]['endpoint_hash'] in excluded
                            for row in rows:
                                row.update(scene=scene,training_overlap=overlap)
                                annotation_file.write(json.dumps(row,allow_nan=False)+'\n')
                                all_rows.append(row)
                            if overlap:
                                excluded_skills += 1
                                continue
                            key = (str(raw),skill['episode_id'])
                            if key not in episode_ids:
                                episode_ids[key] = len(episode_ids)+1
                            skill = dict(skill, episode_id=episode_ids[key],
                                         raw_episode_id=skill['episode_id'], trial_id=plan['trial_id'],
                                         raw_reference=dict(raw_path=rows[0]['raw_path'],skill_id=skill['skill_id'],episode_id=skill['episode_id']))
                            clean.append(skill)
            if clean:
                arrays = replay_arrays(clean)
                arrays.update(duration_low_steps=np.asarray([s['duration_low_steps'] for s in clean],dtype=np.int64),
                              skill_ids=np.asarray([s['skill_id'] for s in clean]), raw_episode_ids=np.asarray([s['raw_episode_id'] for s in clean]),
                              trial_ids=np.asarray([s['trial_id'] for s in clean]))
            else:
                arrays = dict(states=np.empty((0,67),np.float32),actions=np.empty((0,dimension),np.float32),next_states=np.empty((0,67),np.float32),
                              episode_ids=np.empty(0,np.int64),terminated=np.empty(0,bool),truncated=np.empty(0,bool),duration_low_steps=np.empty(0,np.int64),
                              skill_ids=np.empty(0,np.int64),raw_episode_ids=np.empty(0,np.int64),trial_ids=np.empty(0,np.int64))
            scene_counts[scene] = dict(physical_control_ticks=scene_ticks, complete_skill_transitions=len(clean),
                                       object_endpoint_annotations=sum(r['scene'] == scene for r in all_rows))
            np.savez_compressed(directory/'skills.npz',**arrays)
            write_json(directory/'windows_index.json',windows_index(clean))
        finally:
            if env is not None:
                env.close()
    if planned:
        collision_trials = [json.loads(path.read_text()) for scene in SCENES
                            for path in sorted((args.output/scene).rglob('collision_summary.json'))]
    balanced = select_balanced(all_rows,args.quota,args.seed)
    write_json(args.output/'balanced_index.json',balanced)
    manifest.update(physical_control_ticks=physical,incomplete_or_zero_skills=incomplete,training_overlap_skills=excluded_skills,
                    status=balanced['status'], spatial_coverage=spatial_audit(all_rows,plans,args.grid))
    write_json(args.output/'manifest.json',manifest)
    write_json(args.output/'collection_summary.json', dict(status=balanced['status'],
        physical_control_ticks=physical, scene_counts=scene_counts, strata=balanced['strata'],
        balanced_index_samples=len(balanced['samples']), action_dim=dimension,
        native_action_dim=8 if planned else 4, collision_trials=collision_trials))
    print(json.dumps(dict(status=balanced['status'],physical_control_ticks=physical,output=str(args.output),
                          unmet_strata=sum(s['missing']>0 for s in balanced['strata']),
                          action='Increase explicitly approved budget or revise documented sampling parameters if quotas remain unmet')))
    return 0 if args.stage == 'smoke' or balanced['status']=='complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
