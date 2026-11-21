"""Collect a fixed action grid with baseline reset rules and two fresh-scene streams."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from exploration.diagnostics.collect_uniform_grasps import replay_from_outcomes, uniform_cases, _write_jsonl
from exploration.diagnostics.profile_cup_grasp import CONFIG, ROOT, SOURCES, validate_config
from exploration.raps.cpu_sampling import _stream_seed

PROTOCOL = 'cup_uniform_default_before_grasp_two_streams_v3'
BLOCK_CONTROLS = 1000


def collect_block(job):
    import torch
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from exploration.raps.raps_gnn_actor import DEFAULTS, SkillCollector, action_contract

    torch.set_num_threads(1)
    config, cases = job['config'], job['cases']
    settings = {**DEFAULTS, **config['raps_actor']}
    env, _ = make_env(episode_steps=config['episode_steps'],
        low_level_episode_steps=config['episode_steps'], centered_initial_tcp=config['centered_initial_tcp'],
        cabinet_door=config['cabinet_door'], cabinet_item_layout=config['cabinet_item_layout'],
        cabinet_grid=settings['pick_place_grid'], cabinet_grid_layout=settings['grid_layout'],
        action_mode='joint_position')
    outcomes = []
    try:
        seed = job['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        encoder = CabinetSkillStateEncoder(env, latched_gripper=settings['latched_gripper'])
        collector = SkillCollector(env, encoder, SimpleNamespace(settings=settings), Path(job['directory']),
                                   collection_optimizations=config['collection_optimizations'])
        collector._propose_action = lambda policy: np.asarray(cases[len(outcomes)]['action'], np.float32).copy()

        def record(skill):
            case = cases[len(outcomes)]
            outcomes.append(dict(case_id=case['case_id'], cell_id=case['cell_id'],
                orientation_id=case['orientation_id'], z_m=case['z_m'], seed=seed,
                stream_id=job['stream_id'], block_id=job['block_id'],
                local_episode_id=skill['episode_id'], controls=skill['duration_low_steps'],
                duration_low_steps=skill['duration_low_steps'], completed=skill['completed'],
                status=skill['status'], wm_eligible=bool(skill['trainable']),
                state_contract=encoder.contract(), action_contract=action_contract(settings), skill=skill))

        ticks, skills = collector.collect(job['controls'], 'random', seed=seed,
            stage='continuous_uniform', on_skill_end=record, stop_when=lambda: len(outcomes) == len(cases))
        assert sum(s['primitive_type'] == 'grasp' for s in skills) == len(outcomes) and len(ticks) <= job['controls']
        return dict(outcomes=outcomes, controls=len(ticks))
    finally:
        env.close()


def merge_block(outcomes, result, next_episode):
    """Keep real within-block episodes; never join different streams or fresh scenes."""
    mapping = {}
    for row in result['outcomes']:
        local = row['local_episode_id']
        if local not in mapping:
            next_episode += 1
            mapping[local] = next_episode
        row['episode_id'] = mapping[local]
        outcomes.append(row)
    return next_episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-controls', type=int, default=74400)
    parser.add_argument('--seed', type=int, default=2000000)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--cell-id', type=int, choices=range(10),
                        help='restrict proposals before simulation to this target cell')
    args = parser.parse_args(argv)
    if args.max_controls < 2 or args.seed < 0:
        parser.error('max-controls must be >=2 and seed must be nonnegative')
    config = OmegaConf.load(args.config)
    validate_config(config, learning=False)
    from exploration.raps.raps_gnn_actor import action_contract, skill_schema
    settings = config.raps_actor
    schema = skill_schema(settings)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    if args.cell_id is not None:
        cases = [case for case in cases if case['cell_id'] == args.cell_id]
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Expected nonempty unique case IDs')
    canonical = {c['case_id']: c for c in uniform_cases(settings, z_levels=17)}
    for case in cases:
        reference = canonical.get(case['case_id'])
        if reference is None or any(case[k] != reference[k] for k in ('cell_id', 'orientation_id', 'x_m', 'y_m', 'z_m', 'action')):
            raise ValueError('Expected canonical 17-Z-grid cases for coarse subset export')
        decoded = schema.decode(np.asarray(case['action'], np.float32))
        expected = [case['x_m'], case['y_m'], case['z_m']]
        if decoded.primitive_type != 'grasp' or decoded.parameters['orientation_id'] != case['orientation_id'] or not np.allclose(
                [decoded.parameters[k] for k in ('x', 'y', 'z')], expected, atol=1e-7, rtol=0):
            raise ValueError(f"Case {case['case_id']} differs from production action schema")
    args.output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(args.output/'cases.jsonl', cases)
    manifest = dict(protocol=PROTOCOL, status='collecting', cases_planned=len(cases),
        config=OmegaConf.to_container(config, resolve=True), cases_source=str(args.cases.resolve()),
        cases_source_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        seed=args.seed, workers=args.workers, streams=2, block_controls=BLOCK_CONTROLS,
        max_possible_controls=args.max_controls, fixed_cup_cell_id=4, canonical_z_levels=17,
        requested_target_cell_id=args.cell_id, target_cell_ids=sorted({c['cell_id'] for c in cases}),
        action_contract=action_contract(settings),
        schedule='canonical plan order, alternating cases into streams; each case attempted once',
        reset_policy='Production SkillCollector: planned default joints before each grasp, observe and choose action after homing; homing controls count but are excluded from WM replay. Success or held failure reset; unheld failure retains objects. Fresh CPU scene every stream block.',
        seed_policy='SeedSequence([config_seed, base_seed+10000*block_id, stream_id]); shared within each block, not independent per case',
        limitations='Uniform proposals do not imply uniform states; order-dependent trajectories; no fitting. Partial last macro at block boundary is excluded from WM replay.',
        files_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            (*SOURCES, 'exploration/diagnostics/collect_uniform_grasps.py',
             'exploration/diagnostics/collect_continuous_uniform_grasps.py', 'exploration/raps/cpu_sampling.py')
            if (ROOT/p).is_file()})
    manifest_path = args.output/'manifest.json'
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    save()
    queues = [cases[stream::2] for stream in range(2)]
    consumed, used = [0, 0], [0, 0]
    budgets = [args.max_controls//2 + args.max_controls%2, args.max_controls//2]
    outcomes, episode, block = [], 0, 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
            while any(consumed[s] < len(queues[s]) and used[s] < budgets[s] for s in range(2)):
                jobs = [dict(config=manifest['config'], cases=queues[s][consumed[s]:],
                    controls=min(BLOCK_CONTROLS, budgets[s]-used[s]), stream_id=s, block_id=block,
                    seed=_stream_seed(config.seed, args.seed+10000*block, s),
                    directory=str(args.output/'streams'/f'block_{block:03d}'/f'stream_{s}'))
                    for s in range(2) if consumed[s] < len(queues[s]) and used[s] < budgets[s]]
                for job, result in zip(jobs, pool.map(collect_block, jobs)):
                    s=job['stream_id']
                    if not result['outcomes'] and not result['controls']:
                        raise RuntimeError('Collection block produced no outcomes')
                    consumed[s] += len(result['outcomes']); used[s] += result['controls']
                    episode = merge_block(outcomes, result, episode)
                _write_jsonl(args.output/'attempts.jsonl', outcomes)
                manifest.update(cases_attempted=len(outcomes), controls=sum(used), blocks_completed=block+1)
                save()
                print(json.dumps(dict(cases_attempted=len(outcomes), controls=sum(used))), flush=True)
                block += 1
        state_contract = outcomes[0]['state_contract']
        replay = replay_from_outcomes(outcomes, state_contract=state_contract)
        np.savez_compressed(args.output/'wm_replay.npz', **replay)
        success = [row for row in outcomes if row['status']=='success']
        np.savez_compressed(args.output/'wm_success_replay.npz', **replay_from_outcomes(success, state_contract=state_contract))
        coarse = [row for row in outcomes if row['case_id']%17 in (0,4,8,12,16)]
        np.savez_compressed(args.output/'wm_uniform_replay.npz', **replay_from_outcomes(coarse, state_contract=state_contract))
        manifest.update(status='collected' if len(outcomes)==len(cases) else 'budget_exhausted',
            cases_eligible=len(replay['states']), success_count=sum(row['wm_eligible'] for row in success),
            outcome_status_counts={status:sum(row['status']==status for row in outcomes)
                                   for status in sorted({row['status'] for row in outcomes})},
            state_contract=outcomes[0]['state_contract'])
        save()
        if manifest['status'] != 'collected':
            raise RuntimeError('Control budget exhausted before all cases; partial evidence retained, no automatic expansion')
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc)); save()
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
