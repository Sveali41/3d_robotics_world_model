"""Independent GPU split7 cases with complete, bounded trajectory recording."""
import argparse
import copy
from collections import deque
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch

from domain.robocasa.batched_cabinet import BatchedCabinetSensor, quaternion_matrix, reset_slots, verify_sparse_joint_writes
from domain.robocasa.batched_skills import BatchedSkillExecution, STATUS_NAMES
from domain.robocasa.primitive_wrapper import _EEDeltaAction, _EEDeltaStep
from domain.robocasa.primitives import update_cumulative_lifts
from exploration.cee_us.cee_us_cabinet import make_env
from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder, OBJECT_NAMES, OBJECT_TYPES
from exploration.contact_reward import ContactExplorationRewardConfig, ContactExplorationRewardWrapper
from exploration.diagnostics.collect_cabinet_grasp_sequences import SplitSequenceProvider, results
from exploration.diagnostics.collect_cabinet_grasps import BUDGET, Finished, export, stop_callback
from exploration.diagnostics.diagnose_cabinet import ReplayEnv, jsonable, write_json
from exploration.diagnostics.trajectory_store import TrajectoryWriter
from exploration.raps.raps_gnn_actor import canonical_action, skill_schema


def _host(state, observation, inside, failed, execution, low=None, full=None, terminated=None, truncated=None):
    """One batch transfer; Python logging never pulls individual GPU rows."""
    fields = dict(state=state, ee=observation.ee_position,
                  root=observation.object_positions_root.flatten(1),
                  world=observation.object_positions_world.flatten(1),
                  forces=observation.finger_forces.flatten(1), grasped=observation.grasped,
                  inside=inside[:, None], failed=failed, opened=observation.gripper_open[:, None])
    if low is not None:
        fields.update(low=low, full=full, terminated=terminated[:, None], truncated=truncated[:, None])
    widths = [value.shape[1] for value in fields.values()]
    packed = torch.cat([value.double() for value in fields.values()], -1).detach().cpu().numpy()
    return dict(zip(fields, np.split(packed, np.cumsum(widths)[:-1], axis=1)))


def _snapshot(host, index):
    root, world = host['root'][index].reshape(2, 3), host['world'][index].reshape(2, 3)
    forces, grasped = host['forces'][index].reshape(2, 2), host['grasped'][index]
    contacts = {name: dict(left_force=float(forces[i, 0]), right_force=float(forces[i, 1]),
                           grasped=bool(grasped[i])) for i, name in enumerate(OBJECT_NAMES)}
    return dict(ee_position=host['ee'][index].tolist(),
                objects={name: dict(position_root=root[i].tolist(), position_world=world[i].tolist())
                         for i, name in enumerate(OBJECT_NAMES)}, contact_snapshot=dict(objects=contacts),
                inside_region=bool(host['inside'][index, 0]),
                failed_object_ids=[name for i, name in enumerate(OBJECT_NAMES) if host['failed'][index, i]])


def _context(snapshot):
    objects = {name: SimpleNamespace(position_root=np.asarray(obj['position_root'], np.float32),
                                    position_world=np.asarray(obj['position_world'], np.float32))
               for name, obj in snapshot['objects'].items()}
    contact = snapshot['contact_snapshot']['objects']
    return SimpleNamespace(ee_position=np.asarray(snapshot['ee_position'], np.float32), objects=objects,
        finger_contact_force=lambda name: (contact[name]['left_force'], contact[name]['right_force']),
        is_grasping=lambda name: contact[name]['grasped'])


def _tick_outcome(slot, low, before, after):
    primitive = _EEDeltaStep(_EEDeltaAction('ee_delta', dict(zip(('dx', 'dy', 'dz', 'gripper'), low))))
    primitive.reset(_context(before))
    primitive.stable_grasp_frames.update(slot['grasp_frames'])
    primitive.next_action(_context(before))
    contact = primitive.observe(_context(after))
    outcome = primitive.result(_context(after))
    slot['grasp_frames'] = dict(primitive.stable_grasp_frames)
    lifted, displacements = update_cumulative_lifts(slot['lift_reference'],
        outcome['initial_object_positions_root'], outcome['final_object_positions_root'],
        slot['grasp_frames'], outcome['held_object_ids'])
    outcome.update(lifted_object_ids=lifted, lift_reference_heights_root=dict(slot['lift_reference']),
                   object_lift_displacements=displacements)
    return contact, outcome


def _contract(sensor, raw, settings):
    # Reuse the authoritative contract builder without invoking scalar sensors.
    encoder = CabinetSkillStateEncoder.__new__(CabinetSkillStateEncoder)
    encoder.object_names, encoder.object_types = OBJECT_NAMES, OBJECT_TYPES
    encoder.objects = tuple(zip(OBJECT_NAMES, sensor.objects))
    encoder.dt = .05
    encoder.latched_gripper = bool(settings['latched_gripper'])
    encoder.collision_metadata = sensor.collision_metadata
    root = sensor.arm.root_link.pose
    encoder._root_position = root.p[0].double().cpu().numpy()
    encoder._root_rotation = quaternion_matrix(root.q)[0].cpu().numpy()
    return encoder.contract()


def _check_contract(actual, expected):
    actual, expected = copy.deepcopy(actual), copy.deepcopy(expected)
    for field in ('root_transform_world', 'collision_meshes'):
        if field not in actual or field not in expected:
            continue
        if field == 'root_transform_world':
            # Match CabinetStateEncoder._root_transform's existing 1e-5 tolerance.
            # GPU scene offsets incur float32 cancellation in fixed root poses.
            for name in ('position_m', 'quaternion_wxyz'):
                if not np.allclose(actual[field][name], expected[field][name], rtol=0, atol=1e-5):
                    raise ValueError(f'GPU encoder state contract differs: {field}.{name}')
                actual[field][name] = expected[field][name]
        else:
            for name in actual[field]:
                if name not in expected[field] or not np.allclose(actual[field][name]['extents_m'],
                        expected[field][name]['extents_m'], rtol=0, atol=1e-7):
                    raise ValueError(f'GPU encoder collision extents differ: {name}')
                actual[field][name]['extents_m'] = expected[field][name]['extents_m']
    if actual != expected:
        raise ValueError('GPU encoder state contract differs from checkpoint')


def _new_slot(output, case, settings, environment, index):
    case_dir = output/'cases'/f"case_{case['case_id']:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    previous = [int(path.name.split('_')[1]) for path in case_dir.glob('attempt_*') if path.is_dir()]
    attempt = max(previous, default=0)+1
    directory = case_dir/f'attempt_{attempt:06d}'
    directory.mkdir(exist_ok=False)
    x, y, z = case['offsets']
    args = argparse.Namespace(x_offset=x, cup_y_offset=y, bowl_y_offset=y,
                              cup_z_offset=z, bowl_z_offset=z, lift_distance=case['lift_distance'])
    provider = SplitSequenceProvider(settings, case, args)
    write_json(directory/'attempt.json', dict(case=case, attempt=attempt, environment=environment,
        gpu_env_slot=index, runtime_backend='gpu', reset_protocol='same_seed_twice'))
    return dict(case=case, attempt=attempt, directory=directory, provider=provider, ticks=0,
                skill_id=0, skill=None, skills=[], grasp_frames={}, lift_reference={}, empty_skills=0,
                reward=ContactExplorationRewardWrapper(ReplayEnv(), ContactExplorationRewardConfig(**environment['reward'])),
                log=(directory/'skills.jsonl').open('a'))


def _finish_skill(slot, state, after, status, support, *, completed=False, terminated=False, truncated=False):
    current = slot['skill']
    duration = slot['ticks']-current['start_step']
    if duration == 0:
        slot['empty_skills'] += 1
        if slot['empty_skills'] >= 100:
            raise RuntimeError('100 consecutive zero-duration skills; check actor/primitive parameters')
    else:
        slot['empty_skills'] = 0
    skill = dict(skill_id=slot['skill_id'], stage='validation_only', policy='curiosity', episode_id=1,
        state=current['state'], next_state=state, requested_action=current['requested'], action=current['actual'],
        primitive_type=current['decoded'].primitive_type, parameters_physical=current['decoded'].parameters,
        duration_low_steps=duration, terminated=bool(terminated), truncated=bool(truncated),
        completed=completed, trainable=completed and duration > 0,
        status=status if completed else ('terminated' if terminated else 'episode_truncated' if truncated else 'budget_truncated'),
        support_stopped_object_ids=support, start_step=current['start_step'], end_step=slot['ticks'],
        before=current['before'], after=after)
    skill = jsonable(skill)
    slot['log'].write(json.dumps(skill, allow_nan=False)+'\n')
    slot['log'].flush()
    slot['skills'].append(skill)
    slot['provider'].on_end(skill)
    slot['skill'] = None
    return skill


def run_gpu(output, manifest, payload, settings, num_envs, export_every_cases=64,
            max_control_steps=None, max_seconds=None):
    """Run asynchronous cases; benchmark limits leave active attempts unclosed."""
    if settings.get('skill_set') != 'split7' or not settings.get('latched_gripper'):
        raise ValueError('GPU collection requires split7 and the observed latched gripper state')
    if settings.get('motion_execution', 'ee_delta') != 'ee_delta' or settings.get('project_goals_to_region', False):
        raise ValueError('GPU collection requires direct EE execution without goal projection')
    if num_envs < 1 or export_every_cases < 1:
        raise ValueError('num_envs and export_every_cases must be positive')
    if max_control_steps is not None and max_control_steps < 0:
        raise ValueError('max_control_steps must be nonnegative')
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError('max_seconds must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    closed = {}
    for case in manifest['cases']:
        path = output/'cases'/f"case_{case['case_id']:03d}"/'summary.json'
        if path.exists():
            summary = json.loads(path.read_text())
            if summary['case'] != case:
                raise ValueError(f'Case summary mismatch: {path}')
            closed[case['case_id']] = summary
    pending = deque(sorted((case for case in manifest['cases'] if case['case_id'] not in closed),
                           key=lambda case: case['case_id']))
    metrics = {name: 0. for name in ('initialization_seconds', 'reset_seconds', 'sensor_seconds',
        'host_copy_seconds', 'scheduling_seconds', 'physics_seconds', 'row_build_seconds',
        'writer_enqueue_seconds', 'writer_flush_seconds', 'export_seconds')}
    metrics.update(actual_control_ticks=0, batch_control_steps=0, closed_cases=0, completed_skills=0, num_envs=num_envs)
    full_start = time.perf_counter()
    env = writer = None
    slots = [None]*num_envs
    collection_start = None
    halted = False
    try:
        env, config = make_env(num_envs=num_envs, sim_backend='gpu', raw_env=True,
            centered_initial_tcp=True, episode_steps=BUDGET, project_goals_to_region=False,
            object_shift_xy=(0., 0.))
        raw = env.unwrapped
        if torch.device(raw.device).type != 'cuda':
            raise ValueError('GPU collector requires a CUDA simulator; CPU fallback is unsupported')
        if num_envs > 1:
            try:
                verify_sparse_joint_writes(raw)
            except RuntimeError as error:
                write_json(output/'runtime_preflight.json', dict(sparse_joint_writes=False, error=str(error)))
                raise
            write_json(output/'runtime_preflight.json', dict(sparse_joint_writes=True))
        sensor = BatchedCabinetSensor(raw)
        execution = BatchedSkillExecution(num_envs, raw.device, settings)
        schema = skill_schema(settings)
        contract = _contract(sensor, raw, settings)
        if payload is not None:
            _check_contract(contract, payload['state_contract'])
        metrics['runtime_state_contract'] = contract
        metrics['state_contract_tolerances'] = dict(root_transform_world_atol=1e-5, collision_extents_atol=1e-7)
        environment = OmegaConf.to_container(config, resolve=True)
        writer = TrajectoryWriter(output)
        metrics['initialization_seconds'] = time.perf_counter()-full_start
        collection_start = time.perf_counter()

        def synchronize():
            if torch.device(raw.device).type == 'cuda':
                torch.cuda.synchronize(raw.device)

        def read():
            start = time.perf_counter()
            state, obs, inside, failed = sensor.read(execution.gripper_command)
            synchronize()
            metrics['sensor_seconds'] += time.perf_counter()-start
            start = time.perf_counter()
            host = _host(state, obs, inside, failed, execution)
            metrics['host_copy_seconds'] += time.perf_counter()-start
            if state.shape != (num_envs, 67) or not np.isfinite(host['state']).all():
                raise FloatingPointError('GPU collection requires finite batched state67')
            return state, obs, inside, failed, host

        state, obs, inside, failed, host = read()

        def fill():
            nonlocal state, obs, inside, failed, host
            ids = [i for i, slot in enumerate(slots) if slot is None][:len(pending)]
            if not ids:
                return []
            cases = [pending.popleft() for _ in ids]
            start = time.perf_counter()
            seeds = [case['seed'] for case in cases]
            peer_ids = [i for i, slot in enumerate(slots) if slot is not None and i not in ids]
            previous_host = host
            reset_slots(raw, ids, seeds)
            reset_slots(raw, ids, seeds)
            execution.reset(ids)
            sensor.reset_orientation[ids] = sensor.arm.ee_pose_at_base.q[ids]
            synchronize()
            metrics['reset_seconds'] += time.perf_counter()-start
            state, obs, inside, failed, host = read()
            for i in peer_ids:
                if any(not np.array_equal(previous_host[key][i], host[key][i])
                       for key in host):
                    raise RuntimeError(f'Subset reset changed live slot {i} without a control tick; '
                                       'collection stopped to preserve trajectory continuity')
            for i, case in zip(ids, cases):
                slots[i] = _new_slot(output, case, settings, environment, i)
                slots[i]['reward'].env.current = _snapshot(host, i)
                slots[i]['reward'].reset(seed=case['seed'])
            return ids

        def launch(ids):
            if not ids:
                return
            requested = [slots[i]['provider'].action(host['state'][i].astype(np.float32)) for i in ids]
            actual = canonical_action(torch.as_tensor(np.asarray(requested)), schema).numpy()
            execution.start(ids, actual, obs)
            for j, i in enumerate(ids):
                slot = slots[i]
                slot['skill_id'] += 1
                slot['skill'] = dict(requested=np.asarray(requested[j]).tolist(), actual=actual[j].tolist(),
                    decoded=schema.decode(actual[j]), state=host['state'][i].tolist(),
                    before=_snapshot(host, i), start_step=slot['ticks'])

        def write_exports():
            start = time.perf_counter()
            omitted = []
            export(output, omitted, action_dim=13)
            result = results(output, manifest['cases'],
                summaries=[closed[c['case_id']] for c in manifest['cases'] if c['case_id'] in closed])
            result['incomplete_jsonl_tail_omitted'] = omitted
            write_json(output/'result.json', result)
            metrics['export_seconds'] += time.perf_counter()-start

        def close_case(i):
            slot = slots[i]
            start = time.perf_counter()
            writer.flush()
            metrics['writer_flush_seconds'] += time.perf_counter()-start
            report = slot['provider'].report(slot['skills'])
            success = report['calibration_success']
            summary = dict(case=slot['case'], attempt=slot['attempt'], status='success' if success else
                ('failure' if slot['provider'].failure else 'timeout'), success=success,
                control_ticks=slot['ticks'], report=report)
            directory = slot['directory']
            write_json(directory/'phase_trace.json', slot['provider'].trace)
            write_json(directory/'summary.json', summary)
            write_json(directory.parent/'summary.tmp', summary)
            os.replace(directory.parent/'summary.tmp', directory.parent/'summary.json')
            slot['log'].close()
            closed[slot['case']['case_id']] = summary
            slots[i] = None
            execution.reset([i])
            metrics['closed_cases'] += 1
            if metrics['closed_cases'] % export_every_cases == 0:
                write_exports()

        launch(fill())
        while any(slot is not None for slot in slots):
            if ((max_seconds is not None and time.perf_counter()-collection_start >= max_seconds) or
                    (max_control_steps is not None and metrics['actual_control_ticks'] >= max_control_steps)):
                halted = True
                break
            start = time.perf_counter()
            todo = [i for i, slot in enumerate(slots) if slot is not None]
            low = torch.zeros((num_envs, 4), device=raw.device)
            low[:, 3] = execution.gripper_command
            while todo:
                mask = torch.zeros(num_envs, dtype=torch.bool, device=raw.device)
                mask[todo] = True
                candidate = execution.next_action(obs, mask=mask)
                statuses = torch.cat((execution.status[:, None], execution.action_mask[:, None],
                                      execution.support_stopped_objects), -1).detach().cpu().numpy()
                again, ready = [], []
                for i in todo:
                    if statuses[i, 1]:
                        ready.append(i)
                        continue
                    slot = slots[i]
                    skill = _finish_skill(slot, host['state'][i].tolist(), _snapshot(host, i),
                        STATUS_NAMES[int(statuses[i, 0])], [name for j, name in enumerate(OBJECT_NAMES) if statuses[i, j+2]],
                        completed=True)
                    metrics['completed_skills'] += int(skill['duration_low_steps'] > 0)
                    try:
                        stop_callback(slot['provider'], SimpleNamespace(state=host['state'][i].astype(np.float32)))(skill)
                    except Finished:
                        close_case(i)
                    else:
                        again.append(i)
                low[ready] = candidate[ready]
                new_ids = fill()
                again.extend(new_ids)
                launch(again)
                todo = again
            metrics['scheduling_seconds'] += time.perf_counter()-start
            active = [i for i, slot in enumerate(slots) if slot is not None]
            if not active:
                break
            if max_control_steps is not None and metrics['actual_control_ticks']+len(active) > max_control_steps:
                halted = True
                break
            before_host = host
            start = time.perf_counter()
            full = sensor.controller_action(low, execution)
            _, _, terminated, truncated, _ = raw.step(full)
            synchronize()
            metrics['physics_seconds'] += time.perf_counter()-start
            execution.record_action(low)
            start = time.perf_counter()
            state, obs, inside, failed = sensor.read(execution.gripper_command)
            synchronize()
            metrics['sensor_seconds'] += time.perf_counter()-start
            terminated = torch.as_tensor(terminated, device=raw.device, dtype=torch.bool).expand(num_envs)
            truncated = torch.as_tensor(truncated, device=raw.device, dtype=torch.bool).expand(num_envs).clone()
            budgets = torch.zeros(num_envs, dtype=torch.bool, device=raw.device)
            budgets[[i for i in active if slots[i]['ticks']+1 == BUDGET]] = True
            truncated |= budgets
            start = time.perf_counter()
            host = _host(state, obs, inside, failed, execution, low, full, terminated, truncated)
            metrics['host_copy_seconds'] += time.perf_counter()-start
            if not np.isfinite(host['state']).all():
                raise FloatingPointError('Nonfinite GPU skill state')
            start = time.perf_counter()
            rows, ended = [], []
            for i in active:
                slot = slots[i]
                slot['ticks'] += 1
                current = slot['skill']
                before, after = _snapshot(before_host, i), _snapshot(host, i)
                action = host['low'][i].astype(np.float32).tolist()
                contact, outcome = _tick_outcome(slot, action, before, after)
                row = dict(step=slot['ticks'], stage='validation_only', policy='curiosity', episode_id=1,
                    skill_id=slot['skill_id'], skill_type=current['decoded'].primitive_type,
                    skill_action_requested=current['requested'], skill_action=current['actual'],
                    state=before_host['state'][i].tolist(), next_state=host['state'][i].tolist(),
                    requested_action=action, action=action, full_controller_action=host['full'][i].tolist(),
                    terminated=bool(host['terminated'][i, 0]), truncated=bool(host['truncated'][i, 0]),
                    run_budget_exhausted=slot['ticks'] == BUDGET, primitive_type='ee_delta', duration_low_steps=1,
                    parameters_physical=dict(zip(('dx', 'dy', 'dz', 'gripper'), action)),
                    action_units='normalized_controller', status='success', contact_events=[contact],
                    outcome_metrics=outcome, before=before, after=after, planner={},
                    case_id=slot['case']['case_id'], attempt_id=slot['attempt'], gpu_env_slot=i,
                    gpu_batch_step=metrics['batch_control_steps']+1)
                reward = slot['reward']
                reward.env.current, reward.env.next_row = before, row
                _, _, term, trunc, info = reward.step(None)
                row.update(terminated=bool(term), truncated=bool(trunc), reward=info['exploration_reward'])
                slot['provider'].on_step(row)
                rows.append(row)
                if term or trunc:
                    ended.append((i, bool(term), bool(trunc)))
            metrics['row_build_seconds'] += time.perf_counter()-start
            force = torch.zeros(num_envs, dtype=torch.bool, device=raw.device)
            force[[i for i, _, _ in ended]] = True
            execution.observe(obs, force_mask=force)
            metrics['actual_control_ticks'] += len(active)
            metrics['batch_control_steps'] += 1
            start = time.perf_counter()
            writer.append(rows)
            metrics['writer_enqueue_seconds'] += time.perf_counter()-start
            for i, term, trunc in ended:
                slot = slots[i]
                _finish_skill(slot, host['state'][i].tolist(), _snapshot(host, i), '', [],
                              terminated=term, truncated=trunc)
                close_case(i)
            launch(fill())
        # A bounded benchmark retains partial skills and traces without case closure.
        for i, slot in enumerate(slots):
            if slot is not None:
                if slot['skill'] is not None:
                    _finish_skill(slot, host['state'][i].tolist(), _snapshot(host, i), '', [])
                write_json(slot['directory']/'phase_trace.json', slot['provider'].trace)
        start = time.perf_counter()
        writer.close()
        metrics['writer_flush_seconds'] += time.perf_counter()-start
        write_exports()
        metrics['halted_by_benchmark_limit'] = halted
        metrics['collection_seconds'] = time.perf_counter()-collection_start
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            for slot in slots:
                if slot is not None:
                    slot['log'].close()
            if env is not None:
                env.close()
    metrics['active_control_ticks'] = metrics['actual_control_ticks']
    metrics['total_seconds'] = time.perf_counter()-full_start
    metrics['control_ticks_per_second'] = metrics['actual_control_ticks']/max(metrics['collection_seconds'], 1e-12)
    write_json(output/'benchmark_metrics.json', metrics)
    return metrics
