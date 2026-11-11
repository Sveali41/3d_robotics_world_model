"""Record labelled cabinet teleoperation and replay primitive endpoints for validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import select
import sys
import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

NAMES = ('move_delta_ee_pose', 'approach', 'lift', 'drop', 'open_gripper', 'close_gripper')
MOVES = {'w': (0, 1), 's': (0, -1), 'a': (1, 1), 'd': (1, -1),
         'r': (2, 1), 'f': (2, -1)}
MANUAL_OBJECT_SPACING_INCREASE_M = .02


def extract_plan(segments, schema, axis_tolerance=.015, transitions=None):
    """Convert observed root-frame endpoints into existing primitive parameters."""
    from domain.robocasa.primitives import PrimitiveAction
    if not np.isfinite(axis_tolerance) or axis_tolerance < 0:
        raise ValueError('axis tolerance must be finite and nonnegative')
    if not segments:
        raise ValueError('demonstration has no segments')
    plan = []
    for i, segment in enumerate(segments):
        name = segment['name']
        if not segment.get('complete'):
            raise ValueError(f'segment {i} is incomplete')
        if name not in NAMES or name not in schema.spec_by_name:
            raise ValueError(f'segment {i}: unsupported primitive {name}')
        start = np.asarray(segment['start_state'], dtype=float)
        end = np.asarray(segment['end_state'], dtype=float)
        if start.shape != (67,) or end.shape != (67,) or not np.isfinite(start).all() or not np.isfinite(end).all():
            raise ValueError(f'segment {i}: expected finite 67D states')
        delta = end[18:21] - start[18:21]
        before, after = start[32], end[32]
        if before not in (-1., 1.) or after not in (-1., 1.):
            raise ValueError(f'segment {i}: invalid latched gripper command')
        if segment['end_step'] <= segment['start_step']:
            raise ValueError(f'segment {i}: no control ticks')
        params = {}
        if name == 'move_delta_ee_pose':
            if before != after:
                raise ValueError(f'segment {i}: move must preserve gripper')
            params = dict(zip(('dx', 'dy', 'dz'), delta))
        elif name == 'approach':
            if delta[0] < 0 or after != -1:
                raise ValueError(f'segment {i}: approach requires positive x and closed gripper')
            params = dict(zip(('x_forward', 'y_offset', 'z_offset'), delta))
        elif name in ('lift', 'drop'):
            if np.any(np.abs(delta[:2]) > axis_tolerance) or before != after:
                raise ValueError(f'segment {i}: {name} requires vertical motion and preserved gripper')
            distance = delta[2] if name == 'lift' else -delta[2]
            if distance < 0:
                raise ValueError(f'segment {i}: wrong {name} direction')
            params = {'z_dist': distance}
        else:
            expected = 1 if name == 'open_gripper' else -1
            if np.linalg.norm(delta) > axis_tolerance or after != expected:
                raise ValueError(f'segment {i}: {name} requires stationary TCP and expected gripper')
        if transitions is not None and name in ('move_delta_ee_pose', 'lift', 'drop'):
            selected = [row for row in transitions if segment['start_step'] < row['step'] <= segment['end_step']]
            if len(selected) != segment['end_step'] - segment['start_step'] or any(
                    row['next_state'][32] != before for row in selected):
                raise ValueError(f'segment {i}: {name} must preserve gripper throughout all ticks')
        for p in schema.spec_by_name[name].parameters:
            if abs(params[p.name]) > p.scale:
                raise ValueError(f'segment {i}: {name}.{p.name} outside scale {p.scale}: {params[p.name]}')
        params = {k: float(v) for k, v in params.items()}
        plan.append(dict(name=name, parameters=params, action=schema.encode(PrimitiveAction(name, params)).tolist(),
                         source_segment=i, start_step=segment['start_step'], end_step=segment['end_step']))
    return plan


def runtime():
    import torch
    from omegaconf import OmegaConf
    from exploration.cee_us.cee_us_cabinet import make_env, replay_arrays
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from exploration.diagnostics.diagnose_cabinet import jsonable, write_json
    from exploration.raps.raps_gnn_actor import DEFAULTS, SCHEMA, SkillCollector, action_contract_matches, skill_schema
    return locals()


def load_checkpoint(path, r):
    payload = r['torch'].load(path, map_location='cpu', weights_only=False)
    config = payload['config']
    settings = {**r['DEFAULTS'], **config['raps_actor']}
    if (payload.get('schema') != r['SCHEMA'] or settings['skill_set'] != 'compact6'
            or settings['motion_execution'] != 'ee_delta' or not settings['latched_gripper']
            or config.get('action_mode', 'ee_delta') != 'ee_delta'
            or not r['action_contract_matches'](payload['action_contract'], settings)):
        raise ValueError('Requires matching compact6 ee_delta checkpoint with latched gripper; planned and split7 unsupported')
    if config.get('execution', 'reference') != 'reference':
        raise ValueError('Requires reference execution checkpoint')
    if config.get('sim_freq', 100) != 100 or config.get('control_freq', 20) != 20:
        raise ValueError('Checkpoint simulation/control frequency differs from make_env (100/20 Hz)')
    return payload, settings


def build_env(config, r, action_mode='ee_delta', object_spacing_increase=0.0):
    object_spacing_increase = float(object_spacing_increase)
    if not np.isfinite(object_spacing_increase) or object_spacing_increase < 0:
        raise ValueError('Object spacing increase must be finite and nonnegative')
    env, cfg = r['make_env'](episode_steps=config.get('episode_steps', 100),
        centered_initial_tcp=config.get('centered_initial_tcp', False),
        project_goals_to_region=config.get('project_goals_to_region', False),
        object_shift_xy=config.get('object_shift_xy', (0., 0.)), action_mode=action_mode)
    if object_spacing_increase:
        raw = env.unwrapped
        items = raw.cabinet_items
        positions = raw._item_spawn_positions
        if len(items) != len(positions):
            raise ValueError('Cannot adjust manual object spacing: cabinet item/spawn-position lengths differ')
        indices = {item.name: index for index, item in enumerate(items)
                   if item.name in ('cabinet_item_16', 'cabinet_item_17')}
        if set(indices) != {'cabinet_item_16', 'cabinet_item_17'}:
            raise ValueError('Cannot adjust manual object spacing: expected cabinet_item_16 and cabinet_item_17')
        adjusted = [np.asarray(position).copy() for position in positions]
        for name, direction in (('cabinet_item_16', -1), ('cabinet_item_17', 1)):
            position = adjusted[indices[name]]
            if position.ndim != 1 or position.size < 2:
                raise ValueError(f'Cannot adjust manual object spacing: invalid spawn position for {name}')
            position[1] += direction * object_spacing_increase / 2
        raw._item_spawn_positions = adjusted
    # Only this manual diagnostic removes episode time limits; training keeps its limits.
    current = env
    while True:
        for name in ('max_ee_steps_per_episode', 'max_low_level_steps_per_episode', '_max_episode_steps'):
            if name in vars(current):
                setattr(current, name, float('inf'))
        if 'env' not in vars(current):
            break
        current = current.env
    environment = r['OmegaConf'].to_container(cfg, resolve=True)
    environment['max_ee_steps_per_episode'] = None
    environment['max_low_level_steps_per_episode'] = None
    environment['object_spacing_increase_m'] = object_spacing_increase
    return env, environment


def verify_initial(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape or not np.allclose(expected, actual, atol=1e-6, rtol=0):
        raise ValueError('Replay initial encoded state differs from demonstration; no evaluation rows produced')


def verify_physics(expected, actual):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError('Replay initial physics state differs from demonstration')
        for key in expected:
            verify_physics(expected[key], actual[key])
    else:
        a, b = np.asarray(expected), np.asarray(actual)
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-6, rtol=0):
            raise ValueError('Replay initial physics state differs from demonstration')


def render_viewer(raw):
    raw.scene.update_render(update_sensors=False, update_human_render_cameras=True)
    viewer = raw.render_human()
    if viewer.closed:
        raise EOFError('Viewer closed')


def read_command(raw, headless):
    if headless:
        return input('teleop> ')
    print('teleop> ', end='', flush=True)
    while not select.select([sys.stdin], [], [], .03)[0]:
        render_viewer(raw)
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line


def mouse_delta_action(target_world, current_world, root_rotation, translation_scale, speed, gripper):
    target = np.asarray(target_world, dtype=float)
    current = np.asarray(current_world, dtype=float)
    rotation = np.asarray(root_rotation, dtype=float)
    scale = np.broadcast_to(np.asarray(translation_scale, dtype=float), (3,))
    if (target.shape != (3,) or current.shape != (3,) or rotation.shape != (3, 3)
            or not all(np.isfinite(x).all() for x in (target, current, rotation, scale))
            or np.any(scale <= 0) or not np.isfinite(speed) or not 0 < speed <= 1
            or gripper not in (-1., 1.)):
        raise ValueError('Invalid mouse target, controller scale, speed, or gripper')
    delta = (rotation.T @ (target - current)) / scale
    norm = np.linalg.norm(delta)
    if norm > speed:
        delta *= speed / norm
    return np.asarray([*delta, gripper], dtype=np.float32)


def tcp_position(raw):
    from exploration.cee_us.cee_us_state import _numpy
    return _numpy(raw.agent.tcp.pose.p, 'TCP position', 3)


def world_pose_to_root(matrix, root_position, root_rotation):
    matrix = np.asarray(matrix, dtype=float)
    root_position = np.asarray(root_position, dtype=float)
    root_rotation = np.asarray(root_rotation, dtype=float)
    if (matrix.shape != (4, 4) or root_position.shape != (3,) or root_rotation.shape != (3, 3)
            or not all(np.isfinite(x).all() for x in (matrix, root_position, root_rotation))
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or any(not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or
                   not np.isclose(np.linalg.det(rot), 1, atol=1e-5)
                   for rot in (matrix[:3, :3], root_rotation))):
        raise ValueError('Expected finite rigid world and root poses')
    position = root_rotation.T @ (matrix[:3, 3] - root_position)
    quaternion = np.roll(Rotation.from_matrix(root_rotation.T @ matrix[:3, :3]).as_quat(), 1)
    return position, quaternion


def tcp_matrix(raw):
    from exploration.cee_us.cee_us_state import _pose
    position, rotation = _pose(raw.agent.tcp.pose, 'TCP pose')
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, position
    return matrix


def validate_pose_runtime():
    if int(np.__version__.split('.')[0]) >= 2:
        raise RuntimeError('Pose planning requires NumPy < 2 and MPLib 0.1.1; run .runtime/thor-ceeus/bin/python (without -s)')
    from domain.robocasa.motion_planning import _native_modules
    try:
        _native_modules()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError('Pose planning requires MPLib 0.1.1; run .runtime/thor-ceeus/bin/python (without -s)') from exc


def joint_angles(raw):
    from exploration.cee_us.cee_us_state import _numpy
    return _numpy(raw.agent.robot.qpos, 'robot qpos', 9)[:7]


def joint_target_action(targets, limits, gripper):
    targets = np.asarray(targets, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if (targets.shape != (7,) or limits.shape != (7, 2)
            or not np.isfinite(targets).all() or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1]) or gripper not in (-1., 1.)
            or np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1])):
        raise ValueError('Joint targets must be finite and within the seven physical joint limits')
    return np.asarray([*targets, gripper], dtype=np.float32)


def joint_details(raw):
    robot = raw.agent.robot
    limits = robot.get_qlimits().detach().cpu().numpy().reshape(9, 2)[:7].copy()
    names = [joint.name for joint in robot.get_active_joints()][:7]
    if len(names) != 7:
        raise ValueError('Expected seven active Panda arm joints')
    return names, limits


def mouse_ui(raw, encoder, budget, control_mode='ee'):
    import sapien
    from sapien import internal_renderer as R
    from sapien.utils.viewer.plugin import Plugin
    from sapien.utils.viewer.transform_window import TransformWindow

    class MouseTeleop(Plugin):
        def __init__(self):
            self.target = tcp_position(raw)
            self.target_matrix = tcp_matrix(raw) if control_mode == 'pose' else None
            self.busy = False
            self.on_reject = None
            self.status = ('Click Begin segment, edit Translate/Rotate target, then Plan & Execute'
                           if control_mode == 'pose' else 'Ready')
            self.contacts = ['none', 'cup', 'bowl']
            self.contact_selected = 0
            if control_mode == 'joints':
                self.joint_names, self.joint_limits = joint_details(raw)
                self.reset_target()
            self.names = list(NAMES)
            self.selected = 0
            self.queue = deque()
            self.active = None
            self.following = False
            self.step = 0
            self.next_tick = 0.
            self.window = None

        @property
        def gizmo_matrix(self):
            if control_mode == 'pose':
                return self.target_matrix.copy()
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, 3] = self.target
            return matrix

        @gizmo_matrix.setter
        def gizmo_matrix(self, value):
            matrix = np.asarray(value, dtype=float)
            if control_mode == 'pose':
                try:
                    world_pose_to_root(matrix, np.zeros(3), np.eye(3))
                except ValueError:
                    return
                self.target_matrix = matrix.copy()
                self.target = matrix[:3, 3].copy()
            elif np.isfinite(matrix[:3, 3]).all():
                self.target = matrix[:3, 3].copy()

        def reset_target(self, _=None):
            if control_mode == 'joints':
                for i, value in enumerate(joint_angles(raw)):
                    setattr(self, f'joint_{i}', float(value))
            elif control_mode == 'pose':
                self.target_matrix = tcp_matrix(raw)
                self.target = self.target_matrix[:3, 3].copy()
            else:
                self.target = tcp_position(raw)

        @property
        def joint_targets(self):
            return np.array([getattr(self, f'joint_{i}') for i in range(7)])

        def begin(self, _):
            if self.active is None:
                self.queue.append('begin ' + ('joint_position' if control_mode == 'joints' else 'pose_control' if control_mode == 'pose' else self.names[self.selected]))

        def end(self, _):
            if self.active is not None:
                self.following = False
                self.queue.append('end')

        def toggle(self, _):
            if self.active is not None:
                self.following = not self.following
                self.next_tick = time.monotonic()

        def finish(self, _):
            if self.active is None:
                self.queue.append('quit')

        def gripper(self, command):
            if self.active is not None:
                if control_mode == 'pose' and self.busy:
                    self.status = 'Motion in progress; request rejected'
                    if self.on_reject: self.on_reject(command, self.status)
                else:
                    self.queue.append(command)
                    if control_mode == 'pose': self.busy = True
            elif control_mode == 'pose':
                self.status = 'Begin segment first'
                if self.on_reject: self.on_reject(command, self.status)

        def plan(self, _):
            if self.active is not None:
                if self.busy:
                    self.status = 'Motion in progress; request rejected'
                    if self.on_reject: self.on_reject('plan', self.status)
                else:
                    self.busy = True
                    self.queue.append('plan')
            else:
                self.status = 'Begin segment first'
                if self.on_reject: self.on_reject('plan', self.status)

        def cancel(self, _):
            if self.active is not None:
                self.queue.append('cancel')

        def get_ui_windows(self):
            if self.window is None:
                self.window = R.UIWindow().Label('Joint teleoperation' if control_mode == 'joints' else 'Pose teleoperation' if control_mode == 'pose' else 'Mouse teleoperation').Pos(420, 10).Size(400, 600 if control_mode == 'joints' else 450)
                if control_mode == 'joints':
                    self.window.append(R.UIDisplayText().Text('Set independent arm joint targets (rad).'))
                    for i, name in enumerate(self.joint_names):
                        self.window.append(R.UISliderFloat().Label(f'{name} (rad)').Bind(self, f'joint_{i}')
                                           .Min(float(self.joint_limits[i, 0])).Max(float(self.joint_limits[i, 1])))
                        self.window.append(R.UIDisplayText().Bind(
                            lambda i=i, name=name: f'{name} actual: {joint_angles(raw)[i]:.4f} rad'))
                elif control_mode == 'pose':
                    self.gizmo = R.UIGizmo().Bind(self, 'gizmo_matrix')
                    self.window.append(R.UIDisplayText().Text('Translate / rotate the world TCP target.'),
                        R.UIDisplayText().Bind(lambda: f'Segment active: {self.active}' if self.active is not None else 'Not recording — click Begin segment'),
                        R.UIButton().Label('Begin segment').Callback(self.begin),
                        R.UIButton().Label('End segment').Callback(self.end),
                        R.UIOptions().Style('select').Label('Allowed finger contact').BindItems(self, 'contacts').BindIndex(self, 'contact_selected'),
                        R.UIButton().Label('Plan & Execute').Callback(self.plan),
                        R.UIButton().Label('Cancel motion').Callback(self.cancel),
                        R.UIDisplayText().Bind(lambda: self.status))
                else:
                    self.gizmo = R.UIGizmo().Bind(self, 'gizmo_matrix')
                    self.window.append(
                        R.UIDisplayText().Text('Drag the gizmo axes to set the world-space TCP target.'),
                        R.UIDisplayText().Text('Begin a segment, then drag; pause following to hold.'),
                        R.UIOptions().Style('select').Label('Primitive').BindItems(self, 'names').BindIndex(self, 'selected'))
                if control_mode != 'pose':
                    self.window.append(
                        R.UIButton().Label('Begin segment').Callback(self.begin),
                        R.UIButton().Label('End segment').Callback(self.end))
                self.window.append(
                    R.UIButton().Label('Run / Pause following').Callback(self.toggle),
                    R.UIButton().Label('Open gripper').Callback(lambda _: self.gripper('o')),
                    R.UIButton().Label('Close gripper').Callback(lambda _: self.gripper('c')),
                    R.UIButton().Label('Reset target to current joints' if control_mode == 'joints' else 'Reset target to current EE').Callback(self.reset_target),
                    R.UIButton().Label('Finish / save').Callback(self.finish),
                    R.UIDisplayText().Bind(lambda: f'Segment: {self.active or "none"}; following: {self.following}'),
                    R.UIDisplayText().Bind(lambda: f'Ticks: {self.step} / {budget if budget is not None else "unlimited"}; gripper: {encoder.gripper_command:+g}'))
                if control_mode in ('ee', 'pose'): self.window.append(self.gizmo)
            if control_mode in ('ee', 'pose'):
                pose = self.viewer.window.get_camera_pose()
                view = (pose * sapien.Pose([0, 0, 0], [-.5, -.5, .5, .5])).inv().to_transformation_matrix()
                self.gizmo.CameraMatrices(view, self.viewer.window.get_camera_projection_matrix())
                self.gizmo.Matrix(self.gizmo_matrix)
            return [self.window]

        def read(self):
            while True:
                render_viewer(raw)
                if self.queue:
                    return self.queue.popleft()
                now = time.monotonic()
                if self.active is not None and self.following and (control_mode != 'pose' or self.busy) and now >= self.next_tick:
                    self.next_tick = now + .05
                    return 'mouse'
                time.sleep(.005)

    viewer = raw.render_human()
    viewer.plugins = [p for p in viewer.plugins if not isinstance(p, TransformWindow)]
    ui = MouseTeleop()
    ui.init(viewer)
    viewer.plugins.append(ui)
    return ui


def record(args, payload, settings, digest, r):
    control_mode = getattr(args, 'control_mode', 'ee')
    if control_mode not in ('ee', 'joints', 'pose'):
        raise ValueError(f'Unsupported control mode: {control_mode}')
    if control_mode in ('joints', 'pose') and args.headless:
        raise ValueError('Joint and pose control require the mouse interface; --headless is unsupported')
    if control_mode == 'pose':
        validate_pose_runtime()
    args.output.mkdir(parents=True, exist_ok=False)
    env = None
    manifest = dict(split='validation_only', automatic_training=False,
        distribution='human-designed teleoperation-conditioned diagnostic; no broad coverage claim',
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest,
        config=payload['config'], settings=settings, state_contract=payload['state_contract'],
        action_contract=payload['action_contract'] if control_mode == 'ee' else None,
        source_action_contract=payload['action_contract'], control_mode=control_mode,
        seed=args.seed, max_steps=args.max_steps,
        object_spacing_increase_m=MANUAL_OBJECT_SPACING_INCREASE_M,
        speed=args.speed, episode_step_limit=None,
        input_mode='terminal' if args.headless else 'mouse', segments=[], status='recording')
    active = None
    job = None
    request_log = None
    request_id = 0
    state = None
    step = 0
    limit = args.max_steps if args.max_steps is not None else float('inf')
    try:
        env, manifest['environment'] = build_env(
            payload['config'], r,
            action_mode='joint_position' if control_mode in ('joints', 'pose') else 'ee_delta',
            object_spacing_increase=MANUAL_OBJECT_SPACING_INCREASE_M)
        env.reset(seed=args.seed)
        encoder = r['CabinetSkillStateEncoder'](env, latched_gripper=True)
        if encoder.contract() != payload['state_contract']:
            raise ValueError('Environment state contract differs from checkpoint')
        state = encoder.encode()
        manifest['initial_state'] = state.tolist()
        raw = env.unwrapped
        if control_mode in ('joints', 'pose'):
            names, limits = joint_details(raw)
            manifest['action_contract'] = dict(action_mode='joint_position', dimension=8,
                joint_names=names, joint_limits_rad=limits.tolist(), joint_target_units='rad',
                gripper_units='normalized', gripper_bounds=[-1., 1.],
                layout='seven absolute arm joint targets followed by coupled gripper command')
        if control_mode == 'pose':
            manifest['macro_contract'] = dict(target_frame='world', planner_target_frame='robot_root', quaternion_order='wxyz',
                gripper_control_ticks=30, control_frequency_hz=20, max_plan_ticks=400, planning_time_s=3., seed=args.seed)
        manifest['initial_physics_state'] = r['jsonable'](raw.get_state_dict())
        r['write_json'](args.output/'demo.json', manifest)
        ui = None
        if not args.headless:
            if control_mode == 'ee':
                cfg = encoder.controller.config
                lower = np.broadcast_to(np.asarray(cfg.pos_lower, dtype=float), (3,))
                upper = np.broadcast_to(np.asarray(cfg.pos_upper, dtype=float), (3,))
                if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper <= 0) or not np.allclose(lower, -upper):
                    raise ValueError('Mouse control requires finite symmetric controller translation bounds')
                ui = mouse_ui(raw, encoder, args.max_steps)
            else:
                ui = mouse_ui(raw, encoder, args.max_steps, control_mode=control_mode)
        else:
            print('Commands: w/s +/-X, a/d +/-Y, r/f +/-Z, o/c gripper; optional tick count. begin NAME, end, quit')
        if control_mode == 'pose':
            request_log = (args.output/'pose_requests.jsonl').open('x')

        def finish_job(status, error=None):
            nonlocal job
            if job is None: return
            execution = job.get('execution')
            if execution is not None:
                if execution.status == 'running': execution.stop(status)
                job['execution_result'] = execution.result()
            job.update(status=status, error=error, end_state=state.tolist(), end_step=step)
            job.pop('execution', None)
            job.pop('hold_targets', None)
            request_log.write(json.dumps(r['jsonable'](job), allow_nan=False)+'\n')
            request_log.flush()
            ui.status = f'{status}' + (f': {error}' if error else '')
            ui.busy = False
            job = None

        def reject_request(command, reason):
            nonlocal request_id
            request_id += 1
            target = ui.target_matrix.copy()
            position, quaternion = world_pose_to_root(target, encoder._root_position, encoder._root_rotation)
            selected = ui.contacts[ui.contact_selected]
            contact = {'cup': 'cabinet_item_16', 'bowl': 'cabinet_item_17'}.get(selected)
            rejected = dict(request_id=request_id, kind='plan' if command == 'plan' else 'gripper',
                target_world_matrix=target, target_root_position=position, target_root_quaternion_wxyz=quaternion,
                allowed_finger_contact=selected, allowed_contact_objects=[contact] if contact else [],
                start_state=state.tolist(), end_state=state.tolist(), start_step=step, end_step=step,
                physical_ticks=0, status='rejected', error=reason, command=command)
            request_log.write(json.dumps(r['jsonable'](rejected), allow_nan=False)+'\n')
            request_log.flush()

        if control_mode == 'pose': ui.on_reject = reject_request

        with (args.output/'demo_transitions.jsonl').open('x') as log:
            while step < limit:
                try:
                    tokens = (read_command(raw, True) if ui is None else ui.read()).strip().split()
                except EOFError:
                    manifest['status'] = 'input_ended' if ui is None else 'viewer_closed'
                    break
                if not tokens:
                    continue
                cmd = tokens[0].lower()
                if cmd == 'quit' and len(tokens) == 1:
                    if active is not None:
                        print('End the active segment before quit'); continue
                    manifest['status'] = 'complete'; break
                if cmd == 'begin' and len(tokens) == 2:
                    if active is not None or tokens[1] not in (('joint_position',) if control_mode == 'joints' else ('pose_control',) if control_mode == 'pose' else NAMES):
                        print('Invalid or nested segment'); continue
                    active = dict(name=tokens[1], start_state=state.tolist(), start_step=step)
                    if ui is not None:
                        ui.active = tokens[1]
                        if control_mode == 'pose':
                            ui.status = 'Recording pose_control. Edit target, then Plan & Execute.'
                        ui.reset_target()
                        ui.following = True
                        ui.next_tick = time.monotonic() + .05
                    continue
                if cmd == 'end' and len(tokens) == 1:
                    if active is None:
                        print('No active segment'); continue
                    if control_mode == 'pose': finish_job('cancelled_by_end')
                    active.update(end_state=state.tolist(), end_step=step, complete=True)
                    manifest['segments'].append(active); active = None
                    if ui is not None:
                        ui.active = None; ui.following = False
                        if control_mode == 'pose':
                            ui.status = 'Segment saved. Click Begin segment for another.'
                    r['write_json'](args.output/'demo.json', manifest)
                    continue
                if control_mode == 'pose' and cmd == 'cancel':
                    finish_job('cancelled')
                    ui.busy = False
                    continue
                if control_mode == 'pose' and cmd in ('plan', 'o', 'c') and active is not None:
                    if job is not None:
                        ui.status = 'Motion in progress; request rejected'
                        continue
                    ui.following = True
                    ui.next_tick = time.monotonic() + .05
                    request_id += 1
                    target = ui.target_matrix.copy()
                    position, quaternion = world_pose_to_root(target, encoder._root_position, encoder._root_rotation)
                    selected = ui.contacts[ui.contact_selected]
                    # These are the two objects identified by the state encoder.
                    contact_name = {'cup': 'cabinet_item_16', 'bowl': 'cabinet_item_17'}.get(selected)
                    job = dict(request_id=request_id, kind='plan' if cmd == 'plan' else 'gripper',
                        target_world_matrix=target, target_root_position=position,
                        target_root_quaternion_wxyz=quaternion, allowed_finger_contact=selected,
                        allowed_contact_objects=[contact_name] if contact_name else [],
                        start_state=state.tolist(), start_step=step, physical_ticks=0)
                    if cmd == 'plan':
                        ui.status = 'Planning...'
                        render_viewer(raw)
                        try:
                            from domain.robocasa.motion_planning import MotionPlanExecution
                            plan = raw.plan_to_pose(position, quaternion, planning_time_s=3., seed=args.seed,
                                allowed_contact_objects=job['allowed_contact_objects'])
                            job['planning'] = plan.metadata
                            job['plan_file'] = f'pose_plan_{request_id:04d}.npz'
                            np.savez_compressed(args.output/job['plan_file'], position=plan.position, time=plan.time)
                            job['execution'] = MotionPlanExecution(raw, plan, gripper_command=encoder.gripper_command, max_steps=400)
                            ui.status = 'Running plan'
                        except Exception as exc:
                            finish_job(getattr(exc, 'status', 'planning_error'), str(exc))
                    else:
                        encoder.gripper_command = 1. if cmd == 'o' else -1.
                        job['gripper_command'] = encoder.gripper_command
                        job['hold_targets'] = joint_angles(raw).copy()
                        ui.status = 'Running gripper'
                    continue
                if control_mode == 'pose' and (cmd != 'mouse' or job is None or active is None):
                    continue
                valid_commands = ('o', 'c', 'mouse') if control_mode in ('joints', 'pose') else (*MOVES, 'o', 'c', *(['mouse'] if ui is not None else []))
                if cmd not in valid_commands or len(tokens) > 2 or active is None:
                    print('Movement requires an active segment and a valid command'); continue
                try:
                    count = int(tokens[1]) if len(tokens) == 2 else 1
                    if count < 1: raise ValueError()
                except ValueError:
                    print('Tick count must be a positive integer'); continue
                for _ in range(min(count, limit-step)):
                    low = np.zeros(4, dtype=np.float32)
                    if cmd in MOVES:
                        axis, sign = MOVES[cmd]; low[axis] = sign * args.speed
                    elif cmd in ('o', 'c'):
                        encoder.gripper_command = 1. if cmd == 'o' else -1.
                    if control_mode == 'pose':
                        low = (job['execution'].next_action() if job['kind'] == 'plan' else
                               joint_target_action(job['hold_targets'], limits, encoder.gripper_command))
                    elif control_mode == 'joints':
                        low = joint_target_action(ui.joint_targets, limits, encoder.gripper_command)
                    elif cmd == 'mouse':
                        low = mouse_delta_action(ui.target, tcp_position(raw), encoder._root_rotation,
                                                 upper, args.speed, encoder.gripper_command)
                    else:
                        low[3] = encoder.gripper_command
                    before = state.copy()
                    _, _, terminated, truncated, info = env.step(low)
                    rec = info['primitive_record']
                    if rec.duration_low_steps != 1:
                        raise RuntimeError('Expected exactly one control tick')
                    state = encoder.encode(); step += 1
                    feedback = None
                    if control_mode == 'pose':
                        job['physical_ticks'] += 1
                        if job['kind'] == 'plan':
                            try:
                                feedback = job['execution'].record_step(terminated=terminated, truncated=truncated)
                            except Exception as exc:
                                job['feedback_error'] = str(exc)
                    row = dict(step=step, segment=len(manifest['segments']), command=cmd, state=before,
                        next_state=state, requested_action=low, action=rec.low_actions[0],
                        full_controller_action=rec.full_controller_actions[0], status=rec.status,
                        contact_events=rec.contact_events, outcome_metrics=rec.outcome_metrics,
                        terminated=bool(terminated), truncated=bool(truncated))
                    if control_mode == 'pose':
                        row.update(control_mode='pose', action_mode='joint_position', request_id=job['request_id'],
                            target_world_matrix=job['target_world_matrix'], target_root_position=job['target_root_position'],
                            target_root_quaternion_wxyz=job['target_root_quaternion_wxyz'], planner_feedback=feedback)
                    elif control_mode == 'joints':
                        row.update(action_mode='joint_position', action_units='seven_joint_targets_rad_plus_normalized_gripper',
                                   joint_targets_rad=low[:7].copy())
                    elif ui is not None: row['target_world'] = ui.target.copy()
                    if ui is not None: ui.step = step
                    log.write(json.dumps(r['jsonable'](row), allow_nan=False)+'\n'); log.flush()
                    if control_mode == 'pose':
                        if 'feedback_error' in job:
                            finish_job('execution_error', job['feedback_error'])
                        elif terminated or truncated:
                            finish_job('terminated' if terminated else 'truncated')
                        elif job['kind'] == 'plan' and job['execution'].status != 'running':
                            finish_job(job['execution'].status)
                        elif job['kind'] == 'gripper' and job['physical_ticks'] >= 30:
                            finish_job('success')
                    if not args.headless: render_viewer(raw)
                    if terminated or truncated:
                        manifest['status'] = 'terminated' if terminated else 'truncated'
                        break
                if terminated or truncated: break
            else:
                manifest['status'] = 'budget_truncated'
    except EOFError:
        manifest['status'] = 'viewer_closed'
    finally:
        if job is not None:
            finish_job(manifest['status'] if manifest['status'] != 'recording' else 'error')
        if request_log is not None: request_log.close()
        if active is not None:
            active.update(end_state=state.tolist(), end_step=step, complete=False)
            manifest['segments'].append(active)
        manifest['control_ticks'] = step
        r['write_json'](args.output/'demo.json', manifest)
        if env is not None: env.close()


class ReplayFinished(Exception):
    pass


class PlanActor:
    def __init__(self, settings, plan):
        self.settings, self.plan, self.index = settings, plan, 0

    def action(self, state):
        result = np.asarray(self.plan[self.index]['action'], dtype=np.float32)
        self.index += 1
        return result


def replay(args, manifest, payload, settings, r):
    reject_joint_manifest(manifest)
    plan = json.loads((args.output/'plan.json').read_text())['plan']
    if not plan: raise ValueError('Plan is empty')
    out = args.output/'replay'
    out.mkdir(exist_ok=False)
    result = dict(split='validation_only', automatic_training=False,
        distribution='human-designed teleoperation-conditioned diagnostic; no broad coverage claim',
        checkpoint=manifest['checkpoint'], checkpoint_sha256=manifest['checkpoint_sha256'],
        config=payload['config'], state_contract=payload['state_contract'], action_contract=payload['action_contract'],
        seed=manifest['seed'], max_steps=args.max_steps, episode_step_limit=None,
        initial_state_match=False, status='started')
    env = None
    try:
        env, result['environment'] = build_env(
            payload['config'], r,
            object_spacing_increase=manifest.get('object_spacing_increase_m', 0.0))
        env.reset(seed=manifest['seed'])
        encoder = r['CabinetSkillStateEncoder'](env, latched_gripper=True)
        if encoder.contract() != payload['state_contract']:
            raise ValueError('Environment state contract differs from checkpoint')
        actor = PlanActor(settings, plan)
        collector = r['SkillCollector'](env, encoder, actor, out)
        def start(episode, skill, decoded, action, state):
            if skill == 1:
                verify_initial(manifest['initial_state'], state)
                verify_physics(manifest['initial_physics_state'], r['jsonable'](env.unwrapped.get_state_dict()))
                result['initial_state_match'] = True
        def end(skill):
            if skill['terminated'] or skill['truncated'] or actor.index == len(plan):
                raise ReplayFinished()
        try:
            collector.collect(args.max_steps if args.max_steps is not None else float('inf'),
                              'task', seed=manifest['seed'], stage='validation',
                              on_skill_start=start, on_skill_end=end)
        except ReplayFinished:
            pass
        skills = [json.loads(line) for line in (out/'skills.jsonl').read_text().splitlines()]
        valid = [s for s in skills if s['completed'] and s['duration_low_steps'] > 0]
        if valid:
            arrays = r['replay_arrays'](valid)
            arrays['duration_low_steps'] = np.asarray([s['duration_low_steps'] for s in valid], dtype=np.int64)
            np.savez_compressed(out/'evaluation.npz', **arrays)
        result.update(status='complete' if len(skills) == len(plan) and skills[-1]['completed'] else 'incomplete',
                      recorded_skills=len(skills), evaluation_rows=len(valid), control_ticks=collector.total_steps)
    except Exception:
        result['status'] = 'failed'
        raise
    finally:
        r['write_json'](out/'manifest.json', result)
        if env is not None: env.close()


def reject_joint_manifest(manifest):
    if manifest.get('control_mode', 'ee') != 'ee':
        raise ValueError('Joint-position and pose recordings cannot be extracted or replayed as compact6 primitive data')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('record', 'extract', 'replay'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--seed', type=int, default=1000000)
    p.add_argument('--max-steps', type=int, help='Optional control-tick cap; omitted means no cap')
    p.add_argument('--speed', type=float, default=.2)
    p.add_argument('--axis-tolerance', type=float, default=.015)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--control-mode', choices=('ee', 'joints', 'pose'), default='ee', help='Record control mode')
    args = p.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        p.error('--max-steps must be positive when supplied')
    if args.seed < 0 or not np.isfinite(args.speed) or not 0 < args.speed <= 1:
        p.error('seed must be nonnegative and speed in (0, 1]')
    if args.stage == 'record' and args.checkpoint is None:
        p.error('record requires --checkpoint')
    if args.stage != 'record' and args.control_mode != 'ee':
        p.error('--control-mode is only supported for record')
    if args.stage == 'record' and args.control_mode in ('joints', 'pose') and args.headless:
        p.error('joint control requires the mouse slider interface; --headless is unsupported')
    r = runtime()
    if args.stage == 'record':
        payload, settings = load_checkpoint(args.checkpoint, r)
        digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        record(args, payload, settings, digest, r)
    else:
        manifest = json.loads((args.output/'demo.json').read_text())
        reject_joint_manifest(manifest)
        checkpoint = args.checkpoint or Path(manifest['checkpoint'])
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != manifest['checkpoint_sha256']:
            raise ValueError('Checkpoint hash differs from recorded source')
        payload, settings = load_checkpoint(checkpoint, r)
        if args.stage == 'extract':
            path = args.output/'plan.json'
            if path.exists(): raise FileExistsError(f'Refusing existing plan: {path}')
            transitions = [json.loads(line) for line in (args.output/'demo_transitions.jsonl').read_text().splitlines()]
            plan = extract_plan(manifest['segments'], r['skill_schema'](settings), args.axis_tolerance, transitions)
            r['write_json'](path, dict(plan=plan, axis_tolerance=args.axis_tolerance, checkpoint_sha256=digest))
        else:
            replay(args, manifest, payload, settings, r)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
