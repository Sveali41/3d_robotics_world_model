"""Fixed FloorPlan1 state and nearest-visible-target action contracts."""
from __future__ import annotations

import numpy as np
import torch

ACTIONS = ('MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown',
           'Crouch', 'Stand', 'Pickup', 'Put', 'Open', 'Close', 'ToggleOn', 'ToggleOff')
CAPABILITIES = ('pickupable', 'receptacle', 'openable', 'toggleable', 'breakable',
                'canFillWithLiquid', 'dirtyable', 'canBeUsedUp', 'cookable', 'sliceable', 'moveable')
DYNAMIC_FLAGS = ('visible', 'isInteractable', 'isPickedUp', 'isOpen', 'isToggled',
                 'isBroken', 'isFilledWithLiquid', 'isDirty', 'isUsedUp', 'isCooked',
                 'isSliced', 'isMoving')
LIQUIDS = (None, 'water', 'coffee', 'wine')
TEMPERATURES = ('RoomTemp', 'Hot', 'Cold')
ACTION_CAPABILITY = {'Pickup': 'pickupable', 'Put': 'receptacle', 'Open': 'openable',
                     'Close': 'openable', 'ToggleOn': 'toggleable', 'ToggleOff': 'toggleable'}


def _xyz(v):
    return [float(v[k]) for k in ('x', 'y', 'z')]


def _angle(v):
    v = np.deg2rad(v)
    return [float(np.sin(v)), float(np.cos(v))]


def _category(value, choices, name):
    if value not in choices:
        raise ValueError(f'Unsupported THOR {name}: {value!r}')
    return [float(value == x) for x in choices]


class ThorState:
    """Encoder preserves object identity, initial static values and field ordering."""

    def __init__(self, metadata):
        objects = sorted(metadata['objects'], key=lambda o: o['objectId'])
        self.ids = tuple(o['objectId'] for o in objects)
        if len(set(self.ids)) != len(self.ids) or not self.ids:
            raise ValueError('Object IDs must be unique and nonempty')
        self.types = tuple(sorted({o['objectType'] for o in objects}))
        self.n = len(objects)
        self.object_dyn_dim = 3 + 6 + len(DYNAMIC_FLAGS) + 1 + len(LIQUIDS) + len(TEMPERATURES) + self.n
        self.object_stat_dim = len(self.types) + self.n + len(CAPABILITIES) + 4
        self.agent_dim = 9
        self.dynamic_dim = self.agent_dim + self.n * self.object_dyn_dim
        self.state_dim = self.dynamic_dim + self.n * self.object_stat_dim
        self._capabilities = np.asarray([[bool(o[k]) for k in CAPABILITIES] for o in objects], dtype=bool)
        static = []
        for i, o in enumerate(objects):
            static += _category(o['objectType'], self.types, 'objectType')
            static += [float(i == j) for j in range(self.n)]
            static += [float(o[k]) for k in CAPABILITIES]
            static += [float(o['mass']), *_xyz(o['axisAlignedBoundingBox']['size'])]
        self.static = np.asarray(static, dtype=np.float32)
        self.encode(metadata)

    def contract(self):
        return {'version': 1, 'object_ids': list(self.ids), 'object_types': list(self.types),
                'agent_dim': self.agent_dim, 'object_dyn_dim': self.object_dyn_dim,
                'object_stat_dim': self.object_stat_dim, 'dynamic_flags': list(DYNAMIC_FLAGS),
                'capabilities': list(CAPABILITIES), 'liquids': list(LIQUIDS),
                'temperatures': list(TEMPERATURES), 'actions': list(ACTIONS),
                'static': self.static.tolist(), 'distance': 'agent-object position euclidean metres',
                'angle': 'sine cosine degrees', 'relations': 'parentReceptacles multi-hot'}

    def encode(self, metadata, *, last_success=None):
        objects = {o['objectId']: o for o in metadata['objects']}
        if set(objects) != set(self.ids) or len(objects) != len(metadata['objects']):
            raise ValueError('THOR object ID set changed; fixed scene contract violated')
        agent = metadata['agent']
        success = metadata['lastActionSuccess'] if last_success is None else last_success
        values = [*_xyz(agent['position']), *_angle(agent['rotation']['y']),
                  *_angle(agent['cameraHorizon']), float(agent['isStanding']), float(success)]
        for name in self.ids:
            obj = objects[name]
            values += _xyz(obj['position'])
            for axis in ('x', 'y', 'z'):
                values += _angle(obj['rotation'][axis])
            values += [float(obj[k]) for k in DYNAMIC_FLAGS]
            values += [float(obj['openness'])]
            values += _category(obj['fillLiquid'], LIQUIDS, 'fillLiquid')
            values += _category(obj['temperature'], TEMPERATURES, 'temperature')
            parents = set(obj['parentReceptacles'] or [])
            if not parents.issubset(self.ids):
                raise ValueError(f'Unknown parentReceptacles: {parents - set(self.ids)}')
            values += [float(x in parents) for x in self.ids]
        result = np.concatenate((np.asarray(values, dtype=np.float32), self.static))
        if result.shape != (self.state_dim,) or not np.isfinite(result).all():
            raise ValueError('Invalid THOR encoded state')
        return result

    def target_indices(self, states, action_indices):
        """Resolve independently for each ensemble and candidate without mutating state."""
        states = torch.as_tensor(states)
        actions = torch.as_tensor(action_indices, device=states.device).expand(states.shape[:-1])
        obj = states[..., self.agent_dim:self.dynamic_dim].reshape(*states.shape[:-1], self.n, self.object_dyn_dim)
        visible = obj[..., 9] >= .5
        held = obj[..., 11] >= .5
        cap = torch.as_tensor(self._capabilities, device=states.device)
        mask = torch.zeros_like(visible)
        for name, capability in ACTION_CAPABILITY.items():
            allowed = visible & cap[:, CAPABILITIES.index(capability)]
            if name == 'Put':
                allowed = allowed & ~held
            mask |= (actions == ACTIONS.index(name)).unsqueeze(-1) & allowed
        distance = ((obj[..., :3] - states[..., None, :3]) ** 2).sum(dim=-1)
        distance = distance.masked_fill(~mask, float('inf'))
        chosen = distance.argmin(dim=-1)
        return torch.where(mask.any(dim=-1), chosen, torch.full_like(chosen, -1))

    def action_encoding(self, states, action_indices):
        states = torch.as_tensor(states)
        actions = torch.as_tensor(action_indices, device=states.device, dtype=torch.long).expand(states.shape[:-1])
        target = self.target_indices(states, actions)
        out = torch.nn.functional.one_hot(actions, len(ACTIONS)).to(dtype=states.dtype)
        target_hot = torch.nn.functional.one_hot(target.clamp_min(0), self.n).to(dtype=states.dtype)
        return torch.cat((out, target_hot * (target >= 0).unsqueeze(-1)), dim=-1), target


def execute(env, encoder, state, action_index):
    """Execute exactly one action attempt; no target is a local failure."""
    name = ACTIONS[int(action_index)]
    encoded, target = encoder.action_encoding(torch.from_numpy(state), int(action_index))
    target_index = int(target.item())
    if name in ACTION_CAPABILITY and target_index < 0:
        next_state = state.copy()
        next_state[8] = 0.0
        return next_state, encoded.numpy(), False, None, 'no_visible_capable_target', False
    args = {}
    if target_index >= 0:
        args = {'objectId': encoder.ids[target_index], 'forceAction': False}
    if name in ('Pickup', 'Put', 'Open', 'Close', 'ToggleOn', 'ToggleOff'):
        name = {'Pickup': 'PickupObject', 'Put': 'PutObject', 'Open': 'OpenObject',
                'Close': 'CloseObject', 'ToggleOn': 'ToggleObjectOn', 'ToggleOff': 'ToggleObjectOff'}[name]
        if name == 'PutObject':
            args['placeStationary'] = True
        if name == 'OpenObject':
            args['openness'] = 1.0
    elif name in ('MoveAhead', 'MoveBack'):
        args['moveMagnitude'] = .25
    elif name in ('RotateLeft', 'RotateRight'):
        args['degrees'] = 90
    elif name in ('LookUp', 'LookDown'):
        args['degrees'] = 30
    event = env.step(name, **args)
    success = bool(event.metadata['lastActionSuccess'])
    return encoder.encode(event.metadata, last_success=success), encoded.numpy(), success, (encoder.ids[target_index] if target_index >= 0 else None), event.metadata['errorMessage'], True
