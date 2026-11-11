"""Metadata-state coverage accounting for the fixed THOR scene contract."""
from __future__ import annotations

import numpy as np

from exploration.cee_us.thor_state import CAPABILITIES, DYNAMIC_FLAGS


class InstanceCoverage:
    """Count attempted and observed effects for fixed object-instance requirements.

    This class only compares encoded states and never interacts with a simulator.
    """

    def __init__(self, encoder, initial_metadata, min_count=1):
        if min_count < 1:
            raise ValueError('min_count must be at least 1')
        self.encoder = encoder
        self.min_count = int(min_count)
        self._index = {name: i for i, name in enumerate(encoder.ids)}
        objects = {o['objectId']: o for o in initial_metadata['objects']}
        if set(objects) != set(encoder.ids) or len(objects) != len(initial_metadata['objects']):
            raise ValueError('Initial metadata object IDs must match the encoder')
        self._requirements = []
        for object_id in encoder.ids:
            obj = objects[object_id]
            capabilities = {name: bool(obj[name]) for name in CAPABILITIES}
            if capabilities['openable']:
                self._add(object_id, 'Open', 'isOpen: False -> True')
                self._add(object_id, 'Close', 'isOpen: True -> False')
            if capabilities['toggleable']:
                self._add(object_id, 'ToggleOn', 'isToggled: False -> True')
                self._add(object_id, 'ToggleOff', 'isToggled: True -> False')
            if capabilities['pickupable']:
                self._add(object_id, 'Pickup', 'isPickedUp: False -> True')
                self._add(object_id, 'Release', 'isPickedUp: True -> False via Put')
            if capabilities['receptacle']:
                self._add(object_id, 'Put', 'held object gains this parentReceptacles entry')

        self._add(None, 'MoveAhead', 'success with position movement > 1e-4 m')
        self._add(None, 'MoveAhead', 'failure with no position movement <= 1e-4 m')
        self._add(None, 'MoveBack', 'success with position movement > 1e-4 m')
        self._add(None, 'MoveBack', 'failure with no position movement <= 1e-4 m')
        for action in ('RotateLeft', 'RotateRight'):
            self._add(None, action, 'success with yaw sine/cosine change')
        for action in ('LookUp', 'LookDown'):
            self._add(None, action, 'success with horizon sine/cosine change')
        self._add(None, 'Crouch', 'success: standing 1 -> 0')
        self._add(None, 'Stand', 'success: standing 0 -> 1')

    def _add(self, object_id, action, effect):
        if object_id is None:
            readable = f'navigation:{action}:{effect}'
        else:
            readable = f'{action}:{object_id}'
        self._requirements.append({
            'id': readable,
            'key': (action, object_id),
            'object_id': object_id,
            'action': action,
            'effect': effect,
            'attempts': 0,
            'count': 0,
        })

    def _object_field(self, state, object_id, field):
        obj_index = self._index[object_id]
        base = self.encoder.agent_dim + obj_index * self.encoder.object_dyn_dim
        if field == 'position':
            return np.asarray(state[base:base + 3], dtype=float)
        flag = DYNAMIC_FLAGS.index(field)
        return float(state[base + 9 + flag]) >= 0.5

    def _parents(self, state, object_id):
        obj_index = self._index[object_id]
        base = self.encoder.agent_dim + obj_index * self.encoder.object_dyn_dim
        parent_base = base + 3 + 6 + len(DYNAMIC_FLAGS) + 1 + 4 + 3
        return {name for i, name in enumerate(self.encoder.ids)
                if state[parent_base + i] >= 0.5}

    @staticmethod
    def _changed(before, after):
        return not np.array_equal(before, after)

    def observe(self, state, next_state, action_name, target_id_or_None, success):
        """Record one action attempt and any matching, successful state effect."""
        before = np.asarray(state)
        after = np.asarray(next_state)
        if before.shape != (self.encoder.state_dim,) or after.shape != before.shape:
            raise ValueError(f'States must have shape ({self.encoder.state_dim},)')
        if target_id_or_None is not None and target_id_or_None not in self._index:
            raise ValueError(f'Unknown THOR target object ID: {target_id_or_None!r}')
        succeeded = bool(success)
        held_before = [name for name in self.encoder.ids
                       if self._object_field(before, name, 'isPickedUp')]

        for requirement in self._requirements:
            action = requirement['action']
            object_id = requirement['object_id']
            effect = requirement['effect']
            match = False
            changed = False

            if action == 'Release':
                # Put targets the receptacle; the released instance is the held source.
                source = object_id
                match = (action_name == 'Put' and source in held_before)
                changed = match and not self._object_field(after, source, 'isPickedUp')
            elif object_id is not None:
                match = action_name == action and target_id_or_None == object_id
                if match and action in ('Open', 'Close'):
                    old = self._object_field(before, object_id, 'isOpen')
                    new = self._object_field(after, object_id, 'isOpen')
                    changed = (not old and new) if action == 'Open' else (old and not new)
                elif match and action in ('ToggleOn', 'ToggleOff'):
                    old = self._object_field(before, object_id, 'isToggled')
                    new = self._object_field(after, object_id, 'isToggled')
                    changed = (not old and new) if action == 'ToggleOn' else (old and not new)
                elif match and action == 'Pickup':
                    changed = (not self._object_field(before, object_id, 'isPickedUp')
                               and self._object_field(after, object_id, 'isPickedUp'))
                elif match and action == 'Put':
                    changed = any(
                        source in held_before
                        and not self._object_field(after, source, 'isPickedUp')
                        and object_id not in self._parents(before, source)
                        and object_id in self._parents(after, source)
                        for source in self.encoder.ids)
            else:
                match = action_name == action
                if match and action in ('MoveAhead', 'MoveBack'):
                    distance = float(np.linalg.norm(after[:3] - before[:3]))
                    changed = ((succeeded and distance > 1e-4)
                               if 'success' in effect else (not succeeded and distance <= 1e-4))
                elif match and action in ('RotateLeft', 'RotateRight') and succeeded:
                    changed = self._changed(before[3:5], after[3:5])
                elif match and action in ('LookUp', 'LookDown') and succeeded:
                    changed = self._changed(before[5:7], after[5:7])
                elif match and action == 'Crouch' and succeeded:
                    changed = before[7] >= 0.5 and after[7] < 0.5
                elif match and action == 'Stand' and succeeded:
                    changed = before[7] < 0.5 and after[7] >= 0.5

            if match:
                requirement['attempts'] += 1
                blocked_navigation = (action_name in ('MoveAhead', 'MoveBack')
                                     and 'failure' in effect)
                qualifying = changed and (not succeeded if blocked_navigation else succeeded)
                if qualifying:
                    requirement['count'] += 1

    def report(self):
        requirements = []
        for item in self._requirements:
            requirement = {key: value for key, value in item.items() if key != 'key'}
            requirement['covered'] = requirement['count'] >= self.min_count
            requirements.append(requirement)
        missing = [item['id'] for item in requirements if not item['covered']]
        covered = len(requirements) - len(missing)
        return {
            'requirements': requirements,
            'total': len(requirements),
            'covered': covered,
            'complete': not missing,
            'missing': missing,
            'min_count': self.min_count,
        }
