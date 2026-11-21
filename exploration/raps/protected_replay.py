"""Protected replay sampling for new and historical robot transitions."""

import numpy as np


_STRATA = ("dynamic", "contact", "static")
_STRATUM_WEIGHTS = np.array([0.50, 0.25, 0.25], dtype=np.float64)


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _largest_remainder(total, weights):
    """Apportion total integer slots; ties follow dynamic/contact/static order."""
    if total == 0:
        return np.zeros(len(weights), dtype=np.int64)
    scaled = total * weights / weights.sum()
    quotas = np.floor(scaled).astype(np.int64)
    remainder = total - int(quotas.sum())
    order = sorted(range(len(weights)), key=lambda i: (-(scaled[i] - quotas[i]), i))
    for index in order[:remainder]:
        quotas[index] += 1
    return quotas


def _diversity_bins(states, actions, *, z_bin_m, joint_bin_rad):
    branch = np.argmax(actions[:, :-1], axis=1)
    z = 0.732 + 0.04 * actions[:, -1]
    z_bin = np.floor(z / z_bin_m).astype(np.int64)
    joints = np.floor(states[:, :7] / joint_bin_rad).astype(np.int64)
    return [tuple([int(branch[i]), int(z_bin[i]), *joints[i].tolist()])
            for i in range(len(states))]


def _group_by_bins(pool_indices, bins):
    occupied = {}
    for index in pool_indices:
        occupied.setdefault(bins[int(index)], []).append(int(index))
    return occupied


def _draw_balanced(occupied, count):
    if count == 0:
        return np.empty(0, dtype=np.int64)
    keys = list(occupied)
    selected_bins = np.random.randint(0, len(keys), size=count)
    return np.fromiter((occupied[keys[int(bin_index)]][np.random.randint(len(occupied[keys[int(bin_index)]]))]
                        for bin_index in selected_bins), dtype=np.int64, count=count)


def bounded_dynamic_replay_indices(states, next_states, *, agent_dim, object_dyn_dim,
                                   num_objects, updates, batch_size, movement_threshold_m=0.015):
    """Draw individual records with bounded upweighting of object movement."""
    states, next_states = np.asarray(states), np.asarray(next_states)
    if states.ndim != 2 or next_states.shape != states.shape or len(states) == 0:
        raise ValueError('states and next_states must be nonempty matching rank-2 arrays')
    for name, values in (('states', states), ('next_states', next_states)):
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f'{name} must contain only finite numeric values')
    agent_dim = _positive_int(agent_dim, 'agent_dim')
    object_dyn_dim = _positive_int(object_dyn_dim, 'object_dyn_dim')
    num_objects = _positive_int(num_objects, 'num_objects')
    updates = _positive_int(updates, 'updates')
    batch_size = _positive_int(batch_size, 'batch_size')
    movement_threshold_m = _positive_finite(movement_threshold_m, 'movement_threshold_m')
    if object_dyn_dim < 3 or states.shape[1] < agent_dim + num_objects * object_dyn_dim:
        raise ValueError('states do not contain the requested object XYZ dynamics')
    moving = np.zeros(len(states), dtype=bool)
    for obj in range(num_objects):
        start = agent_dim + obj * object_dyn_dim
        moving |= np.linalg.norm(next_states[:, start:start + 3] - states[:, start:start + 3], axis=1) >= movement_threshold_m
    moving_count = int(moving.sum())
    static_count = len(states) - moving_count
    moving_weight = min(4.0, max(1.0, static_count / moving_count)) if moving_count else 1.0
    weights = np.where(moving, moving_weight, 1.0)
    indices = np.random.choice(len(states), size=updates * batch_size, replace=True, p=weights / weights.sum())
    stats = dict(mode='bounded_dynamic', num_records=len(states), num_samples=len(indices),
                 updates=updates, batch_size=batch_size, moving_records=moving_count,
                 static_records=static_count, moving_weight=moving_weight,
                 sampled_moving=int(moving[indices].sum()),
                 movement_threshold_m=movement_threshold_m)
    return indices, stats


def protected_replay_indices(states, actions, next_states, *, agent_dim, object_dyn_dim,
                             num_objects, new_mask, contact_mask, updates, batch_size,
                             movement_threshold_m=0.015, z_bin_m=0.005,
                             joint_bin_rad=np.pi / 36):
    """Sample indices with replacement while preserving fresh data and rare outcomes.

    Each update assigns ``ceil(batch_size / 2)`` slots to new data and the rest
    to history; unavailable sources donate their quota to the available source.
    Historical quota is split 50/25/25 across dynamic/contact/static outcomes,
    with absent strata removed and the remaining weights renormalized. Largest
    remainder ties use dynamic, then contact, then static order.

    Within each source or historical outcome, occupied action/Z/joint bins are
    selected uniformly, then a record is selected uniformly within that bin.
    The NumPy global RNG is used so its existing checkpoint state fully
    determines replay draws. Returned indices refer to original rows; no rows
    are removed or rewritten.
    """
    states = np.asarray(states)
    actions = np.asarray(actions)
    next_states = np.asarray(next_states)
    if states.ndim != 2 or actions.ndim != 2 or next_states.ndim != 2:
        raise ValueError("states, actions, and next_states must be rank-2 arrays")
    n = states.shape[0]
    if actions.shape[0] != n or next_states.shape != states.shape:
        raise ValueError("states, actions, and next_states must have matching record counts and state dimensions")
    if actions.shape[1] < 2:
        raise ValueError("actions must contain at least one categorical value and one final Z value")
    agent_dim = _positive_int(agent_dim, "agent_dim")
    object_dyn_dim = _positive_int(object_dyn_dim, "object_dyn_dim")
    num_objects = _positive_int(num_objects, "num_objects")
    updates = _positive_int(updates, "updates")
    batch_size = _positive_int(batch_size, "batch_size")
    movement_threshold_m = _positive_finite(movement_threshold_m, "movement_threshold_m")
    z_bin_m = _positive_finite(z_bin_m, "z_bin_m")
    joint_bin_rad = _positive_finite(joint_bin_rad, "joint_bin_rad")
    if agent_dim < 7 or object_dyn_dim < 3:
        raise ValueError("agent_dim must include seven arm joints and object_dyn_dim must include xyz")
    if states.shape[1] < agent_dim + object_dyn_dim * num_objects:
        raise ValueError("states do not contain the requested agent and object dynamics dimensions")
    for name, values in (("states", states), ("actions", actions), ("next_states", next_states)):
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite numeric values")
    new_mask = np.asarray(new_mask)
    contact_mask = np.asarray(contact_mask)
    if new_mask.dtype != np.bool_ or new_mask.shape != (n,):
        raise ValueError(f"new_mask must be a boolean vector with shape ({n},)")
    if contact_mask.dtype != np.bool_ or contact_mask.shape != (n,):
        raise ValueError(f"contact_mask must be a boolean vector with shape ({n},)")
    if n == 0:
        raise ValueError("at least one replay record is required")

    dynamic = np.zeros(n, dtype=np.bool_)
    for obj in range(num_objects):
        start = agent_dim + obj * object_dyn_dim
        displacement = np.linalg.norm(next_states[:, start:start + 3] - states[:, start:start + 3], axis=1)
        dynamic |= displacement >= movement_threshold_m
    labels = np.full(n, 2, dtype=np.int8)  # static
    labels[contact_mask] = 1
    labels[dynamic] = 0  # movement has precedence over contact

    bins = _diversity_bins(states, actions, z_bin_m=z_bin_m, joint_bin_rad=joint_bin_rad)
    new_pool = np.flatnonzero(new_mask)
    history_pool = np.flatnonzero(~new_mask)
    new_groups = _group_by_bins(new_pool, bins)
    history_groups = {
        stratum_index: _group_by_bins(history_pool[labels[history_pool] == stratum_index], bins)
        for stratum_index in range(len(_STRATA))
    }
    total = updates * batch_size
    sampled = np.empty(total, dtype=np.int64)
    source_counts = {"new": 0, "history": 0}
    stratum_counts = {name: 0 for name in _STRATA}
    per_source_bins = {"new": set(), "history": set()}
    per_stratum_bins = {name: set() for name in _STRATA}
    new_requested_per_update = (batch_size + 1) // 2

    offset = 0
    for _ in range(updates):
        new_quota = new_requested_per_update
        history_quota = batch_size - new_quota
        if len(history_pool) == 0:
            new_quota, history_quota = batch_size, 0
        elif len(new_pool) == 0:
            new_quota, history_quota = 0, batch_size

        drawn_new = _draw_balanced(new_groups, new_quota)
        sampled[offset:offset + new_quota] = drawn_new
        offset += new_quota
        source_counts["new"] += new_quota
        per_source_bins["new"].update(bins[int(i)] for i in drawn_new)

        if history_quota:
            active = np.array([bool(history_groups[index]) for index in range(3)])
            weights = _STRATUM_WEIGHTS * active
            quotas = _largest_remainder(history_quota, weights)
            for stratum_index, quota in enumerate(quotas):
                if quota == 0:
                    continue
                name = _STRATA[stratum_index]
                drawn = _draw_balanced(history_groups[stratum_index], int(quota))
                sampled[offset:offset + quota] = drawn
                offset += quota
                source_counts["history"] += int(quota)
                stratum_counts[name] += int(quota)
                per_source_bins["history"].update(bins[int(i)] for i in drawn)
                per_stratum_bins[name].update(bins[int(i)] for i in drawn)

    stats = {
        "num_records": int(n),
        "num_samples": int(total),
        "updates": int(updates),
        "batch_size": int(batch_size),
        "source_counts": {key: int(value) for key, value in source_counts.items()},
        "historical_stratum_counts": {key: int(value) for key, value in stratum_counts.items()},
        "occupied_bins": {
            "new": int(len(new_groups)),
            "history": int(len(set().union(*(set(groups) for groups in history_groups.values())))),
            "sampled_new": int(len(per_source_bins["new"])),
            "sampled_history": int(len(per_source_bins["history"])),
            "sampled_history_by_stratum": {name: int(len(values)) for name, values in per_stratum_bins.items()},
        },
        "parameters": {
            "movement_threshold_m": movement_threshold_m,
            "z_bin_m": z_bin_m,
            "joint_bin_rad": joint_bin_rad,
            "new_fraction_per_update": "ceil(batch_size / 2) slots, approximately 50%",
            "historical_stratum_weights": {name: float(weight) for name, weight in zip(_STRATA, _STRATUM_WEIGHTS)},
            "dynamic_precedence": "dynamic when any object xyz displacement meets threshold; else contact; else static",
            "z_definition": "0.732 + 0.04 * final action component, floor-binned in meters",
            "joint_definition": "first seven state components, floor-binned in radians",
        },
        "ratio_scope": "counts describe the complete sampled stream; upstream independently shuffles rows, so individual optimizer minibatches need not match these ratios",
    }
    return sampled, stats
