from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from domain.robocasa.primitives import (
    AlignYZPrimitive, ApproachXPrimitive, PrimitiveAction, RAPS_SPLIT_ACTION_SCHEMA,
)
from domain.robocasa.skill_execution import RAPSSkillExecution
from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.raps.raps_gnn_actor import (
    DEFAULTS, SCHEMA, RAPSImaginationActor, action_contract, action_contract_matches,
    canonical_action, load, save, skill_schema,
)


def context():
    return SimpleNamespace(
        ee_position=np.array([.35, -.06, .72], dtype=np.float32), gripper_open=1.,
        objects={"cup": SimpleNamespace(position_root=np.zeros(3), position_world=np.zeros(3))},
        finger_contact_force=lambda _: (0., 0.), is_grasping=lambda _: False,
    )


@pytest.mark.parametrize("name,parameters,delta", [
    ("align_yz", dict(y_offset=.04, z_offset=-.03), [0., .04, -.03]),
    ("approach_x", dict(x_forward=.10), [.10, 0., 0.]),
    ("approach_x", dict(x_forward=-.10), [.10, 0., 0.]),
])
@pytest.mark.parametrize("gripper", [-1., 1.])
def test_split_moves_independently_to_relative_goal_without_changing_gripper(name, parameters, delta, gripper):
    ctx = context()
    execution = RAPSSkillExecution(latched_gripper=True, grasp_stop_on_contact=False)
    execution.gripper_command = gripper
    action = PrimitiveAction(name, parameters)
    execution.start(action, ctx)
    initial = ctx.ee_position.copy()
    goal = initial + np.array(delta, dtype=np.float32)
    np.testing.assert_allclose(execution.primitive.goal, goal)
    emitted = []
    for _ in range(160):
        low = execution.next_action(ctx)
        if low is None:
            break
        emitted.append(low.copy())
        assert low[3] == gripper
        execution.record_action(low)
        ctx.ee_position += .10 * low[:3]
    assert emitted
    assert execution.primitive.status == "success"
    np.testing.assert_allclose(ctx.ee_position, goal, atol=.015)
    inactive = [0] if name == "align_yz" else [1, 2]
    np.testing.assert_array_equal(ctx.ee_position[inactive], initial[inactive])
    assert execution.next_action(ctx) is None
    result = execution.primitive.result(ctx, schema=RAPS_SPLIT_ACTION_SCHEMA)
    np.testing.assert_array_equal(result["high_level_action"], action.to_vector(RAPS_SPLIT_ACTION_SCHEMA))


@pytest.mark.parametrize("stop_on_contact", [False, True])
def test_x_contact_stopping_never_transitions_to_closure(stop_on_contact):
    ctx = context()
    execution = RAPSSkillExecution(latched_gripper=True, grasp_stop_on_contact=stop_on_contact)
    execution.start(PrimitiveAction("approach_x", dict(x_forward=.1)), ctx)
    ctx.finger_contact_force = lambda _: (.1, 0.)
    execution.observe(ctx)
    assert not execution.primitive.done
    ctx.finger_contact_force = lambda _: (.1, .1)
    execution.observe(ctx)
    assert execution.primitive.stage == "approach"
    assert execution.primitive.done == stop_on_contact
    low = execution.next_action(ctx)
    if stop_on_contact:
        assert low is None
        assert execution.primitive.status == "contact_stop"
    else:
        assert low[3] == 1.
    assert execution.gripper_command == 1.


def test_split_schema_masks_and_old_checkpoint_contracts_stay_separate():
    schema = skill_schema(dict(skill_set="split7"))
    assert schema is RAPS_SPLIT_ACTION_SCHEMA
    assert schema.primitive_names == ("move_delta_ee_pose", "align_yz", "approach_x", "lift", "drop", "open_gripper", "close_gripper")
    assert schema.action_dim == 13
    raw = torch.ones(2, 13)
    raw[:, :7] = 0
    raw[0, 1] = raw[1, 2] = 1
    canonical = canonical_action(raw, schema)
    assert canonical[0, 7:].tolist() == [0., 0., 0., 0., 1., 1.]
    assert canonical[1, 7:].tolist() == [0., 0., 0., 1., 0., 0.]
    action = PrimitiveAction.from_vector(canonical[0], schema)
    assert action.primitive_type == "align_yz"
    np.testing.assert_array_equal(action.to_vector(schema), canonical[0])
    settings = dict(DEFAULTS, skill_set="split7", latched_gripper=True)
    contract = action_contract(settings)
    assert action_contract_matches(contract, settings)
    assert not action_contract_matches(action_contract(dict(settings, skill_set="compact6")), settings)
    assert not action_contract_matches(contract, dict(settings, skill_set="compact6"))
    assert contract["split_motion"]["automatic_gripper_closure"] is False


def test_split_actor_wm_training_and_checkpoint_roundtrip(tmp_path):
    settings = dict(skill_set="split7", latched_gripper=True, hidden_size=8,
                    actor_layers=1, critic_layers=1, imagination_horizon=3,
                    imagination_batch_size=4, updates_per_round=2)
    backend = CEEUSBackend(agent_dim=33, action_dim=13, hidden_dim=8, horizon=3,
                           candidates=8, iterations=2, elites=4, batch_size=4)
    with pytest.raises(ValueError, match="split7 requires latched_gripper"):
        unlatched_backend = SimpleNamespace(env=SimpleNamespace(agent_dim=32))
        RAPSImaginationActor(unlatched_backend, dict(settings, latched_gripper=False))
    actor = RAPSImaginationActor(backend, settings)
    states = np.random.default_rng(4).normal(size=(8, 67)).astype(np.float32)
    next_states = states.copy()
    next_states[:, :57] += .001
    raw = torch.zeros(8, 13)
    raw[torch.arange(8), torch.arange(8) % 7] = 1
    raw[:, 7:] = .5
    actions = canonical_action(raw, actor.schema).numpy()
    replay = dict(states=states, next_states=next_states, actions=actions, terminated=np.zeros(8, bool))
    backend.train(replay, epochs=1)
    assert actor.imagine(torch.tensor(states[:2]))["actions"].shape == (3, 2, 13)
    assert actor.train(replay)["total_updates"] == 2
    rows = [dict(state=states[i], next_state=next_states[i], action=actions[i],
                 episode_id=1, terminated=False, truncated=False, trainable=True, duration_low_steps=3)
            for i in range(8)]
    encoder = SimpleNamespace(contract=lambda: {"dimension": 67})
    path = tmp_path / "split.pt"
    save(path, backend, actor, encoder, OmegaConf.create({}),
         [{**r, "action": np.zeros(4)} for r in rows], rows)
    expected = actor.action(states[0])
    load(path, backend, actor, encoder)
    np.testing.assert_array_equal(actor.action(states[0]), expected)
    torch.save(dict(schema=SCHEMA, state_contract=encoder.contract(),
                    action_contract=action_contract(dict(DEFAULTS, skill_set="compact6", latched_gripper=True))), path)
    with pytest.raises(ValueError, match="contract differs"):
        load(path, backend, actor, encoder)
