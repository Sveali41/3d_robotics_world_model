"""Interface contracts; these tests do not substitute for a Habitat smoke run."""

import json
from types import SimpleNamespace

import pytest

from domain.household.cli import ci_demo_actions, parse_action
from domain.household.partnr import HouseholdEnv, graph_snapshot, skill_argument


def test_place_uses_upstream_five_argument_format():
    assert skill_argument("Place", "cup_0", "cabinet_1", "within") == (
        "cup_0,within,cabinet_1,none,none"
    )
    assert parse_action("Place cup_0 counter_1") == {
        "skill": "Place", "target": "cup_0", "destination": "counter_1", "relation": "on"
    }


@pytest.mark.parametrize("command", ["Pick", "Pick cup_0 extra", "Place cup_0",
                                         "Place cup_0 table_0 inside", "Teleport room_0"])
def test_invalid_commands_are_rejected(command):
    with pytest.raises(ValueError):
        parse_action(command)


def test_graph_keeps_directed_relations_and_omits_region_handle():
    class Vector:
        def __getitem__(self, index):
            return (1.0, 2.0, 3.0)[index]

    class Node:
        def __init__(self, name):
            self.name, self.sim_handle = name, name + "_handle"
            self.properties = {"type": "test", "region": object(), "translation": Vector()}

    cup, table = Node("cup"), Node("table")
    table.properties["last_held_object"] = cup
    graph = SimpleNamespace(graph={table: {cup: "under"}, cup: {table: "on"}})
    snapshot = graph_snapshot(graph)
    assert snapshot["edges"] == [
        {"source": "cup", "target": "table", "relation": "on"},
        {"source": "table", "target": "cup", "relation": "under"},
    ]
    assert all("region" not in node["properties"] for node in snapshot["nodes"])
    assert snapshot["nodes"][0]["properties"]["translation"] == [1.0, 2.0, 3.0]
    assert snapshot["nodes"][1]["properties"]["last_held_object"] == {
        "name": "cup", "sim_handle": "cup_handle"
    }
    json.dumps(snapshot)


def fake_env(tmp_path, responses, *, done_at=None, max_steps=10, no_actions=False):
    env = object.__new__(HouseholdEnv)
    env.output_dir = tmp_path
    env.max_skill_steps = max_steps
    env.save_video = False
    env.frame_callback = None
    env._decision, env._done = 0, False
    observed, calls, resets = [], [], []
    graph = SimpleNamespace(get_node_from_name=lambda name: name)
    interface = SimpleNamespace(full_world_graph=graph, batch={"step": 0})
    interface.get_observations = lambda: interface.batch
    interface.parse_observations = lambda obs: obs
    interface.filter_obs_space = lambda obs, uid: obs

    def step(actions):
        calls.append(actions)
        count = len(calls)
        return {"step": count}, 0, count == done_at, {}

    interface.step = step
    env.interface = interface
    response_iter = iter(responses)

    def dispatch(skill, argument, observations):
        observed.append(observations["step"])
        return (None if no_actions else "control"), next(response_iter)

    env.agent = SimpleNamespace(
        process_high_level_action=dispatch,
        reset=lambda: resets.append(True),
    )
    env.state = lambda: {"step": len(calls)}
    return env, observed, calls, resets


def test_transition_and_next_skill_use_current_observation(tmp_path):
    env, observed, calls, _ = fake_env(tmp_path, ["", "Successful execution!", "Successful execution!"])
    first = env.execute_skill("Pick", "cup")
    second = env.execute_skill("Navigate", "table")
    assert observed == [0, 1, 2]
    assert first["state_before"] == {"step": 0}
    assert first["state_after"] == second["state_before"] == {"step": 2}
    assert first["control_steps"] == 2 and len(calls) == 3
    assert first["success"] and second["success"]
    rows = [json.loads(line) for line in (tmp_path / "transitions.jsonl").read_text().splitlines()]
    assert [row["decision"] for row in rows] == [0, 1]


def test_skill_failure_is_a_recorded_transition(tmp_path):
    env, _, calls, _ = fake_env(tmp_path, ["Unexpected failure! - Not close enough"])
    result = env.execute_skill("Pick", "cup")
    assert not result["success"] and not result["timed_out"]
    assert len(calls) == 1
    assert (tmp_path / "transitions.jsonl").exists()


def test_timeout_counts_real_steps_and_clears_skill_state(tmp_path):
    env, _, calls, resets = fake_env(tmp_path, ["", ""], max_steps=2)
    result = env.execute_skill("Navigate", "table")
    assert result["timed_out"] and not result["success"]
    assert result["control_steps"] == len(calls) == 2
    assert resets == [True]


def test_episode_boundary_prevents_further_actions(tmp_path):
    env, _, calls, _ = fake_env(tmp_path, [""], done_at=1)
    result = env.execute_skill("Navigate", "table")
    assert result["episode_done"] and not result["timed_out"]
    with pytest.raises(RuntimeError, match="Episode ended"):
        env.execute_skill("Pick", "cup")
    assert len(calls) == 1


def test_infrastructure_failure_is_not_a_training_label(tmp_path):
    env, _, _, resets = fake_env(tmp_path, [""], no_actions=True)
    with pytest.raises(RuntimeError, match="neither"):
        env.execute_skill("Pick", "cup")
    assert not (tmp_path / "transitions.jsonl").exists()
    assert env._done and resets == [True]


def test_ci_demo_does_not_invent_entity_names():
    with pytest.raises(ValueError, match="lacks"):
        ci_demo_actions({"nodes": []})


def test_render_receives_every_real_step_without_advancing_simulator(tmp_path):
    env, _, calls, _ = fake_env(tmp_path, ["", "Successful execution!"])
    frames = []
    env.frame_callback = lambda observations, actions: frames.append(observations["step"])
    result = env.execute_skill("Navigate", "table")
    assert frames == [1, 2]
    assert len(calls) == result["control_steps"] == 2


def test_closing_render_stops_skill_without_writing_a_training_label(tmp_path):
    env, _, calls, resets = fake_env(tmp_path, [""])

    def close_window(*_):
        raise InterruptedError("window closed")

    env.frame_callback = close_window
    with pytest.raises(InterruptedError, match="window closed"):
        env.execute_skill("Navigate", "table")
    assert len(calls) == 1 and resets == [True] and env._done
    assert not (tmp_path / "transitions.jsonl").exists()
