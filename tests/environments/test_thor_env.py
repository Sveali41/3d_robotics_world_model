"""Recording contracts; these tests do not validate Unity or generated scenes."""

import json
from types import SimpleNamespace

import pytest

from domain.thor.env import ThorEnv, require_success


def test_failed_action_is_recorded_with_actual_before_and_after_state(tmp_path):
    before = SimpleNamespace(metadata={"lastActionSuccess": True, "objects": [{"isOpen": False}]})
    after = SimpleNamespace(metadata={"lastActionSuccess": False, "errorMessage": "too far",
                                      "objects": [{"isOpen": False}]})
    called = []

    def step(**kwargs):
        called.append(kwargs)
        return after

    env = object.__new__(ThorEnv)
    env.output_dir = tmp_path
    env.decision, env.resets = 0, 1
    env.controller = SimpleNamespace(last_event=before, step=step)
    assert env.step("OpenObject", objectId="fridge", forceAction=False) is after
    record = json.loads((tmp_path / "transitions.jsonl").read_text())
    assert called == [{"action": "OpenObject", "objectId": "fridge", "forceAction": False}]
    assert record["state_before"]["objects"] == record["state_after"]["objects"]
    assert record["state_after"]["lastActionSuccess"] is False
    assert record["reset"] == 1


def test_initialization_failure_is_explicit():
    event = SimpleNamespace(metadata={"lastActionSuccess": False, "errorMessage": "bad pose"})
    with pytest.raises(RuntimeError, match="Initial pose: bad pose"):
        require_success(event, "Initial pose")
