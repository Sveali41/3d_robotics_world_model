import numpy as np
import pytest
from omegaconf import OmegaConf

from exploration.diagnostics.profile_cup_grasp import CONFIG, validate_config, validate_replay


def test_historical_five_step_profile_configuration_and_required_budget():
    config = OmegaConf.load(CONFIG)
    config.cabinet_door = True
    config.raps_actor.imagination_horizon = 5
    config.raps_actor.discount = .8
    validate_config(config)
    config.raps_actor.updates_per_round = 101
    with pytest.raises(ValueError, match='updates_per_round'):
        validate_config(config)


def test_replay_requires_two_real_interface_rows():
    replay = dict(states=np.zeros((2, 50)), next_states=np.zeros((2, 50)),
                  actions=np.zeros((2, 31)), episode_ids=np.zeros(2),
                  terminated=np.zeros(2), truncated=np.zeros(2))
    validate_replay(replay)
    replay['actions'] = np.zeros((1, 31))
    with pytest.raises(ValueError, match='actions'):
        validate_replay(replay)
