"""Offline fitting keeps provenance explicit and monitoring out of normalizers."""
import json

import pytest
import torch

from exploration.cee_us.cee_us_thor import file_hash, save_dataset
from exploration.cee_us.thor_state import ThorState
from scripts.fit_thor_offline import main
from tests.cee_us.test_thor_training import replay
from tests.cee_us.test_thor_world_model import metadata


@pytest.mark.parametrize('gns', [False, True])
@pytest.mark.parametrize('prediction_mode', ['mixed', 'mixed-relations'])
def test_offline_fit_has_distinct_checkpoint_schema_and_training_only_normalization(
        tmp_path, gns, prediction_mode):
    initial = metadata()
    encoder = ThorState(initial)
    train = replay(encoder, initial)
    monitor = replay(encoder, initial)
    monitor['states'][:, 0] = 99
    monitor['next_states'][:, 0] = 99
    dataset = tmp_path / 'train.npz'
    monitoring = tmp_path / 'monitor.npz'
    save_dataset(dataset, encoder, initial, train)
    save_dataset(monitoring, encoder, initial, monitor)
    original_hash = file_hash(dataset)
    output = tmp_path / 'fit'
    assert main(['--data', str(dataset), '--monitor-data', str(monitoring),
                 '--output', str(output), '--epochs', '1', '--hidden-dim', '8', '--device', 'cpu',
                 '--prediction-mode', prediction_mode] + (['--gns', '--gns-radius', '0.5'] if gns else [])) == 0
    history = json.loads((output / 'history.json').read_text())
    assert [r['epoch'] for r in history] == [0, 1]
    assert history[1]['fit']['finitegrad']
    report = json.loads((output / 'full_fitting_report.json').read_text())
    assert report['conditional_one_step']['count'] == 6
    checkpoint = torch.load(output / 'checkpoint.pt', weights_only=False)
    assert checkpoint['schema'] == 'thor-offline-fit-v1'
    assert checkpoint['prediction_mode'] == prediction_mode
    assert checkpoint['graph'] == {'mode': 'gns' if gns else 'full', 'radius': .5 if gns else None}
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['graph'] == checkpoint['graph']
    assert manifest['prediction_mode'] == prediction_mode
    if prediction_mode == 'mixed-relations':
        assert 'separate relation head' in manifest['prediction_objective']
        assert 'excluding self targets' in manifest['prediction_objective']
    else:
        assert 'separate relation head' not in manifest['prediction_objective']
    assert 'not held-out' in checkpoint['purpose']
    assert checkpoint['backend']['normalizers']['input_normalizer_agent']['mean'][0, 0] == 0
    assert file_hash(dataset) == original_hash
