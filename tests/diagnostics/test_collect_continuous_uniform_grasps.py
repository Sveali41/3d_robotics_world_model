"""Continuous export preserves measured episode boundaries across fresh scenes."""
import numpy as np

from exploration.diagnostics.collect_continuous_uniform_grasps import merge_block
from exploration.diagnostics.collect_uniform_grasps import replay_from_outcomes


def test_merge_preserves_continuation_and_isolates_streams_and_blocks():
    outcomes = []
    def row(case, local):
        return dict(case_id=case, local_episode_id=local)
    episode = merge_block(outcomes, dict(outcomes=[row(0, 1), row(2, 1), row(4, 2)]), 0)
    episode = merge_block(outcomes, dict(outcomes=[row(1, 1), row(3, 1)]), episode)
    episode = merge_block(outcomes, dict(outcomes=[row(6, 1)]), episode)
    assert episode == 4
    assert [r['episode_id'] for r in outcomes] == [1, 1, 2, 3, 3, 4]


def test_replay_preserves_continuous_episode_and_pre_reset_endpoint():
    rows=[]
    for case,episode,state,next_state in [(10, 1, 0., 1.), (11, 1, 1., 2.), (12, 2, 0., 3.)]:
        rows.append(dict(case_id=case,episode_id=episode,cell_id=4,orientation_id=2,z_m=.742,
            wm_eligible=True,skill=dict(state=np.full(50,state),next_state=np.full(50,next_state),
                action=np.zeros(31),terminated=False,truncated=False,duration_low_steps=10)))
    replay=replay_from_outcomes(rows)
    assert replay['episode_ids'].tolist()==[1,1,2]
    np.testing.assert_array_equal(replay['next_states'][0],replay['states'][1])
    assert replay['next_states'][1,0]==2. and replay['states'][2,0]==0.


def test_noncanonical_plan_is_rejected_before_simulation(tmp_path):
    import json
    import pytest
    from omegaconf import OmegaConf
    from exploration.diagnostics.collect_uniform_grasps import uniform_cases
    from exploration.diagnostics.collect_continuous_uniform_grasps import main
    config=OmegaConf.load('configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml')
    case=uniform_cases(config.raps_actor,z_levels=17)[0]
    case['cell_id']=4
    path=tmp_path/'bad.jsonl'
    path.write_text(json.dumps(case)+'\n')
    with pytest.raises(ValueError,match='canonical'):
        main(['--cases',str(path),'--output',str(tmp_path/'output')])
    assert not (tmp_path/'output').exists()


def test_cell4_restriction_filters_proposals_before_any_simulation(monkeypatch,tmp_path):
    import json
    import pytest
    from omegaconf import OmegaConf
    from exploration.diagnostics import collect_continuous_uniform_grasps as module
    from exploration.diagnostics.collect_uniform_grasps import uniform_cases
    config=OmegaConf.load('configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml')
    cases=uniform_cases(config.raps_actor,z_levels=17)
    path=tmp_path/'all_cases.jsonl'
    path.write_text(''.join(json.dumps(c)+'\n' for c in cases))
    class BeforeSimulation(Exception):pass
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,fn,jobs):
            jobs=list(jobs)
            assert len(jobs)==2
            assert sum(len(j['cases']) for j in jobs)==51
            assert all(c['cell_id']==4 for j in jobs for c in j['cases'])
            raise BeforeSimulation()
    monkeypatch.setattr(module,'ProcessPoolExecutor',Pool)
    out=tmp_path/'data'
    with pytest.raises(BeforeSimulation):
        module.main(['--cases',str(path),'--output',str(out),'--cell-id','4'])
    selected=[json.loads(s) for s in (out/'cases.jsonl').read_text().splitlines()]
    assert len(selected)==51 and {c['cell_id'] for c in selected}=={4}
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['requested_target_cell_id']==4 and manifest['target_cell_ids']==[4]
