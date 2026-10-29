"""Actor-critic warm-start ablation with a shared old WM and old/new replay.

Explicit compare starts training. Smoke is capped at 200 physical control steps.
No original baseline configuration or strict resume behavior is replaced.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from exploration.cee_us_backend import CEEUSBackend
from exploration.cee_us_cabinet import ROOT, make_env, prepare_output, rng_state, restore_rng
from exploration.cee_us_state import CabinetSkillStateEncoder, OBJECT_NAMES
from exploration.diagnose_cabinet import write_json
from exploration.diagnostic_metrics import motion_bucket
from exploration.raps_gnn_actor import (
    DEFAULTS, SCHEMA, RAPSImaginationActor, SkillCollector, action_contract, load, save, skill_arrays,
)

SOURCE = ROOT / 'logs/raps_compact6_train_400k_episode40s_centered_20261026T173840/curiosity/round_398.pt'
CONFIG = ROOT / 'configs/raps_gnn_actor_cabinet_400k_episode40s_centered.yaml'


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def initialize(config, payload, encoder, reuse_policy):
    """Only policy learner state differs; WM including Adam is identical."""
    seed_all(int(config.seed))
    backend = CEEUSBackend(agent_dim=33, action_dim=12, device=str(config.device),
                           seed=int(config.seed), object_heuristic=OmegaConf.to_container(config.object_heuristic))
    actor = RAPSImaginationActor(backend, OmegaConf.to_container(config.raps_actor))
    if payload.get('schema') != SCHEMA or payload['state_contract'] != encoder.contract():
        raise ValueError('Source checkpoint state/schema does not match the two-object scene')
    if payload['action_contract'] != action_contract(actor.settings):
        raise ValueError('Source action/controller/reward contract differs')
    backend.load_state_dict(deepcopy(payload['backend']), restore_planner=False)
    if reuse_policy: actor.load_state_dict(deepcopy(payload['actor']))
    return backend, actor


def load_old_replay(path):
    with np.load(path) as data: replay = {key: data[key].copy() for key in data.files}
    if replay['states'].ndim != 2 or replay['states'].shape[1] != 67 or replay['actions'].shape != (len(replay['states']), 12):
        raise ValueError('Expected 67D complete-skill states and 12D primitive actions')
    if len(replay['states']) < 2 or not np.isfinite(replay['states']).all() or not np.isfinite(replay['next_states']).all() or not np.isfinite(replay['actions']).all():
        raise ValueError('Invalid old replay')
    if not np.array_equal(replay['states'][:,57:], replay['next_states'][:,57:]):
        raise ValueError('Old replay static attributes changed')
    return replay


def balanced_arrays(old, new, rng):
    """Fixed 500-sample virtual dataset; keep full replay separately."""
    old_ids = rng.choice(len(old['states']), 250, replace=len(old['states']) < 250)
    new_ids = rng.choice(len(new['states']), 250, replace=len(new['states']) < 250)
    keys = ('states','actions','next_states','terminated')
    wm = {key: np.concatenate([old[key][old_ids], new[key][new_ids]]) for key in keys}
    # Actor starts use the full pools, with exactly 32+32 per imagined batch.
    starts = {key: np.concatenate([old[key], new[key]]) for key in ('states','terminated')}
    labels = np.r_[np.zeros(len(old['states']),dtype=np.int64),np.ones(len(new['states']),dtype=np.int64)]
    return wm, starts, labels, old_ids, new_ids


def update(backend, actor, old, skills, directory, label, sampler, smoke=False):
    new = skill_arrays(skills)
    wm, starts, groups, old_ids, new_ids = balanced_arrays(old,new,sampler)
    began=time.perf_counter()
    model = backend.train(wm,epochs=1 if smoke else 25)
    expected_updates = 4 if smoke else 100
    if model['updates'] != expected_updates:
        raise AssertionError(f'Unexpected WM step count: {model["updates"]} vs {expected_updates}')
    policy = actor.train(starts,updates=2 if smoke else 100,start_groups=groups)
    result=dict(label=label,model=model,actor=policy,old_samples=250,new_samples=250,
                old_available=len(old['states']),new_available=len(new['states']),seconds=time.perf_counter()-began)
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/f'{label}.sampling.npz',old_indices=old_ids,new_indices=new_ids)
    with (directory/'training.jsonl').open('a') as handle:handle.write(json.dumps(result,allow_nan=False)+'\n')
    return result


def events(ticks, skills):
    per={name:dict(contact_steps=0,grasp_events=0,lift_events=0,first_contact_step=None,
                   first_grasp_step=None,first_lift_step=None) for name in OBJECT_NAMES}
    streak={n:0 for n in OBJECT_NAMES};zref={};lifted={};previous=None;contact=failures=0
    for step,row in enumerate(ticks,1):
        boundary=(row['stage'],row['episode_id'])
        if boundary!=previous:streak={n:0 for n in OBJECT_NAMES};lifted={};previous=boundary
        any_touch=False
        failures+=bool(row['after']['failed_object_ids'])
        for name in OBJECT_NAMES:
            obj=row['contact_events'][-1]['objects'][name];stat=per[name]
            touch=max(obj['left_force'],obj['right_force'])>row['reward']['weights']['contact_force_threshold']
            any_touch|=touch;stat['contact_steps']+=int(touch)
            if touch and stat['first_contact_step'] is None:stat['first_contact_step']=step
            held=row['action'][3]<0 and obj['grasped'];z=row['after']['objects'][name]['position_root'][2]
            if held:
                if streak[name]==0:zref[name]=z;lifted[name]=False
                streak[name]+=1
                if streak[name]==4:
                    stat['grasp_events']+=1
                    if stat['first_grasp_step'] is None:stat['first_grasp_step']=step
                if streak[name]>=4 and z-zref[name]>=.01 and not lifted[name]:
                    stat['lift_events']+=1;lifted[name]=True
                    if stat['first_lift_step'] is None:stat['first_lift_step']=step
            else:streak[name]=0;lifted[name]=False
        contact+=any_touch
    failed_steps={name:{r['step'] for r in ticks if name in r['after']['failed_object_ids']} for name in OBJECT_NAMES}
    buckets=set();safe=set();visits=repeats=motions=0;curve=[]
    for skill in skills:
        if not skill['trainable']:continue
        for name in OBJECT_NAMES:
            delta=np.array(skill['after']['objects'][name]['position_root'])-skill['before']['objects'][name]['position_root']
            bucket=motion_bucket(name,skill['primitive_type'],delta)
            if bucket:
                motions+=1;buckets.add(bucket)
                if not any(step in failed_steps[name] for step in range(skill['start_step']+1,skill['end_step']+1)):
                    visits+=1;repeats+=bucket in safe;safe.add(bucket)
        curve.append(dict(step=skill['end_step'],coverage=len(buckets),recoverable_coverage=len(safe)))
    n=len(ticks)
    return dict(real_steps=n,contact_time_percent=100*contact/n if n else 0.,
                grasp_events=sum(v['grasp_events'] for v in per.values()),lift_events=sum(v['lift_events'] for v in per.values()),
                motion_object_skills=motions,coverage=len(buckets),recoverable_coverage=len(safe),
                repeat_fraction=repeats/visits if visits else 0.,failure_step_fraction=failures/n if n else 0.,
                per_object=per,coverage_curve=curve,
                skill_selections={name:sum(s['primitive_type']==name for s in skills) for name in set(s['primitive_type'] for s in skills)})


def frozen_digest(backend, actor):
    digest=hashlib.sha256()
    def visit(value):
        if isinstance(value,dict):
            for key in sorted(value):digest.update(str(key).encode());visit(value[key])
        elif isinstance(value,(tuple,list)):
            for v in value:visit(v)
        elif torch.is_tensor(value):digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value,np.ndarray):digest.update(value.tobytes())
        else:digest.update(repr(value).encode())
    visit(backend.state_dict());visit(actor.state_dict())
    return digest.hexdigest()


def evaluate(env,encoder,backend,actor,directory,steps,shift,on_step):
    before_rng=rng_state();before=frozen_digest(backend,actor)
    old_shift=env.unwrapped.object_shift_xy.copy()
    try:
        seed_all(100000);env.unwrapped.object_shift_xy=np.array(shift,dtype=np.float32)
        collector=SkillCollector(env,encoder,actor,directory)
        ticks,skills=collector.collect(steps,'curiosity',seed=100000,stage='evaluation',on_step=on_step)
        result=events(ticks,skills)
        if frozen_digest(backend,actor)!=before:raise AssertionError('Evaluation changed training state')
        result.update(frozen_state_verified=True,seed=100000,object_shift_xy_m=list(shift))
        write_json(directory/'evaluation.json',result)
        return result
    finally:
        env.unwrapped.object_shift_xy=old_shift;restore_rng(before_rng)


def checkpoint(path,backend,actor,encoder,config,ticks,skills,sampler):
    save(path,backend,actor,encoder,config,ticks,skills)
    # The source old replay remains immutable; checkpoints contain only this arm's new data.
    write_json(path.with_suffix('.sampler.json'),sampler.bit_generator.state)


def fixed_wm_test(backend, dataset, train):
    """Identical held-out endpoints for both arms; filter any exact training overlap."""
    def key(s,a,y):return hashlib.sha256(s.tobytes()+a.tobytes()+y.tobytes()).digest()
    training={key(s,a,y) for r in train for s,a,y in zip(r['states'],r['actions'],r['next_states'])}
    keep=np.array([key(s,a,y) not in training for s,a,y in zip(dataset['states'],dataset['actions'],dataset['next_states'])])
    if not keep.any():return {'status':'no_held_out_segments'}
    s=dataset['states'][keep];a=dataset['actions'][keep];y=dataset['next_states'][keep]
    p=backend.predict(s,a).mean(0)
    error=np.stack([np.linalg.norm(p[:,i:i+3]-y[:,i:i+3],axis=1) for i in (33,45)],1)
    baseline=np.stack([np.linalg.norm(s[:,i:i+3]-y[:,i:i+3],axis=1) for i in (33,45)],1)
    moving=baseline>=.015
    return dict(segments=int(keep.sum()),excluded_overlap=int((~keep).sum()),moving_endpoints=int(moving.sum()),
                moving_error_cm=float(error[moving].mean()*100) if moving.any() else None,
                no_motion_error_cm=float(baseline[moving].mean()*100) if moving.any() else None,
                per_object_error_cm=(error.mean(0)*100).tolist())


def prediction_report(output):
    """Compare all checkpoints on the SAME old and new diagnostic trajectories."""
    torch.set_num_threads(1)
    manifest=json.loads((output/'manifest.json').read_text())
    old=load_old_replay(Path(manifest['source_checkpoint']).with_suffix('.replay.npz'))
    training=[old];datasets={};result={}
    old_test=ROOT/'logs/ee_primitive_wm_validation_20261028/primitive_88000_predictions.npz'
    if old_test.exists():
        with np.load(old_test) as z:
            datasets['old_fixed']=dict(states=z['states'].copy(),actions=z['skill_actions'].copy(),next_states=z['targets'].copy())
    heldout=[]
    for arm in ('continue','fresh'):
        root=output/arm
        initial_eval=root/'evaluation/step_000000/shifted/skills.jsonl'
        if initial_eval.exists():
            heldout.extend(json.loads(line) for line in initial_eval.open() if json.loads(line)['trainable'])
        paths=sorted((root/'curiosity').glob('round_*.replay.npz'))
        if paths:
            with np.load(paths[-1]) as z:training.append({k:z[k].copy() for k in ('states','actions','next_states')})
    if len(heldout)>=2:datasets['new_shared_initial_frozen']=skill_arrays(heldout)
    # Dataset union is collected before fitting, never from either training buffer.
    for arm in ('continue','fresh'):
        root=output/arm
        paths=[root/'initial.pt']+[root/'curiosity'/f'round_{step//1000-1:03d}.pt' for step in (10000,30000,60000)]
        for path in paths:
            if not path.exists():continue
            payload=torch.load(path,map_location='cpu',weights_only=False)
            backend=CEEUSBackend(agent_dim=33,action_dim=12,device='cpu',object_heuristic=payload['backend']['object_heuristic'])
            backend.load_state_dict(payload['backend'],restore_planner=False)
            label='0' if path.name=='initial.pt' else str(payload['real_steps'])
            result.setdefault(arm,{})[label]={name:fixed_wm_test(backend,data,training) for name,data in datasets.items()}
    result['limitations']=['Initial-policy trajectories are a shared independent cohort, not a balanced grasp/lift benchmark',
                          'Missing moving samples cannot establish interaction dynamics quality',
                          'Exact overlaps with either arm training replay are excluded identically for both arms']
    write_json(output/'wm_validation.json',result)
    return result


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=dict(arms={},limitation='Single seed; policy learner including optimizers/counters is the ablation; no isolated actor attribution')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5));csvrows=[]
    for arm in ('continue','fresh'):
        root=output/arm
        if not root.exists():continue
        rows=[json.loads(line) for line in (root/'learning.jsonl').read_text().splitlines()] if (root/'learning.jsonl').exists() else []
        evaluations={p.parent.parent.name+'/'+p.parent.name:json.loads(p.read_text()) for p in (root/'evaluation').glob('*/*/evaluation.json')}
        result['arms'][arm]=dict(training=rows[-1] if rows else None,evaluations=evaluations)
        for ax,metric in zip(axes,('contact_time_percent','grasp_events','lift_events')):
            ax.plot([r['real_steps'] for r in rows],[r[metric] for r in rows],label=arm);ax.set(xlabel='New training control steps',ylabel=metric);ax.grid(alpha=.2)
        for row in rows:csvrows.append(dict(arm=arm,**{k:row[k] for k in ('real_steps','contact_time_percent','grasp_events','lift_events','recoverable_coverage')}))
    if csvrows:
        axes[0].legend();fig.tight_layout();fig.savefig(output/'learning_curves.png',dpi=160)
        with (output/'learning_curves.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    plt.close(fig)
    write_json(output/'report.json',result)
    return result


def run(args):
    if args.stage=='report':
        prediction_report(args.output);report(args.output);return 0
    if args.output.exists():raise FileExistsError(f'Refusing to overwrite {args.output}')
    payload=torch.load(args.source,map_location='cpu',weights_only=False)
    config=OmegaConf.create(payload['config']);config.raps_actor=OmegaConf.merge(DEFAULTS,config.raps_actor)
    if config.agent!='raps_gnn_actor' or config.raps_actor.skill_set!='compact6' or config.raps_actor.disagreement_weight!=1.:
        raise ValueError('Requires the compact6 disagreement+heuristic checkpoint')
    if config.episode_steps!=800 or not config.centered_initial_tcp or config.execution!='reference':
        raise ValueError('Requires the centered 40s reference baseline')
    config.device=args.device;config.rounds=59;config.evaluation_steps=1000
    config.replay_experiment=dict(source=str(args.source.resolve()),old_new_ratio=[.5,.5],wm_samples=500,
        policy_ablation='restore_full_policy_learner_vs_fresh',object_shift_xy_m=[.03,0.],new_training_steps=60000,
        note='Balanced 500-sample replay and fixed update counts differ from full-history baseline')
    old=load_old_replay(args.source.with_suffix('.replay.npz'))
    if config.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent fallback')
    torch.set_num_threads(1);seed_all(0)
    env=None;physical=0;smoke=args.stage=='smoke';cap=200 if smoke else 135000
    def count_step(row):
        nonlocal physical
        physical+=1
        if physical>cap:raise AssertionError('Physical budget exceeded')
    try:
        env,env_config=make_env(episode_steps=800,centered_initial_tcp=True,object_shift_xy=(.03,0.))
        env.reset(seed=0);encoder=CabinetSkillStateEncoder(env,latched_gripper=True)
        prepare_output(args.output,config,env_config)
        write_json(args.output/'state_contract.json',encoder.contract())
        manifest=json.loads((args.output/'manifest.json').read_text())
        manifest.update(source_checkpoint=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
            source_replay_sha256=hashlib.sha256(args.source.with_suffix('.replay.npz').read_bytes()).hexdigest(),
            old_complete_skills=len(old['states']),old_replay_in_both_arms=True,physical_budget=cap,
            policy_initialization={'continue':'actor critic target optimizers counters restored','fresh':'all policy learner state fresh'},
            wm_initialization='identical model normalizers optimizer counters',normalization='500-sample old/new balanced dataset per round',
            log_flush_interval_steps=32,imagination_horizon_skills=5,prediction_horizon_seconds=None)
        write_json(args.output/'manifest.json',manifest)
        common_backend,common_actor=initialize(config,payload,encoder,False)
        bootstrap=SkillCollector(env,encoder,common_actor,args.output/'bootstrap')
        ticks,skills=bootstrap.collect(160 if smoke else 1000,'random',seed=0,stage='bootstrap',on_step=count_step)
        skill_arrays(skills);common_rng=rng_state()
        del common_backend,common_actor
        for arm in ('continue','fresh'):
            backend,actor=initialize(config,payload,encoder,arm=='continue');restore_rng(common_rng)
            sampler=np.random.default_rng(0)
            root=args.output/arm;root.mkdir()
            for folder in ('bootstrap','curiosity'):(root/folder).mkdir()
            # Per-arm logs/checkpoints contain only new-scene data, independently copied.
            history,endpoints=deepcopy(ticks),deepcopy(skills)
            for name,rows in [('transitions.jsonl',history),('skills.jsonl',endpoints)]:
                with (root/'bootstrap'/name).open('w') as handle:
                    for row in rows:handle.write(json.dumps(row,allow_nan=False)+'\n')
            checkpoint(root/'initial.pt',backend,actor,encoder,config,history,endpoints,sampler)
            for scene,shift in [('original',(0.,0.)),('shifted',(.03,0.))]:
                evaluate(env,encoder,backend,actor,root/f'evaluation/step_000000/{scene}',8 if smoke else 1000,shift,count_step)
            update(backend,actor,old,endpoints,root/'bootstrap','bootstrap',sampler,smoke)
            checkpoint(root/'bootstrap/checkpoint.pt',backend,actor,encoder,config,history,endpoints,sampler)
            collector=SkillCollector(env,encoder,actor,root/'curiosity')
            collector.total_steps=len(history);collector.episode_id=max(r['episode_id'] for r in history)
            collector.skill_id=max(s['skill_id'] for s in endpoints)
            for index in range(1,2 if smoke else 60):
                write_json(args.output/'progress.json',dict(stage='collecting',arm=arm,round=index,new_training_steps=len(history),new_physical_steps=physical))
                latest,macros=collector.collect(4 if smoke else 1000,'curiosity',seed=10000*index,stage=f'round_{index}',on_step=count_step)
                history.extend(latest);endpoints.extend(macros)
                write_json(args.output/'progress.json',dict(stage='training',arm=arm,round=index,new_training_steps=len(history),new_physical_steps=physical))
                training=update(backend,actor,old,endpoints,root/'curiosity',f'round_{index}',sampler,smoke)
                path=root/'curiosity'/f'round_{index:03d}.pt'
                checkpoint(path,backend,actor,encoder,config,history,endpoints,sampler)
                metrics=events(history,endpoints)
                with (root/'learning.jsonl').open('a') as handle:handle.write(json.dumps(metrics,allow_nan=False)+'\n')
                write_json(path.with_suffix('.json'),dict(training=training,interactions=metrics))
                print(json.dumps(dict(arm=arm,steps=len(history),grasp_events=metrics['grasp_events'],lift_events=metrics['lift_events'])),flush=True)
                if not smoke and len(history) in (10000,30000,60000):
                    for scene,shift in [('original',(0.,0.)),('shifted',(.03,0.))]:
                        write_json(args.output/'progress.json',dict(stage='evaluating',arm=arm,steps=len(history),scene=scene,new_physical_steps=physical))
                        evaluate(env,encoder,backend,actor,root/f'evaluation/step_{len(history):06d}/{scene}',1000,shift,count_step)
            if smoke:
                arrays=skill_arrays(endpoints);expected=backend.predict(arrays['states'][:2],arrays['actions'][:2])
                before_rng=rng_state();action=actor.action(arrays['states'][0]);restore_rng(before_rng)
                checkpoint(path,backend,actor,encoder,config,history,endpoints,sampler)
                load(path,backend,actor,encoder)
                np.testing.assert_array_equal(expected,backend.predict(arrays['states'][:2],arrays['actions'][:2]))
                np.testing.assert_array_equal(action,actor.action(arrays['states'][0]))
                write_json(root/'restore_check.json',dict(predictions_exact=True,action_rng_exact=True))
            write_json(root/'result.json',dict(status='completed',new_training_steps=len(history),metrics=events(history,endpoints)))
        if physical!=cap:raise AssertionError(f'Unexpected physical count {physical}, expected {cap}')
        write_json(args.output/'result.json',dict(status='completed',smoke=smoke,new_physical_steps=physical,
            interaction_dynamics='not_validated_by_smoke' if smoke else 'inspect_frozen_evaluations'))
        write_json(args.output/'progress.json',dict(stage='completed',new_physical_steps=physical))
        if not smoke:prediction_report(args.output)
        report(args.output)
    except BaseException as exc:
        if args.output.exists():write_json(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),new_physical_steps=physical))
        raise
    finally:
        if env is not None:env.close()
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('smoke','compare','report'))
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    return run(parser.parse_args(argv))


if __name__=='__main__':raise SystemExit(main())
