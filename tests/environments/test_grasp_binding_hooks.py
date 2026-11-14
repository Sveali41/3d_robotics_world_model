"""Binding lifecycle ordering in the project-owned ManiSkill environment."""
from types import SimpleNamespace
import torch
from domain.wall_cabinet_env import WallCabinetEnv
from mani_skill.envs.sapien_env import BaseEnv

class BareWall(WallCabinetEnv):
    def __init__(self): pass
    def close(self): pass


def test_open_releases_before_simulation_and_closed_command_keeps_binding():
    env=BareWall();events=[]
    binding=SimpleNamespace(bound_object_id='object',release=lambda:events.append('release'), begin_control=lambda:None)
    env._grasp_binding=binding;env._grasp_retention_mode='constraint'
    gripper=SimpleNamespace(_target_qpos=torch.tensor([[-.01]]),config=SimpleNamespace(lower=-.01,upper=.04))
    env.agent=SimpleNamespace(controller=SimpleNamespace(controllers={'gripper':gripper}))
    env._before_control_step()
    assert events==[]
    gripper._target_qpos=torch.tensor([[.04]])
    env._before_control_step();events.append('physics')
    assert events==['release','physics']


def test_automatic_binding_waits_for_four_closed_grasp_controls():
    env=BareWall();events=[];count=[0]
    planner=SimpleNamespace(gripper_command=-1.)
    planner.observe_grasp=lambda:count.__setitem__(0,count[0]+1)
    planner.attachment_state=lambda: {'name':'object'} if count[0]>=4 else None
    env._cabinet_motion_planner=planner
    binding=SimpleNamespace(bound_object_id=None)
    def bind(name):
        events.append((name,count[0]));binding.bound_object_id=name
    binding.bind_confirmed=bind
    env._grasp_binding=binding
    for _ in range(5):env._after_control_step()
    assert events==[('object',4)]


def test_reset_clears_constraint_before_body_reset(monkeypatch):
    env=BareWall();events=[]
    env._grasp_retention_mode='constraint'
    env._grasp_binding=SimpleNamespace(release=lambda:events.append('release'))
    monkeypatch.setattr(BaseEnv,'reset',lambda *args,**kwargs:events.append('reset_bodies'))
    env.reset(seed=3)
    assert events==['release','reset_bodies']
    assert '_grasp_binding' not in env.__dict__
    assert env._grasp_retention_mode=='constraint'


def test_friction_mode_removes_existing_constraint():
    env=BareWall();events=[]
    env._grasp_binding=SimpleNamespace(release=lambda:events.append('release'))
    env.set_grasp_retention('friction')
    assert events==['release'] and '_grasp_binding' not in env.__dict__
    assert env._grasp_retention_mode=='friction'


def test_native_finger_hold_runs_after_controller_substep_update(monkeypatch):
    env=BareWall();events=[]
    monkeypatch.setattr(BaseEnv,'_before_simulation_step',lambda *a:events.append('base'))
    env._grasp_binding=SimpleNamespace(apply_finger_hold=lambda:events.append('hold'))
    env._before_simulation_step()
    assert events==['base','hold']
