import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from domain.robocasa.grasp_binding import GraspBinding
from domain.robocasa.motion_planning import MotionPlanningError


class Pose:
    def __init__(self, p=(0., 0., 0.), q=(1., 0., 0., 0.)):
        self.p = np.asarray(p, dtype=float)
        self.q = np.asarray(q, dtype=float)

    def inv(self):
        inverse = Rotation.from_quat(np.roll(self.q, -1)).inv()
        return Pose(inverse.apply(-self.p), np.roll(inverse.as_quat(), 1))

    def __mul__(self, other):
        rotation = Rotation.from_quat(np.roll(self.q, -1))
        orientation = rotation * Rotation.from_quat(np.roll(other.q, -1))
        return Pose(self.p + rotation.apply(other.p), np.roll(orientation.as_quat(), 1))


class ReadOnlyPose:
    def __init__(self, pose):
        self._pose = pose

    @property
    def pose(self):
        return self._pose


class Entity:
    def __init__(self, scene):
        self.components = []
        self.scene = scene
        self.removed = []

    def remove_component(self, component):
        self.components.remove(component)
        self.removed.append(component)
        component.entity = None


class Drive:
    def __init__(self, entity):
        self.entity = entity
        self.is_enabled = True
        self.limits = {}
        entity.components.append(self)

    def set_limit_x(self, *values):
        self.limits["x"] = values

    def set_limit_y(self, *values):
        self.limits["y"] = values

    def set_limit_z(self, *values):
        self.limits["z"] = values

    def set_drive_property_slerp(self, stiffness, damping, **kwargs):
        self.angular_drive = (stiffness, damping, kwargs)

    def set_drive_target(self, pose):
        self.target = pose


def environment(monkeypatch):
    monkeypatch.setitem(sys.modules, "sapien", SimpleNamespace(Pose=Pose))
    calls = []
    scene = SimpleNamespace()
    entity = Entity(scene)
    actor = ReadOnlyPose(Pose([.45, .1, .74]))
    actor.linear_velocity = np.zeros(3)
    actor.name = "generic"
    actor._bodies = [SimpleNamespace(entity=entity)]
    parent = ReadOnlyPose(Pose([.31, .08, .72], [np.cos(.1), 0., 0., np.sin(.1)]))
    parent.name = "panda_hand"
    parent._objs = [object()]
    tcp = ReadOnlyPose(Pose([.42, .1, .74]))

    def create_drive(*args):
        calls.append(args)
        return Drive(args[2].entity)

    scene.create_drive = create_drive
    planner = SimpleNamespace(
        gripper_command=-1., grasp_frames={"generic": 4},
        _grasp_contacts=lambda: {"generic": True},
        attachment_state=lambda: {"name": "generic", "gripper_command": planner.gripper_command})
    env = SimpleNamespace(num_envs=1, gpu_sim_enabled=False, motion_planner=planner,
        cabinet_items=[actor], scene=SimpleNamespace(sub_scenes=[scene]),
        agent=SimpleNamespace(tcp=tcp, robot=SimpleNamespace(get_links=lambda: [parent])))
    return env, actor, parent, entity, calls


def test_confirmed_binding_holds_measured_pose_with_explicit_angular_drive_without_pose_writes(monkeypatch):
    env, actor, parent, entity, calls = environment(monkeypatch)
    original_pose = actor.pose
    binding = GraspBinding(env)
    result = binding.bind_confirmed("generic")
    assert binding.bound_object_id == "generic"
    assert result["physical_binding"]
    assert actor.pose is original_pose
    assert len(calls) == 1
    body0, anchor0, body1, anchor1 = calls[0]
    assert body0 is parent._objs[0] and body1 is actor._bodies[0]
    np.testing.assert_allclose((parent.pose * anchor0).p, actor.pose.p)
    np.testing.assert_allclose((parent.pose * anchor0).q, actor.pose.q, atol=1e-15)
    np.testing.assert_array_equal(anchor1.p, np.zeros(3))
    np.testing.assert_array_equal(anchor1.q, [1., 0., 0., 0.])
    assert entity.components[0].limits == {
        "x": (0., 0.), "y": (0., 0.), "z": (0., 0.)}
    assert entity.components[0].angular_drive == (1e6, 2e3, {"mode": "acceleration"})
    np.testing.assert_array_equal(entity.components[0].target.q, [1., 0., 0., 0.])


@pytest.mark.parametrize("failure", ["no_attachment", "different_attachment", "opening", "short_confirmation", "no_contact"])
def test_binding_requires_existing_closed_confirmed_raw_grasp(monkeypatch, failure):
    env, _, _, entity, calls = environment(monkeypatch)
    planner = env.motion_planner
    if failure == "no_attachment":
        planner.attachment_state = lambda: None
    elif failure == "different_attachment":
        planner.attachment_state = lambda: {"name": "other", "gripper_command": -1.}
    elif failure == "opening":
        planner.gripper_command = 1.
    elif failure == "short_confirmation":
        planner.grasp_frames["generic"] = 3
    else:
        planner._grasp_contacts = lambda: {"generic": False}
    with pytest.raises(MotionPlanningError, match="unconfirmed_grasp"):
        GraspBinding(env).bind_confirmed("generic")
    assert not calls and not entity.components


def test_matching_bind_is_idempotent_second_object_rejected_release_and_reset_remove_component(monkeypatch):
    env, _, _, entity, calls = environment(monkeypatch)
    binding = GraspBinding(env)
    binding.bind_confirmed("generic")
    binding.bind_confirmed("generic")
    assert len(calls) == 1
    with pytest.raises(MotionPlanningError, match="grasp_binding_conflict"):
        binding.bind_confirmed("other")
    binding.release()
    binding.release()
    assert binding.bound_object_id is None and binding.feedback() is None
    assert len(entity.removed) == 1 and not entity.components
    binding.bind_confirmed("generic")
    binding.reset()
    assert len(entity.removed) == 2 and not entity.components
    assert binding.bound_object_id is None


@pytest.mark.parametrize("failure", ["disabled", "removed_drive", "removed_entity", "removed_actor", "position_slip", "angle_slip"])
def test_feedback_detects_missing_binding_or_tcp_relative_slip(monkeypatch, failure):
    env, actor, _, entity, _ = environment(monkeypatch)
    binding = GraspBinding(env)
    binding.bind_confirmed("generic")
    if failure == "disabled":
        entity.components[0].is_enabled = False
    elif failure == "removed_drive":
        entity.remove_component(entity.components[0])
    elif failure == "removed_entity":
        entity.scene = None
    elif failure == "removed_actor":
        env.cabinet_items.clear()
    elif failure == "position_slip":
        actor._pose = Pose(actor.pose.p + [0., 0., .003])
    else:
        actor._pose = Pose(actor.pose.p, np.roll(Rotation.from_euler("z", 3., degrees=True).as_quat(), 1))
    with pytest.raises(MotionPlanningError, match="grasp_binding_(lost|slip)"):
        binding.feedback()
    binding.release()
    assert binding.bound_object_id is None


def test_feedback_allows_common_robot_object_motion_and_small_constraint_error(monkeypatch):
    env, actor, _, _, _ = environment(monkeypatch)
    binding = GraspBinding(env)
    binding.bind_confirmed("generic")
    actor._pose = Pose(actor.pose.p + [.1, .2, .3])
    env.agent.tcp._pose = Pose(env.agent.tcp.pose.p + [.1, .2, .3])
    assert binding.feedback()["position_error_m"] < 1e-12
    actor._pose = Pose(actor.pose.p + [.001, 0., 0.])
    assert binding.feedback()["position_error_m"] == pytest.approx(.001)


@pytest.mark.parametrize("settings", [{"num_envs": 2}, {"gpu_sim_enabled": True}])
def test_binding_rejects_unsupported_simulation(monkeypatch, settings):
    env, _, _, _, _ = environment(monkeypatch)
    env.__dict__.update(settings)
    with pytest.raises(MotionPlanningError, match="single-environment CPU"):
        GraspBinding(env)


def test_supported_finger_hold_keeps_logical_closure_and_binding_until_release(monkeypatch):
    import torch
    env, _, _, entity, _ = environment(monkeypatch)
    targets = []
    gripper = SimpleNamespace(qpos=torch.tensor([[.028, .029]]),
        _target_qpos=torch.tensor([[-.01, -.01]]),
        set_drive_targets=lambda target: targets.append(target.clone()))
    env.agent.controller = SimpleNamespace(controllers={"gripper": gripper})
    binding = GraspBinding(env)
    with pytest.raises(MotionPlanningError, match="missing_grasp"):
        binding.hold_fingers_at_current_width()
    binding.bind_confirmed("generic")
    drive = entity.components[0]
    binding.hold_fingers_at_current_width()
    gripper.qpos += .001
    binding.apply_finger_hold()
    torch.testing.assert_close(targets[-1], torch.tensor([[.028, .029]]))
    assert torch.all(gripper._target_qpos < 0)
    assert binding.bound_object_id == "generic" and drive in entity.components
    binding.release()
    binding.apply_finger_hold()
    assert len(targets) == 1 and not entity.components


def test_pose_speed_uses_peak_substep_motion_and_resets_each_control(monkeypatch):
    env, actor, _, _, _ = environment(monkeypatch)
    actor.linear_velocity[:] = [.23, 0., 0.]
    binding = GraspBinding(env);binding.bind_confirmed("generic")
    start = actor.pose.p.copy()
    binding.begin_control()
    assert binding.feedback()["max_substep_pose_speed_m_s"] is None
    actor._pose = Pose(start + [.0002, 0., 0.])
    binding.observe_substep(.01)
    actor._pose = Pose(start)
    binding.observe_substep(.01)
    result = binding.feedback()
    assert result["max_substep_pose_speed_m_s"] == pytest.approx(.02)
    assert result["native_speed_m_s"] == pytest.approx(.23)
    assert result["pose_speed_samples"] == 2
    binding.begin_control();binding.observe_substep(.01)
    assert binding.feedback()["max_substep_pose_speed_m_s"] == 0.
    with pytest.raises(MotionPlanningError, match="Invalid physics"):
        binding.observe_substep(0.)
