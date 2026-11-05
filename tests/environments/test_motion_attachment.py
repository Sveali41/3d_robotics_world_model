"""Focused, native-free checks for motion-planner grasp tracking."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from domain.robocasa.motion_planning import (
    CabinetMotionPlanner, MotionPlanningError,
)


class Pose:
    """Small wxyz pose helper with the composition used by SAPIEN poses."""

    def __init__(self, p=(0., 0., 0.), q=(1., 0., 0., 0.)):
        self.p = np.asarray(p, dtype=float)
        self.q = np.asarray(q, dtype=float)

    def inv(self):
        rotation = Rotation.from_quat(np.roll(self.q, -1))
        inverse = rotation.inv()
        return Pose(-inverse.apply(self.p), np.roll(inverse.as_quat(), 1))

    def __mul__(self, other):
        first = Rotation.from_quat(np.roll(self.q, -1))
        second = Rotation.from_quat(np.roll(other.q, -1))
        return Pose(self.p + first.apply(other.p), np.roll((first * second).as_quat(), 1))


def planner_with_items(names=("cup",)):
    items = [SimpleNamespace(name=name, pose=Pose()) for name in names]
    force_map = {}
    native_grasps = set()
    agent = SimpleNamespace(
        is_grasping=lambda item: [item.name in native_grasps],
        finger1_link="left", finger2_link="right",
        tcp=SimpleNamespace(pose=Pose()),
    )
    gripper = SimpleNamespace(
        config=SimpleNamespace(lower=0., upper=1.),
        _target_qpos=np.array([1.]),
    )
    agent.controller = SimpleNamespace(controllers={"gripper": gripper})
    scene = SimpleNamespace(
        get_pairwise_contact_forces=lambda link, item: [force_map.get((link, item.name), (0., 0., 0.))]
    )
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.env = SimpleNamespace(agent=agent, cabinet_items=items, scene=scene)
    planner.grasp_frames = {item.name: 0 for item in items}
    planner.attached_name = None
    planner.gripper_command = 1.
    return planner, items, force_map, native_grasps, gripper


def test_grasp_contacts_accept_native_grasp_or_bilateral_threshold():
    planner, items, forces, native, _ = planner_with_items(("native", "bilateral", "weak"))
    native.add("native")
    forces[("left", "bilateral")] = (0., .05, 0.)
    forces[("right", "bilateral")] = (0., 0., -.05)
    forces[("left", "weak")] = (.05, 0., 0.)
    forces[("right", "weak")] = (0., .049, 0.)

    assert planner._grasp_contacts() == {"native": True, "bilateral": True, "weak": False}


def test_observe_grasp_requires_four_closed_contact_ticks_and_resets_streak():
    planner, _, forces, _, gripper = planner_with_items()
    forces[("left", "cup")] = (.05, 0., 0.)
    forces[("right", "cup")] = (.05, 0., 0.)
    gripper._target_qpos[:] = 0.  # normalized closed command

    for _ in range(3):
        planner.observe_grasp()
    assert planner.attached_name is None
    planner.observe_grasp()
    assert planner.attached_name == "cup"
    assert planner.gripper_command == -1.

    forces[("right", "cup")] = (0., 0., 0.)
    planner.observe_grasp()
    assert planner.grasp_frames["cup"] == 0
    assert planner.attached_name is None


def test_opening_keeps_attachment_until_contact_is_lost():
    planner, _, forces, _, gripper = planner_with_items()
    forces[("left", "cup")] = (.1, 0., 0.)
    forces[("right", "cup")] = (.1, 0., 0.)
    gripper._target_qpos[:] = 0.
    for _ in range(4):
        planner.observe_grasp()
    gripper._target_qpos[:] = 1.
    planner.observe_grasp()
    assert planner.attached_name == "cup"
    assert planner.gripper_command == 1.


def test_attachment_state_reports_measured_tcp_relative_pose_and_rejects_two_items():
    planner, items, forces, _, gripper = planner_with_items(("cup", "bowl"))
    planner.env.agent.tcp.pose = Pose((1., 2., 3.))
    items[0].pose = Pose((1.2, 2., 3.))
    planner.grasp_frames["cup"] = 4
    planner.attached_name = "cup"
    planner.gripper_command = -1.
    state = planner.attachment_state()
    assert state["name"] == "cup"
    np.testing.assert_allclose(state["tcp_to_object_pose"], [.2, 0., 0., 1., 0., 0., 0.], atol=1e-12)

    planner.grasp_frames["bowl"] = 4
    with pytest.raises(MotionPlanningError) as error:
        planner.attachment_state()
    assert error.value.status == "multiple_grasped_objects"


def test_planning_attachment_rejects_unconfirmed_contact_and_opening_held_item():
    planner, _, forces, _, gripper = planner_with_items()
    forces[("left", "cup")] = (.1, 0., 0.)
    forces[("right", "cup")] = (.1, 0., 0.)
    with pytest.raises(MotionPlanningError) as error:
        planner._planning_attachment()
    assert error.value.status == "unconfirmed_grasp"

    planner.grasp_frames["cup"] = 4
    planner.attached_name = "cup"
    gripper._target_qpos[:] = 1.
    planner.gripper_command = 1.
    with pytest.raises(MotionPlanningError) as error:
        planner._planning_attachment()
    assert error.value.status == "release_in_progress"


def test_attachment_feedback_checks_loss_slip_unexpected_contact_and_valid_state():
    planner, _, forces, _, _ = planner_with_items()
    plan = SimpleNamespace(metadata={"attachment": None})
    forces[("left", "cup")] = (.1, 0., 0.)
    forces[("right", "cup")] = (.1, 0., 0.)
    with pytest.raises(MotionPlanningError) as error:
        planner.check_attachment_feedback(plan)
    assert error.value.status == "unexpected_attachment"

    forces.clear()
    plan.metadata["attachment"] = {
        "name": "cup", "tcp_to_object_pose": [0., 0., 0., 1., 0., 0., 0.]
    }
    with pytest.raises(MotionPlanningError) as error:
        planner.check_attachment_feedback(plan)
    assert error.value.status == "grasp_lost"

    planner.attached_name = "cup"
    planner.grasp_frames["cup"] = 4
    planner.gripper_command = -1.
    forces[("left", "cup")] = (.1, 0., 0.)
    forces[("right", "cup")] = (.1, 0., 0.)
    planner.env.agent.tcp.pose = Pose()
    planner.env.cabinet_items[0].pose = Pose((.003, 0., 0.))
    with pytest.raises(MotionPlanningError) as error:
        planner.check_attachment_feedback(plan)
    assert error.value.status == "attachment_slip"

    planner.env.cabinet_items[0].pose = Pose((.001, 0., 0.))
    feedback = planner.check_attachment_feedback(plan)
    assert feedback["name"] == "cup"
    assert feedback["position_error_m"] == pytest.approx(.001)
    assert feedback["angle_error_deg"] == pytest.approx(0.)


def payload_collision_planner(penetration_depth=0.):
    class PhysxCollisionShapeSphere:
        radius = .1

    shape = PhysxCollisionShapeSphere()
    payload_geom = "payload"
    obstacle_geom = "obstacle"
    link_geoms = {"panda_leftfinger": "leftfinger", "panda_hand": "hand"}
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.kin = SimpleNamespace(pmodel=SimpleNamespace(
        compute_forward_kinematics=lambda _: None,
        get_link_pose=lambda _: Pose(),
    ), end_link_idx=0)
    planner.root = Pose()
    planner.link_indices = {name: i for i, name in enumerate(link_geoms)}
    planner.robot_shapes = [dict(link=name, shape=shape, geom=geom, local=Pose(), groups=[1, 1, 0, 0])
                            for name, geom in link_geoms.items()]
    planner._shape_pose = lambda _shape, pose: pose
    planner._object = lambda geometry, _pose: SimpleNamespace(geom=geometry)
    planner.adjacent_pairs = {frozenset(("panda_leftfinger", "panda_hand"))}
    planner.env = SimpleNamespace(ground=SimpleNamespace(name="ground"))
    planner.allowed_contacts = ("cup",)
    planner.obstacles = [dict(name="cup/0", actor="cup", shape=shape, geom=obstacle_geom,
                              pose=Pose(), groups=[1, 1, 0, 0])]
    planner.payload_shapes = [dict(name="cup/0", shape=shape, geom=payload_geom,
                                   local=Pose(), groups=[1, 1, 0, 0])]
    planner.tcp_to_payload = Pose()
    planner.support_contacts = {}

    def collide(first, second, _request):
        pair = {first.geom, second.geom}
        is_collision = (pair == {payload_geom, obstacle_geom}
                        or pair == {payload_geom, link_geoms["panda_hand"]}
                        or pair == {payload_geom, link_geoms["panda_leftfinger"]})
        contact = SimpleNamespace(penetration_depth=penetration_depth, normal=np.array([0., 0., 1.]))
        return SimpleNamespace(is_collision=lambda: is_collision,
                               get_contacts=lambda: [contact] if is_collision else [])

    planner.fcl = SimpleNamespace(CollisionRequest=lambda *_args: None, collide=collide)
    return planner


def test_payload_checks_allowed_object_environment_and_only_exempts_fingers():
    planner = payload_collision_planner()
    contacts = planner._check(np.zeros(1))
    assert {tuple(contact.values()) for contact in contacts} == {
        ("cup/0", "cup/0", "payload_environment"),
        ("cup/0", "panda_hand", "payload_self"),
    }


def test_designated_support_allows_shallow_touch_but_rejects_deeper_penetration():
    planner = payload_collision_planner(penetration_depth=1e-5)
    planner.robot_shapes = []
    planner.support_contacts["cup/0"] = 1e-5
    assert planner._check(np.zeros(1)) == []

    planner.fcl.collide = lambda *_args: SimpleNamespace(
        is_collision=lambda: True,
        get_contacts=lambda: [SimpleNamespace(penetration_depth=1.01e-5, normal=np.array([0., 0., 1.]))],
    )
    contacts = planner._check(np.zeros(1))
    assert [contact["kind"] for contact in contacts] == ["payload_environment"]
