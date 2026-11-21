"""Native-free checks of opt-in geometry reuse and planner profiling."""
import inspect
import random
from types import SimpleNamespace

import numpy as np
import pytest

from domain.robocasa import motion_planning as motion


class Pose:
    def __init__(self, p=(0., 0., 0.), q=(1., 0., 0., 0.)):
        self.p, self.q = np.asarray(p), np.asarray(q)

    def __mul__(self, other):
        return Pose(self.p + other.p, self.q.copy())


class PhysxCollisionShapeBox:
    def __init__(self):
        self.half_size = np.array([.1, .2, .3], dtype=np.float32)
        self.local_pose = Pose()
        self.groups = [1, 1, 0, 0]

    def get_collision_groups(self):
        return self.groups.copy()


class PhysxCollisionShapeConvexMesh:
    def __init__(self):
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.scale = np.array([.1, .2, .3], dtype=np.float32)
        self.triangles = np.array([[0, 1, 2]], dtype=np.int64)


def planner(enabled=None, profile=None):
    result = motion.CabinetMotionPlanner.__new__(motion.CabinetMotionPlanner)
    result.env = SimpleNamespace(_collection_profile=profile)
    if enabled is not None:
        result.env.collection_optimizations = enabled
    result._geometry_cache = {}
    result.constructed = []

    def construct(*args):
        value = SimpleNamespace(args=args)
        result.constructed.append(value)
        return value

    result.fcl = SimpleNamespace(Box=construct, Sphere=construct, Cylinder=construct, Convex=construct)
    result.sapien = SimpleNamespace(Pose=Pose)
    return result


@pytest.mark.parametrize('enabled', [None, False])
def test_default_constructs_each_time_without_cache(enabled):
    p = planner(enabled)
    shape = PhysxCollisionShapeBox()
    assert p._geometry(shape) is not p._geometry(shape)
    assert not p._geometry_cache


def test_identity_margin_dimensions_and_new_instance():
    profile = {}
    p = planner(True, profile)
    shape = PhysxCollisionShapeBox()
    first = p._geometry(shape)
    assert p._geometry(shape) is first
    same_dimensions = PhysxCollisionShapeBox()
    assert p._geometry(same_dimensions) is not first
    inflated = p._geometry(shape, .001)
    assert inflated is not first
    assert p._geometry(shape, .001) is inflated
    shape.half_size[0] += .01
    changed = p._geometry(shape)
    assert changed is not first
    assert p._geometry(shape, .001) is not inflated
    np.testing.assert_array_equal(changed.args, 2 * shape.half_size)
    assert p._geometry_cache[(id(shape), 0.)][0] is shape
    fresh = planner(True)
    assert fresh._geometry(shape) is not changed
    assert len(fresh.constructed) == 1
    assert profile['geometry_cache_hits'] == 2
    assert profile['geometry_cache_misses'] == 5
    assert profile['planner_geometry_s'] >= 0


def test_convex_content_scale_dtype_and_triangles_invalidate_exactly():
    p = planner(True)
    shape = PhysxCollisionShapeConvexMesh()
    previous = p._geometry(shape)
    vertices, triangles, convex = previous.args
    np.testing.assert_array_equal(vertices, np.asarray(shape.vertices * shape.scale, dtype=np.float64))
    np.testing.assert_array_equal(triangles, np.asarray(shape.triangles, dtype=np.int32))
    assert vertices.dtype == np.float64 and triangles.dtype == np.int32 and convex is True
    assert p._geometry(shape) is previous
    mutations = [lambda: shape.scale.__setitem__(0, .15),
                 lambda: shape.vertices.__setitem__((0, 0), .01),
                 lambda: shape.triangles.__setitem__((0, 0), 2),
                 lambda: setattr(shape, 'vertices', shape.vertices.astype(np.float64))]
    for mutate in mutations:
        mutate()
        current = p._geometry(shape)
        assert current is not previous
        assert p._geometry(shape) is current
        previous = current
    assert p._geometry(shape, .001) is not previous


def test_shape_type_change_invalidates():
    p = planner(True)
    shape = PhysxCollisionShapeBox()
    first = p._geometry(shape)
    class PhysxCollisionShapeSphere:
        radius = .1
    shape.__class__ = PhysxCollisionShapeSphere
    assert p._geometry(shape) is not first


def test_scene_refreshes_pose_local_groups_attachment_and_order():
    p = planner(True)
    robot_shape, wall_shape, item_shape = [PhysxCollisionShapeBox() for _ in range(3)]
    body = lambda shape: SimpleNamespace(get_collision_shapes=lambda: [shape])
    link = SimpleNamespace(name='robot', _objs=[body(robot_shape)])
    wall = SimpleNamespace(name='wall', pose=Pose(), _bodies=[body(wall_shape)])
    item = SimpleNamespace(name='cup', pose=Pose(), _bodies=[body(item_shape)])
    ground = SimpleNamespace(name='ground', pose=Pose(), _bodies=[SimpleNamespace(get_collision_shapes=lambda: [])])
    p.robot = SimpleNamespace(get_links=lambda: [link])
    p.link_indices = {'robot': 0}
    p.env.cabinet = SimpleNamespace(get_links=lambda: [])
    p.env.wall, p.env.ground, p.env.cabinet_items = wall, ground, [item]
    p.attachment = None
    p._scene_geometry()
    first_robot, first_wall = p.robot_shapes[0]['geom'], p.obstacles[0]['geom']
    wall.pose = Pose([1., 0., 0.])
    wall_shape.local_pose = Pose([0., 2., 0.])
    robot_shape.local_pose = Pose([0., 0., 3.])
    wall_shape.groups = [2, 2, 0, 0]
    p._scene_geometry()
    assert [o['name'] for o in p.obstacles] == ['wall/0', 'cup/0']
    assert p.robot_shapes[0]['geom'] is first_robot
    assert p.robot_shapes[0]['local'] is robot_shape.local_pose
    assert p.obstacles[0]['geom'] is first_wall
    np.testing.assert_array_equal(p.obstacles[0]['pose'].p, [1., 2., 0.])
    assert p.obstacles[0]['groups'] == [2, 2, 0, 0]
    p.attachment = dict(name='cup', tcp_to_object_pose=[0., 0., 0., 1., 0., 0., 0.])
    p._object = lambda *args: args
    p._scene_geometry()
    assert [o['name'] for o in p.obstacles] == ['wall/0']
    assert p.payload_shapes[0]['geom'] is p._geometry(item_shape)


@pytest.mark.parametrize('enabled', [False, True])
def test_failure_total_and_nested_timings_do_not_change_rng(monkeypatch, enabled):
    profile = {}
    p = planner(enabled, profile)
    ticks = iter(range(100))
    monkeypatch.setattr(motion, 'perf_counter', lambda: next(ticks))
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    with pytest.raises(motion.MotionPlanningError):
        p.plan_to_pose([0., 0.], [1., 0., 0., 0.])
    assert profile['planner_total_s'] == 1.
    assert 'start_qpos' in inspect.signature(p.plan_to_pose).parameters
    for key in ('planner_ik_s', 'planner_path_search_s', 'planner_topp_s'):
        def fail():
            raise RuntimeError('native failure')
        with pytest.raises(RuntimeError, match='native failure'):
            motion._profile_call(p, key, fail)
        assert profile[key] == 1.
    p._geometry(PhysxCollisionShapeBox())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert random.getstate() == python_before


@pytest.mark.parametrize('fail_topp', [False, True])
def test_real_plan_wrapper_preserves_ik_seeds_and_times_nested_topp(monkeypatch, fail_topp):
    seeds = []
    results = []
    for enabled in (False, True):
        profile = {}
        p = planner(enabled, profile)
        p.env.cabinet_items, p.env.control_freq = [], 20
        p.robot = SimpleNamespace(get_qpos=lambda: np.zeros((1, 9)),
            get_qlimits=lambda: np.tile([-3., 3.], (1, 9, 1)))
        p.pindices, p.controlled = np.arange(9), np.arange(7)
        p.root_link, p.links = SimpleNamespace(pose=Pose()), []
        calls = []
        def inverse(*args, **kwargs):
            calls.append(kwargs['initial_qpos'].copy())
            return np.zeros(9), True, None
        p.kin = SimpleNamespace(end_link_idx=0, qmask=np.ones(9), pmodel=SimpleNamespace(
            compute_forward_kinematics=lambda _: None, get_link_pose=lambda _: Pose(),
            compute_inverse_kinematics=inverse))
        p._planning_attachment = lambda: None
        p.snapshot_scene = lambda: {}
        p._scene_geometry = lambda: None
        p._check = lambda _: []
        p.obstacles, p.support_contacts = [], {}
        def topp(*args):
            if fail_topp:
                raise RuntimeError('Fail to parameterize path')
            return np.array([0., .05]), np.zeros((2, 9)), None, None, .05
        upstream = SimpleNamespace(move_group_joint_indices=np.arange(9),
                                   set_base_pose=lambda _: None, TOPP=topp)
        def path(**kwargs):
            times, positions, _, _, _ = upstream.TOPP(np.zeros((2, 9)), .05)
            return dict(status='Success', time=times, position=positions)
        upstream.plan_qpos_to_qpos = path
        p._make_planner = lambda: upstream
        monkeypatch.setattr(motion, '_native_modules', lambda: (
            SimpleNamespace(__version__='test'), p.fcl, p.sapien))
        if fail_topp:
            with pytest.raises(motion.MotionPlanningError, match='time_parameterization_failed'):
                p.plan_to_pose([0., 0., 0.], [1., 0., 0., 0.], seed=7)
        else:
            results.append(p.plan_to_pose([0., 0., 0.], [1., 0., 0., 0.], seed=7))
        assert len(calls) == 16
        seeds.append(calls)
        assert profile['planner_ik_s'] > 0
        assert profile['planner_total_s'] >= profile['planner_path_search_s'] >= profile['planner_topp_s'] > 0
    np.testing.assert_array_equal(seeds[0], seeds[1])
    if not fail_topp:
        np.testing.assert_array_equal(results[0].position, results[1].position)
        assert results[0].metadata == results[1].metadata
