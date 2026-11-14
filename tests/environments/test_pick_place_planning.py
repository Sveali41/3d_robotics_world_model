"""Narrow lower-shelf collision permission checks without native simulation."""
from types import SimpleNamespace
import numpy as np
import pytest
from domain.robocasa.motion_planning import _lower_support_contact_allowed


def test_placement_bounds_include_local_shape_pose_grasp_offset_and_target_rotation():
    import sapien
    from scipy.spatial.transform import Rotation
    from domain.robocasa.motion_planning import CabinetMotionPlanner, MotionPlanningError

    shape = type("PhysxCollisionShapeBox", (), {})()
    shape.half_size = np.array([.02, .04, .03])
    shape.local_pose = sapien.Pose([.02, 0., -.01])
    actor = SimpleNamespace(name="payload", _bodies=[SimpleNamespace(get_collision_shapes=lambda: [shape])])
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.sapien = sapien
    planner.env = SimpleNamespace(cabinet_items=[actor])
    attachment = dict(name="payload", tcp_to_object_pose=[.01, -.03, -.05, 1., 0., 0., 0.])
    quaternion = np.roll(Rotation.from_euler("z", 90, degrees=True).as_quat(), 1)
    bounds = planner.placement_bounds(attachment, quaternion)
    np.testing.assert_allclose(bounds, [[-.01, .01, -.09], [.07, .05, -.03]], atol=1e-7)
    # The actor origin is not the collision geometry center; a grasp rotation matters too.
    attachment["tcp_to_object_pose"][3:] = quaternion.tolist()
    rotated = planner.placement_bounds(attachment, [1., 0., 0., 0.])
    np.testing.assert_allclose(rotated, [[-.03, -.03, -.09], [.05, .01, -.03]], atol=1e-7)
    actor._bodies[0].get_collision_shapes = lambda: []
    with pytest.raises(MotionPlanningError, match="finite payload collision bounds"):
        planner.placement_bounds(attachment, quaternion)


def contact(position=(.45,.15,.330),normal=(0.,0.,-1.),depth=.0005):
    return SimpleNamespace(pos=np.array(position),normal=np.array(normal),penetration_depth=depth)


def test_shallow_lower_top_contact_only():
    assert _lower_support_contact_allowed([contact()],np.eye(4),.330)
    for entry in [contact(depth=.0011),contact(normal=(1.,0.,0.)),
                  contact(normal=(0.,0.,1.)),contact(position=(.45,.15,.672)),
                  contact(depth=float('nan'))]:
        assert not _lower_support_contact_allowed([entry],np.eye(4),.330)
    assert not _lower_support_contact_allowed([],np.eye(4),.330)


def test_permission_uses_root_frame_contact_geometry():
    transform=np.eye(4)
    transform[:3,3]=[.4,.12,-.2]
    assert _lower_support_contact_allowed([contact(position=(.05,.03,.530))],transform,.330)
    assert not _lower_support_contact_allowed([contact()],transform,.330)


def test_orientation_projection_preserves_tcp_path_and_endpoints():
    from scipy.spatial.transform import Rotation
    from domain.robocasa.motion_planning import CabinetMotionPlanner, MotionPlanningError
    class Model:
        def compute_forward_kinematics(self,q): self.q=np.asarray(q)
        def get_link_pose(self,_):
            return SimpleNamespace(p=np.array([self.q[0],0.,0.]),
                q=np.roll(Rotation.from_rotvec([0.,0.,self.q[1]]).as_quat(),1))
        def compute_inverse_kinematics(self,_,goal,**kwargs):
            return np.array([goal.p[0],0.]),True,0.
    model=Model()
    planner=CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.kin=SimpleNamespace(pmodel=model,end_link_idx=0,qmask=np.ones(2))
    planner.pindices=np.array([0,1]);planner.controlled=np.array([0,1])
    planner.sapien=SimpleNamespace(Pose=lambda p,q:SimpleNamespace(p=p,q=q))
    original=np.array([[0.,0.],[.1,.2],[.2,0.]])
    output=planner._fixed_orientation_path(original,[1.,0.,0.,0.],np.array([[-1.,1.],[-1.,1.]]))
    np.testing.assert_array_equal(output[[0,-1]],original[[0,-1]])
    np.testing.assert_allclose(output[:,1],0.)
    assert np.all(np.diff(output[:,0])>=0)
    model.compute_inverse_kinematics=lambda *args,**kwargs:(np.zeros(2),False,1.)
    with pytest.raises(MotionPlanningError,match='orientation_projection_failed'):
        planner._fixed_orientation_path(original,[1.,0.,0.,0.],np.array([[-1.,1.],[-1.,1.]]))
