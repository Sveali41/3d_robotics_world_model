"""Simulator-independent cabinet specification and optional OpenUSD export."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from domain.cabinet_items import build_cabinet_items

CABINET_POSITION = [0.125, 0.0, 0.66]
ROBOT_POSITION = [-0.40, -0.12, 0.0]
PREGRASP_QPOS = [0.076, -1.100, -0.170, -2.660, 0.010, 3.130, 0.620, 0.04, 0.04]
LAYOUTS = {
    "full": None,
    "empty": set(),
    "single_upper_second_row": {(1, 1, 0)},
    "two_upper_second_row": {(1, 1, 0), (1, 1, 1)},
}


class _Pose:
    def __init__(self, p=None, q=None):
        self.p = np.asarray([0., 0., 0.] if p is None else p)
        self.q = np.asarray([1., 0., 0., 0.] if q is None else q)


class _Builder:
    def __init__(self):
        self.shapes = []
        self.initial_pose = _Pose()

    def __getattr__(self, name):
        if not name.startswith("add_"):
            raise AttributeError(name)
        _, kind, purpose = name.split("_")
        if kind not in {"box", "sphere", "cylinder"} or purpose not in {"visual", "collision"}:
            raise AttributeError(name)

        def record(pose=None, **kwargs):
            pose = pose or _Pose()
            self.shapes.append(dict(kind=kind, purpose=purpose,
                                    position=pose.p.tolist(), quaternion=pose.q.tolist(), **kwargs))
        return record

    def build(self, name):
        return dict(name=name, position=self.initial_pose.p.tolist(),
                    quaternion=self.initial_pose.q.tolist(), shapes=self.shapes)


class _Scene:
    @staticmethod
    def create_actor_builder():
        return _Builder()


def _box(position, half_size, color, density=None):
    shape = dict(kind="box", position=position, quaternion=[1., 0., 0., 0.],
                 half_size=half_size, material=color, purpose="both")
    if density is not None:
        shape["density"] = density
    return shape


def build_scene_spec(layout="full"):
    """Return all geometry, physics, and reset metadata as JSON-compatible data."""
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown cabinet layout {layout!r}; expected one of {list(LAYOUTS)}")
    capture = SimpleNamespace(scene=_Scene(), CABINET_POSE=_Pose(CABINET_POSITION),
                              _material=lambda color: list(color))
    build_cabinet_items(capture, _Pose, selected_slots=LAYOUTS[layout])
    items = capture.cabinet_items
    for item, kind in zip(items, capture.cabinet_item_types):
        item.update(type=kind, dynamic=True)
    wood, inside = [0.63, 0.39, 0.20, 1.0], [0.74, 0.55, 0.34, 1.0]
    frame_parts = [
        ([0.238, 0., 0.], [0.012, 0.30, 0.36], inside),
        ([0., 0., 0.345], [0.25, 0.30, 0.015], wood),
        ([0., 0., -0.345], [0.25, 0.30, 0.015], wood),
        ([0., 0.285, 0.], [0.25, 0.015, 0.33], wood),
        ([0., -0.285, 0.], [0.25, 0.015, 0.33], wood),
        ([0., 0., 0.], [0.235, 0.27, 0.012], inside),
        ([0.2575, 0.20, 0.22], [0.0075, 0.04, 0.04], wood),
        ([0.2575, -0.20, 0.22], [0.0075, 0.04, 0.04], wood),
    ]
    fixtures = [
        dict(name="floor", position=[0., 0., -0.05], shapes=[_box([0., 0., 0.], [2., 2., 0.05], [0.5, 0.5, 0.5, 1.])]),
        dict(name="wall", position=[0., 0., 0.], shapes=[_box([0.40, 0., 1.20], [0.025, 1.50, 1.20], [0.82, 0.84, 0.86, 1.])]),
        dict(name="frame", position=CABINET_POSITION, shapes=[_box(*part) for part in frame_parts]),
    ]
    hinge = (np.asarray(CABINET_POSITION) + [-0.265, 0.275, 0.]).tolist()
    angle = np.deg2rad(-70.)
    door = dict(name="door", dynamic=True, position=hinge,
                quaternion=[float(np.cos(angle / 2)), 0., 0., float(np.sin(angle / 2))],
                shapes=[_box([0., -0.275, 0.], [0.012, 0.27, 0.34], [0.19, 0.43, 0.62, 1.], 1000.),
                        _box([-0.035, -0.49, 0.], [0.025, 0.018, 0.095], [0.10, 0.11, 0.12, 1.], 1000.)])
    return dict(schema_version=1, layout=layout, meters_per_unit=1., up_axis="Z",
                source="domain/cabinet_items.py",
                source_sha256=hashlib.sha256(inspect.getsource(build_cabinet_items).encode()).hexdigest(),
                cabinet_position=CABINET_POSITION, robot_base_position=ROBOT_POSITION,
                pregrasp_qpos=np.asarray(PREGRASP_QPOS, dtype=np.float32).tolist(),
                exploration_bounds_local=np.asarray([[-.20, -.25, .04], [.10, .03, .30]], dtype=np.float32).tolist(),
                recovery_bounds_local=np.asarray([[-.27, -.29, -.37], [.27, .29, .36]], dtype=np.float32).tolist(),
                physics=dict(static_friction=.3, dynamic_friction=.3, restitution=0., gravity=[0., 0., -9.81]),
                fixtures=fixtures, door=door,
                door_hinge=dict(position=hinge, axis=[0., 0., -1.], initial_angle_degrees=70.,
                                limits_degrees=[0., 110.], friction=.06, damping=.20),
                items=items)


def export_scene(output_path, layout="full"):
    """Write a standalone USD and sibling JSON manifest, requiring ``usd-core``."""
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:
        raise RuntimeError("USD export requires pxr; install the usd-core package in the export environment") from exc
    spec = build_scene_spec(layout)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    physics = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
    physics.CreateGravityMagnitudeAttr(9.81)
    material = UsdShade.Material.Define(stage, "/World/physicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(.3)
    api.CreateDynamicFrictionAttr(.3)
    api.CreateRestitutionAttr(0.)

    def quat(values):
        return Gf.Quatf(float(values[0]), Gf.Vec3f(*values[1:]))

    def pose(xform, position, quaternion):
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddOrientOp().Set(quat(quaternion))

    for body in spec["fixtures"] + [spec["door"]] + spec["items"]:
        path = "/World/" + body["name"]
        root = UsdGeom.Xform.Define(stage, path)
        pose(root, body["position"], body.get("quaternion", [1., 0., 0., 0.]))
        if body.get("dynamic"):
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        for index, shape in enumerate(body["shapes"]):
            shape_path = f"{path}/shape_{index:03d}"
            if shape["kind"] == "box":
                geom = UsdGeom.Cube.Define(stage, shape_path)
                geom.CreateSizeAttr(2.)
            elif shape["kind"] == "cylinder":
                geom = UsdGeom.Cylinder.Define(stage, shape_path)
                # Match SAPIEN's local-X cylinder before applying its supplied pose.
                geom.CreateAxisAttr(UsdGeom.Tokens.x)
                geom.CreateRadiusAttr(shape["radius"])
                geom.CreateHeightAttr(2 * shape["half_length"])
            else:
                geom = UsdGeom.Sphere.Define(stage, shape_path)
                geom.CreateRadiusAttr(shape["radius"])
            pose(geom, shape["position"], shape["quaternion"])
            if shape["kind"] == "box":
                geom.AddScaleOp().Set(Gf.Vec3f(*shape["half_size"]))
            prim = geom.GetPrim()
            if shape["purpose"] in {"collision", "both"}:
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
                if "density" in shape:
                    UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr(float(shape["density"]))
            if shape["purpose"] == "collision":
                geom.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            else:
                geom.CreateDisplayColorAttr([Gf.Vec3f(*shape["material"][:3])])
                geom.CreateDisplayOpacityAttr([float(shape["material"][3])])
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/door_hinge")
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/door")])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*spec["door_hinge"]["position"]))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
    # Joint +Z is world -Z, matching the source's SAPIEN local +X hinge.
    joint.CreateLocalRot0Attr(Gf.Quatf(0., Gf.Vec3f(1., 0., 0.)))
    joint.CreateLocalRot1Attr(Gf.Quatf(0., Gf.Vec3f(1., 0., 0.)))
    joint.CreateLowerLimitAttr(0.)
    joint.CreateUpperLimitAttr(110.)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.)
    # USD angular drive damping uses torque/(degree/second), SAPIEN uses radians.
    drive.CreateDampingAttr(float(.20 * np.pi / 180.))
    # PhysX exposes joint friction outside core UsdPhysics.
    joint.GetPrim().CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(.06)
    stage.GetRootLayer().Save()
    output.with_suffix(".json").write_text(json.dumps(spec, indent=2) + "\n")
    return spec
