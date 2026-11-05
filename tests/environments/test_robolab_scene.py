"""Geometry parity and standalone USD checks, without importing ManiSkill."""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from domain.cabinet_items import build_cabinet_items
from domain.robolab_cabinet import build_scene_spec, export_scene


def test_extracted_item_body_matches_original_ast():
    # SHA256 of the original method body with only sapien.Pose -> pose_type.
    # Captures exact numerical expressions and all primitive-builder calls.
    body = ast.parse(inspect.getsource(build_cabinet_items)).body[0].body
    normalized = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
    assert hashlib.sha256(normalized.encode()).hexdigest() == (
        "95aab75b2aad0d849c3581dd9f6b491839083501dfe349589aa591e9b5af3951"
    )


@pytest.mark.parametrize("layout,count", [("full", 24), ("empty", 0),
                                          ("single_upper_second_row", 1),
                                          ("two_upper_second_row", 2)])
def test_layouts_preserve_original_slots(layout, count):
    spec = build_scene_spec(layout)
    json.dumps(spec)
    assert len(spec["items"]) == count
    full = {item["name"]: item for item in build_scene_spec()["items"]}
    assert all(item == full[item["name"]] for item in spec["items"])
    assert len(spec["fixtures"][2]["shapes"]) == 8
    if layout in {"single_upper_second_row", "two_upper_second_row"}:
        mug = spec["items"][0]
        assert (mug["name"], mug["type"]) == ("cabinet_item_16", "mug")
        np.testing.assert_array_equal(mug["position"], np.asarray([.055, -.18, .744], dtype=np.float32))
        assert len([s for s in mug["shapes"] if s["purpose"] == "collision"]) == 4
    if count == 2:
        bowl = spec["items"][1]
        assert (bowl["name"], bowl["type"]) == ("cabinet_item_17", "bowl")
        assert bowl["shapes"][0]["radius"] == .036
        np.testing.assert_array_equal(bowl["position"], np.asarray([.055, -.06, .712], dtype=np.float32))


def test_reset_and_hinge_metadata():
    spec = build_scene_spec()
    assert spec["robot_base_position"] == [-.40, -.12, 0.]
    assert spec["door_hinge"]["limits_degrees"] == [0., 110.]
    assert spec["door_hinge"]["initial_angle_degrees"] == 70.
    np.testing.assert_allclose(spec["door"]["position"], [-.14, .275, .66])
    assert spec["physics"] == dict(static_friction=.3, dynamic_friction=.3,
                                  restitution=0., gravity=[0., 0., -9.81])
    with pytest.raises(ValueError, match="Unknown cabinet layout"):
        build_scene_spec("unknown")


def test_usd_export_has_dynamic_compounds_and_revolute_hinge(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "cabinet.usda"
    spec = export_scene(path)
    assert json.loads(path.with_suffix(".json").read_text()) == spec
    stage = Usd.Stage.Open(str(path))
    assert stage.GetDefaultPrim().GetPath() == "/World"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    joint = UsdPhysics.RevoluteJoint.Get(stage, "/World/door_hinge")
    assert joint and joint.GetLowerLimitAttr().Get() == 0.
    assert joint.GetUpperLimitAttr().Get() == 110.
    assert str(joint.GetBody1Rel().GetTargets()[0]) == "/World/door"
    assert not joint.GetBody0Rel().GetTargets()  # Fixed world anchor.
    assert stage.GetPrimAtPath("/World/door").HasAPI(UsdPhysics.RigidBodyAPI)
    for item in spec["items"]:
        prim = stage.GetPrimAtPath("/World/" + item["name"])
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        collisions = [child for child in prim.GetChildren() if child.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(collisions) == sum(s["purpose"] == "collision" for s in item["shapes"])
        for child in collisions:
            assert UsdPhysics.MassAPI(child).GetDensityAttr().Get() > 0.
    cylinder = UsdGeom.Cylinder.Get(stage, "/World/cabinet_item_16/shape_000")
    assert cylinder.GetAxisAttr().Get() == "X"
    matrix = UsdGeom.Xformable(cylinder).GetLocalTransformation()
    axis = matrix.TransformDir(Gf.Vec3d(1., 0., 0.)).GetNormalized()
    np.testing.assert_allclose(np.abs(list(axis)), [0., 0., 1.], atol=1e-6)
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(cylinder.GetPrim()).ComputeAlignedRange()
    np.testing.assert_allclose(list(bound.GetSize()), [.068, .068, .14], atol=1e-6)
