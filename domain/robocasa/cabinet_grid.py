"""Shared row-major shelf locations in robot-root meters."""
import numpy as np


GRID_X_ROOT_M = (.395, .515, .635)
GRID_Y_ROOT_M = (-.06, .06, .18, .30)
GRID_XY_ROOT_M = tuple((x, y) for x in GRID_X_ROOT_M for y in GRID_Y_ROOT_M)
GRID_INITIAL_CELLS = {"cabinet_item_16": 4, "cabinet_item_17": 5}
COMPACT_GRID_XY_ROOT_M = tuple((x, y) for x in (.375, .405, .435) for y in (0., .08, .16, .24))
COMPACT_GRID_INITIAL_CELLS = {"cabinet_item_16": 4, "cabinet_item_17": 7}
TRAPEZOID_GRID_XY_ROOT_M = ((.335, -.06), (.335, .06), (.335, .18), (.335, .30),
                            (.410, 0.), (.410, .12), (.410, .24),
                            (.485, .06), (.485, .18), (.560, .12))
TRAPEZOID_GRID_INITIAL_CELLS = {"cabinet_item_16": 4, "cabinet_item_17": 6}


def resolve_grid_layout(name="legacy_3x4"):
    if name == "legacy_3x4":
        return GRID_XY_ROOT_M, GRID_INITIAL_CELLS
    if name == "compact_3x4":
        return COMPACT_GRID_XY_ROOT_M, COMPACT_GRID_INITIAL_CELLS
    if name == "trapezoid_4_3_2_1":
        return TRAPEZOID_GRID_XY_ROOT_M, TRAPEZOID_GRID_INITIAL_CELLS
    raise ValueError(f"Unknown cabinet grid layout: {name!r}")


def local_collision_bounds(actor):
    """Reuse exact native-shape support, including PhysX cylinder axes."""
    import sapien
    from domain.robocasa.motion_planning import CabinetMotionPlanner, _plane_min_distance

    h = 2**-.5
    positive = ([h, 0, h, 0], [h, -h, 0, 0], [1, 0, 0, 0])
    negative = ([h, 0, -h, 0], [h, h, 0, 0], [0, 1, 0, 0])
    bounds = []
    for shape in actor._bodies[0].get_collision_shapes():
        kind = CabinetMotionPlanner._kind(shape)
        pose = shape.local_pose
        if kind in {"Cylinder", "Plane"}:
            pose = pose * sapien.Pose(q=[h, 0, h, 0])
        lo = [_plane_min_distance(shape, pose, sapien.Pose(q=q), kind) for q in positive]
        hi = [-_plane_min_distance(shape, pose, sapien.Pose(q=q), kind) for q in negative]
        bounds.append([lo, hi])
    bounds = np.asarray(bounds)
    return np.array([bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)])
