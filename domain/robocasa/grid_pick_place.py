"""Discrete shared cabinet XY cells with one grasp-only continuous Z argument."""
from dataclasses import replace

import numpy as np

from domain.robocasa.cabinet_grid import resolve_grid_layout
from domain.robocasa.grasp_orientations import (
    GRASP_ORIENTATION_YAW_PITCH_DEG, validate_grasp_orientation_id,
)
from domain.robocasa.pick_place import PICK_PLACE_ACTION_SCHEMA, PickPlaceActionSchema
from domain.robocasa.primitives import PrimitiveAction, PrimitiveSpec, RAPSActionSchema


class GridPickPlaceActionSchema(RAPSActionSchema):
    def __init__(self, grasp_z=None, *, grasp_only=False, grasp_orientation_candidates=False,
                 grid_layout="legacy_3x4"):
        self.grasp_orientation_candidates = grasp_orientation_candidates
        self.grasp_z = grasp_z or PICK_PLACE_ACTION_SCHEMA.spec_by_name["grasp"].parameters[2]
        if grasp_z is None and grid_layout == "trapezoid_4_3_2_1":
            self.grasp_z = replace(self.grasp_z, center=.732, scale=.04)
        self.grid_layout = grid_layout
        self.grid_xy_root_m = tuple(resolve_grid_layout(grid_layout)[0])
        self.logical_primitive_names = ("grasp",) if grasp_only else ("grasp", "place")
        # The current trapezoid keeps the three horizontal poses with full-grasp evidence.
        # Preserve the five-pose legacy/compact experiment interfaces.
        orientation_count = 3 if grid_layout == "trapezoid_4_3_2_1" else len(GRASP_ORIENTATION_YAW_PITCH_DEG)
        self.grasp_orientation_ids = tuple(range(orientation_count)) if grasp_orientation_candidates else ()
        orientations = self.grasp_orientation_ids if grasp_orientation_candidates else (None,)
        branches = [("grasp", cell, orientation) for cell in range(len(self.grid_xy_root_m))
                    for orientation in orientations]
        if not grasp_only:
            branches.extend(("place", cell, None) for cell in range(len(self.grid_xy_root_m)))
        self.branch_primitive_types = tuple(name for name, _, _ in branches)
        self.branch_cell_indices = tuple(cell for _, cell, _ in branches)
        self.branch_orientation_indices = tuple(orientation for _, _, orientation in branches)
        self.num_grasp_branches = sum(name == "grasp" for name, _, _ in branches)
        self.primitive_names = tuple(
            f"{name}_{cell:02d}_orientation_{orientation}" if orientation is not None else f"{name}_{cell:02d}"
            for name, cell, orientation in branches)
        self.specs = tuple(PrimitiveSpec(name, (self.grasp_z,) if logical == "grasp" else ())
                           for name, logical in zip(self.primitive_names, self.branch_primitive_types))
        self.spec_by_name = {spec.name: spec for spec in self.specs}
        self.num_primitives = len(self.specs)
        self.primitive_name_to_action_idx = {
            name: [0] if logical == "grasp" else []
            for name, logical in zip(self.primitive_names, self.branch_primitive_types)}
        self.parameter_slices = {name: (0, len(indices))
                                 for name, indices in self.primitive_name_to_action_idx.items()}
        self.max_arg_len = self.num_parameters = 1
        self.action_dim = self.num_primitives + 1
        self.low = np.concatenate((np.zeros(self.num_primitives), [-1.])).astype(np.float32)
        self.high = np.ones(self.action_dim, dtype=np.float32)

    def physical_action_schema(self):
        """Validate grid TCP targets without inheriting legacy continuous XY bounds."""
        xy = np.asarray(self.grid_xy_root_m)
        grasp = PICK_PLACE_ACTION_SCHEMA.spec_by_name["grasp"]
        parameters = tuple(replace(parameter, center=float((xy[:, axis].min() + xy[:, axis].max()) / 2),
                                   scale=float((xy[:, axis].max() - xy[:, axis].min()) / 2))
                           for axis, parameter in enumerate(grasp.parameters[:2])) + (self.grasp_z,)
        return PickPlaceActionSchema((replace(grasp, parameters=parameters),
                                      PICK_PLACE_ACTION_SCHEMA.spec_by_name["place"]))

    def encode(self, action):
        if action.primitive_type not in self.logical_primitive_names:
            raise KeyError(f"Unknown primitive {action.primitive_type!r}; available={self.logical_primitive_names}")
        expected = {"x", "y", "z"} if action.primitive_type == "grasp" else {"x", "y"}
        if action.primitive_type == "grasp" and self.grasp_orientation_candidates:
            expected.add("orientation_id")
        if set(action.parameters) != expected:
            raise ValueError(f"{action.primitive_type} requires exactly {sorted(expected)}")
        xy = np.asarray([action.parameters[axis] for axis in "xy"], dtype=float)
        # Permit only float32 representation error, never snap continuous proposals.
        matches = np.flatnonzero(np.all(np.isclose(self.grid_xy_root_m, xy, rtol=0., atol=3e-8), axis=1))
        if not np.isfinite(xy).all() or len(matches) != 1:
            raise ValueError("Position XY must be a configured cabinet grid cell")
        orientation = (validate_grasp_orientation_id(action.parameters["orientation_id"])
                       if action.primitive_type == "grasp" and self.grasp_orientation_candidates else None)
        if orientation is not None and orientation not in self.grasp_orientation_ids:
            raise ValueError(f"orientation_id must be one of {self.grasp_orientation_ids} for {self.grid_layout}")
        index = next(index for index, branch in enumerate(zip(
            self.branch_primitive_types, self.branch_cell_indices, self.branch_orientation_indices))
            if branch == (action.primitive_type, int(matches[0]), orientation))
        vector = np.zeros(self.action_dim, dtype=np.float32)
        vector[index] = 1.
        if action.primitive_type == "grasp":
            vector[-1] = self.grasp_z.normalize(action.parameters["z"])
        return vector

    def sample_parameters(self, primitive_type, rng):
        return {p.name: p.denormalize(float(rng.uniform(-1., 1.)))
                for p in self.spec_by_name[primitive_type].parameters}

    def decode(self, vector):
        branch = super().decode(vector)
        index = self.primitive_names.index(branch.primitive_type)
        x, y = self.grid_xy_root_m[self.branch_cell_indices[index]]
        parameters = {"x": x, "y": y, **branch.parameters}
        if self.branch_orientation_indices[index] is not None:
            parameters["orientation_id"] = self.branch_orientation_indices[index]
        return PrimitiveAction(self.branch_primitive_types[index], parameters)
