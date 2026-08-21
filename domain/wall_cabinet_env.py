"""A self-contained ManiSkill task with a wall-mounted cabinet and Panda.

Importing this module registers ``WallCabinet-v1`` with Gymnasium.  The
cabinet is built from primitive collision shapes, so no external asset bundle
or PartNet download is required.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.panda import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import SimConfig


@register_env("WallCabinet-v1", max_episode_steps=200, override=True)
class WallCabinetEnv(BaseEnv):
    """Explore a wall-mounted open cabinet with a Panda robot.

    The Panda is fixed to the floor directly in front of the cabinet.  The
    cabinet frame is fixed to a vertical wall, while its door is a physical
    revolute link that opens towards the robot.

    The door starts visibly open at 70 degrees and is not itself a task goal.
    The base environment is reward-free so object-interaction rewards can be
    supplied explicitly by a high-level exploration wrapper.
    """

    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: PandaWristCam

    CABINET_POSE = sapien.Pose([0.125, 0.0, 0.66])
    # Slight y offset keeps the arm clear of the door's opening arc while the
    # closer x position makes the rear item row reachable.
    # Move the fixed base toward the cabinet so the selected second-row items
    # are inside Panda's reliable forward IK workspace.
    ROBOT_POSE = sapien.Pose([-0.40, -0.12, 0.0])
    INITIAL_OPEN_ANGLE = np.deg2rad(70.0)
    MAX_OPEN_ANGLE = np.deg2rad(110.0)
    EXPLORATION_REGION_BOUNDS_LOCAL = np.array(
        [[-0.20, -0.25, 0.04], [0.10, 0.03, 0.30]], dtype=np.float32
    )
    OBJECT_RECOVERY_BOUNDS_LOCAL = np.array(
        [[-0.27, -0.29, -0.37], [0.27, 0.29, 0.36]], dtype=np.float32
    )
    TWO_ITEM_OBSERVATION_NAMES = ("cabinet_item_16", "cabinet_item_17")
    # Horizontal side-grasp reset: the open gripper starts near the cabinet
    # opening, in front of the upper-shelf objects.  TCP ~= [0.35, -0.06,
    # 0.72] in robot-base coordinates.  The finger extension axis points into
    # the cabinet (+robot-base x), while the two fingers open along y.
    PREGRASP_TCP_POSITION = np.array([0.35, -0.06, 0.72], dtype=np.float32)
    # wxyz quaternion. This maps TCP local +z (finger extension) to +x.
    PREGRASP_TCP_QUATERNION = np.array(
        [0.0, 0.70710678, 0.0, 0.70710678], dtype=np.float32
    )
    PREGRASP_QPOS = np.array(
        [
            0.076,
            -1.100,
            -0.170,
            -2.660,
            0.010,
            3.130,
            0.620,
            0.04,
            0.04,
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        *args,
        robot_uids: str = "panda_wristcam",
        populate_cabinet: bool = True,
        cabinet_item_layout: str | None = None,
        visualize_exploration_region: bool = False,
        **kwargs,
    ):
        self.populate_cabinet = populate_cabinet
        self.visualize_exploration_region = bool(visualize_exploration_region)
        self.exploration_region_bounds_local = (
            self.EXPLORATION_REGION_BOUNDS_LOCAL.copy()
        )
        if cabinet_item_layout is None:
            cabinet_item_layout = "full" if populate_cabinet else "empty"
        if cabinet_item_layout not in {
            "full",
            "empty",
            "single_upper_second_row",
            "two_upper_second_row",
        }:
            raise ValueError(f"Unknown cabinet_item_layout={cabinet_item_layout!r}")
        self.cabinet_item_layout = cabinet_item_layout
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=20)

    @property
    def _default_sensor_configs(self):
        # A fixed overview camera complements Panda's wrist camera.
        pose = sapien_utils.look_at([-0.75, -0.80, 1.10], [-0.02, 0.0, 0.58])
        return [
            CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        # Close, elevated view into the open cabinet. This is also the camera
        # used by ``render_mode=rgb_array`` when generating grasp GIFs.
        pose = sapien_utils.look_at([-0.82, -0.46, 1.22], [0.04, -0.08, 0.74])
        return CameraConfig("render_camera", pose, 640, 512, 0.82, 0.01, 100)

    def render_rgb_array(self, camera_name: str | None = None):
        """Render viewer markers, then restore their sensor-hidden state."""
        try:
            return super().render_rgb_array(camera_name)
        finally:
            if self.exploration_region is not None:
                self.exploration_region.hide_visual()

    def _load_agent(self, options: dict):
        super()._load_agent(options, self.ROBOT_POSE)

    @staticmethod
    def _material(color):
        return sapien.render.RenderMaterial(
            base_color=np.asarray(color, dtype=np.float32),
            roughness=0.65,
            specular=0.15,
        )

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene, floor_width=4, floor_length=4)

        wall_material = self._material([0.82, 0.84, 0.86, 1.0])
        wall_builder = self.scene.create_actor_builder()
        wall_builder.add_box_collision(
            pose=sapien.Pose([0.40, 0.0, 1.20]),
            half_size=[0.025, 1.50, 1.20],
        )
        wall_builder.add_box_visual(
            pose=sapien.Pose([0.40, 0.0, 1.20]),
            half_size=[0.025, 1.50, 1.20],
            material=wall_material,
        )
        wall_builder.initial_pose = sapien.Pose()
        self.wall = wall_builder.build_static("wall")

        cabinet_builder = self.scene.create_articulation_builder()
        cabinet_builder.initial_pose = self.CABINET_POSE
        cabinet_builder.disable_self_collisions = True

        wood = self._material([0.63, 0.39, 0.20, 1.0])
        inside = self._material([0.74, 0.55, 0.34, 1.0])
        door_material = self._material([0.19, 0.43, 0.62, 1.0])
        handle_material = self._material([0.10, 0.11, 0.12, 1.0])

        frame = cabinet_builder.create_link_builder(parent=None)
        frame.set_name("frame")

        # Cabinet outer dimensions: 0.50 m deep, 0.60 m wide, 0.72 m high.
        frame_parts = [
            # pose, half size, material
            ([0.238, 0.0, 0.0], [0.012, 0.30, 0.36], inside),  # back
            ([0.0, 0.0, 0.345], [0.25, 0.30, 0.015], wood),  # top
            ([0.0, 0.0, -0.345], [0.25, 0.30, 0.015], wood),  # bottom
            ([0.0, 0.285, 0.0], [0.25, 0.015, 0.33], wood),  # left
            ([0.0, -0.285, 0.0], [0.25, 0.015, 0.33], wood),  # right
            ([0.0, 0.0, 0.0], [0.235, 0.27, 0.012], inside),  # shelf
            # Two short mounting blocks visibly connect the frame to the wall.
            ([0.2575, 0.20, 0.22], [0.0075, 0.04, 0.04], wood),
            ([0.2575, -0.20, 0.22], [0.0075, 0.04, 0.04], wood),
        ]
        for position, half_size, material in frame_parts:
            part_pose = sapien.Pose(position)
            frame.add_box_collision(part_pose, half_size=half_size)
            frame.add_box_visual(part_pose, half_size=half_size, material=material)

        door = cabinet_builder.create_link_builder(parent=frame)
        door.set_name("door")
        door.set_joint_name("door_hinge")

        # SAPIEN revolute joints rotate about their local x-axis.  This
        # quaternion maps it to -z, so a positive angle swings the door out
        # towards the robot instead of through the cabinet.
        hinge_orientation = [0.7071068, 0.0, 0.7071068, 0.0]
        door.set_joint_properties(
            type="revolute",
            limits=[[0.0, self.MAX_OPEN_ANGLE]],
            pose_in_parent=sapien.Pose(
                [-0.265, 0.275, 0.0], hinge_orientation
            ),
            pose_in_child=sapien.Pose([0.0, 0.0, 0.0], hinge_orientation),
            friction=0.06,
            damping=0.20,
        )

        door_pose = sapien.Pose([0.0, -0.275, 0.0])
        door.add_box_collision(door_pose, half_size=[0.012, 0.27, 0.34])
        door.add_box_visual(
            door_pose,
            half_size=[0.012, 0.27, 0.34],
            material=door_material,
        )

        # A protruding vertical handle near the free edge of the door.
        handle_pose = sapien.Pose([-0.035, -0.49, 0.0])
        door.add_box_collision(handle_pose, half_size=[0.025, 0.018, 0.095])
        door.add_box_visual(
            handle_pose,
            half_size=[0.025, 0.018, 0.095],
            material=handle_material,
        )

        self.cabinet = cabinet_builder.build("wall_cabinet", fix_root_link=True)
        self.door_link = self.cabinet.links_map["door"]
        self.door_joint = self.cabinet.active_joints_map["door_hinge"]

        with torch.device(self.device):
            handle_offset = torch.tensor([[-0.035, -0.49, 0.0]])
            self._handle_local_pose = Pose.create_from_pq(p=handle_offset)

        if self.cabinet_item_layout == "full":
            self._build_cabinet_items()
        elif self.cabinet_item_layout == "single_upper_second_row":
            # Reuse the exact first slot/object from the full layout's upper
            # shelf (level 2), second depth row: a handled mug.
            self._build_cabinet_items(selected_slots={(1, 1, 0)})
        elif self.cabinet_item_layout == "two_upper_second_row":
            # The mug plus its original immediate neighbor, a bowl.
            self._build_cabinet_items(selected_slots={(1, 1, 0), (1, 1, 1)})
        else:
            self.cabinet_items = []
            self.cabinet_item_types = []
            self._item_spawn_positions = []

        self._two_item_observation_actors = {}
        if self.cabinet_item_layout == "two_upper_second_row":
            actors_by_name = {item.name: item for item in self.cabinet_items}
            missing_names = set(self.TWO_ITEM_OBSERVATION_NAMES) - set(actors_by_name)
            if missing_names:
                raise RuntimeError(
                    "Two-item cabinet layout is missing its observation actors: "
                    f"{sorted(missing_names)}"
                )
            self._two_item_observation_actors = {
                name: actors_by_name[name] for name in self.TWO_ITEM_OBSERVATION_NAMES
            }

        self.exploration_region = None
        if self.visualize_exploration_region:
            self._build_exploration_region()

    def _build_exploration_region(self):
        """Build a viewer-only wireframe around the two-item search volume.

        Bounds are expressed in the cabinet frame so the same values can later
        be reused by an EE workspace constraint and its reward. The marker has
        no collision geometry and is hidden from robot sensors / point clouds
        through ManiSkill's ``_hidden_objects`` mechanism.
        """
        lower, upper = self.exploration_region_bounds_local

        if not self.scene.can_render():
            return

        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        thickness = 0.0035
        marker_material = self._material([0.05, 0.90, 0.95, 0.85])
        builder = self.scene.create_actor_builder()

        # Four edges parallel to each cabinet-local axis form a lightweight
        # wireframe without adding collision surfaces around the objects.
        for y in (-half[1], half[1]):
            for z in (-half[2], half[2]):
                builder.add_box_visual(
                    pose=sapien.Pose(center + [0.0, y, z]),
                    half_size=[half[0], thickness, thickness],
                    material=marker_material,
                )
        for x in (-half[0], half[0]):
            for z in (-half[2], half[2]):
                builder.add_box_visual(
                    pose=sapien.Pose(center + [x, 0.0, z]),
                    half_size=[thickness, half[1], thickness],
                    material=marker_material,
                )
        for x in (-half[0], half[0]):
            for y in (-half[1], half[1]):
                builder.add_box_visual(
                    pose=sapien.Pose(center + [x, y, 0.0]),
                    half_size=[thickness, thickness, half[2]],
                    material=marker_material,
                )

        builder.initial_pose = self.CABINET_POSE
        self.exploration_region = builder.build_kinematic(
            "two_item_exploration_region"
        )
        self._hidden_objects.append(self.exploration_region)
        self.exploration_region.hide_visual()

    def ee_position_in_cabinet_frame(self) -> np.ndarray:
        """Return the Panda TCP position in the fixed cabinet-local frame."""
        ee_world = self.agent.tcp.pose.p[0].detach().cpu().numpy()
        cabinet_world = np.asarray(
            self.CABINET_POSE.to_transformation_matrix(), dtype=np.float32
        )
        ee_world_h = np.concatenate(
            [np.asarray(ee_world, dtype=np.float32), np.ones(1, dtype=np.float32)]
        )
        return (np.linalg.inv(cabinet_world) @ ee_world_h)[:3]

    def is_ee_inside_exploration_region(self) -> bool:
        """Whether the TCP is inside the viewer's two-object search volume."""
        position = self.ee_position_in_cabinet_frame()
        lower, upper = self.exploration_region_bounds_local
        return bool(np.all(position >= lower) and np.all(position <= upper))

    def failed_object_ids(self) -> list[str]:
        """Return objects that left the recoverable cabinet volume."""
        lower, upper = self.OBJECT_RECOVERY_BOUNDS_LOCAL
        cabinet_matrix = np.asarray(
            self.CABINET_POSE.to_transformation_matrix(), dtype=np.float32
        )
        cabinet_inv = np.linalg.inv(cabinet_matrix)
        failed = []
        for item in self.cabinet_items:
            position = item.pose.p[0].detach().cpu().numpy().astype(np.float32)
            homogeneous = np.concatenate([position, np.ones(1, dtype=np.float32)])
            local_position = (cabinet_inv @ homogeneous)[:3]
            if np.any(local_position < lower) or np.any(local_position > upper):
                failed.append(item.name)
        return sorted(failed)

    def _build_cabinet_items(self, selected_slots=None):
        """Fill both cabinet compartments with movable procedural objects."""
        colors = [
            [0.84, 0.20, 0.18, 1.0],
            [0.95, 0.55, 0.12, 1.0],
            [0.96, 0.82, 0.20, 1.0],
            [0.25, 0.65, 0.32, 1.0],
            [0.12, 0.62, 0.72, 1.0],
            [0.25, 0.42, 0.82, 1.0],
            [0.54, 0.30, 0.72, 1.0],
            [0.88, 0.36, 0.58, 1.0],
            [0.75, 0.68, 0.52, 1.0],
            [0.35, 0.72, 0.62, 1.0],
            [0.68, 0.32, 0.18, 1.0],
            [0.82, 0.82, 0.86, 1.0],
        ]
        label_color = self._material([0.94, 0.93, 0.84, 1.0])
        cap_color = self._material([0.12, 0.13, 0.15, 1.0])
        # SAPIEN cylinders point along local x by default; rotate them to z.
        upright_cylinder_pose = sapien.Pose(q=[0.7071068, 0.0, 0.7071068, 0.0])

        # Three depth rows x four width columns x two shelf levels.  Small
        # clearances keep all 24 dynamic items stable while making the cabinet
        # look full from the open front.
        # Keep all three rows in the front half of the deep cabinet. Together
        # with the lower mounting height and closer robot base, every item's
        # center remains inside Panda's reachable workspace.
        x_rows = [-0.19, -0.07, 0.05]
        y_columns = [-0.18, -0.06, 0.06, 0.18]
        shelf_surfaces = [-0.330, 0.012]

        self.cabinet_items = []
        self.cabinet_item_types = []
        self._item_spawn_positions = []
        item_type_names = (
            "carton",
            "can",
            "book_stack",
            "bottle",
            "mug",
            "bowl",
            "fruit",
            "lidded_jar",
            "plate_stack",
            "toy_blocks",
            "thermos",
            "tissue_box",
            "vase",
            "cooking_pot",
        )
        item_index = 0
        for level, surface_z in enumerate(shelf_surfaces):
            for row, local_x in enumerate(x_rows):
                for column, local_y in enumerate(y_columns):
                    if selected_slots is not None and (
                        level,
                        row,
                        column,
                    ) not in selected_slots:
                        item_index += 1
                        continue
                    builder = self.scene.create_actor_builder()
                    color = colors[item_index % len(colors)]
                    material = self._material(color)
                    item_kind = (item_index + row + level) % len(item_type_names)

                    if item_kind == 0:
                        # Carton/package, with a bright label on its front.
                        half_x = 0.042 + 0.004 * (column % 2)
                        half_y = 0.034 + 0.004 * (row % 2)
                        half_z = 0.105
                        half_size = [half_x, half_y, half_z]
                        builder.add_box_collision(
                            half_size=half_size, density=280
                        )
                        builder.add_box_visual(
                            half_size=half_size, material=material
                        )
                        builder.add_box_visual(
                            pose=sapien.Pose([-half_x - 0.001, 0.0, 0.0]),
                            half_size=[0.001, half_y * 0.72, half_z * 0.42],
                            material=label_color,
                        )
                        center_height = half_z
                    elif item_kind == 1:
                        # Food can.
                        radius = 0.032 + 0.003 * (column % 2)
                        half_height = 0.088 + 0.006 * (row % 2)
                        builder.add_cylinder_collision(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            density=320,
                        )
                        builder.add_cylinder_visual(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            material=material,
                        )
                        builder.add_cylinder_visual(
                            pose=sapien.Pose(
                                [0.0, 0.0, half_height + 0.002],
                                upright_cylinder_pose.q,
                            ),
                            radius=radius * 0.92,
                            half_length=0.002,
                            material=cap_color,
                        )
                        center_height = half_height
                    elif item_kind == 2:
                        # A short stack of differently colored books.
                        book_half_size = [0.045, 0.040, 0.010]
                        book_heights = [-0.060, -0.036, -0.012, 0.012, 0.036, 0.060]
                        for book_index, book_height in enumerate(book_heights):
                            book_pose = sapien.Pose([0.0, 0.0, book_height])
                            book_material = self._material(
                                colors[(item_index + book_index) % len(colors)]
                            )
                            builder.add_box_collision(
                                pose=book_pose,
                                half_size=book_half_size,
                                density=250,
                            )
                            builder.add_box_visual(
                                pose=book_pose,
                                half_size=book_half_size,
                                material=book_material,
                            )
                        center_height = 0.072
                    elif item_kind == 3:
                        # Bottle with a narrower neck/cap, represented as one
                        # rigid actor so it remains graspable and stable.
                        radius = 0.030
                        body_half_height = 0.078
                        cap_half_height = 0.020
                        cap_z = body_half_height + cap_half_height
                        builder.add_cylinder_collision(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=body_half_height,
                            density=260,
                        )
                        builder.add_cylinder_visual(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=body_half_height,
                            material=material,
                        )
                        builder.add_cylinder_collision(
                            pose=sapien.Pose(
                                [0.0, 0.0, cap_z], upright_cylinder_pose.q
                            ),
                            radius=0.018,
                            half_length=cap_half_height,
                            density=260,
                        )
                        builder.add_cylinder_visual(
                            pose=sapien.Pose(
                                [0.0, 0.0, cap_z], upright_cylinder_pose.q
                            ),
                            radius=0.018,
                            half_length=cap_half_height,
                            material=cap_color,
                        )
                        center_height = body_half_height
                    elif item_kind == 4:
                        # Cup with a three-piece handle.
                        radius = 0.034
                        half_height = 0.070
                        builder.add_cylinder_collision(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            density=240,
                        )
                        builder.add_cylinder_visual(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            material=material,
                        )
                        handle_parts = [
                            ([0.0, 0.054, 0.0], [0.011, 0.009, 0.034]),
                            ([0.0, 0.044, 0.039], [0.011, 0.019, 0.007]),
                            ([0.0, 0.044, -0.039], [0.011, 0.019, 0.007]),
                        ]
                        for part_position, part_half_size in handle_parts:
                            part_pose = sapien.Pose(part_position)
                            builder.add_box_collision(
                                pose=part_pose,
                                half_size=part_half_size,
                                density=240,
                            )
                            builder.add_box_visual(
                                pose=part_pose,
                                half_size=part_half_size,
                                material=material,
                            )
                        center_height = half_height
                    elif item_kind == 5:
                        # A shallow bowl with a contrasting inner surface.
                        radius = 0.044
                        half_height = 0.038
                        builder.add_cylinder_collision(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            density=280,
                        )
                        builder.add_cylinder_visual(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            material=material,
                        )
                        builder.add_cylinder_visual(
                            pose=sapien.Pose(
                                [0.0, 0.0, half_height + 0.002],
                                upright_cylinder_pose.q,
                            ),
                            radius=radius * 0.72,
                            half_length=0.002,
                            material=cap_color,
                        )
                        center_height = half_height
                    elif item_kind == 6:
                        # Apple/orange-like fruit with a small stem. A hidden
                        # flat foot prevents it rolling out during reset.
                        radius = 0.043
                        builder.add_sphere_collision(radius=radius, density=260)
                        builder.add_sphere_visual(radius=radius, material=material)
                        foot_pose = sapien.Pose([0.0, 0.0, -0.043])
                        builder.add_box_collision(
                            pose=foot_pose,
                            half_size=[0.027, 0.027, 0.006],
                            density=260,
                        )
                        stem_pose = sapien.Pose([0.0, 0.0, 0.049])
                        builder.add_box_visual(
                            pose=stem_pose,
                            half_size=[0.006, 0.006, 0.012],
                            material=cap_color,
                        )
                        center_height = 0.049
                    elif item_kind == 7:
                        # Squat glass/jam jar with a broad lid.
                        radius = 0.040
                        half_height = 0.062
                        lid_pose = sapien.Pose(
                            [0.0, 0.0, half_height + 0.008],
                            upright_cylinder_pose.q,
                        )
                        builder.add_cylinder_collision(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            density=300,
                        )
                        builder.add_cylinder_visual(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            material=material,
                        )
                        builder.add_cylinder_collision(
                            pose=lid_pose,
                            radius=radius * 1.04,
                            half_length=0.008,
                            density=300,
                        )
                        builder.add_cylinder_visual(
                            pose=lid_pose,
                            radius=radius * 1.04,
                            half_length=0.008,
                            material=cap_color,
                        )
                        center_height = half_height
                    elif item_kind == 8:
                        # Stack of shallow plates.
                        plate_radius = 0.046
                        plate_half_height = 0.004
                        plate_heights = [-0.020, -0.010, 0.0, 0.010, 0.020]
                        for plate_index, plate_height in enumerate(plate_heights):
                            plate_pose = sapien.Pose(
                                [0.0, 0.0, plate_height],
                                upright_cylinder_pose.q,
                            )
                            plate_material = self._material(
                                colors[(item_index + plate_index) % len(colors)]
                            )
                            builder.add_cylinder_collision(
                                pose=plate_pose,
                                radius=plate_radius,
                                half_length=plate_half_height,
                                density=260,
                            )
                            builder.add_cylinder_visual(
                                pose=plate_pose,
                                radius=plate_radius,
                                half_length=plate_half_height,
                                material=plate_material,
                            )
                        center_height = 0.026
                    elif item_kind == 9:
                        # A small multicolor toy/block construction.
                        block_specs = [
                            ([-0.020, 0.0, -0.020], [0.020, 0.038, 0.030]),
                            ([0.022, 0.0, -0.020], [0.018, 0.038, 0.030]),
                            ([0.0, 0.0, 0.030], [0.038, 0.022, 0.020]),
                        ]
                        for block_index, (block_position, block_size) in enumerate(
                            block_specs
                        ):
                            block_pose = sapien.Pose(block_position)
                            block_material = self._material(
                                colors[(item_index + block_index) % len(colors)]
                            )
                            builder.add_box_collision(
                                pose=block_pose,
                                half_size=block_size,
                                density=220,
                            )
                            builder.add_box_visual(
                                pose=block_pose,
                                half_size=block_size,
                                material=block_material,
                            )
                        center_height = 0.052
                    elif item_kind == 10:
                        # Tall thermos with a cap and a contrasting grip band.
                        radius = 0.032
                        half_height = 0.100
                        builder.add_cylinder_collision(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            density=300,
                        )
                        builder.add_cylinder_visual(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            material=material,
                        )
                        cap_pose = sapien.Pose(
                            [0.0, 0.0, 0.112], upright_cylinder_pose.q
                        )
                        builder.add_cylinder_collision(
                            pose=cap_pose,
                            radius=0.027,
                            half_length=0.012,
                            density=300,
                        )
                        builder.add_cylinder_visual(
                            pose=cap_pose,
                            radius=0.027,
                            half_length=0.012,
                            material=cap_color,
                        )
                        builder.add_cylinder_visual(
                            pose=sapien.Pose(
                                [0.0, 0.0, 0.025], upright_cylinder_pose.q
                            ),
                            radius=radius + 0.002,
                            half_length=0.018,
                            material=label_color,
                        )
                        center_height = half_height
                    elif item_kind == 11:
                        # Tissue box with one raised sheet.
                        half_size = [0.050, 0.040, 0.045]
                        builder.add_box_collision(
                            half_size=half_size, density=180
                        )
                        builder.add_box_visual(
                            half_size=half_size, material=material
                        )
                        builder.add_box_visual(
                            pose=sapien.Pose([0.0, 0.0, 0.068]),
                            half_size=[0.020, 0.004, 0.025],
                            material=label_color,
                        )
                        center_height = half_size[2]
                    elif item_kind == 12:
                        # Rounded vase with a narrow neck.
                        body_pose = sapien.Pose([0.0, 0.0, -0.014])
                        neck_pose = sapien.Pose(
                            [0.0, 0.0, 0.046], upright_cylinder_pose.q
                        )
                        builder.add_sphere_collision(
                            pose=body_pose, radius=0.043, density=260
                        )
                        builder.add_sphere_visual(
                            pose=body_pose, radius=0.043, material=material
                        )
                        builder.add_cylinder_collision(
                            pose=neck_pose,
                            radius=0.021,
                            half_length=0.035,
                            density=260,
                        )
                        builder.add_cylinder_visual(
                            pose=neck_pose,
                            radius=0.021,
                            half_length=0.035,
                            material=material,
                        )
                        # Flat foot keeps the rounded vase upright.
                        builder.add_box_collision(
                            pose=sapien.Pose([0.0, 0.0, -0.058]),
                            half_size=[0.027, 0.027, 0.005],
                            density=260,
                        )
                        center_height = 0.064
                    else:
                        # Small cooking pot with two side handles and a lid.
                        radius = 0.044
                        half_height = 0.043
                        builder.add_cylinder_collision(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            density=310,
                        )
                        builder.add_cylinder_visual(
                            pose=upright_cylinder_pose,
                            radius=radius,
                            half_length=half_height,
                            material=material,
                        )
                        for handle_y in (-0.057, 0.057):
                            handle_pose = sapien.Pose([0.0, handle_y, 0.005])
                            builder.add_box_collision(
                                pose=handle_pose,
                                half_size=[0.018, 0.014, 0.010],
                                density=310,
                            )
                            builder.add_box_visual(
                                pose=handle_pose,
                                half_size=[0.018, 0.014, 0.010],
                                material=cap_color,
                            )
                        lid_pose = sapien.Pose(
                            [0.0, 0.0, half_height + 0.004],
                            upright_cylinder_pose.q,
                        )
                        builder.add_cylinder_visual(
                            pose=lid_pose,
                            radius=radius * 0.92,
                            half_length=0.004,
                            material=cap_color,
                        )
                        center_height = half_height

                    world_position = np.array(
                        [
                            self.CABINET_POSE.p[0] + local_x,
                            self.CABINET_POSE.p[1] + local_y,
                            self.CABINET_POSE.p[2]
                            + surface_z
                            + center_height
                            + 0.002,
                        ],
                        dtype=np.float32,
                    )
                    builder.initial_pose = sapien.Pose(world_position)
                    item = builder.build(f"cabinet_item_{item_index:02d}")
                    self.cabinet_items.append(item)
                    self.cabinet_item_types.append(item_type_names[item_kind])
                    self._item_spawn_positions.append(world_position)
                    item_index += 1

    @property
    def handle_position(self):
        return (self.door_link.pose * self._handle_local_pose).p

    @property
    def door_angle(self):
        return self.door_joint.qpos

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            batch_size = len(env_idx)

            initial_angle = torch.full(
                (batch_size, 1), float(self.INITIAL_OPEN_ANGLE)
            )
            self.cabinet.set_qpos(initial_angle)
            self.cabinet.set_qvel(torch.zeros((batch_size, 1)))

            for item, spawn_position in zip(
                self.cabinet_items, self._item_spawn_positions
            ):
                positions = torch.as_tensor(spawn_position).repeat(batch_size, 1)
                item.set_pose(Pose.create_from_pq(p=positions))
                item.set_linear_velocity(torch.zeros((batch_size, 3)))
                item.set_angular_velocity(torch.zeros((batch_size, 3)))

            # A collision-free ready configuration facing the cabinet.
            qpos = self.PREGRASP_QPOS.copy()
            qpos = np.repeat(qpos[None, :], batch_size, axis=0)
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(self.ROBOT_POSE)

    def evaluate(self):
        return {
            # This is an exploration scene, not a door-opening task. A grasp
            # exploration wrapper can replace this with its own success event.
            "success": torch.zeros_like(self.door_angle, dtype=torch.bool),
            "door_angle": self.door_angle,
            "handle_position": self.handle_position,
        }

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "door_angle": info["door_angle"],
        }
        if "state" in self.obs_mode:
            obs.update(
                handle_position=info["handle_position"],
                tcp_to_handle=info["handle_position"] - self.agent.tcp.pose.p,
            )
            if self._two_item_observation_actors:
                # PrimitiveContext controls in the arm controller's base
                # frame. Keep object-relative observations in that same frame
                # so a positive policy delta has a consistent meaning.
                arm_controller = self.agent.controller.controllers["arm"]
                tcp_position = arm_controller.ee_pose_at_base.p
                root_inv = arm_controller.root_link.pose.inv()
                obs.update(
                    tcp_to_mug=(
                        (root_inv
                         * self._two_item_observation_actors["cabinet_item_16"].pose).p
                        - tcp_position
                    ),
                    tcp_to_bowl=(
                        (root_inv
                         * self._two_item_observation_actors["cabinet_item_17"].pose).p
                        - tcp_position
                    ),
                )
        return obs

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        return torch.zeros_like(self.door_angle)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info)


@register_env("WallCabinetEmpty-v1", max_episode_steps=200, override=True)
class WallCabinetEmptyEnv(WallCabinetEnv):
    """The same open wall cabinet and Panda setup, without cabinet items."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, populate_cabinet=False, **kwargs)


@register_env("WallCabinetSingleItem-v1", max_episode_steps=200, override=True)
class WallCabinetSingleItemEnv(WallCabinetEnv):
    """Open cabinet with one original second-row mug on the upper shelf."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            populate_cabinet=True,
            cabinet_item_layout="single_upper_second_row",
            **kwargs,
        )


@register_env("WallCabinetTwoItems-v1", max_episode_steps=200, override=True)
class WallCabinetTwoItemsEnv(WallCabinetEnv):
    """Open cabinet with the original neighboring mug and bowl pair."""

    def __init__(self, *args, **kwargs):
        visualize_region = kwargs.pop("visualize_exploration_region", True)
        super().__init__(
            *args,
            populate_cabinet=True,
            cabinet_item_layout="two_upper_second_row",
            visualize_exploration_region=visualize_region,
            **kwargs,
        )
