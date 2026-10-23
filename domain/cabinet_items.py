"""Shared procedural geometry for the ManiSkill and USD cabinet scenes."""

import numpy as np


def build_cabinet_items(self, pose_type, selected_slots=None):
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
    upright_cylinder_pose = pose_type(q=[0.7071068, 0.0, 0.7071068, 0.0])

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
                        pose=pose_type([-half_x - 0.001, 0.0, 0.0]),
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
                        pose=pose_type(
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
                        book_pose = pose_type([0.0, 0.0, book_height])
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
                        pose=pose_type(
                            [0.0, 0.0, cap_z], upright_cylinder_pose.q
                        ),
                        radius=0.018,
                        half_length=cap_half_height,
                        density=260,
                    )
                    builder.add_cylinder_visual(
                        pose=pose_type(
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
                        part_pose = pose_type(part_position)
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
                    radius = 0.036  # Leave clearance inside Panda's 8 cm opening.
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
                        pose=pose_type(
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
                    foot_pose = pose_type([0.0, 0.0, -0.043])
                    builder.add_box_collision(
                        pose=foot_pose,
                        half_size=[0.027, 0.027, 0.006],
                        density=260,
                    )
                    stem_pose = pose_type([0.0, 0.0, 0.049])
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
                    lid_pose = pose_type(
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
                        plate_pose = pose_type(
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
                        block_pose = pose_type(block_position)
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
                    cap_pose = pose_type(
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
                        pose=pose_type(
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
                        pose=pose_type([0.0, 0.0, 0.068]),
                        half_size=[0.020, 0.004, 0.025],
                        material=label_color,
                    )
                    center_height = half_size[2]
                elif item_kind == 12:
                    # Rounded vase with a narrow neck.
                    body_pose = pose_type([0.0, 0.0, -0.014])
                    neck_pose = pose_type(
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
                        pose=pose_type([0.0, 0.0, -0.058]),
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
                        handle_pose = pose_type([0.0, handle_y, 0.005])
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
                    lid_pose = pose_type(
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
                builder.initial_pose = pose_type(world_position)
                item = builder.build(f"cabinet_item_{item_index:02d}")
                self.cabinet_items.append(item)
                self.cabinet_item_types.append(item_type_names[item_kind])
                self._item_spawn_positions.append(world_position)
                item_index += 1
