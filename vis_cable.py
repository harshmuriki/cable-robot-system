import open3d as o3d
import numpy as np
import time

#! (0, 0, 0) is the center of the bounding box

BOX_SIZE = 0.3
BOUNDING_BOX_LENGTH = 2.0
BOUNDING_BOX_WIDTH = 2.0
BOUNDING_BOX_HEIGHT = BOX_SIZE

ANCHORS = np.array([
    # bottom corners
    [-BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],
    [-BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],

    # upper corners
    [-BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
    [-BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
])

MARGIN = 0.2
MOVING_BOUNDARY_X = BOUNDING_BOX_LENGTH - BOX_SIZE/2 - MARGIN
MOVING_BOUNDARY_Y = BOUNDING_BOX_WIDTH - BOX_SIZE/2 - MARGIN

# 5 * 5 grid of plants' centers
PLANTS = np.linspace(-MOVING_BOUNDARY_X, MOVING_BOUNDARY_X, 5)
PLANTS = np.array([[x, y] for x in PLANTS for y in PLANTS])
PLANTS = np.hstack((PLANTS, np.zeros((len(PLANTS), 1))))[::-1]

TIME_TAKEN_PER_PLANT = 2.0  # seconds

# This is the randomly generated new position of the box


def sample_new_position():
    x = np.random.uniform(-MOVING_BOUNDARY_X, MOVING_BOUNDARY_X)
    y = np.random.uniform(-MOVING_BOUNDARY_X, MOVING_BOUNDARY_X)
    z = 0
    return [x, y, z]

# This is the frame of the cable-robot system


def create_bounding_box():
    bbox = o3d.geometry.LineSet()
    bbox_points = ANCHORS

    # Relationship between the bbox_points
    bbox_lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    bbox.colors = o3d.utility.Vector3dVector(
        [[0.5, 0.5, 0.5] for _ in bbox_lines])  # Gray lines
    bbox.points = o3d.utility.Vector3dVector(bbox_points)
    bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
    return bbox

# This is the box that moves around (wooden box)


def create_box(size=BOX_SIZE, color=[92/255, 29/255, 16/255]):
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=size, height=size, depth=size)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

# This is the points/conners of the box that moves around


def get_box_corners(box_center):
    half_size = BOX_SIZE / 2
    box_corners = np.array([
        # lower corners
        [box_center[0] - half_size, box_center[1] -
            half_size, box_center[2] - half_size],
        [box_center[0] + half_size, box_center[1] -
            half_size, box_center[2] - half_size],
        [box_center[0] - half_size, box_center[1] +
            half_size, box_center[2] - half_size],
        [box_center[0] + half_size, box_center[1] +
            half_size, box_center[2] - half_size],
        # upper corners
        [box_center[0] - half_size, box_center[1] -
            half_size, box_center[2] + half_size],
        [box_center[0] + half_size, box_center[1] -
            half_size, box_center[2] + half_size],
        [box_center[0] - half_size, box_center[1] +
            half_size, box_center[2] + half_size],
        [box_center[0] + half_size, box_center[1] +
            half_size, box_center[2] + half_size]
    ])
    return box_corners


def create_cables(box_center):
    box_corners = get_box_corners(box_center)

    lines = [[i, i + 8] for i in range(8)]
    colors = [[1, 0, 0] for _ in lines]  # Red cables

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(
        np.vstack((ANCHORS, box_corners)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# Generate trajectory points between start and end


def generate_positions(start, end, num_steps=10, spacing_fn=np.linspace):
    start = np.array(start)
    end = np.array(end)
    positions = np.array([spacing_fn(start[i], end[i], num_steps)
                         for i in range(3)]).T
    return positions


def create_plants_visualization(size=0.1, color=[0, 1, 0], plants=PLANTS):
    plant_meshes = []
    for plant_center in plants:
        x, y, z = plant_center
        mesh = o3d.geometry.TriangleMesh.create_box(
            width=size, height=size, depth=size)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        mesh.translate([x-size/2, y-size/2, z+0.5])
        plant_meshes.append(mesh)
    return plant_meshes


def create_hydroponic_plate(size=0.1, color=[0.5, 0.5, 0.5]):
    print(BOUNDING_BOX_WIDTH, BOUNDING_BOX_LENGTH)
    mesh = o3d.geometry.TriangleMesh.create_box(width=4, height=4, depth=size)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    mesh.translate([-BOUNDING_BOX_WIDTH, -BOUNDING_BOX_LENGTH, 0.6])
    return mesh


def remove_plant(plants, pos=[[0, 0]]):
    # Top left corner of the plant is [0, 0]
    # pos is the position of the plant to be removed
    # returns the new plants array and the visualisation of the new plants
    if pos is None:
        return plants, create_plants_visualization()
    new_plants = plants.copy()
    for pos_indv in pos:
        index = pos_indv[0] * 5 + pos_indv[1]
        new_plants = np.delete(new_plants, index, axis=0)

    return np.asarray(new_plants), create_plants_visualization(plants=new_plants)


def main():
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    bbox = create_bounding_box()
    box = create_box()
    hydroponic_plate = create_hydroponic_plate()
    box.translate([0, 0, 0])  # Start position
    cables = create_cables(np.array([0, 0, 0]))
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0])

    # To edit when plants are removed
    new_plants, plants_viz = remove_plant(
        plants=PLANTS, pos=None)  # Eg: [[0, 0], [3, 4]]

    vis.add_geometry(coordinate_frame)
    vis.add_geometry(bbox)
    vis.add_geometry(box)
    for plant in plants_viz:
        vis.add_geometry(plant)
    vis.add_geometry(hydroponic_plate)
    vis.add_geometry(cables)

    # Start from top-left corner
    for (x, y, z) in new_plants:

        start_pos = list(box.get_center())
        end_pos = [x, y, z]

        positions = generate_positions(start_pos, end_pos, num_steps=100)

        print("Moving to a new plant: ", end_pos)

        for pos in positions:
            # print(f"Moving to: {pos}, Current: {box.get_center()}")
            box.translate(pos, relative=False)

            new_cable_points = np.vstack((ANCHORS, get_box_corners(pos)))
            cables.points = o3d.utility.Vector3dVector(new_cable_points)
            vis.update_geometry(box)
            vis.update_geometry(cables)
            vis.poll_events()
            vis.update_renderer()

        # Non-blocking pause
        pause_start_time = time.time()
        while time.time() - pause_start_time < TIME_TAKEN_PER_PLANT:
            # Keep the visualization responsive during the pause
            vis.poll_events()
            vis.update_renderer()


if __name__ == "__main__":
    main()
    # pass
