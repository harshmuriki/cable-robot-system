import open3d as o3d
import numpy as np

BOX_SIZE = 0.3
BOUNDING_BOX_LENGTH = 2.0
BOUNDING_BOX_WIDTH = 2.0
BOUNDING_BOX_HEIGHT = BOX_SIZE

ANCHORS = np.array([
    # bottom corners
    [-BOUNDING_BOX_LENGTH/2, -BOUNDING_BOX_WIDTH/2, -BOUNDING_BOX_HEIGHT/2], 
    [BOUNDING_BOX_LENGTH/2, -BOUNDING_BOX_WIDTH/2, -BOUNDING_BOX_HEIGHT/2], 
    [-BOUNDING_BOX_LENGTH/2, BOUNDING_BOX_WIDTH/2, -BOUNDING_BOX_HEIGHT/2], 
    [BOUNDING_BOX_LENGTH/2, BOUNDING_BOX_WIDTH/2, -BOUNDING_BOX_HEIGHT/2],
    # upper corners
    [-BOUNDING_BOX_LENGTH/2, -BOUNDING_BOX_WIDTH/2, BOUNDING_BOX_HEIGHT/2], 
    [BOUNDING_BOX_LENGTH/2, -BOUNDING_BOX_WIDTH/2, BOUNDING_BOX_HEIGHT/2], 
    [-BOUNDING_BOX_LENGTH/2, BOUNDING_BOX_WIDTH/2, BOUNDING_BOX_HEIGHT/2], 
    [BOUNDING_BOX_LENGTH/2, BOUNDING_BOX_WIDTH/2, BOUNDING_BOX_HEIGHT/2],
])


MARGIN = 0.2
MOVING_BOUNDARY_X = BOUNDING_BOX_LENGTH / 2 - MARGIN 
MOVING_BOUNDARY_X = BOUNDING_BOX_WIDTH / 2 - MARGIN

def sample_new_position():
    x = np.random.uniform(-MOVING_BOUNDARY_X, MOVING_BOUNDARY_X)
    y = np.random.uniform(-MOVING_BOUNDARY_X, MOVING_BOUNDARY_X)
    # z = np.random.uniform(0.3, ANCHOR_HEIGHT - MARGIN)
    z = 0
    return [x, y, z]



def create_bounding_box():
    bbox = o3d.geometry.LineSet()
    bbox_points = ANCHORS

    bbox_lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    bbox.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in bbox_lines])  # Gray lines
    bbox.points = o3d.utility.Vector3dVector(bbox_points)
    bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
    return bbox

def create_box(size=BOX_SIZE, color=[0, 0, 1]):
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)  # Blue box
    return mesh

def get_box_corners(box_center):
    half_size = BOX_SIZE / 2
    box_corners = np.array([
        # lower corners
        [box_center[0] - half_size, box_center[1] - half_size, box_center[2] - half_size], 
        [box_center[0] + half_size, box_center[1] - half_size, box_center[2] - half_size], 
        [box_center[0] - half_size, box_center[1] + half_size, box_center[2] - half_size], 
        [box_center[0] + half_size, box_center[1] + half_size, box_center[2] - half_size],
        # upper corners
        [box_center[0] - half_size, box_center[1] - half_size, box_center[2] + half_size], 
        [box_center[0] + half_size, box_center[1] - half_size, box_center[2] + half_size], 
        [box_center[0] - half_size, box_center[1] + half_size, box_center[2] + half_size], 
        [box_center[0] + half_size, box_center[1] + half_size, box_center[2] + half_size]
    ])
    return box_corners

def create_cables(box_center):
    box_corners = get_box_corners(box_center)

    lines = [[i, i + 8] for i in range(8)]
    colors = [[1, 0, 0] for _ in lines]  # Red cables

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack((ANCHORS, box_corners)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def generate_positions(start, end, num_steps=10, spacing_fn=np.linspace):
    start = np.array(start)
    end = np.array(end)
    positions = np.array([spacing_fn(start[i], end[i], num_steps) for i in range(3)]).T
    return positions

vis = o3d.visualization.Visualizer()
vis.create_window()

bbox = create_bounding_box()
box = create_box()
box.translate([0, 0, 0])  # Start position
cables = create_cables(np.array([0, 0, 0]))
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

vis.add_geometry(coordinate_frame)
vis.add_geometry(bbox)
vis.add_geometry(box)
vis.add_geometry(cables)


while True:
    start_pos = list(box.get_center())
    end_pos = sample_new_position()

    positions = generate_positions(start_pos, end_pos, num_steps=100)

    for pos in positions:
        print(f"Moving to: {pos}, Current: {box.get_center()}")
        box.translate(pos, relative=False)
        # box.translate(pos - np.array(box.get_center()), relative=False)

        new_cable_points = np.vstack((ANCHORS, get_box_corners(pos)))
        cables.points = o3d.utility.Vector3dVector(new_cable_points)
        vis.update_geometry(box)
        vis.update_geometry(cables)
        vis.poll_events()
        vis.update_renderer()

