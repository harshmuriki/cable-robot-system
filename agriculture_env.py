import gymnasium as gym
import numpy as np
from gymnasium import spaces
import open3d as o3d
import time
BOX_SIZE = 0.3
BOUNDING_BOX_LENGTH = BOUNDING_BOX_WIDTH = 1.0
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

# GRID_SIZE * GRID_SIZE grid of plants' centers
GRID_SIZE = 4
CAMERA_RADIUS = 0.2  # radius of the camera
PLANTS = np.linspace(-MOVING_BOUNDARY_X, MOVING_BOUNDARY_X, GRID_SIZE)
PLANTS = np.array([[x, y] for x in PLANTS for y in PLANTS])
PLANTS = np.hstack((PLANTS, np.zeros((len(PLANTS), 1))))[::-1]

TIME_TAKEN_PER_PLANT = 1  # seconds


class AgricultureEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, enable_viz=True, max_steps=64):
        super().__init__()
        self.action_space = spaces.Discrete(GRID_SIZE * GRID_SIZE)
        self.observation_space = spaces.Dict(
            {
                'last_grid_map': spaces.Box(
                    low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.uint8
                ),
                'unvisited_plants_map': spaces.Box(
                    low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.uint8
                ),
                'current_position': spaces.Box(
                    low=np.array([-MOVING_BOUNDARY_X, -MOVING_BOUNDARY_Y, 0]),
                    high=np.array([MOVING_BOUNDARY_X, MOVING_BOUNDARY_Y, 0]),
                    shape=(3,),
                    dtype=np.float32
                ),
            }
        )

        self.all_grid_positions = np.linspace(
            -MOVING_BOUNDARY_X, MOVING_BOUNDARY_X, GRID_SIZE)
        self.all_grid_positions = np.array(
            [[x, y] for x in self.all_grid_positions for y in self.all_grid_positions])
        self.all_grid_positions = np.hstack(
            (self.all_grid_positions, np.zeros((len(self.all_grid_positions), 1))))[::-1]

        self.target_pose = None
        self.delta_t = 0.1
        self.T_max = 60.0
        self.step_count = 0
        self.max_steps = max_steps  # For truncation
        self.enable_viz = enable_viz
        self.reset()
        if enable_viz:
            self.init_viusalization()

    def mark_plant_as_visited(self, idx, color=[0, 1, 0]):
        # Set visited plant to green
        self.plants_viz[idx].paint_uniform_color(color)
        self.vis.update_geometry(self.plants_viz[idx])

    def generate_random_plant_set(self, percentage=0.5):
        num_total = GRID_SIZE * GRID_SIZE
        num_plants = int(num_total * percentage)
        chosen_indices = np.random.choice(
            num_total, size=num_plants, replace=False)
        plant_positions = self.all_grid_positions[chosen_indices]
        return plant_positions, chosen_indices

    def step(self, action):
        self.target_pose = self.all_grid_positions[action]

        start_pos = self.moving_box_center
        end_pos = self.target_pose
        self.moving_box_center = end_pos

        dist = np.linalg.norm(end_pos - start_pos)

        # print("Moving to a new position: ", end_pos)
        if self.enable_viz:
            positions = self.interpolate_traj(
                start_pos, end_pos, num_steps=100)
            for pos in positions:
                self.moving_box_viz.translate(pos, relative=False)
                self.camera_zone_viz.translate(
                    pos + np.array([0, 0, 0.6]), relative=False)
                new_cable_points = np.vstack(
                    (ANCHORS, self.get_box_corners(pos)))
                self.cables_viz.points = o3d.utility.Vector3dVector(
                    new_cable_points)
                self.vis.update_geometry(self.moving_box_viz)
                self.vis.update_geometry(self.cables_viz)
                self.vis.update_geometry(self.camera_zone_viz)
                self.vis.poll_events()
                self.vis.update_renderer()
                self.capture_positions.append(pos)

        else:
            self.capture_positions.append(self.moving_box_center.tolist())

        pause_start_time = time.time()
        if self.enable_viz:
            while time.time() - pause_start_time < TIME_TAKEN_PER_PLANT:
                pass

        self.cycle_time += self.delta_t

        # --- Reward logic ---
        revisit_penalty = -1.0  # Penalty for revisiting
        step_penalty = -1.0     # Small penalty for each step
        no_plant_penalty = -2.0  # Penalty for going to a location with no plant
        new_plant_reward = 5.0  # Reward for visiting a new plant
        reward = step_penalty
        reward += - dist * 2.0   # Penalty for distance moved

        if action not in self.active_plant_indices:
            reward += no_plant_penalty  # No plant at this location
            # print("No plant at this location")
        elif self.unvisited_plants_map[action] == 1:
            # print("Visiting a new plant")
            reward += new_plant_reward

            self.unvisited_plants_map[action] = 0
            self.num_plants_captured += 1

            if self.enable_viz:
                idx = list(self.active_plant_indices).index(action)
                self.mark_plant_as_visited(idx)
        else:
            # print("Already visited this plant")
            reward += revisit_penalty  # Penalize revisiting

        last_grid_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        last_grid_map[action // GRID_SIZE, action % GRID_SIZE] = 1
        observation = {
            'last_grid_map': last_grid_map,
            'unvisited_plants_map': self.unvisited_plants_map.reshape((GRID_SIZE, GRID_SIZE)),
            'current_position': np.array(self.moving_box_center, dtype=np.float32),
        }
        terminated = False
        if self.unvisited_plants_map.sum() == 0:
            terminated = True
        self.step_count += 1
        # Truncate if max steps reached
        truncated = (self.step_count >= self.max_steps)
        # print("Current reward is: ", reward)
        info = {}
        self.last_grid = action
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.plant_positions, self.active_plant_indices = self.generate_random_plant_set()
        # Start position is the bottom right corner
        start_idx = 0
        self.moving_box_center = self.all_grid_positions[start_idx]
        self.last_grid = start_idx

        self.capture_positions = []
        self.current_target = None
        # Remove wall-clock based cycle time and use simulated time:
        self.cycle_time = 0.0
        self.num_plants_captured = 0
        self.step_count = 0  # Reset step count for the new episode
        self.arm_state = "closed"
        self.target_pose = self.moving_box_center.copy()
        self.unvisited_plants_map = np.zeros(
            (GRID_SIZE * GRID_SIZE), dtype=np.int8)
        # Mark the chosen plants as unvisited
        self.unvisited_plants_map[self.active_plant_indices] = 1
        last_grid_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        last_grid_map[start_idx // GRID_SIZE, start_idx % GRID_SIZE] = 1
        observation = {
            'last_grid_map': last_grid_map,
            'unvisited_plants_map': self.unvisited_plants_map.reshape((GRID_SIZE, GRID_SIZE)),
            'current_position': np.array(self.moving_box_center, dtype=np.float32),
        }
        if self.enable_viz and hasattr(self, 'plants_viz'):
            for plant in self.plants_viz:
                self.vis.remove_geometry(plant, reset_bounding_box=False)
            # Unvisited plants are gray
            self.plants_viz = self.create_plants_visualization(
                plants=self.plant_positions, color=[0.7, 0.7, 0.7])
            for plant in self.plants_viz:
                self.vis.add_geometry(plant)

        # Set the visualization of the cable robot
        # try:
        #     cam = o3d.io.read_pinhole_camera_parameters("camera.json")
        #     self.vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
        # except Exception as e:
        #     print("camera.json not found run once to generate it:", e)
        if self.enable_viz and hasattr(self, 'vis'):  # self.vis is not None:
            # now override the camera once
            ctr = self.vis.get_view_control()
            ctr.set_front([0.0, 0.0, -1.0])
            ctr.set_lookat([0.0, 0.0, -1.0])
            ctr.set_up([0.0, -1.0, 0.0])
            ctr.set_zoom(0.5)
            self.vis.poll_events()
            self.vis.update_renderer()
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    def get_reward(self, action):
        base_reward = 10.0
        if self.unvisited_plants_map[action] == 1:
            current_grid = np.array(divmod(action, GRID_SIZE))  # (row, col)
            last_grid_coord = np.array(divmod(self.last_grid, GRID_SIZE))

            # Get remaining unvisited grid coordinates
            unvisited_indices = np.where(self.unvisited_plants_map == 1)[0]
            assert len(unvisited_indices) > 0
            unvisited_coords = np.array(
                [divmod(i, GRID_SIZE) for i in unvisited_indices])

            # Distances from last to unvisited
            dists_from_last = np.linalg.norm(
                unvisited_coords - last_grid_coord, axis=1)
            closest_dist = np.min(dists_from_last)
            current_dist = np.linalg.norm(current_grid - last_grid_coord)

            # Scale reward
            if current_dist > 1e-6:
                coef = closest_dist / current_dist
                coef = np.clip(coef, 0.0, 2.0)
                reward = base_reward * coef
            else:
                reward = base_reward

            self.unvisited_plants_map[action] = 0
            self.num_plants_captured += 1

            if self.enable_viz:
                idx = list(self.active_plant_indices).index(action)
                self.mark_plant_as_visited(idx)
        else:
            reward = 0.0

        return reward

    def init_viusalization(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.moving_space_viz = self.create_bounding_box()
        self.moving_box_viz = self.create_box()
        self.hydroponic_plate_viz = self.create_hydroponic_plate()
        self.camera_zone_viz = self.create_camera_zone()
        self.moving_box_viz.translate(self.moving_box_center)  # Start position
        self.camera_zone_viz.translate(
            self.moving_box_center + np.array([0, 0, 0.6]))  # Start position
        self.cables_viz = self.create_cables(self.moving_box_center)
        coordinate_frame_viz = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0])
        self.plants_viz = self.create_plants_visualization(
            plants=self.plant_positions)
        self.vis.add_geometry(coordinate_frame_viz)
        self.vis.add_geometry(self.moving_space_viz)
        self.vis.add_geometry(self.moving_box_viz)
        self.vis.add_geometry(self.camera_zone_viz)
        for plant in self.plants_viz:
            self.vis.add_geometry(plant)
        self.vis.add_geometry(self.hydroponic_plate_viz)
        self.vis.add_geometry(self.cables_viz)

    def create_bounding_box(self):
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

    def create_box(self, size=BOX_SIZE, color=[92/255, 29/255, 16/255]):
        mesh = o3d.geometry.TriangleMesh.create_box(
            width=size, height=size, depth=size)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh

    def create_camera_zone(self, resolution=50, color=[0.9, 0.8, 0.2]):
        circle_zone = o3d.geometry.TriangleMesh.create_cylinder(
            radius=CAMERA_RADIUS, height=0.001, resolution=resolution)
        circle_zone.compute_vertex_normals()
        circle_zone.paint_uniform_color(color)
        return circle_zone

    # This is the points/conners of the box that moves around
    def get_box_corners(self, box_center):
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

    def create_cables(self, box_center):
        box_corners = self.get_box_corners(box_center)

        lines = [[i, i + 8] for i in range(8)]
        colors = [[1, 0, 0] for _ in lines]  # Red cables

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(
            np.vstack((ANCHORS, box_corners)))
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    # Generate trajectory points between start and end

    def interpolate_traj(self, start, end, num_steps=10, spacing_fn=np.linspace):
        positions = np.array([spacing_fn(start[i], end[i], num_steps)
                              for i in range(3)]).T
        return positions

    def create_plants_visualization(self, size=0.1, color=[0.7, 0.7, 0.7], plants=PLANTS):
        # Default color is gray for unvisited
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

    def create_hydroponic_plate(self, size=0.1, color=[0.5, 0.5, 0.5]):
        print(BOUNDING_BOX_WIDTH, BOUNDING_BOX_LENGTH)
        mesh = o3d.geometry.TriangleMesh.create_box(
            width=BOUNDING_BOX_WIDTH*2, height=BOUNDING_BOX_WIDTH*2, depth=size)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        mesh.translate([-BOUNDING_BOX_WIDTH, -BOUNDING_BOX_LENGTH, 0.6])
        return mesh

    def remove_plant(self, plants, pos=[[0, 0]]):
        # Top left corner of the plant is [0, 0]
        # pos is the position of the plant to be removed
        # returns the new plants array and the visualisation of the new plants
        if pos is None:
            return plants, self.create_plants_visualization()
        plant_positions = plants.copy()
        for pos_indv in pos:
            index = pos_indv[0] * 5 + pos_indv[1]
            plant_positions = np.delete(plant_positions, index, axis=0)

        return np.asarray(plant_positions), self.create_plants_visualization(plants=plant_positions)
