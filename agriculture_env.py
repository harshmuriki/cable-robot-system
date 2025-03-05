import gymnasium as gym
import numpy as np
from gymnasium import spaces
import open3d as o3d
import time
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


class AgricultureEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, enable_viz=True):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(
            low=np.array([-MOVING_BOUNDARY_X, -MOVING_BOUNDARY_Y, 0]),
            high=np.array([MOVING_BOUNDARY_X, MOVING_BOUNDARY_Y, 0]),
            dtype=np.float32
        )
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Dict(
            {
                'moving_box_centers': spaces.Box(
                    low=np.array([-MOVING_BOUNDARY_X, -MOVING_BOUNDARY_Y, 0]),
                    high=np.array([MOVING_BOUNDARY_X, MOVING_BOUNDARY_Y, 0]),
                    dtype=np.float32
                ),
                'plant_positions': spaces.Box(
                    low=np.tile(
                        np.array([-MOVING_BOUNDARY_X, -MOVING_BOUNDARY_Y, 0]), (len(PLANTS), 1)),
                    high=np.tile(
                        np.array([MOVING_BOUNDARY_X, MOVING_BOUNDARY_Y, 0]), (len(PLANTS), 1)),
                    shape=(len(PLANTS), 3),
                    dtype=np.float32
                )
            }
        )

        self.plant_positions = []  # Your plant positions
        self.target_pose = None
        self.delta_t = 0.1
        self.T_max = 60.0
        self.reset()
        self.enable_viz = enable_viz
        if enable_viz:
            self.init_viusalization()

    def step(self, action):
        # action: [x, y, z]
        self.target_pose = np.array(action)
        start_pos = self.moving_box_viz.get_center()
        end_pos = self.target_pose
        self.moving_box_center = end_pos

        print("Moving to a new plant: ", end_pos)
        if self.enable_viz:
            positions = self.interpolate_traj(
                start_pos, end_pos, num_steps=100)
            for pos in positions:
                # print(f"Moving to: {pos}, Current: {box.get_center()}")
                self.moving_box_viz.translate(pos, relative=False)

                new_cable_points = np.vstack(
                    (ANCHORS, self.get_box_corners(pos)))
                self.cables_viz.points = o3d.utility.Vector3dVector(
                    new_cable_points)
                self.vis.update_geometry(self.moving_box_viz)
                self.vis.update_geometry(self.cables_viz)
                self.vis.poll_events()
                self.vis.update_renderer()
                self.capture_positions.append(pos)

            # Non-blocking pause
            pause_start_time = time.time()
            while time.time() - pause_start_time < TIME_TAKEN_PER_PLANT:
                # Keep the visualization responsive during the pause
                self.vis.poll_events()
                self.vis.update_renderer()
                self.capture_positions.append(self.moving_box_center.tolist())
                # time.sleep(self.delta_t)
        self.cycle_time = time.time() - self.cycle_start_time
        observation = {'moving_box_centers': self.moving_box_center,
                       'plant_positions': self.plant_positions}
        terminated = False
        truncated = False
        reward = self.get_reward()
        print("Current reward is: ", reward)
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.moving_box_center = np.array([0, 0, 0])
        self.plant_positions = PLANTS
        self.capture_positions = []
        self.current_target = None
        self.cycle_start_time = time.time()
        self.num_plants_captured = 0
        self.arm_state = "closed"
        self.target_pose = self.moving_box_center.copy()
        self.visited_plants = []
        observation = {'moving_box_centers': self.moving_box_center,
                       'plant_positions': self.plant_positions}
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    def get_reward(self):
        # Params
        alpha = 10.0
        r_c = 0.2
        beta = 0.2
        efficiency_const = 5.0
        idle_penalty_factor = 1.0
        velocity_threshold = 0.1
        arm_reward_in = 100
        arm_reward_out = -100

        # Capture region <= might be more useful for 2nd phase
        # N = int(10.0 / self.delta_t)
        # capture_success = 0
        # in_region = []
        # for pos in self.capture_positions[-N:]:
        #     distance = np.linalg.norm(np.array(pos) - np.array(self.target_pose))
        #     if distance <= r_c:
        #         in_region.append(1)
        #     else:
        #         in_region.append(0)
        # if sum(in_region) == N:
        #     self.num_plants_captured += 1
        #     capture_success = 1
        # else:
        #     capture_success = 0
        # R_success = alpha * capture_success

        new_plant_reward = 10.0
        detection_threshold = 0.5  # threshold distance to consider the plant "seen"
        R_new_plant = 0.0

        for idx, plant in enumerate(self.plant_positions):
            if idx not in self.visited_plants:
                distance = np.linalg.norm(self.moving_box_center - plant)
                if distance <= detection_threshold:
                    print(f"New plant detected: {plant}, Distance: {distance}")
                    R_new_plant += new_plant_reward
                    self.visited_plants.append(idx)

        # Time penalty
        T_measured = self.cycle_time
        R_T = beta * max(0, T_measured - self.T_max)

        # Efficiency reward
        R_eff = efficiency_const * \
            (self.num_plants_captured / T_measured) if T_measured > 0 else 0

        # Idle time penalty
        # If there is no significant movement in the 1st and the last steps, penalize
        idle_penalty = 0.0
        if len(self.capture_positions) >= 2:
            p_prev = np.array(self.capture_positions[0])
            p_current = np.array(self.capture_positions[-1])
            velocity = np.linalg.norm(p_current - p_prev) / (self.cycle_time - TIME_TAKEN_PER_PLANT)
            print(f"Velocity: {velocity}")
            if velocity < velocity_threshold:
                idle_penalty = idle_penalty_factor

        print(f"R_new_plant: {R_new_plant}, R_T: {R_T}, R_eff: {R_eff}, idle_penalty: {idle_penalty}")
        total_reward = R_new_plant - R_T + R_eff - idle_penalty
        return total_reward

    def init_viusalization(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.moving_space_viz = self.create_bounding_box()
        self.moving_box_viz = self.create_box()
        self.hydroponic_plate_viz = self.create_hydroponic_plate()
        self.moving_box_viz.translate(self.moving_box_center)  # Start position
        self.cables_viz = self.create_cables(np.array(self.moving_box_center))
        coordinate_frame_viz = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0])
        self.plants_viz = self.create_plants_visualization(
            plants=self.plant_positions)
        self.vis.add_geometry(coordinate_frame_viz)
        self.vis.add_geometry(self.moving_space_viz)
        self.vis.add_geometry(self.moving_box_viz)
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

    def create_plants_visualization(self, size=0.1, color=[0, 1, 0], plants=PLANTS):
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
            width=4, height=4, depth=size)
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
