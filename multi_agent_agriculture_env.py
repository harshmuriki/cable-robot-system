import gymnasium as gym
import numpy as np
import open3d as o3d
import time
from gymnasium import spaces

BOX_SIZE = 0.3
BOUNDING_BOX_LENGTH = 2.0
BOUNDING_BOX_WIDTH = 2.0
BOUNDING_BOX_HEIGHT = BOX_SIZE
Z_OFFSET = -0.6

ANCHORS_R1 = np.array([
    [-BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],
    [-BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],

    [-BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
    [-BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
])

ANCHORS_R2 = ANCHORS_R1 + np.array([0, 0, Z_OFFSET])

MARGIN = 0.2
MOVING_BOUNDARY_X = BOUNDING_BOX_LENGTH - BOX_SIZE/2 - MARGIN
MOVING_BOUNDARY_Y = BOUNDING_BOX_WIDTH - BOX_SIZE/2 - MARGIN

GRID_SIZE = 5
TIME_TAKEN_PER_PLANT = 2.0

ROBOT_STATE_IDLE = 0
ROBOT_STATE_MOVING = 1
ROBOT_STATE_REACHING = 2

class MultiAgentAgricultureEnv(gym.Env):
    """
    Custom Multi-Agent Environment for two cable robots capturing plants.
    Assumes a shared reward and centralized observation for a CTDE setup. (I guess)
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, enable_viz=True, max_steps=128):
        super().__init__()

        self.num_agents = 2
        self.grid_size = GRID_SIZE
        self.num_grid_locations = self.grid_size * self.grid_size

        self.action_space = spaces.MultiDiscrete([self.num_grid_locations] * self.num_agents)

        self.observation_space = spaces.Dict({
            'unvisited_plants_map': spaces.Box(
                low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.uint8
            ),
            'robot_1_position': spaces.Box(
                low=np.array([-MOVING_BOUNDARY_X, -MOVING_BOUNDARY_Y, -BOUNDING_BOX_HEIGHT]),
                high=np.array([MOVING_BOUNDARY_X, MOVING_BOUNDARY_Y, BOUNDING_BOX_HEIGHT]),
                shape=(3,), dtype=np.float32
            ),
            'robot_1_last_target_grid_idx': spaces.Box(
                low=0, high=self.num_grid_locations - 1, shape=(1,), dtype=np.int64
            ),
             'robot_1_state': spaces.Box( 
                low=0, high=2, shape=(1,), dtype=np.uint8
            ),
            'robot_2_position': spaces.Box(
                 low=np.array([-MOVING_BOUNDARY_X, -MOVING_BOUNDARY_Y, -BOUNDING_BOX_HEIGHT + Z_OFFSET]), # Adjust bounds for R2 Z
                 high=np.array([MOVING_BOUNDARY_X, MOVING_BOUNDARY_Y, BOUNDING_BOX_HEIGHT + Z_OFFSET]), # Adjust bounds for R2 Z
                 shape=(3,), dtype=np.float32
            ),
            'robot_2_last_target_grid_idx': spaces.Box(
                low=0, high=self.num_grid_locations - 1, shape=(1,), dtype=np.int64
            ),
             'robot_2_state': spaces.Box( 
                low=0, high=2, shape=(1,), dtype=np.uint8
            ),
            'current_step': spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int64)
        })

        self.single_agent_observation_space = spaces.Dict({
            'unvisited_plants_map': spaces.Box(
                low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.uint8
            ),
            'current_position': spaces.Box(
                low=np.array([-MOVING_BOUNDARY_X, -MOVING_BOUNDARY_Y, -BOUNDING_BOX_HEIGHT*2]),
                high=np.array([MOVING_BOUNDARY_X, MOVING_BOUNDARY_Y, BOUNDING_BOX_HEIGHT*2]),
                shape=(3,), dtype=np.float32
            ),
            'last_target_grid_idx': spaces.Box(
                low=0, high=self.num_grid_locations - 1, shape=(1,), dtype=np.int64
            ),
            'other_robot_position': spaces.Box(
                 low=np.array([-MOVING_BOUNDARY_X, -MOVING_BOUNDARY_Y, -BOUNDING_BOX_HEIGHT*2]),
                 high=np.array([MOVING_BOUNDARY_X, MOVING_BOUNDARY_Y, BOUNDING_BOX_HEIGHT*2]),
                 shape=(3,), dtype=np.float32
            ),
             'other_robot_state': spaces.Box(
                low=0, high=2, shape=(1,), dtype=np.uint8
            ),
             'current_step': spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int64)
        })


        self.all_grid_positions_xy = np.linspace(-MOVING_BOUNDARY_X, MOVING_BOUNDARY_X, GRID_SIZE)
        self.all_grid_positions_xy = np.array([[x, y] for x in self.all_grid_positions_xy for y in self.all_grid_positions_xy])
        self.all_grid_positions_base = np.hstack((self.all_grid_positions_xy, np.zeros((len(self.all_grid_positions_xy), 1))))[::-1]

        self.all_grid_target_positions_r1 = self.all_grid_positions_base.copy()
        self.all_grid_target_positions_r2 = self.all_grid_positions_base.copy() + np.array([0, 0, Z_OFFSET])


        self.enable_viz = enable_viz
        self.max_steps = max_steps

        self.robot_positions = [None] * self.num_agents # [R1_pos, R2_pos]
        self.robot_states = [ROBOT_STATE_IDLE] * self.num_agents # [R1_state, R2_state]
        self.robot_last_target_grid_idx = [0] * self.num_agents # [R1_last_idx, R2_last_idx]
        self.robot_target_positions = [None] * self.num_agents # [R1_target_XYZ, R2_target_XYZ]

        self.plant_positions_xyz = None
        self.active_plant_grid_indices = None
        self.unvisited_plants_map = None 
        self.current_step = 0
        self.num_plants_captured = 0

        self.vis = None
        self.moving_space_viz_r1 = None
        self.moving_box_viz_r1 = None
        self.camera_zone_viz_r1 = None
        self.cables_viz_r1 = None

        self.moving_space_viz_r2 = None
        self.moving_box_viz_r2 = None
        self.camera_zone_viz_r2 = None
        self.cables_viz_r2 = None

        self.hydroponic_plate_viz = None
        self.plants_viz_list = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.plant_positions_xyz, self.active_plant_grid_indices = self._generate_random_plant_set(percentage=0.5)
        self.unvisited_plants_map = np.zeros((self.num_grid_locations,), dtype=np.int8)
        self.unvisited_plants_map[self.active_plant_grid_indices] = 1
        self.unvisited_plants_map = self.unvisited_plants_map.reshape((self.grid_size, self.grid_size))

        start_grid_idx_r1 = 0
        self.robot_positions[0] = self.all_grid_target_positions_r1[start_grid_idx_r1].copy()
        self.robot_last_target_grid_idx[0] = start_grid_idx_r1
        self.robot_target_positions[0] = self.robot_positions[0].copy()
        self.robot_states[0] = ROBOT_STATE_IDLE

        start_grid_idx_r2 = self.num_grid_locations - 1
        self.robot_positions[1] = self.all_grid_target_positions_r2[start_grid_idx_r2].copy()
        self.robot_last_target_grid_idx[1] = start_grid_idx_r2
        self.robot_target_positions[1] = self.robot_positions[1].copy()
        self.robot_states[1] = ROBOT_STATE_IDLE

        self.current_step = 0
        self.num_plants_captured = 0

        if self.enable_viz:
             if self.vis is None:
                 self._init_visualization()
             else:
                 self._reset_visualization()

        central_obs = self._get_centralized_observation()

        info = {}
        return central_obs, info

    def step(self, actions):

        actions = np.asarray(actions).flatten()
        rewards = [0.0] * self.num_agents
        shared_reward = 0.0
        penalty_collision = -200.0
        reward_new_plant = 10.0
        penalty_revisit = -1.0
        penalty_no_plant = -2.0
        penalty_step = -0.1

        for i in range(self.num_agents):
            target_grid_idx = actions[i]
            self.robot_last_target_grid_idx[i] = target_grid_idx

            if i == 0:
                 self.robot_target_positions[i] = self.all_grid_target_positions_r1[target_grid_idx].copy()
            else:
                 self.robot_target_positions[i] = self.all_grid_target_positions_r2[target_grid_idx].copy()

            self.robot_positions[i] = self.robot_target_positions[i].copy()
            self.robot_states[i] = ROBOT_STATE_REACHING


        # Collision Check Logic ---
        # When two robots are at the same XY place AND one is reaching,the other must avoid.
        # In this discrete grid environment with instantaneous movement to target,
        # collision happens if they target the same XY grid cell simultaneously.
        # We simplify "reaching" to mean "at the target location this step".

        collision = False
        target_grid_xy_r1 = (actions[0] // self.grid_size, actions[0] % self.grid_size)
        target_grid_xy_r2 = (actions[1] // self.grid_size, actions[1] % self.grid_size)

        # Check if both robots targeted the same XY grid cell
        if target_grid_xy_r1 == target_grid_xy_r2:
             # Since they both just arrived at their targets, they are both in
             # the "REACHING" state at this same location. This constitutes a collision.
             collision = True
             print("Collision detected: Robots targeted the same grid location.")

        if collision:
            shared_reward += penalty_collision
            # Maybe terminate the episode immediately on collision
            terminated = True
            truncated = False # Assuming collision isn't a planned truncation
            capture_rewards = 0.0
        else:
            capture_rewards = 0.0
            # Check plants captured by each robot if no collision
            for i in range(self.num_agents):
                target_grid_idx = actions[i]
                target_grid_xy = (target_grid_idx // self.grid_size, target_grid_idx % self.grid_size)

                if target_grid_idx in self.active_plant_grid_indices:
                    if self.unvisited_plants_map[target_grid_xy[0], target_grid_xy[1]] == 1:
                        capture_rewards += reward_new_plant
                        self.unvisited_plants_map[target_grid_xy[0], target_grid_xy[1]] = 0 # Mark as visited
                        self.num_plants_captured += 1
                        if self.enable_viz:
                             original_plant_index = np.where((self.all_grid_positions_base[:, :2] == self.all_grid_positions_base[target_grid_idx, :2]).all(axis=1))[0][0]
                             active_plant_list_index = list(self.active_plant_grid_indices).index(target_grid_idx)
                             self._mark_plant_as_visited_viz(active_plant_list_index) # Use the index in the active list

                    else:
                        capture_rewards += penalty_revisit
                else:
                    capture_rewards += penalty_no_plant

            shared_reward += capture_rewards
            terminated = self.unvisited_plants_map.sum() == 0

        shared_reward += penalty_step
        self.current_step += 1
        truncated = (self.current_step >= self.max_steps)

        if self.enable_viz:
             self._update_visualization()
             time.sleep(TIME_TAKEN_PER_PLANT)

        central_obs = self._get_centralized_observation()

        info = {}
        return central_obs, shared_reward, terminated, truncated, info

    def render(self):
        if self.enable_viz:
             self.vis.poll_events()
             self.vis.update_renderer()

    def close(self):
        if self.enable_viz and self.vis:
            self.vis.destroy_window()
            self.vis = None

    def _generate_random_plant_set(self, percentage=0.5):
        """Generates a random set of plants on the grid."""
        num_total = self.num_grid_locations
        num_plants = int(num_total * percentage)
        chosen_indices = np.random.choice(
            num_total, size=num_plants, replace=False)
        plant_positions_xyz = self.all_grid_target_positions_r1[chosen_indices]
        return plant_positions_xyz, chosen_indices

    def _get_centralized_observation(self):
        """Constructs the observation dictionary for the centralized critic."""
        obs = {
            'unvisited_plants_map': self.unvisited_plants_map.copy(),
            'robot_1_position': self.robot_positions[0].astype(np.float32),
            'robot_2_position': self.robot_positions[1].astype(np.float32),
            'robot_1_last_target_grid_idx': np.array([self.robot_last_target_grid_idx[0]], dtype=np.int64),
            'robot_2_last_target_grid_idx': np.array([self.robot_last_target_grid_idx[1]], dtype=np.int64),
            'robot_1_state': np.array([self.robot_states[0]], dtype=np.uint8),
            'robot_2_state': np.array([self.robot_states[1]], dtype=np.uint8),
            'current_step': np.array([self.current_step], dtype=np.int64)
        }
        return obs

    def _init_visualization(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Robot 1
        self.moving_space_viz_r1 = self.create_bounding_box(bbox_points=ANCHORS_R1, colors=[0.5, 0.5, 0.5])
        self.moving_box_viz_r1 = self._create_box(color=[255/255, 165/255, 93/255])
        self.camera_zone_viz_r1 = self._create_camera_zone(color=[0.9, 0.8, 0.2]) # Example camera zone viz
        self.cables_viz_r1 = self._create_cables(self.robot_positions[0], anchors=ANCHORS_R1, colors=[1, 0, 0])

        self.vis.add_geometry(self.moving_space_viz_r1)
        self.vis.add_geometry(self.moving_box_viz_r1)
        self.vis.add_geometry(self.camera_zone_viz_r1)
        self.vis.add_geometry(self.cables_viz_r1)

        # Robot 2
        self.moving_space_viz_r2 = self.create_bounding_box(bbox_points=ANCHORS_R2, colors=[0.33, 0.33, 0.33])
        self.moving_box_viz_r2 = self._create_box(color=[255/255, 223/255, 136/255])
        self.camera_zone_viz_r2 = self._create_camera_zone(color=[0.9, 0, 0.2]) # Example camera zone viz
        self.cables_viz_r2 = self._create_cables(self.robot_positions[1], anchors=ANCHORS_R2, colors=[0, 0, 1])

        self.vis.add_geometry(self.moving_space_viz_r2)
        self.vis.add_geometry(self.moving_box_viz_r2)
        self.vis.add_geometry(self.camera_zone_viz_r2)
        self.vis.add_geometry(self.cables_viz_r2)

        self.hydroponic_plate_viz = self._create_hydroponic_plate()
        coordinate_frame_viz = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        self.plants_viz_list = self._create_plants_visualization(plants=self.plant_positions_xyz) # Store list of viz objects

        self.vis.add_geometry(coordinate_frame_viz)
        for plant_viz in self.plants_viz_list:
            self.vis.add_geometry(plant_viz)
        self.vis.add_geometry(self.hydroponic_plate_viz)

        self._update_visualization()
        self._set_initial_camera_view()

    def reset_visualization(self):
        """Updates visualization on reset (e.g., plants, robot positions)."""
        for plant_viz in self.plants_viz_list:
            self.vis.remove_geometry(plant_viz, reset_bounding_box=False)

        self.plants_viz_list = self._create_plants_visualization(plants=self.plant_positions_xyz, color=[0.7, 0.7, 0.7])
        for plant_viz in self.plants_viz_list:
            self.vis.add_geometry(plant_viz)

        self.moving_box_viz_r1.translate(self.robot_positions[0], relative=False)
        self.camera_zone_viz_r1.translate(self.robot_positions[0] + np.array([0, 0, 0.6]), relative=False) # Adjust camera zone height
        self.cables_viz_r1.points = o3d.utility.Vector3dVector(np.vstack((ANCHORS_R1, self._get_box_corners(self.robot_positions[0]))))

        self.moving_box_viz_r2.translate(self.robot_positions[1], relative=False)
        self.camera_zone_viz_r2.translate(self.robot_positions[1] + np.array([0, 0, 0.6]), relative=False) # Adjust camera zone height
        self.cables_viz_r2.points = o3d.utility.Vector3dVector(np.vstack((ANCHORS_R2, self._get_box_corners(self.robot_positions[1]))))

        self.vis.update_geometry(self.moving_box_viz_r1)
        self.vis.update_geometry(self.camera_zone_viz_r1)
        self.vis.update_geometry(self.cables_viz_r1)
        self.vis.update_geometry(self.moving_box_viz_r2)
        self.vis.update_geometry(self.camera_zone_viz_r2)
        self.vis.update_geometry(self.cables_viz_r2)

        self.vis.poll_events()
        self.vis.update_renderer()

    def update_visualization(self):
        """Updates visualization during a step (e.g., robot movement)."""

        self.moving_box_viz_r1.translate(self.robot_positions[0], relative=False)
        self.camera_zone_viz_r1.translate(self.robot_positions[0] + np.array([0, 0, 0.6]), relative=False) # Adjust camera zone height
        self.cables_viz_r1.points = o3d.utility.Vector3dVector(np.vstack((ANCHORS_R1, self._get_box_corners(self.robot_positions[0]))))

        self.moving_box_viz_r2.translate(self.robot_positions[1], relative=False)
        self.camera_zone_viz_r2.translate(self.robot_positions[1] + np.array([0, 0, 0.6]), relative=False) # Adjust camera zone height
        self.cables_viz_r2.points = o3d.utility.Vector3dVector(np.vstack((ANCHORS_R2, self._get_box_corners(self.robot_positions[1]))))

        self.vis.update_geometry(self.moving_box_viz_r1)
        self.vis.update_geometry(self.camera_zone_viz_r1)
        self.vis.update_geometry(self.cables_viz_r1)
        self.vis.update