import os
import argparse
import wandb
import statistics
import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
# Support different RLlib versions for DefaultCallbacks import
try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except ImportError:
    from ray.rllib.agents.callbacks import DefaultCallbacks
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from gym.spaces import Dict as GymDict, Box, Discrete
import open3d as o3d
import time
from matplotlib import pyplot as plt


policy_mapping_dict = {
    "agriculture": {
        "description": "multi-agent agriculture task",
        "team_prefix": ("robot_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    }
}

BOX_SIZE = 0.3
BOUNDING_BOX_LENGTH = BOUNDING_BOX_WIDTH = 1.0
BOUNDING_BOX_HEIGHT = BOX_SIZE

ANCHORS = np.array([
    [-BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],
    [-BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, -BOUNDING_BOX_HEIGHT],
    [-BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, -BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
    [-BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
    [BOUNDING_BOX_LENGTH, BOUNDING_BOX_WIDTH, BOUNDING_BOX_HEIGHT],
])

MARGIN = 0.2
MOVING_BOUNDARY_X = BOUNDING_BOX_LENGTH - BOX_SIZE/2 - MARGIN
MOVING_BOUNDARY_Y = BOUNDING_BOX_WIDTH - BOX_SIZE/2 - MARGIN

GRID_SIZE = 4
CAMERA_RADIUS = 0.2
PLANTS = np.linspace(-MOVING_BOUNDARY_X, MOVING_BOUNDARY_X, GRID_SIZE)
PLANTS = np.array([[x, y] for x in PLANTS for y in PLANTS])
PLANTS = np.hstack((PLANTS, np.zeros((len(PLANTS), 1))))[::-1]

TIME_TAKEN_PER_PLANT = 1
CMAT = plt.get_cmap('tab10').colors

class MultiAgentAgricultureEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.grid_size = env_config.get("grid_size", 4)
        self.num_agents = env_config.get("num_agents", 2)
        self.max_steps = env_config.get("max_steps", 64)
        self.agents = [f"robot_{i}" for i in range(self.num_agents)]
        self.total_grids = self.grid_size * self.grid_size

        self.action_space = Discrete(self.total_grids)
        # self.observation_space = GymDict({
        #     'last_grid_map': Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.uint8),
        #     'unvisited_plants_map': Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.uint8),
        #     'obstacle_map': Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.uint8),
        # })
        self.observation_space = GymDict({
            'obs': Box(low=0, high=1, shape=(self.grid_size * self.grid_size * 3, ), dtype=np.uint8),
            # 'obs': Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 3), dtype=np.uint8), # only suitable for cases with large grid size
        })

        self.all_grid_positions = np.linspace(
            -MOVING_BOUNDARY_X, MOVING_BOUNDARY_X, GRID_SIZE)
        self.all_grid_positions = np.array(
            [[x, y] for x in self.all_grid_positions for y in self.all_grid_positions])
        self.all_grid_positions = np.hstack(
            (self.all_grid_positions, np.zeros((len(self.all_grid_positions), 1))))[::-1]

        self.env_config = env_config
        self.enable_viz = env_config.get("enable_viz", False)
        self.reset()
        if self.enable_viz:
            self.init_viusalization()
    def mark_plant_as_visited(self, idx, color=[0, 1, 0]):
        # Set visited plant to green
        self.plants_viz[idx].paint_uniform_color(color)
        self.vis.update_geometry(self.plants_viz[idx])
    def generate_random_plant_set(self, percentage=0.5):
        num_plants = int(self.total_grids * percentage)
        indices = np.random.choice(self.total_grids, size=num_plants, replace=False)
        return self.all_grid_positions[indices], indices

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.plant_positions, self.active_indices = self.generate_random_plant_set()
        self.unvisited_map = np.zeros((self.total_grids,), dtype=np.uint8)
        self.unvisited_map[self.active_indices] = 1

        # should start at different locations
        self.agent_grid_idxs = np.random.choice(
            self.total_grids, size=self.num_agents, replace=False)
        self.agent_pos = [self.all_grid_positions[i] for i in self.agent_grid_idxs]

        self.last_grid_map = {agent: self._build_grid_map(0) for agent in self.agents}

        obs = {}
        for i, agent in enumerate(self.agents):
            obstacle_map = self._build_obstacle_map(exclude=i)
            # obs[agent] = {
            #     'last_grid_map': self.last_grid_map[agent],
            #     'unvisited_plants_map': self.unvisited_map.reshape((self.grid_size, self.grid_size)),
            #     'obstacle_map': obstacle_map
            # }
            obs[agent] = {
                'obs': np.concatenate((self.last_grid_map[agent], self.unvisited_map.reshape((self.grid_size, self.grid_size)), obstacle_map), axis=-1).reshape(-1).astype(np.uint8)
                # 'obs': np.concatenate((self.last_grid_map[agent], self.unvisited_map.reshape((self.grid_size, self.grid_size)), obstacle_map), axis=-1)
            }
        if self.enable_viz and hasattr(self, 'plants_viz'):
            for plant in self.plants_viz:
                self.vis.remove_geometry(plant, reset_bounding_box=False)
            # Unvisited plants are gray
            self.plants_viz = self.create_plants_visualization(
                plants=self.plant_positions, color=[0.7, 0.7, 0.7])
            for plant in self.plants_viz:
                self.vis.add_geometry(plant)
            self.update_visualization(num_steps=1)
            # override the camera once
            ctr = self.vis.get_view_control()
            ctr.set_front([0.0, 0.0, -1.0])
            ctr.set_lookat([0.0, 0.0, -1.0])
            ctr.set_up([0.0, -1.0, 0.0])
            ctr.set_zoom(0.5)
            self.vis.poll_events()
            self.vis.update_renderer()
        return obs

    def step(self, action_dict):
        obs, rewards, dones, infos = {}, {}, {}, {}
        self.step_count += 1

        next_grid_idxs = [action_dict[agent] for agent in self.agents]
        collision_set = set([idx for idx in next_grid_idxs if next_grid_idxs.count(idx) > 1])

        self.prev_positions = self.agent_pos.copy()

        # --- Reward logic ---
        revisit_penalty = -1.0  # Penalty for revisiting
        step_penalty = -1.0     # Small penalty for each step
        no_plant_penalty = -2.0  # Penalty for going to a location with no plant
        new_plant_reward = 5.0  # Reward for visiting a new plant
        visited_plant_idx = []
        for i, agent in enumerate(self.agents):
            action = next_grid_idxs[i]
            reward = 0.0

            if action in collision_set:
                reward = -10.0
            else:
                start_pos = self.agent_pos[i]
                self.agent_grid_idxs[i] = action
                target = self.all_grid_positions[action]
                dist = np.linalg.norm(target - start_pos)
                self.agent_pos[i] = target

                reward = step_penalty - dist * 2.0 # Penalty for moving
                if action not in self.active_indices:
                    reward += no_plant_penalty
                elif self.unvisited_map[action] == 1:
                    reward += new_plant_reward
                    self.unvisited_map[action] = 0
                    if self.enable_viz:
                        idx = list(self.active_indices).index(action)
                        visited_plant_idx.append(idx)
                else:
                    reward += revisit_penalty

            last_grid_map = self._build_grid_map(self.agent_grid_idxs[i]).reshape(-1, 1)
            obstacle_map = self._build_obstacle_map(exclude=i).reshape(-1, 1)
            obs[agent] = {
                'obs': np.concatenate([last_grid_map, self.unvisited_map.reshape(-1, 1), obstacle_map], axis=1).reshape(-1).astype(np.uint8)
            }
            rewards[agent] = reward
            dones[agent] = False
            infos[agent] = {}

        if self.enable_viz:
            self.update_visualization()
            for idx in visited_plant_idx:
                self.mark_plant_as_visited(idx)
            pause_start_time = time.time()
            while time.time() - pause_start_time < TIME_TAKEN_PER_PLANT:
                pass


        dones["__all__"] = self.unvisited_map.sum() == 0 or self.step_count >= self.max_steps
        return obs, rewards, dones, infos

    def _build_grid_map(self, index):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        i, j = divmod(index, self.grid_size)
        grid[i, j] = 1
        return grid

    def _build_obstacle_map(self, exclude):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for i, idx in enumerate(self.agent_grid_idxs):
            if i != exclude:
                x, y = divmod(idx, self.grid_size)
                grid[x, y] = 1
        return grid

    def render(self):
        pass

    def close(self):
        pass

    def get_env_info(self):
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_steps,
            "policy_mapping_info": policy_mapping_dict
        }
    def init_viusalization(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        coordinate_frame_viz = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0])
        self.hydroponic_plate_viz = self.create_hydroponic_plate()

        self.agent_boxes = []
        self.agent_cameras = []
        self.agent_cables = []
        self.moving_spaces = []

        z_offsets = -np.array([i * BOUNDING_BOX_HEIGHT for i in range(self.num_agents)])

        for i, pos in enumerate(self.agent_pos):
            offset = np.array([0, 0, z_offsets[i]])
            box = self.create_box(color=CMAT[i]) # different color for each box
            box.translate(pos + offset)
            camera = self.create_camera_zone()
            camera.translate(pos + np.array([0, 0, 0.6]))
            # camera.translate(pos + offset + np.array([0, 0, 0.6]))
            cables = self.create_cables(pos + offset)
            space = self.create_bounding_box()
            space.translate(offset)

            self.agent_boxes.append(box)
            self.agent_cameras.append(camera)
            self.agent_cables.append(cables)
            self.moving_spaces.append(space)

            self.vis.add_geometry(box)
            self.vis.add_geometry(camera)
            self.vis.add_geometry(cables)
            self.vis.add_geometry(space)

        self.plants_viz = self.create_plants_visualization(plants=self.plant_positions)

        self.vis.add_geometry(coordinate_frame_viz)
        for plant in self.plants_viz:
            self.vis.add_geometry(plant)
        self.vis.add_geometry(self.hydroponic_plate_viz)

    def update_visualization(self, num_steps=50):
        if num_steps == 1:
            agent_trajectories = np.array(self.agent_pos)[:, np.newaxis]  # No interpolation, just current positions
        else:
            agent_trajectories = [
                self.interpolate_traj(self.prev_positions[i], self.agent_pos[i], num_steps=num_steps)
                for i in range(self.num_agents)
            ]
        for step in range(num_steps):
            for i in range(self.num_agents):
                pos = agent_trajectories[i][step]
                offset = np.array([0, 0, -i * BOX_SIZE])
                self.agent_boxes[i].translate(pos + offset, relative=False)
                self.agent_cameras[i].translate(pos + np.array([0, 0, 0.6]), relative=False)
                # self.agent_cameras[i].translate(pos + offset + np.array([0, 0, 0.6]), relative=False)
                cable_points = np.vstack((ANCHORS, self.get_box_corners(pos + offset)))
                self.agent_cables[i].points = o3d.utility.Vector3dVector(cable_points)
                self.vis.update_geometry(self.agent_boxes[i])
                self.vis.update_geometry(self.agent_cameras[i])
                self.vis.update_geometry(self.agent_cables[i])
            self.vis.poll_events()
            self.vis.update_renderer()
    def interpolate_traj(self, start, end, num_steps=10, spacing_fn=np.linspace):
        positions = np.array([spacing_fn(start[i], end[i], num_steps)
                              for i in range(3)]).T
        return positions

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

class CustomMetrics(DefaultCallbacks):
    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data["collisions"] = 0
        episode.user_data["rewards"] = {agent: 0.0 for agent in episode._agent_to_policy}

    def on_episode_step(self, *, episode, **kwargs):
        for agent_id, r in episode.last_reward_for().items():
            episode.user_data["rewards"][agent_id] += r
            info = episode.last_info_for(agent_id) or {}
            if info.get("collision", False):
                episode.user_data["collisions"] += 1

    def on_episode_end(self, *, episode, **kwargs):
        rewards = list(episode.user_data["rewards"].values())
        mean_r = statistics.mean(rewards)
        std_r = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        collision_rate = episode.user_data["collisions"] / max(episode.length, 1)
        success = 1.0 if episode._dones.get("__all__", False) else 0.0
        episode.custom_metrics["per_agent_reward_mean"] = mean_r
        episode.custom_metrics["per_agent_reward_std"] = std_r
        episode.custom_metrics["collision_rate"] = collision_rate
        episode.custom_metrics["success_rate"] = success

class AgricultureWandbCallback(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        # Log the mean episode reward
        if "episode_reward_mean" in result:
            wandb.log({"episode_reward_mean": result["episode_reward_mean"]}, step=trainer.iteration)

        # Log max episode reward
        if "episode_reward_max" in result:
            wandb.log({"episode_reward_max": result["episode_reward_max"]}, step=trainer.iteration)

        # Log min episode reward
        if "episode_reward_min" in result:
            wandb.log({"episode_reward_min": result["episode_reward_min"]}, step=trainer.iteration)

        # Log policy and VF losses, KL divergence, and entropy for each agent
        if "policy_loss" in result:
            for agent_id, loss in result["policy_loss"].items():
                wandb.log({f"{agent_id}/policy_loss": loss}, step=trainer.iteration)

        if "vf_loss" in result:
            for agent_id, loss in result["vf_loss"].items():
                wandb.log({f"{agent_id}/vf_loss": loss}, step=trainer.iteration)

        if "kl_divergence" in result:
            for agent_id, kl in result["kl_divergence"].items():
                wandb.log({f"{agent_id}/kl_divergence": kl}, step=trainer.iteration)

        if "entropy" in result:
            for agent_id, entropy_val in result["entropy"].items():
                wandb.log({f"{agent_id}/entropy": entropy_val}, step=trainer.iteration)

        # Log gradient norm (if available in the results)
        if "grad_norm" in result:
            wandb.log({"grad_norm": result["grad_norm"]}, step=trainer.iteration)
        elif "agent_0/grad_norm_policy" in result: # Example of agent-specific grad norm
            for agent_id in trainer.get_policy_ids():
                if f"{agent_id}/grad_norm_policy" in result:
                    wandb.log({f"{agent_id}/grad_norm_policy": result[f"{agent_id}/grad_norm_policy"]}, step=trainer.iteration)
                if f"{agent_id}/grad_norm_vf" in result:
                    wandb.log({f"{agent_id}/grad_norm_vf": result[f"{agent_id}/grad_norm_vf"]}, step=trainer.iteration)

ENV_REGISTRY["agriculture"] = MultiAgentAgricultureEnv
ray.tune.registry.register_env("agriculture", lambda cfg: MultiAgentAgricultureEnv(cfg))


import socket

def get_node_ip_address():
    """Returns the primary IP address of the current machine."""
    try:
        # Create a socket object and connect to a known external host.
        # Google DNS (8.8.8.8) is a common choice.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
    except socket.error as e:
        print(f"Error getting IP address: {e}")
        ip_address = "127.0.0.1"  # Fallback to localhost
    finally:
        if 's' in locals():
            s.close()
    return ip_address

if __name__ == "__main__":
    ip = get_node_ip_address()
    print(f"The IP address of this node is: {ip}")

def main(algo_name):
    # Ensure Ray is cleanly started
    try:
        ray.shutdown()
    except Exception:
        pass

    # Initialize W&B run
    run = wandb.init(
        project="multiagent-agri",
        name=f"{algo_name}-mlp-128x128",
        config={
            "algo": algo_name,
            "env": "agriculture",
            "core_arch": "mlp",
            "encode_layer": "128-128",
            "num_agents": 2,
            "grid_size": 4,
        }
    )

    # Prepare model directory
    models_dir = f"models/{algo_name}"
    os.makedirs(models_dir, exist_ok=True)

    # Build environment and algorithm
    env = marl.make_env(
        environment_name="agriculture",
        map_name="agriculture",
        abs_path="../../../config/env_config/agriculture.yaml",
    )
    ALGOS = {
        "IPPO": marl.algos.ippo,
        "MAPPO": marl.algos.mappo,
        "VDPPO": marl.algos.vdppo,
        "HAPPO": marl.algos.happo,
        "IQL": marl.algos.iql,
        "IA2C": marl.algos.ia2c,
        "IDDPG": marl.algos.iddpg
    }
    algo = ALGOS[algo_name](hyperparam_source="test")
    model = marl.build_model(env, algo, {
        "core_arch": "mlp",
        "encode_layer": "128-128",
    })

    # Train and collect results via MARLlib
    analysis = algo.fit(
        env,
        model,
        stop={
            "episode_reward_mean": 1000,
            "timesteps_total": 200000
        },
        local_mode=True,
        num_gpus=1,
        num_workers=8,
        share_policy="all",
        checkpoint_freq=100
    )

    # Extract the latest training result
    trial = analysis.trials[0]
    result = trial.last_result

    # Log core episode metrics
    metrics = {
        "episode_reward_mean": result.get("episode_reward_mean"),
        "episode_reward_max":  result.get("episode_reward_max"),
        "episode_reward_min":  result.get("episode_reward_min"),
    }
    # Log per-policy (agent) statistics
    learner_info = result.get("info", {}).get("learner", {})
    for policy_id, info in learner_info.items():
        stats = info.get("learner_stats", {})
        metrics[f"{policy_id}/policy_loss"]   = stats.get("policy_loss")
        metrics[f"{policy_id}/vf_loss"]       = stats.get("vf_loss")
        metrics[f"{policy_id}/kl_divergence"] = stats.get("kl")
        metrics[f"{policy_id}/entropy"]       = stats.get("entropy")
    # Log all metrics to W&B
    wandb.log(metrics, step=result.get("training_iteration", 0))

    # Retrieve best checkpoint and save
    best_trial = analysis.get_best_trial(
        metric="episode_reward_mean", mode="max"
    )
    checkpoint_path = analysis.get_best_checkpoint(
        best_trial,
        metric="episode_reward_mean",
        mode="max"
    )
    print(f"Best checkpoint for {algo_name}: {checkpoint_path}")
    wandb.save(checkpoint_path)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", required=True,
        choices=["IPPO", "MAPPO", "VDPPO", "HAPPO", "IQL", "IA2C", "IDDPG"],
    )
    args = parser.parse_args()
    main(args.algo)


