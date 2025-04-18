import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from multi_agent_agriculture_env import MultiAgentAgricultureEnv

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, central_obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(central_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, central_obs):
        return self.net(central_obs)

def preprocess_single_agent_obs(obs, agent_idx):
    if agent_idx == 0:
        pos = obs['robot_1_position']
        last_target = obs['robot_1_last_target_grid_idx']
        state = obs['robot_1_state']
        other_pos = obs['robot_2_position']
        other_state = obs['robot_2_state']
    else:
        pos = obs['robot_2_position']
        last_target = obs['robot_2_last_target_grid_idx']
        state = obs['robot_2_state']
        other_pos = obs['robot_1_position']
        other_state = obs['robot_1_state']
    flat = torch.cat([
        torch.from_numpy(obs['unvisited_plants_map'].flatten()).float(),
        torch.from_numpy(pos).float(),
        torch.from_numpy(last_target).float(),
        torch.from_numpy(other_pos).float(),
        torch.from_numpy(other_state).float(),
        torch.from_numpy(obs['current_step']).float()
    ])
    return flat

def preprocess_central_obs(obs):
    flat = torch.cat([
        torch.from_numpy(obs['unvisited_plants_map'].flatten()).float(),
        torch.from_numpy(obs['robot_1_position']).float(),
        torch.from_numpy(obs['robot_1_last_target_grid_idx']).float(),
        torch.from_numpy(obs['robot_1_state']).float(),
        torch.from_numpy(obs['robot_2_position']).float(),
        torch.from_numpy(obs['robot_2_last_target_grid_idx']).float(),
        torch.from_numpy(obs['robot_2_state']).float(),
        torch.from_numpy(obs['current_step']).float()
    ])
    return flat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ppo_model = PPO.load("ppo_unvisited_4x4_v2", device=device)
ppo_actor_params = dict(ppo_model.policy.state_dict())
env = MultiAgentAgricultureEnv(enable_viz=False)
num_agents = 2
action_dim = env.num_grid_locations
obs_dim = preprocess_single_agent_obs(env.reset()[0], 0).shape[0]
central_obs_dim = preprocess_central_obs(env.reset()[0]).shape[0]
actors = [Actor(obs_dim, action_dim).to(device) for _ in range(num_agents)]
critic = Critic(central_obs_dim).to(device)
for actor in actors:
    actor.load_state_dict(ppo_actor_params, strict=False)
actor_optimizers = [optim.Adam(actor.parameters(), lr=1e-5) for actor in actors]
critic_optimizer = optim.Adam(critic.parameters(), lr=5e-4)
num_epochs = 1000
n_steps = 50
gamma = 0.99
clip_eps = 0.2
reward_history = []

for epoch in range(num_epochs):
    obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = [], [], [], [], []
    agent_obs_batches = [[] for _ in range(num_agents)]
    total_reward = 0
    obs, _ = env.reset()
    for _ in range(n_steps):
        single_obs_tensor = [preprocess_single_agent_obs(obs, i).to(device) for i in range(num_agents)]
        logits = [actors[i](single_obs_tensor[i].unsqueeze(0)) for i in range(num_agents)]
        dists = [torch.distributions.Categorical(logits=logit) for logit in logits]
        actions = [dist.sample() for dist in dists]
        actions_np = np.array([a.cpu().numpy() for a in actions])
        next_obs, reward, done, trunc, info = env.step(actions_np)
        total_reward += reward
        obs_batch.append(preprocess_central_obs(obs).unsqueeze(0))
        actions_batch.append(torch.stack(actions).unsqueeze(0))
        rewards_batch.append(torch.tensor([reward], dtype=torch.float32).unsqueeze(0))
        next_obs_batch.append(preprocess_central_obs(next_obs).unsqueeze(0))
        dones_batch.append(torch.tensor([done], dtype=torch.float32).unsqueeze(0))
        for agent_idx in range(num_agents):
            agent_obs_batches[agent_idx].append(single_obs_tensor[agent_idx].unsqueeze(0))
        obs = next_obs
        if done or trunc:
            obs, _ = env.reset()
    obs_batch = torch.cat(obs_batch).to(device)
    actions_batch = torch.cat(actions_batch).to(device)
    rewards_batch = torch.cat(rewards_batch).to(device)
    next_obs_batch = torch.cat(next_obs_batch).to(device)
    dones_batch = torch.cat(dones_batch).to(device)
    agent_obs_batches = [torch.cat(agent_obs_batches[i]).to(device) for i in range(num_agents)]
    values = critic(obs_batch)
    next_values = critic(next_obs_batch)
    targets = rewards_batch + gamma * next_values * (1 - dones_batch)
    critic_loss = ((values - targets.detach()) ** 2).mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    advantages = (targets - values).detach()
    for agent_idx in range(num_agents):
        logits = actors[agent_idx](agent_obs_batches[agent_idx])
        dists = torch.distributions.Categorical(logits=logits)
        log_probs = dists.log_prob(actions_batch[:, agent_idx])
        old_log_probs = log_probs.detach()
        ratio = torch.exp(log_probs - old_log_probs)
        surrogate1 = ratio * advantages.squeeze()
        surrogate2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages.squeeze()
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        actor_optimizers[agent_idx].zero_grad()
        actor_loss.backward()
        actor_optimizers[agent_idx].step()
    reward_history.append(total_reward)
    print(f"Epoch {epoch} | Critic Loss: {critic_loss.item():.3f} | Total Reward: {total_reward:.2f}")

torch.save({'actors': [actor.state_dict() for actor in actors], 'critic': critic.state_dict()}, 'mappo_model.pth')

plt.plot(reward_history)
plt.xlabel('Epoch')
plt.ylabel('Total Reward')
plt.title('Training Reward Over Epochs')
plt.grid(True)
plt.show()
