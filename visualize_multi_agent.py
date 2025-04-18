import torch
import numpy as np
from multi_agent_agriculture_env import MultiAgentAgricultureEnv
import time
from train_mappo import Actor, preprocess_single_agent_obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_data = torch.load("mappo_model.pth", map_location=device)

env = MultiAgentAgricultureEnv(enable_viz=True)
action_dim = env.num_grid_locations
obs_dim = preprocess_single_agent_obs(env.reset()[0], 0).shape[0]
actors = [Actor(obs_dim, action_dim).to(device) for _ in range(2)]

for i in range(2):
    actors[i].load_state_dict(model_data['actors'][i])

obs, _ = env.reset()
print("Starting MAPPO model visualization")

for _ in range(1000):
    obs_tensor = [preprocess_single_agent_obs(obs, i).unsqueeze(0).to(device) for i in range(2)]
    with torch.no_grad():
        logits = [actors[i](obs_tensor[i]) for i in range(2)]
        dists = [torch.distributions.Categorical(logits=logit) for logit in logits]
        actions = [dist.sample().cpu().numpy() for dist in dists]
    obs, reward, terminated, truncated, info = env.step(np.array(actions))
    if terminated or truncated:
        time.sleep(0.5)
        obs, _ = env.reset()

env.close()
