from stable_baselines3 import PPO
from agriculture_env import AgricultureEnv

env = AgricultureEnv(enable_viz=True)
model = PPO.load("ppo_agriculture")

obs, _ = env.reset()  # Unpack the tuple to get the observation
print("Starting visualization")
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()  # Unpack the tuple to get the observation
env.close()
