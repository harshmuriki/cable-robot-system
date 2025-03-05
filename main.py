from stable_baselines3 import PPO
from agriculture_env import AgricultureEnv

env = AgricultureEnv()
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10)
model.save("ppo_agriculture")
del model
model = PPO.load("ppo_agriculture")
obs = env.reset()
print("Done with training")
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()
env.close()

# for plant_pos in env.plant_positions:
#     env.step(plant_pos)
