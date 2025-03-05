from agriculture_env import AgricultureEnv
env = AgricultureEnv()
for plant_pos in env.plant_positions:
    env.step(plant_pos)