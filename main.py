from stable_baselines3 import PPO
from agriculture_env import AgricultureEnv
import matplotlib.pyplot as plt
import time
from stable_baselines3.common.vec_env import DummyVecEnv

############################ Training ############################
print("Starting training")


def make_env():
    return AgricultureEnv(enable_viz=False, max_steps=64)


exp_id = 'ppo_unvisited_4x4_v2'
vec_env = DummyVecEnv([make_env for _ in range(8)])  # 8 parallel copies
model = PPO("MultiInputPolicy", vec_env, verbose=1, n_steps=256,
            batch_size=64, device="auto", tensorboard_log="./log/",)

# model = PPO("MultiInputPolicy", env, verbose=1, device="auto", tensorboard_log="./log/", n_steps=64)   # TODO: adjust n_steps and total_timesteps
model.learn(total_timesteps=200000, progress_bar=True,
            log_interval=1, tb_log_name=exp_id)
model.save(exp_id)
del model


############################ Evaluation ############################
def visualize_rewards(total_rewards, cycle_times):
    """
    Visualizes total rewards and cycle times per episode.

    Args:
        total_rewards (list of float): Total reward for each episode.
        cycle_times (list of float): Cycle time (in seconds) for each episode.
    """
    episodes = range(1, len(total_rewards) + 1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot total reward per episode.
    axs[0].plot(episodes, total_rewards, marker='o', label="Total Reward")
    axs[0].set_title("Total Reward per Episode")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].grid(True)

    # Plot cycle time per episode.
    axs[1].plot(episodes, cycle_times, marker='o',
                color='red', label="Cycle Time")
    axs[1].set_title("Cycle Time per Episode")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Time (sec)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


env = AgricultureEnv(enable_viz=True)
model = PPO.load(exp_id, device="cuda")
obs, _ = env.reset()  # Unpack the tuple to get the observation
print("Done with training")

all_total_rewards = []
all_cycle_times = []
num_episodes = 5  # Set the number of episodes to evaluate

for episode in range(num_episodes):
    episode_reward = 0.0
    episode_start_time = time.time()
    obs, _ = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    cycle_time = time.time() - episode_start_time
    all_total_rewards.append(episode_reward)
    all_cycle_times.append(cycle_time)
    print(
        f"Episode {episode+1}: Total Reward = {episode_reward:.2f}, Cycle Time = {cycle_time:.2f} sec")

env.close()
visualize_rewards(all_total_rewards, all_cycle_times)
