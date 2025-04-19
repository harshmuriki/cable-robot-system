# evaluate_agriculture.py
import time
from marllib import marl
from ma_agriculture_env import *
# === Load the environment (with visualization) ===
env = marl.make_env(
    environment_name="agriculture",
    map_name="agriculture",
    abs_path="../../../config/env_config/agriculture.yaml",  # <- Update this path if needed
    enable_viz=True  # <-- Enable visualization
)

# === Load algorithm and model ===
algo = marl.algos.mappo(hyperparam_source="test")  # change to your algo name
model = marl.build_model(env, algo, {
    "core_arch": "mlp",
    "encode_layer": "128-128"
})

# === Restore from checkpoint ===
checkpoint_path = "exp_results/mappo_mlp_agriculture/MAPPOTrainer_agriculture_agriculture_3a9e9_00000_0_2025-04-18_21-33-00/checkpoint_000782/checkpoint-782"  # <-- update this
algo.render(env, model,
             stop={'timesteps_total': 40000000},
             restore_path={'params_path': "/home/cassie/Workspace/cable-robot-system/exp_results/mappo_mlp_agriculture/MAPPOTrainer_agriculture_agriculture_3a9e9_00000_0_2025-04-18_21-33-00/params.json",  # experiment configuration
                           'model_path': checkpoint_path,
                           'render': True},  # checkpoint path
             local_mode=True,
             num_workers=1,
             share_policy="all",
             checkpoint_end=False,)