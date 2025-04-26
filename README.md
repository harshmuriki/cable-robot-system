# Cable Robot System for Agricultural Applications

This repository contains reinforcement learning implementations for cable-driven parallel robots designed for agricultural monitoring and inspection tasks. The project features both single-agent and multi-agent approaches to optimize path planning and coordination in hydroponic farming environments.
![Cable Robot System for Agriculture](./MRS%20Poster%20Presentation.png)
_Visualization of our cable-driven parallel robot system operating on a hydroponic farming environment_

## Project Overview

The system simulates cable robots that move above a grid of plants, with the goal of efficiently visiting and monitoring all plants while minimizing movement costs and avoiding collisions in multi-agent scenarios.

### Key Features

- 3D visualization of the cable robot system using Open3D
- Single-agent reinforcement learning using Stable Baselines3
- Multi-agent reinforcement learning using MARLlib (various algorithms supported)
- Customizable grid size and environment parameters
- Reward system optimized for efficient plant visitation

## Installation

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/harshmuriki/cable-robot-system.git
cd cable-robot-system

# Create and activate conda environment
conda env create -f environment.yml
conda activate cable-robot
```

### Dependencies

- Python 3.8+
- Gymnasium
- Stable Baselines3
- Open3D
- NumPy
- Matplotlib
- Ray/RLlib
- MARLlib
- Wandb (for tracking experiments)

## Usage

### Single-Agent Training

```bash
python main.py
```

This will train a PPO agent to navigate the cable robot system and evaluate its performance across multiple episodes.

### Multi-Agent Training

```bash
python train.py --algo IPPO
```

Available algorithms:

- `IPPO`: Independent PPO
- `MAPPO`: Multi-Agent PPO
- `VDPPO`: Value Decomposition PPO
- `HAPPO`: Heterogeneous-Agent PPO
- `IQL`: Independent Q-Learning
- `IA2C`: Independent A2C
- `IDDPG`: Independent DDPG

### Evaluation

```bash
# For single-agent evaluation
python eval.py

# For multi-agent evaluation
python eval_ma.py
```

## Environment Description

### Single-Agent Environment (AgricultureEnv)

A Gymnasium environment where a single cable robot navigates a grid of plants. The robot receives rewards for visiting new plants and penalties for revisiting plants or unnecessary movements.

### Multi-Agent Environment (MultiAgentAgricultureEnv)

Extends the single-agent environment to support multiple cable robots operating simultaneously. Includes additional coordination challenges like collision avoidance and efficient task allocation.

## Visualization

The environments include 3D visualization capabilities using Open3D, displaying:

- Cable robot position and movement
- Plant locations and visitation status
- Camera coverage zones
- Cable connections

## Configuration

Environment parameters can be adjusted in the respective environment files or through configuration files in the `config/env_config/` directory.

## Results

Training results are logged to the `log/` directory and can be visualized using TensorBoard:

```bash
tensorboard --logdir=log
```

Trained models are saved in the project root directory with the `.zip` extension.

## License

This project is licensed under the MIT License.
