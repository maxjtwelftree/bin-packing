# rl/agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment import BinPackingEnv
from box import Box

def create_env():
    # Define the container dimensions
    container_dims = (10, 10, 10)  # Adjust as needed

    # Create boxes to be packed
    boxes = [
        Box(3, 2, 1, box_id=1),
        Box(2, 2, 2, box_id=2),
        Box(1, 3, 2, box_id=3),
        Box(2, 1, 1, box_id=4),
        Box(1, 1, 1, box_id=5),
        # Add more boxes as needed
    ]
    env = BinPackingEnv(container_dims, boxes)
    return env

# Initialize the environment
env = create_env()

# Check the environment
check_env(env, warn=True)

# Create the agent
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_bin_packing_tensorboard/")
