# rl/test.py

from stable_baselines3 import PPO
from agent import create_env
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def test_agent():
    # Load the trained model
    model = PPO.load("ppo_bin_packing_model")

    # Create a new environment instance
    env = create_env()

    # Reset the environment
    obs, _ = env.reset()
    done = False

    while not done:
        # Predict the action
        action, _states = model.predict(obs, deterministic=True)
        # Take the action
        obs, reward, done, truncated, info = env.step(action)
    
    # Render the final state
    env.render()
    env.close()

if __name__ == "__main__":
    test_agent()
