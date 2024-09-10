import gym
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
from stable_baselines3 import PPO

class AussieFootballEnv(gym.Env):
    def __init__(self):
        super(AussieFootballEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.physics_client = p.connect(p.GUI)  
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.initial_ball_pos = None
        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        plane_id = p.loadURDF("plane.urdf")
        self.ball_id = p.loadURDF("sphere2.urdf", [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.1)
        self.player_id = p.loadURDF("r2d2.urdf", [0, -1, 0.1])  
        self.initial_ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        self.state = self._get_observation()
        return self.state

    def step(self, action):
        linear_velocity = action * 5 
        p.resetBaseVelocity(self.player_id, linearVelocity=[linear_velocity[0], linear_velocity[1], 0])
        p.stepSimulation()
        self.state = self._get_observation()
        reward = self.compute_reward()
        done = self._check_done()
        self.render()
        return self.state, reward, done, {}

    def _get_observation(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        player_pos, _ = p.getBasePositionAndOrientation(self.player_id)
        return np.array(ball_pos + player_pos)

    def compute_reward(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        player_pos, _ = p.getBasePositionAndOrientation(self.player_id)

        distance_to_ball = np.linalg.norm(np.array(ball_pos) - np.array(player_pos))

        ball_movement = np.linalg.norm(np.array(ball_pos) - np.array(self.initial_ball_pos))

        if distance_to_ball < 0.5:
            reward = 10
        else:
            reward = -distance_to_ball

        reward += ball_movement * 10

        return reward

    def _check_done(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        if ball_pos[0] > 10:
            return True
        return False

    def render(self, mode='human'):
        p.stepSimulation()

    def close(self):
        p.disconnect(self.physics_client)

env = AussieFootballEnv()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
