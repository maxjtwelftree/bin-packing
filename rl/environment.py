# rl/environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from state import State
from box import Box

class BinPackingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, container_dims, boxes):
        super(BinPackingEnv, self).__init__()

        self.container_dims = container_dims  # (width, height, depth)
        self.boxes = boxes  # List of Box objects
        self.state = None

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.boxes))  # Selecting which box to place next

        # Observation space: positions and sizes of placed boxes
        obs_size = len(self.boxes) * 6  # Each box: position (3) + dimensions (3)
        self.observation_space = spaces.Box(
            low=0, high=max(container_dims), shape=(obs_size,), dtype=np.float32
        )

        self.current_step = 0
        self.fig = None
        self.ax = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = State(*self.container_dims, boxes_to_place=self.boxes.copy())
        self.placed_boxes = []
        self.current_step = 0
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        box_index = action

        if box_index >= len(self.state.boxes_to_place):
            reward = -10
            terminated = True
            truncated = False
            observation = self._get_observation()
            info = {}
            return observation, reward, terminated, truncated, info

        box = self.state.boxes_to_place[box_index]
        rotation = (box.width, box.height, box.depth)  # Default rotation
        position = self._find_position_for_box(box)

        if position:
            # Update state
            self.state = self.state.perform_action((box, position, rotation))
            self.placed_boxes.append((box, position, rotation))
            self.state.boxes_to_place.remove(box)
            reward = self.state.calculate_reward()
            terminated = len(self.state.boxes_to_place) == 0
        else:
            # Couldn't find a valid position
            reward = -1
            terminated = True

        truncated = False

        # Provide episode information when the episode ends
        if terminated or truncated:
            info = {
                'episode': {
                    'r': reward,
                    'l': self.current_step,
                    'total_volume_packed': self._calculate_total_volume_packed(),
                    'num_boxes_used': len(self.placed_boxes),
                }
            }
        else:
            info = {}

        observation = self._get_observation()
        self.current_step += 1
        return observation, reward, terminated, truncated, info

    def _find_position_for_box(self, box):
        # Iterate through possible positions with a certain step size for efficiency
        step_size = 1  # Adjust as needed
        for x in range(0, int(self.container_dims[0] - box.width + 1), step_size):
            for y in range(0, int(self.container_dims[1] - box.depth + 1), step_size):
                for z in range(0, int(self.container_dims[2] - box.height + 1), step_size):
                    position = (x, y, z)
                    if self.state.can_place_item_at_position(box, position, (box.width, box.height, box.depth)):
                        return position
        # If no valid position found, try rotating the box and attempt placement
        rotations = [
            (box.width, box.depth, box.height),
            (box.depth, box.width, box.height),
            (box.height, box.depth, box.width)
        ]
        for rot in rotations:
            for x in range(0, int(self.container_dims[0] - rot[0] + 1), step_size):
                for y in range(0, int(self.container_dims[1] - rot[1] + 1), step_size):
                    for z in range(0, int(self.container_dims[2] - rot[2] + 1), step_size):
                        position = (x, y, z)
                        if self.state.can_place_item_at_position(box, position, rot):
                            return position
        # If still no position, return None
        return None

    def _get_observation(self):
        obs = []
        for box, position, rotation in self.placed_boxes:
            obs.extend(position)
            obs.extend(rotation)
        # Pad the observation if necessary
        obs += [0] * (self.observation_space.shape[0] - len(obs))
        return np.array(obs, dtype=np.float32)

    def _calculate_total_volume_packed(self):
        total_volume = 0
        for box, _, _ in self.placed_boxes:
            volume = box.width * box.height * box.depth
            total_volume += volume
        return total_volume

    def _draw_container_edges(self, ax):
        w, d, h = self.container_dims
        # Define the corners of the container
        corners = np.array([
            [0, 0, 0],
            [w, 0, 0],
            [w, d, 0],
            [0, d, 0],
            [0, 0, h],
            [w, 0, h],
            [w, d, h],
            [0, d, h]
        ])
        # Define the edges connecting the corners
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
        ]
        for edge in edges:
            x_vals = [corners[edge[0]][0], corners[edge[1]][0]]
            y_vals = [corners[edge[0]][1], corners[edge[1]][1]]
            z_vals = [corners[edge[0]][2], corners[edge[1]][2]]
            ax.plot(x_vals, y_vals, z_vals, color='black')

    def _draw_box(self, ax, box, position, rotation):
        x, y, z = position
        dx, dy, dz = rotation
        # Create a list of the box's corner coordinates
        corners = np.array([
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz]
        ])
        # List of box faces defined by the corner indices
        faces = [
            [0, 1, 2, 3],  # Bottom face
            [4, 5, 6, 7],  # Top face
            [0, 1, 5, 4],  # Front face
            [1, 2, 6, 5],  # Right face
            [2, 3, 7, 6],  # Back face
            [3, 0, 4, 7]   # Left face
        ]
        # Plot each face
        for face in faces:
            square = corners[face]
            poly = Poly3DCollection([square], facecolors=np.random.rand(3,), linewidths=1, edgecolors='r', alpha=0.5)
            ax.add_collection3d(poly)

    def close(self):
        if self.fig is not None:
            plt.ioff()  # Turn off interactive mode
            plt.show()  # Keep the final plot open
            plt.close(self.fig)
            self.fig = None
            self.ax = None
