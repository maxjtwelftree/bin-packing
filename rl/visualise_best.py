# rl/visualize_best_packing.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from box import Box

def visualize_best_packing(placed_boxes, container_dims):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, container_dims[0])
    ax.set_ylim(0, container_dims[1])
    ax.set_zlim(0, container_dims[2])
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Height')
    ax.set_title('Best Packing Arrangement')

    # Draw the container edges
    _draw_container_edges(ax, container_dims)

    # Plot each placed box
    for box, position, rotation in placed_boxes:
        _draw_box(ax, box, position, rotation)

    plt.show()

def _draw_container_edges(ax, container_dims):
    w, d, h = container_dims
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
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        x_vals = [corners[edge[0]][0], corners[edge[1]][0]]
        y_vals = [corners[edge[0]][1], corners[edge[1]][1]]
        z_vals = [corners[edge[0]][2], corners[edge[1]][2]]
        ax.plot(x_vals, y_vals, z_vals, color='black')

def _draw_box(ax, box, position, rotation):
    x, y, z = position
    dx, dy, dz = rotation
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
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7]
    ]
    for face in faces:
        square = corners[face]
        poly = Poly3DCollection([square], facecolors=np.random.rand(3,), linewidths=1, edgecolors='r', alpha=0.5)
        ax.add_collection3d(poly)

if __name__ == "__main__":
    # Load the best state
    placed_boxes = np.load('best_packing_state.npy', allow_pickle=True)
    # Define container dimensions (should match your environment)
    container_dims = (10, 10, 10)  # Adjust as needed
    visualize_best_packing(placed_boxes, container_dims)
