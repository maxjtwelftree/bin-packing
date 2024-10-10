import random
from typing import List

from state import State, Box
from monte import mcts

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
import numpy as np

# Function to generate random boxes
def generate_random_boxes(n_boxes: int, max_width: int, max_height: int, max_depth: int):
    boxes = []
    for i in range(1, n_boxes + 1):
        width = random.randint(1, max_width)
        height = random.randint(1, max_height)
        depth = random.randint(1, max_depth)
        boxes.append(Box(width, height, depth, i))
    return boxes

# Function to visualize the current state using Matplotlib 3D
def plot_state_3d(state: State, title: str, save_filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    container_width = state.width
    container_height = state.height
    container_depth = state.depth

    # Generate a list of colors for the boxes
    color_list = list(mcolors.TABLEAU_COLORS.values())
    num_colors = len(color_list)
    
    # Plot each box based on the action history
    for idx, action in enumerate(state.action_history):
        box, position, rotation = action
        x0, y0, z0 = position
        box_width, box_height, box_depth = rotation
        color = color_list[idx % num_colors]
        
        # Define the corners of the box
        x = [x0, x0 + box_width]
        y = [y0, y0 + box_height]
        z = [z0, z0 + box_depth]
        # Create a mesh grid
        xx, yy = np.meshgrid(x, y)
        # Plot surfaces of the box
        ax.plot_surface(xx, yy, np.full_like(xx, z[0]), color=color, alpha=0.7)
        ax.plot_surface(xx, yy, np.full_like(xx, z[1]), color=color, alpha=0.7)
        zz, yy = np.meshgrid(z, y)
        ax.plot_surface(np.full_like(zz, x[0]), yy, zz, color=color, alpha=0.7)
        ax.plot_surface(np.full_like(zz, x[1]), yy, zz, color=color, alpha=0.7)
        xx, zz = np.meshgrid(x, z)
        ax.plot_surface(xx, np.full_like(xx, y[0]), zz, color=color, alpha=0.7)
        ax.plot_surface(xx, np.full_like(xx, y[1]), zz, color=color, alpha=0.7)
        # Add box ID at the center
        ax.text(
            x0 + box_width / 2,
            y0 + box_height / 2,
            z0 + box_depth / 2,
            f'ID:{box.id}',
            ha='center',
            va='center',
            color='black',
            fontsize=8
        )

    # Set plot boundaries and labels
    ax.set_xlim(0, container_width)
    ax.set_ylim(0, container_height)
    ax.set_zlim(0, container_depth)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Depth')
    ax.set_title(title)
    ax.view_init(elev=20., azim=-35)
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename)
        plt.close(fig)
    else:
        plt.show()

def mcts_packing(boxes, width, height, depth, iterations_per_move=100):
    state = State(width, height, depth)
    for box in boxes:
        state.add_box(box)
    step = 0
    while state.boxes_to_place and state.get_possible_actions():
        # Use MCTS to select the best action
        best_action = mcts(state, iterations=iterations_per_move)
        if best_action:
            state = state.perform_action(best_action)
            # Plot the current state and save it as an image
            step += 1
            print(f"Step {step}: Placed box {best_action[0].id}")
            plot_state_3d(state, f"Step {step}", save_filename=f"step_{step}.png")
        else:
            break  # No valid actions available
    return state

def main():
    # Generate random boxes
    num_boxes = 10  # Reduced for better visualization
    max_box_width = 5
    max_box_height = 5
    max_box_depth = 5
    boxes = generate_random_boxes(num_boxes, max_box_width, max_box_height, max_box_depth)
    # Set container dimensions
    container_width, container_height, container_depth = 10, 10, 10
    # Run the MCTS packing simulation
    final_state = mcts_packing(boxes, container_width, container_height, container_depth, iterations_per_move=1000)
    # Plot the final state
    plot_state_3d(final_state, "Final Packing State")

if __name__ == "__main__":
    main()
