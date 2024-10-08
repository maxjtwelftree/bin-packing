import random
import matplotlib.pyplot as plt
from typing import List

from state import State, Box
from monte import mcts

# Function to generate random boxes
def generate_random_boxes(n_boxes: int, max_width: int, max_height: int):
    boxes = []
    for i in range(1, n_boxes + 1):
        width = random.randint(1, max_width)
        height = random.randint(1, max_height)
        boxes.append(Box(width, height, i))
    return boxes

# Function to visualize the current state using matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

def plot_state(state: State, title: str, save_filename=None):
    fig, ax = plt.subplots()
    container_height = state.height
    container_width = state.width

    # Generate a list of colors for the boxes
    color_list = list(mcolors.TABLEAU_COLORS.values())
    num_colors = len(color_list)
    
    # Create a mapping from box IDs to colors
    box_colors = {}
    for idx, action in enumerate(state.action_history):
        box_id = action[0].id
        if box_id not in box_colors:
            box_colors[box_id] = color_list[idx % num_colors]

    # Plot each box based on the action history
    for action in state.action_history:
        box, layer, interval, rotation = action
        box_width, box_height = rotation
        x0 = interval[0]
        y0 = layer
        # Create a rectangle representing the placed box
        rect = patches.Rectangle(
            (x0, y0),
            box_width,
            box_height,
            linewidth=1,
            edgecolor='black',
            facecolor=box_colors[box.id],
            label=f'Box {box.id}'
        )
        ax.add_patch(rect)
        # Add box ID and dimensions in the center
        ax.text(
            x0 + box_width / 2,
            y0 + box_height / 2,
            f'ID:{box.id}\n{box_width}x{box_height}',
            ha='center',
            va='center',
            color='white',
            fontsize=8
        )

    # Set plot boundaries and labels
    ax.set_xlim(0, container_width)
    ax.set_ylim(0, container_height)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.grid(True)

    # Create a legend
    handles = [patches.Patch(color=box_colors[box_id], label=f'Box {box_id}') for box_id in box_colors]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename)
        plt.close(fig)
    else:
        plt.show()

def mcts_packing(boxes, width, height, iterations_per_move=100):
    state = State(width, height)
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
            plot_state(state, f"Step {step}", save_filename=f"step_{step}.png")
        else:
            break  # No valid actions available
    return state

def main():
    # Generate random boxes
    num_boxes = 30
    max_box_width = 5
    max_box_height = 5
    boxes = generate_random_boxes(num_boxes, max_box_width, max_box_height)
    # Set container dimensions
    container_width, container_height = 10, 10
    # Run the MCTS packing simulation
    final_state = mcts_packing(boxes, container_width, container_height, iterations_per_move=100000)
    # Plot the final state
    plot_state(final_state, "Final Packing State")

if __name__ == "__main__":
    main()
