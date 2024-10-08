import random
import matplotlib.pyplot as plt
from typing import List

from state import State, Box

# Function to generate random boxes
def generate_random_boxes(n_boxes: int, max_width: int, max_height: int):
    boxes = []
    for i in range(1, n_boxes + 1):
        width = random.randint(1, max_width)
        height = random.randint(1, max_height)
        boxes.append(Box(width, height, i))
    return boxes

# Function to visualize the current state using matplotlib
def plot_state(state: State, title: str, save_filename=None):
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors

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

    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename)
        plt.close(fig)
    else:
        plt.show()

def ffdh_packing(boxes, width, height):
    # Sort boxes in decreasing order of height (or area)
    boxes_sorted = sorted(boxes, key=lambda b: b.height * b.width, reverse=True)
    state = State(width, height)
    for box in boxes_sorted:
        state.add_box(box)
    step = 0
    while state.boxes_to_place:
        box = state.boxes_to_place[0]  # Always select the next box in the sorted list
        placed = False
        rotations = box.get_rotations()
        # Try to place the box in existing layers
        for rotation in rotations:
            for layer in sorted(state.available_spaces.keys()):
                intervals = state.available_spaces[layer]
                for interval in intervals:
                    if state.can_place_item(layer, interval, rotation):
                        action = (box, layer, interval, rotation)
                        state = state.perform_action(action)
                        # Visualization
                        step += 1
                        plot_state(state, f"Step {step}", save_filename=f"step_{step}.png")
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            # Cannot place the box; remove it
            state.boxes_to_place.remove(box)
    return state

def main():
    # Generate random boxes
    num_boxes = 30
    max_box_width = 5
    max_box_height = 5
    boxes = generate_random_boxes(num_boxes, max_box_width, max_box_height)
    # Set container dimensions
    container_width, container_height = 10, 10
    # Run the FFDH packing simulation
    final_state = ffdh_packing(boxes, container_width, container_height)
    # Plot the final state
    plot_state(final_state, "Final Packing State")

if __name__ == "__main__":
    main()
