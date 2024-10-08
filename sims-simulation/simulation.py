import random
import matplotlib.pyplot as plt
from typing import List, Tuple
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
    fig, ax = plt.subplots()
    container_height = state.height
    container_width = state.width

    # Plot each box based on the action history
    for action in state.action_history:
        box, layer, interval, rotation = action
        box_width, box_height = rotation
        # Create a rectangle representing the placed box
        rect = plt.Rectangle((interval[0], layer), box_width, box_height, fill=True, edgecolor='black',
                             facecolor='blue', label=f'Box {box.id}')
        ax.add_patch(rect)

    # Set plot boundaries and labels
    ax.set_xlim(0, container_width)
    ax.set_ylim(0, container_height)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    if save_filename:
        plt.savefig(save_filename)
        plt.close(fig)
    else:
        plt.show()

def random_packing(boxes, width, height):
    state = State(width, height)
    for box in boxes:
        state.add_box(box)
    step = 0
    while state.boxes_to_place:
        box = random.choice(state.boxes_to_place)
        possible_actions = []
        rotations = box.get_rotations()
        # For the selected box, find possible actions
        for layer, intervals in state.available_spaces.items():
            for interval in intervals:
                for rotation in rotations:
                    if state.can_place_item(layer, interval, rotation):
                        possible_actions.append((box, layer, interval, rotation))
        if possible_actions:
            action = random.choice(possible_actions)
            state = state.perform_action(action)
            # Plot the current state and save it as an image
            step += 1
            plot_state(state, f"Step {step}", save_filename=f"step_{step}.png")
        else:
            # Cannot place the box, remove it
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
    # Run the random packing simulation
    final_state = random_packing(boxes, container_width, container_height)
    # Plot the final state
    plot_state(final_state, "Final Packing State")

if __name__ == "__main__":
    main()
