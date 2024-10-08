import random

import matplotlib.pyplot as plt
import copy
from typing import List, Dict, Tuple

from State import State, Box

# Function to visualize the current state using matplotlib
def plot_state(state: State, title: str):
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
    plt.show()

def random_one_step(pre_boxes, width, height):
    boxes = [item for item in pre_boxes]
    state = State(width, height)
    # Perform first action if available and generate a new state
    for box in boxes:
        state.add_box(box)
        possible_actions = state.get_possible_actions()
        action = random.choice(possible_actions)
        state = state.perform_action(action)
    return state

def get_best(pre_boxes, width, height, epoch=5000):
    # Example setup to simulate bin packing
    best_state = None
    # Perform first action if available and generate a new state
    for i in range(epoch):
        state = random_one_step(pre_boxes, width, height)
        if best_state is None or best_state < state:
            best_state = state
    # Plot the new state after action
    plot_state(best_state, "2D Bin Packing After First Action")

# Example 1: Small boxes
boxes1 = [Box(1, 1, 1), Box(1, 1, 2), Box(2, 2, 3)]
get_best(boxes1, 10, 10)

# Example 2: Larger boxes
boxes2 = [Box(3, 3, 1), Box(4, 5, 2), Box(2, 2, 3)]
get_best(boxes2, 10, 10)

# Example 3: Mixed size boxes
boxes3 = [Box(1, 2, 1), Box(5, 3, 2), Box(2, 1, 3), Box(3, 3, 4)]
get_best(boxes3, 10, 10)

# Example 4: More boxes with different sizes
boxes4 = [Box(2, 3, 1), Box(3, 2, 2), Box(4, 4, 3), Box(1, 1, 4), Box(2, 5, 5)]
get_best(boxes4, 10, 10)

# Example 5: More complex set with different sizes
boxes5 = [Box(3, 3, 1), Box(1, 1, 2), Box(5, 2, 3), Box(2, 6, 4), Box(4, 3, 5), Box(2, 2, 6)]
get_best(boxes5, 10, 10)

# Example 6: Larger container
boxes6 = [Box(5, 5, 1), Box(3, 3, 2), Box(7, 2, 3), Box(4, 4, 4)]
get_best(boxes6, 15, 15)
