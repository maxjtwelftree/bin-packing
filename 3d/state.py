from typing import List, Tuple
from itertools import permutations

# Define the penalty factor globally
penalty_factor = 10  # Adjust this value as needed

# Box class representing a 3D box to be packed
class Box:
    def __init__(self, width: int, height: int, depth: int, box_id: int):
        self.width = width
        self.height = height
        self.depth = depth
        self.id = box_id

    # Returns possible rotations (all unique permutations of dimensions)
    def get_rotations(self):
        return list(set(permutations((self.width, self.height, self.depth))))

    # Define equality comparison for Box based on id
    def __eq__(self, other):
        if isinstance(other, Box):
            return self.id == other.id
        return False

    # Define hash for using Box in sets or dicts
    def __hash__(self):
        return hash(self.id)

# State class representing the packing state of the 3D container
class State:
    def __init__(self, width: int, height: int, depth: int, boxes_to_place=None, action_history=None, available_spaces=None):
        if boxes_to_place is None:
            boxes_to_place = []
        if action_history is None:
            action_history = []
        if available_spaces is None:
            # Initially, the entire container is available as a single space
            available_spaces = [((0, 0, 0), (width, height, depth))]
        self.width = width
        self.height = height
        self.depth = depth
        self.boxes_to_place = boxes_to_place.copy()
        self.action_history = action_history.copy()  # Keeps track of placed boxes
        self.available_spaces = available_spaces.copy()  # List of available spaces (each space is defined by min and max coordinates)

    # Clone the state (deepcopy)
    def clone(self):
        return State(
            self.width,
            self.height,
            self.depth,
            self.boxes_to_place.copy(),
            self.action_history.copy(),
            self.available_spaces.copy()
        )

    def add_box(self, box: Box):
        self.boxes_to_place.append(box)

    # Get all possible actions (box, position, rotation) that can be performed
    def get_possible_actions(self):
        actions = []
        for box in self.boxes_to_place:
            rotations = box.get_rotations()  # Get possible rotations
            for rotation in rotations:
                for space in self.available_spaces:
                    if self.can_place_item(space, rotation):
                        # Place the box at the minimum coordinates of the space
                        position = space[0]
                        actions.append((box, position, rotation))  # Possible actions
        return actions

    # Check if the box can be placed in the specified space with the given rotation
    def can_place_item(self, space, rotation):
        (x0, y0, z0), (x1, y1, z1) = space
        box_width, box_height, box_depth = rotation
        # Check if the box fits within the space
        if (box_width <= x1 - x0 and
            box_height <= y1 - y0 and
            box_depth <= z1 - z0):
            return True
        return False

    # Split the space after placing the box
    def split(self, space, box, position, rotation):
        (x0, y0, z0), (x1, y1, z1) = space
        box_width, box_height, box_depth = rotation
        px, py, pz = position

        # Remove the occupied space
        self.available_spaces.remove(space)

        # Calculate the remaining spaces after placing the box
        new_spaces = []

        # Right space
        if px + box_width < x1:
            new_space = ((px + box_width, y0, z0), (x1, y1, z1))
            new_spaces.append(new_space)

        # Front space
        if py + box_height < y1:
            new_space = ((px, py + box_height, z0), (px + box_width, y1, z1))
            new_spaces.append(new_space)

        # Top space
        if pz + box_depth < z1:
            new_space = ((px, py, pz + box_depth), (px + box_width, py + box_height, z1))
            new_spaces.append(new_space)

        # Left space
        if x0 < px:
            new_space = ((x0, y0, z0), (px, y1, z1))
            new_spaces.append(new_space)

        # Back space
        if y0 < py:
            new_space = ((px, y0, z0), (px + box_width, py, z1))
            new_spaces.append(new_space)

        # Bottom space
        if z0 < pz:
            new_space = ((px, py, z0), (px + box_width, py + box_height, pz))
            new_spaces.append(new_space)

        # Add the new spaces to the available_spaces list
        self.available_spaces.extend(new_spaces)

    # Perform an action by placing a box, and return a new State
    def perform_action(self, action):
        # Clone the current state to avoid modifying the original
        new_state = self.clone()
        box, position, rotation = action
        new_state.action_history.append((box, position, rotation))  # Record the action
        # Find the space where the box is placed
        space = None
        for s in new_state.available_spaces:
            if s[0] == position and new_state.can_place_item(s, rotation):
                space = s
                break
        if space is None:
            # This should not happen, but just in case
            return new_state
        new_state.split(space, box, position, rotation)
        new_state.boxes_to_place.remove(box)
        # Optionally merge adjacent spaces
        new_state.merge_spaces()
        return new_state  # Return the new state

    # Merge adjacent free spaces (optional, not implemented)
    def merge_spaces(self):
        # Implement logic to merge adjacent or overlapping spaces if desired
        pass

    # Evaluate the current state by computing total volume and penalizing the number of boxes placed
    def evaluation(self):
        total_volume = sum((action[0].width * action[0].height * action[0].depth for action in self.action_history))
        num_boxes_placed = len(self.action_history)
        # Subtract the number of boxes placed multiplied by penalty factor to prioritize fewer boxes
        return total_volume - (num_boxes_placed * penalty_factor)

    # Define less-than comparison for State based on evaluation value
    def __lt__(self, other):
        if not isinstance(other, State):
            return NotImplemented

        return self.evaluation() < other.evaluation()

    # Define equality comparison for State
    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented

        return self.evaluation() == other.evaluation()
