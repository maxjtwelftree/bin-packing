from typing import List, Dict, Tuple

# Box class representing a rectangle to be packed
class Box:
    def __init__(self, width: int, height: int, box_id: int):
        self.width = width
        self.height = height
        self.id = box_id

    # Returns possible rotations (width, height) and (height, width)
    def get_rotations(self):
        return [(self.width, self.height), (self.height, self.width)]

    # Define equality comparison for Box based on id
    def __eq__(self, other):
        if isinstance(other, Box):
            return self.id == other.id
        return False

    # Define hash for using Box in sets or dicts
    def __hash__(self):
        return hash(self.id)

# State class representing the packing state of the container
class State:
    def __init__(self, width: int, height: int, boxes_to_place=None, action_history=None, available_spaces=None):
        if boxes_to_place is None:
            boxes_to_place = []
        if action_history is None:
            action_history = []
        if available_spaces is None:
            available_spaces = {0: [(0, width)]}
        self.width = width
        self.height = height
        self.boxes_to_place = [item for item in boxes_to_place]
        self.action_history = [item for item in action_history]  # Keeps track of placed boxes
        self.available_spaces = {key:available_spaces.get(key) for key in available_spaces.keys()} # Initially entire width available at layer 0

    # Clone the state (deepcopy)
    def clone(self):
        return State(
            self.width,
            self.height,
            self.boxes_to_place,
            self.action_history,
            self.available_spaces
        )

    def add_box(self, box: Box):
        self.boxes_to_place.append(box)

    # Get all possible actions (box, layer, interval, rotation) that can be performed
    def get_possible_actions(self):
        actions = []
        for box in self.boxes_to_place:
            rotations = box.get_rotations()  # Get possible rotations
            for layer, intervals in self.available_spaces.items():
                for interval in intervals:
                    for rotation in rotations:
                        if self.can_place_item(layer, interval, rotation):
                            actions.append((box, layer, interval, rotation))  # Possible actions
        return actions

    # Check if the box can be placed in a specified layer and interval
    def can_place_item(self, layer, interval: Tuple[int, int], rotation: Tuple[int, int]):
        box_width, box_height = rotation
        # Check width and height constraints
        if box_width <= interval[1] - interval[0] and box_height + layer <= self.height:
            return True
        return False

    # Split the space after placing the box
    def split(self, layer, interval, box, rotation):
        box_width, box_height = rotation
        new_intervals = []

        # Update the space at the current layer (horizontal split)
        if interval[0] + box_width < interval[1]:
            new_intervals.append((interval[0] + box_width, interval[1]))  # Remaining space in the interval

        self.available_spaces[layer].remove(interval)
        self.available_spaces[layer].extend(new_intervals)

        # Add new space to the next layer if box extends vertically
        if box_height > 0:
            new_layer = layer + box_height
            if new_layer not in self.available_spaces:
                self.available_spaces[new_layer] = []
            self.available_spaces[new_layer].append((interval[0], interval[0] + box_width))

    # Perform an action by placing a box, and return a new State
    def perform_action(self, action):
        # Clone the current state to avoid modifying the original
        new_state = self.clone()
        box, layer, interval, rotation = action
        new_state.action_history.append(action)  # Record the action
        new_state.split(layer, interval, box, rotation)
        new_state.boxes_to_place.remove(box)
        new_state.merge()  # Merge adjacent intervals
        return new_state  # Return the new state

    # Merge adjacent free intervals
    def merge(self):
        for layer in self.available_spaces:
            merged_intervals = []
            intervals = sorted(self.available_spaces[layer])  # Sort the intervals by start point
            if not intervals:
                continue
            current_start, current_end = intervals[0]

            for start, end in intervals[1:]:
                if start == current_end:  # Merge intervals if they are adjacent
                    current_end = end
                else:
                    merged_intervals.append((current_start, current_end))
                    current_start, current_end = start, end

            merged_intervals.append((current_start, current_end))  # Append the last interval
            self.available_spaces[layer] = merged_intervals

    # Evaluate the current state by computing total area and remaining available space
    def evaluation(self):
        total_area = sum((action[0].width * action[0].height for action in self.action_history))  # Area of placed boxes
        nums_available = sum(len(spaces) for spaces in self.available_spaces.values())  # Number of available intervals
        least_layer = min(self.available_spaces.keys())  # The highest occupied layer
        return total_area, nums_available, least_layer

    # Define less-than comparison for State based on evaluation rules
    def __lt__(self, other):
        if not isinstance(other, State):
            return NotImplemented

        # Get evaluation values for both states
        self_eval = self.evaluation()
        other_eval = other.evaluation()

        # Compare based on the three criteria
        if self_eval[0] != other_eval[0]:  # Compare total_area
            return self_eval[0] < other_eval[0]
        elif self_eval[1] != other_eval[1]:  # If total_area is equal, compare nums_available
            return self_eval[1] > other_eval[1]
        else:  # If both are equal, compare least_layer
            return self_eval[2] < other_eval[2]

    # Define equality comparison for State
    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented

        # Compare based on evaluation
        return self.evaluation() == other.evaluation()