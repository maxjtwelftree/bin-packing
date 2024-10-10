# state.py

class State:
    def __init__(self, width, height, depth, boxes_to_place=None):
        self.width = width
        self.height = height
        self.depth = depth
        self.boxes_to_place = boxes_to_place.copy() if boxes_to_place else []
        self.placed_boxes = []

    def can_place_item_at_position(self, box, position, rotation):
        x, y, z = position
        dx, dy, dz = rotation
        new_box_bounds = [x, x + dx, y, y + dy, z, z + dz]

        for placed_box, pos, rot in self.placed_boxes:
            px, py, pz = pos
            pdx, pdy, pdz = rot
            placed_box_bounds = [px, px + pdx, py, py + pdy, pz, pz + pdz]

            if self._boxes_overlap(new_box_bounds, placed_box_bounds):
                return False
        if (x + dx > self.width) or (y + dy > self.height) or (z + dz > self.depth):
            return False
        return True

    def _boxes_overlap(self, box1, box2):
        return (box1[0] < box2[1] and box1[1] > box2[0] and
                box1[2] < box2[3] and box1[3] > box2[2] and
                box1[4] < box2[5] and box1[5] > box2[4])

    def perform_action(self, action):
        box, position, rotation = action
        self.placed_boxes.append((box, position, rotation))
        return self

    def calculate_reward(self):
        # Reward based on the total volume packed
        return self._calculate_total_volume_packed()

    def _calculate_total_volume_packed(self):
        # Calculate the total volume of all placed boxes
        total_volume = 0
        for box, _, _ in self.placed_boxes:
            total_volume += box.width * box.height * box.depth
        return total_volume
