# rl/box.py

class Box:
    def __init__(self, width, height, depth, box_id):
        self.width = width
        self.height = height
        self.depth = depth
        self.id = box_id

    def __eq__(self, other):
        return (self.width == other.width and
                self.height == other.height and
                self.depth == other.depth and
                self.id == other.id)

    def __hash__(self):
        return hash((self.width, self.height, self.depth, self.id))
