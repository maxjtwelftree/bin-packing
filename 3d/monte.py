import math
import random
from state import State, Box

class MCTSNode:
    def __init__(self, state: State, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # The action that led to this state
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.untried_actions = state.get_possible_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.reward / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.perform_action(action)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def rollout_policy(self, possible_actions):
        return random.choice(possible_actions)

    def rollout(self):
        current_state = self.state.clone()
        while current_state.boxes_to_place and current_state.get_possible_actions():
            action = self.rollout_policy(current_state.get_possible_actions())
            current_state = current_state.perform_action(action)
        return self.evaluate_state(current_state)

    def backpropagate(self, reward):
        self.visits += 1
        self.reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def tree_policy(self):
        current_node = self
        while current_node.state.boxes_to_place and current_node.state.get_possible_actions():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def evaluate_state(self, state):
        # Define a reward function based on the total volume of placed boxes
        total_volume = sum((action[0].width * action[0].height * action[0].depth for action in state.action_history))
        return total_volume

def mcts(root_state, iterations=1000):
    root_node = MCTSNode(root_state)
    for _ in range(iterations):
        node = root_node.tree_policy()
        reward = node.rollout()
        node.backpropagate(reward)
    # Select the action corresponding to the best child
    best_child = root_node.best_child(c_param=0)
    return best_child.action
