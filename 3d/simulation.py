import random
import time
from typing import List, Tuple

from state import State, Box
from monte import mcts
from features import extract_features

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

def generate_random_boxes(n_boxes: int, max_width: int, max_height: int, max_depth: int) -> List[Box]:
    boxes = []
    for i in range(1, n_boxes + 1):
        width = random.randint(1, max_width)
        height = random.randint(1, max_height)
        depth = random.randint(1, max_depth)
        boxes.append(Box(width, height, depth, i))
    return boxes

def plot_state_3d(state: State, title: str, save_filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    container_width = state.width
    container_height = state.height
    container_depth = state.depth

    color_list = list(mcolors.TABLEAU_COLORS.values())
    num_colors = len(color_list)
    
    for idx, action in enumerate(state.action_history):
        box, position, rotation = action
        x0, y0, z0 = position
        box_width, box_height, box_depth = rotation
        color = color_list[idx % num_colors]
        
        # Define the corners of the box
        corners = [
            [x0, y0, z0],
            [x0 + box_width, y0, z0],
            [x0 + box_width, y0 + box_height, z0],
            [x0, y0 + box_height, z0],
            [x0, y0, z0 + box_depth],
            [x0 + box_width, y0, z0 + box_depth],
            [x0 + box_width, y0 + box_height, z0 + box_depth],
            [x0, y0 + box_height, z0 + box_depth]
        ]
        
        # Define the edges of the box
        edges = [
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[3]],
            [corners[3], corners[0]],
            [corners[4], corners[5]],
            [corners[5], corners[6]],
            [corners[6], corners[7]],
            [corners[7], corners[4]],
            [corners[0], corners[4]],
            [corners[1], corners[5]],
            [corners[2], corners[6]],
            [corners[3], corners[7]]
        ]
        
        # Plot the edges
        for edge in edges:
            xs, ys, zs = zip(*edge)
            ax.plot(xs, ys, zs, color=color)
        
        # Add box ID and dimensions at the center
        ax.text(
            x0 + box_width / 2,
            y0 + box_height / 2,
            z0 + box_depth / 2,
            f'ID:{box.id}\n{box_width}x{box_height}x{box_depth}',
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

def mcts_packing_with_timing_and_reward(boxes: List[Box], width: int, height: int, depth: int, iterations_per_move: int = 1000) -> Tuple[State, List[float], List[float], List[np.ndarray]]:
    state = State(width, height, depth)
    for box in boxes:
        state.add_box(box)
    step = 0
    time_costs = []
    rewards = []
    feature_vectors = []
    
    while state.boxes_to_place and state.get_possible_actions():
        start_time = time.time()
        best_action = mcts(state, iterations=iterations_per_move)
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_costs.append(elapsed_time)
        
        if best_action:
            state = state.perform_action(best_action)
            step += 1
            reward = state.evaluation()
            rewards.append(reward)
            features = extract_features(state)
            feature_vectors.append(features)
            print(f"Step {step}: Placed box {best_action[0].id} in {elapsed_time:.4f} seconds with reward {reward}")
            plot_state_3d(state, f"Step {step}", save_filename=f"step_{step}.png")
        else:
            print("No valid actions available. Terminating packing.")
            break
    
    return state, time_costs, rewards, feature_vectors

def plot_time_cost(time_costs: List[float]):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(time_costs) + 1), time_costs, marker='o', linestyle='-', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Time Cost (seconds)')
    plt.title('MCTS Time Cost per Step')
    plt.grid(True)
    plt.savefig('mcts_time_cost.png')
    plt.close()
    print("MCTS time cost plot saved as 'mcts_time_cost.png'.")

def plot_reward_curve(rewards: List[float]):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, marker='x', linestyle='-', color='green')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward vs. Number of Steps')
    plt.grid(True)
    plt.savefig('reward_curve.png')
    plt.close()
    print("Reward curve plot saved as 'reward_curve.png'.")

def main():
    num_boxes = 30
    max_box_width = 5
    max_box_height = 5
    max_box_depth = 5
    boxes = generate_random_boxes(num_boxes, max_box_width, max_box_height, max_box_depth)
    
    visualize_boxes_2d(boxes)
    
    container_width, container_height, container_depth = 10, 10, 10
    
    final_state, time_costs, rewards, feature_vectors = mcts_packing_with_timing_and_reward(
        boxes,
        container_width,
        container_height,
        container_depth,
        iterations_per_move=1000
    )
    
    plot_state_3d(final_state, "Final Packing State", save_filename="final_packing_state.png")
    print("Final packing state plot saved as 'final_packing_state.png'.")
    
    plot_time_cost(time_costs)
    plot_reward_curve(rewards)
    
    np.save('feature_vectors.npy', feature_vectors)
    print("Feature vectors saved as 'feature_vectors.npy'.")

def visualize_boxes_2d(boxes: List[Box]):
    fig, ax = plt.subplots(figsize=(12, 6))
    color_list = list(mcolors.TABLEAU_COLORS.values())
    spacing = 1
    current_x = 0
    
    print("Initial Boxes:")
    for box in boxes:
        width, height, depth, box_id = box.width, box.height, box.depth, box.id
        print(f"Box ID: {box_id}, Dimensions: w {width}x h {height}x d{depth}")
        rect = patches.Rectangle((current_x, 0), width, height, linewidth=1, edgecolor='black', facecolor=random.choice(color_list), alpha=0.7)
        ax.add_patch(rect)
        ax.text(current_x + width / 2, height / 2, f'ID:{box_id}\n{width}x{height}x{depth}', 
                ha='center', va='center', fontsize=8, color='black')
        current_x += width + spacing
    
    max_x = current_x
    max_y = max(box.height for box in boxes) + 5
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Initial Boxes to be Packed, width, height, depth')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('initial_boxes_2d.png')
    plt.close(fig)
    print("Initial boxes 2D visualization saved as 'initial_boxes_2d.png'.")

if __name__ == "__main__":
    main()
