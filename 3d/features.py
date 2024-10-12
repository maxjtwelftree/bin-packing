# features.py

import numpy as np
from state import State

def extract_features(state: State):
    features = []
    
    # Container dimensions
    features.extend([state.width, state.height, state.depth])
    
    # Placed boxes statistics
    num_placed = len(state.action_history)
    total_volume_placed = sum(box.width * box.height * box.depth for box, _, _ in state.action_history)
    features.extend([num_placed, total_volume_placed])
    
    # Occupied volume ratio
    container_volume = state.width * state.height * state.depth
    occupied_volume_ratio = total_volume_placed / container_volume if container_volume > 0 else 0
    features.append(occupied_volume_ratio)
    
    # Positions statistics
    if num_placed > 0:
        positions = np.array([position for _, position, _ in state.action_history])
        mean_position = positions.mean(axis=0)
        std_position = positions.std(axis=0)
        features.extend(mean_position.tolist())
        features.extend(std_position.tolist())
    else:
        features.extend([0, 0, 0])  
        features.extend([0, 0, 0])  
    
    # Orientations statistics
    orientations = np.array([rotation for _, _, rotation in state.action_history])
    if num_placed > 0:
        mean_orientation = orientations.mean(axis=0)
        std_orientation = orientations.std(axis=0)
        features.extend(mean_orientation.tolist())
        features.extend(std_orientation.tolist())
    else:
        features.extend([0, 0, 0]) 
        features.extend([0, 0, 0])  
    
    # Remaining boxes statistics
    num_remaining = len(state.boxes_to_place)
    total_volume_remaining = sum(box.width * box.height * box.depth for box in state.boxes_to_place)
    avg_width = np.mean([box.width for box in state.boxes_to_place]) if num_remaining > 0 else 0
    avg_height = np.mean([box.height for box in state.boxes_to_place]) if num_remaining > 0 else 0
    avg_depth = np.mean([box.depth for box in state.boxes_to_place]) if num_remaining > 0 else 0
    variance_width = np.var([box.width for box in state.boxes_to_place]) if num_remaining > 0 else 0
    variance_height = np.var([box.height for box in state.boxes_to_place]) if num_remaining > 0 else 0
    variance_depth = np.var([box.depth for box in state.boxes_to_place]) if num_remaining > 0 else 0
    features.extend([
        num_remaining,
        total_volume_remaining,
        avg_width,
        avg_height,
        avg_depth,
        variance_width,
        variance_height,
        variance_depth
    ])
    
    # Potential rotations
    total_potential_rotations = sum(len(box.get_rotations()) for box in state.boxes_to_place)
    avg_potential_rotations = total_potential_rotations / num_remaining if num_remaining > 0 else 0
    features.append(avg_potential_rotations)
    
    # Available spaces statistics
    num_spaces = len(state.available_spaces)
    total_available_volume = sum((x1 - x0) * (y1 - y0) * (z1 - z0) for (x0, y0, z0), (x1, y1, z1) in state.available_spaces)
    largest_space = max([(x1 - x0) * (y1 - y0) * (z1 - z0) for (x0, y0, z0), (x1, y1, z1) in state.available_spaces], default=0)
    features.extend([num_spaces, total_available_volume, largest_space])
    
    # Aspect ratios of available spaces
    if num_spaces > 0:
        aspect_ratios = []
        for (x0, y0, z0), (x1, y1, z1) in state.available_spaces:
            width = x1 - x0
            height = y1 - y0
            depth = z1 - z0
            if height != 0 and depth != 0:
                aspect_ratios.append(width / height)
                aspect_ratios.append(width / depth)
                aspect_ratios.append(height / depth)
        if aspect_ratios:
            mean_aspect_ratio = np.mean(aspect_ratios)
            std_aspect_ratio = np.std(aspect_ratios)
        else:
            mean_aspect_ratio = 0
            std_aspect_ratio = 0
    else:
        mean_aspect_ratio = 0
        std_aspect_ratio = 0
    features.extend([mean_aspect_ratio, std_aspect_ratio])
    
    # Last placed box ID (normalized)
    if state.action_history:
        last_box_id = state.action_history[-1][0].id
    else:
        last_box_id = -1 
    features.append(last_box_id)
    
    max_box_id = 1000  
    normalized_last_box_id = last_box_id / max_box_id if last_box_id >=0 else 0
    features[-1] = normalized_last_box_id  
    
    return np.array(features)
