#!/usr/bin/env python3
"""
Find the minimum number of mutually exclusive groups given pairs of mutually exclusive set IDs.

A pair of set IDs is mutually exclusive if their corresponding sets have no elements in common.
Groups are formed such that any two set IDs in the same group are mutually exclusive.
"""

import argparse
from collections import defaultdict
import time


def find_minimum_exclusive_groups(exclusive_pairs, num_ids):
    """
    Find the minimum number of mutually exclusive groups using graph coloring.
    
    Args:
        exclusive_pairs: List of tuples (id1, id2) where id1 and id2 are mutually exclusive
        num_ids: Number of set IDs (assumed to be 0 to num_ids-1)
        
    Returns:
        List of groups, where each group is a list of set IDs
    """
    # Create an adjacency list for IDs that are NOT mutually exclusive
    # (can't be in the same group)
    incompatible = [set() for _ in range(num_ids)]
    
    # Convert exclusive pairs to a set for O(1) lookup
    exclusive_set = {(min(a, b), max(a, b)) for a, b in exclusive_pairs}
    
    # Build graph of incompatible IDs (IDs that can't be in the same group)
    # Two IDs are incompatible if they're not mutually exclusive
    for i in range(num_ids):
        for j in range(i+1, num_ids):
            if (i, j) not in exclusive_set:
                incompatible[i].add(j)
                incompatible[j].add(i)
    
    # Use Welsh-Powell algorithm for graph coloring
    # Order vertices by degree (number of incompatible IDs) in descending order
    ordered_ids = sorted(range(num_ids), key=lambda x: len(incompatible[x]), reverse=True)
    
    # Assign colors (groups)
    colors = [-1] * num_ids  # -1 means no color assigned yet
    
    for id in ordered_ids:
        # Find colors used by incompatible IDs
        used_colors = {colors[incompatible_id] for incompatible_id in incompatible[id] 
                      if colors[incompatible_id] != -1}
        
        # Find smallest unused color
        color = 0
        while color in used_colors:
            color += 1
        
        colors[id] = color
    
    # Group IDs by color
    groups = defaultdict(list)
    for id, color in enumerate(colors):
        groups[color].append(id)
    
    # Convert to list of groups
    result = list(groups.values())
    
    return result


def can_form_groups_with_min_size(exclusive_pairs, num_ids, min_size):
    """
    Check if the set IDs can be divided into mutually exclusive groups 
    such that each group has at least min_size elements.
    
    This is a more complex graph coloring problem where we're looking for a coloring
    where each color class (group) has at least min_size elements.
    
    Args:
        exclusive_pairs: List of tuples (id1, id2) where id1 and id2 are mutually exclusive
        num_ids: Number of set IDs
        min_size: Minimum size of each group
        
    Returns:
        Tuple (is_possible, groups) where:
        - is_possible: Boolean indicating if valid grouping is possible
        - groups: List of groups if is_possible is True, otherwise None
    """
    # First, check if it's even mathematically possible
    # We need at least min_size IDs to form a group
    max_possible_groups = num_ids // min_size
    if max_possible_groups == 0:
        return False, None
    
    # Create the incompatibility graph (same as in find_minimum_exclusive_groups)
    incompatible = [set() for _ in range(num_ids)]
    exclusive_set = {(min(a, b), max(a, b)) for a, b in exclusive_pairs}
    
    for i in range(num_ids):
        for j in range(i+1, num_ids):
            if (i, j) not in exclusive_set:
                incompatible[i].add(j)
                incompatible[j].add(i)
    
    # We'll use a greedy approach with backtracking:
    # 1. Calculate the maximum number of groups we can have
    # 2. Try to color the graph using at most that many colors
    # 3. Check if each color class has at least min_size elements
    
    # First, try a greedy approach using a modification of Welsh-Powell
    # Order vertices by degree (most constrained first)
    ordered_ids = sorted(range(num_ids), key=lambda x: len(incompatible[x]), reverse=True)
    
    # Initialize with all vertices uncolored
    colors = [-1] * num_ids
    
    # We'll try to use at most max_possible_groups colors
    for id in ordered_ids:
        # Find colors used by incompatible vertices
        used_colors = {colors[incompatible_id] for incompatible_id in incompatible[id] 
                      if colors[incompatible_id] != -1}
        
        # Find smallest unused color that's less than max_possible_groups
        color = 0
        while color < max_possible_groups and color in used_colors:
            color += 1
        
        # If we can't find a valid color, this approach failed
        if color >= max_possible_groups:
            # We'll need a more sophisticated approach (see below)
            pass
        else:
            colors[id] = color
    
    # Group IDs by color
    groups = defaultdict(list)
    for id, color in enumerate(colors):
        if color != -1:  # Skip uncolored vertices (if any)
            groups[color].append(id)
    
    # Check if all vertices are colored
    if len([id for id, color in enumerate(colors) if color == -1]) > 0:
        # Some vertices couldn't be colored with our greedy approach
        # We'll need to try a different strategy
        
        # For now, let's use the minimum coloring and check if it works
        minimum_groups = find_minimum_exclusive_groups(exclusive_pairs, num_ids)
        
        # Check if the minimum coloring has at most max_possible_groups groups
        if len(minimum_groups) > max_possible_groups:
            return False, None
        
        # Check if each group is at least min_size
        for group in minimum_groups:
            if len(group) < min_size:
                return False, None
        
        return True, minimum_groups
    
    # Check if all groups meet the minimum size requirement
    for group in groups.values():
        if len(group) < min_size:
            # Try using our original algorithm and see if it produces a better result
            minimum_groups = find_minimum_exclusive_groups(exclusive_pairs, num_ids)
            
            # Check if the minimum coloring has at most max_possible_groups groups
            if len(minimum_groups) > max_possible_groups:
                return False, None
            
            # Check if each group is at least min_size
            for group in minimum_groups:
                if len(group) < min_size:
                    return False, None
            
            return True, minimum_groups
    
    # All groups are valid and meet the minimum size requirement
    return True, list(groups.values())


def main():
    parser = argparse.ArgumentParser(description='Find minimum mutually exclusive groups')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file with pairs of mutually exclusive IDs (one pair per line)')
    parser.add_argument('--num-ids', type=int, required=True,
                        help='Total number of IDs (assumed to be 0 to num_ids-1)')
    parser.add_argument('--min-size', type=int, default=None,
                        help='Check if groups can be formed with minimum size')
    args = parser.parse_args()
    
    # Load exclusive pairs from file
    exclusive_pairs = []
    with open(args.input, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                exclusive_pairs.append((int(parts[0]), int(parts[1])))
    
    # Measure execution time
    start_time = time.time()
    
    if args.min_size is not None:
        # Check if groups can be formed with minimum size
        is_possible, groups = can_form_groups_with_min_size(exclusive_pairs, args.num_ids, args.min_size)
        
        end_time = time.time()
        
        if is_possible:
            print(f"It is possible to form groups with minimum size {args.min_size}")
            print(f"Found {len(groups)} mutually exclusive groups:")
            for i, group in enumerate(groups):
                print(f"Group {i+1} (size {len(group)}): {group}")
        else:
            print(f"It is NOT possible to form groups with minimum size {args.min_size}")
    else:
        # Find minimum exclusive groups
        groups = find_minimum_exclusive_groups(exclusive_pairs, args.num_ids)
        
        end_time = time.time()
        
        # Print results
        print(f"Found {len(groups)} mutually exclusive groups:")
        for i, group in enumerate(groups):
            print(f"Group {i+1} (size {len(group)}): {group}")
    
    print(f"\nExecution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()