#!/usr/bin/env python3
"""
Simple test for subchain generation logic
"""

def extract_subchains_with_config(chain: str, window_size: int, shift: int):
    """
    Extract subchains using sliding window approach
    
    Args:
        chain: Input chain (e.g., "cat1>cat2>cat3>cat4")
        window_size: Size of the sliding window
        shift: Step size for sliding window
        
    Returns:
        List of subchains
    """
    # Split chain into categories
    categories = chain.split('>')
    categories = [cat.strip() for cat in categories if cat.strip()]
    
    if len(categories) < window_size:
        # If chain is shorter than window, return the whole chain
        return [chain] if categories else []
    
    subchains = []
    
    # Sliding window extraction
    for i in range(0, len(categories) - window_size + 1, shift):
        window_categories = categories[i:i + window_size]
        subchain = '>'.join(window_categories)
        subchains.append(subchain)
        
    return subchains

def test_subchain_logic():
    """Test the core subchain generation logic"""
    
    # Test cases
    test_chains = [
        "Technology>Software>Web Development>Frontend",
        "Finance>Investment>Private Equity",
        "Healthcare>Pharmaceuticals>Drug Discovery>Research"
    ]
    
    # Test configurations
    configs = [
        {"window_size": 1, "shift": 1, "name": "Individual categories"},
        {"window_size": 3, "shift": 2, "name": "3-word subchains, step 2"},
        {"window_size": 2, "shift": 1, "name": "2-word subchains, step 1"}
    ]
    
    print("Testing Subchain Generation Logic")
    print("=" * 50)
    
    for chain in test_chains:
        print(f"\nOriginal chain: {chain}")
        categories = chain.split('>')
        print(f"Categories: {categories} (length: {len(categories)})")
        
        for config in configs:
            window_size = config["window_size"]
            shift = config["shift"]
            name = config["name"]
            
            subchains = extract_subchains_with_config(chain, window_size, shift)
            print(f"\n  {name} (window={window_size}, shift={shift}):")
            for i, subchain in enumerate(subchains):
                print(f"    {i+1}. {subchain}")
        
        print("-" * 40)
    
    # Test deduplication across all chains
    print("\nDeduplication Test")
    print("=" * 30)
    
    all_subchains = []
    for chain in test_chains:
        for config in configs:
            subchains = extract_subchains_with_config(chain, config["window_size"], config["shift"])
            all_subchains.extend(subchains)
    
    print(f"Total subchains before deduplication: {len(all_subchains)}")
    unique_subchains = list(set(all_subchains))
    print(f"Unique subchains after deduplication: {len(unique_subchains)}")
    
    print("\nUnique subchains:")
    for i, subchain in enumerate(sorted(unique_subchains)):
        print(f"  {i+1}. {subchain}")
    
    return unique_subchains

if __name__ == "__main__":
    test_subchain_logic() 