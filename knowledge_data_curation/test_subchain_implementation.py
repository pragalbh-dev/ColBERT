#!/usr/bin/env python3
"""
Test script for subchain implementation
"""
import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processors.subchain_generator import SubchainGenerator

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_subchain_generation():
    """Test the subchain generation functionality"""
    
    # Test configuration
    config = {
        "paths": {
            "output_dir": "/tmp/test_subchain_output"
        },
        "subchain_generation": {
            "enabled": True,
            "configurations": [
                {
                    "window_size": 1,
                    "shift": 1,
                    "enabled": True
                },
                {
                    "window_size": 3,
                    "shift": 2,
                    "enabled": True
                }
            ],
            "deduplicate_subchains": True
        }
    }
    
    # Test data
    test_cleaned_chains = {
        "Technology>Software>Web Development>Frontend": "Technology>Software>Web Development>Frontend",
        "Technology>Software>Web Development>Backend": "Technology>Software>Web Development>Backend", 
        "Finance>Investment>Private Equity": "Finance>Investment>Private Equity",
        "Healthcare>Pharmaceuticals>Drug Discovery": "Healthcare>Pharmaceuticals>Drug Discovery"
    }
    
    logger.info("Testing SubchainGenerator...")
    
    # Initialize generator
    generator = SubchainGenerator(config)
    
    # Test single chain subchain generation
    test_chain = "Technology>Software>Web Development>Frontend"
    subchains = generator.generate_subchains(test_chain)
    
    logger.info(f"Single chain test - Input: {test_chain}")
    for config_id, chain_subchains in subchains.items():
        logger.info(f"  Config {config_id}: {chain_subchains}")
    
    # Test full subchain generation
    subchain_data = generator.generate_all_subchains(test_cleaned_chains)
    
    logger.info("\nFull subchain generation results:")
    logger.info(f"Enabled: {subchain_data['enabled']}")
    logger.info(f"Stats: {subchain_data['stats']}")
    logger.info(f"Total unique subchains: {len(subchain_data['deduplicated_subchains'])}")
    
    # Display some examples
    logger.info("\nExample subchains:")
    for i, subchain in enumerate(subchain_data['deduplicated_subchains'][:10]):
        logger.info(f"  {i+1}. {subchain}")
    
    logger.info("\nSubchain generation test completed successfully!")
    
    return subchain_data

if __name__ == "__main__":
    test_subchain_generation() 