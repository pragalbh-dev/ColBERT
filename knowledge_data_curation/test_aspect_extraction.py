#!/usr/bin/env python3
"""
Test script for Industry Aspect Extraction
Demonstrates the functionality with sample industry chains
"""

import sys
from pathlib import Path
import json

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processors.industry_aspect_extractor import IndustryAspectExtractor

def create_test_config():
    """Create a minimal test configuration"""
    return {
        "aspect_extraction": {
            "parallel_threads": 2,
            "batch_size": 5
        },
        "openai": {
            "model_aspect_extractor": "gpt-4",
            "rate_limit": 10
        },
        "paths": {
            "output_dir": "test_output"
        }
    }

def main():
    """Test the aspect extraction functionality"""
    print("🔧 Testing Industry Aspect Extraction")
    print("=" * 50)
    
    # Sample industry chains
    sample_chains = {
        "original_tech_chain": "Technology>Software>Web Development>Frontend",
        "original_finance_chain": "Finance>Banking>Digital Banking>Mobile Payments",
        "original_health_chain": "Healthcare>Medical Devices>Diagnostics>AI Diagnostics"
    }
    
    # Sample subchains
    sample_subchains = [
        "Technology>Software",
        "Web Development>Frontend", 
        "Finance>Banking",
        "Digital Banking",
        "Healthcare>Medical Devices",
        "AI Diagnostics"
    ]
    
    # Create test configuration
    config = create_test_config()
    
    # Initialize extractor
    print("Initializing aspect extractor...")
    extractor = IndustryAspectExtractor(config)
    
    # Test single chain extraction
    print("\n🧪 Testing single chain extraction:")
    test_chain = "Technology>Software>Web Development>Frontend"
    print(f"Input: {test_chain}")
    
    try:
        aspects = extractor.extract_aspects_from_chain(test_chain, is_subchain=False)
        if aspects:
            print("✅ Aspects extracted:")
            print(json.dumps(aspects, indent=2))
        else:
            print("❌ Failed to extract aspects")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test subchain extraction
    print("\n🧪 Testing subchain extraction:")
    test_subchain = "Web Development>Frontend"
    print(f"Input: {test_subchain}")
    
    try:
        aspects = extractor.extract_aspects_from_chain(test_subchain, is_subchain=True)
        if aspects:
            print("✅ Subchain aspects extracted:")
            print(json.dumps(aspects, indent=2))
        else:
            print("❌ Failed to extract subchain aspects")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test batch extraction
    print("\n🧪 Testing batch extraction:")
    print(f"Processing {len(list(sample_chains.values()))} chains...")
    
    try:
        batch_results = extractor.extract_aspects_batch(list(sample_chains.values()), is_subchain=False)
        print(f"✅ Batch processing completed: {len(batch_results)} results")
        
        for chain, aspects in batch_results.items():
            print(f"\nChain: {chain}")
            print(f"Industry: {aspects.get('industry', [])}")
            print(f"Target Audience: {aspects.get('target_audience', [])}")
            print(f"Technology: {aspects.get('technology_used', [])}")
            print(f"Products: {aspects.get('products_solutions', [])}")
            print(f"Business Model: {aspects.get('business_model', [])}")
            print(f"Revenue Model: {aspects.get('revenue_model', [])}")
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
    
    print("\n=" * 50)
    print("🎯 Test completed!")
    
    # Show expected output format
    print("\n📋 Expected Output Format:")
    expected_format = {
        "industry": ["Technology", "Software"],
        "target_audience": ["Developers", "Businesses"],
        "technology_used": ["Web", "Frontend"],
        "products_solutions": ["Development", "Solutions"],
        "business_model": ["B2B", "SaaS"],
        "revenue_model": ["Subscription", "Licensing"]
    }
    print(json.dumps(expected_format, indent=2))

if __name__ == "__main__":
    main() 