#!/usr/bin/env python3
"""
Standalone script for Strategy 3: Industry Aspect Extraction
This script processes cleaned chains and subchains to extract structured aspects
"""

import argparse
import sys
import logging
from pathlib import Path
import json

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processors.industry_aspect_extractor import IndustryAspectExtractor
from src.utils.io import load_json

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/aspect_extraction.log')
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_cleaned_chains(output_dir: str) -> dict:
    """Load cleaned chains from pipeline output"""
    cleaned_chains_path = Path(output_dir) / "cleaned_chains" / "cleaned_chains.json"
    if not cleaned_chains_path.exists():
        raise FileNotFoundError(f"Cleaned chains file not found: {cleaned_chains_path}")
    
    return load_json(str(cleaned_chains_path))

def load_subchain_data(output_dir: str) -> dict:
    """Load subchain data from pipeline output"""
    subchain_data_path = Path(output_dir) / "subchains" / "subchain_data.json"
    if not subchain_data_path.exists():
        logging.warning(f"Subchain data file not found: {subchain_data_path}")
        return {"enabled": False}
    
    return load_json(str(subchain_data_path))

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run Industry Aspect Extraction (Strategy 3) independently"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/integrated_strategies_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--chains-only",
        action="store_true",
        help="Process only cleaned chains (skip subchains)"
    )
    parser.add_argument(
        "--subchains-only",
        action="store_true",
        help="Process only subchains (skip cleaned chains)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Starting Strategy 3: Industry Aspect Extraction")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Chains only: {args.chains_only}")
    logger.info(f"Subchains only: {args.subchains_only}")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        
        # Override output directory if provided
        if args.output_dir:
            config["paths"]["output_dir"] = args.output_dir
            logger.info(f"Output directory overridden to: {args.output_dir}")
        
        output_dir = config["paths"]["output_dir"]
        
        # Check if aspect extraction is enabled
        if not config.get("aspect_extraction", {}).get("enabled", False):
            logger.error("Aspect extraction is disabled in configuration. Enable it to proceed.")
            sys.exit(1)
        
        # Initialize the aspect extractor
        logger.info("Initializing aspect extractor...")
        extractor = IndustryAspectExtractor(config)
        
        # Load required data
        cleaned_chains = None
        subchain_data = None
        
        if not args.subchains_only:
            logger.info("Loading cleaned chains...")
            cleaned_chains = load_cleaned_chains(output_dir)
            logger.info(f"Loaded {len(cleaned_chains)} cleaned chain mappings")
        
        if not args.chains_only:
            logger.info("Loading subchain data...")
            subchain_data = load_subchain_data(output_dir)
            if subchain_data.get("enabled", False):
                subchain_count = len(subchain_data.get("deduplicated_subchains", []))
                logger.info(f"Loaded {subchain_count} subchains")
            else:
                logger.info("Subchain data not available or disabled")
        
        # Process aspects
        logger.info("Starting aspect extraction...")
        
        if args.chains_only and cleaned_chains:
            # Process only cleaned chains
            logger.info("Processing cleaned chains only...")
            chain_aspects = extractor.process_cleaned_chains(cleaned_chains)
            results = {
                "cleaned_chains_aspects": chain_aspects,
                "subchains_aspects": {},
                "stats": {
                    "cleaned_chains_processed": len(chain_aspects),
                    "subchains_processed": 0,
                    "total_aspects_extracted": len(chain_aspects)
                }
            }
        elif args.subchains_only and subchain_data:
            # Process only subchains
            logger.info("Processing subchains only...")
            subchain_aspects = extractor.process_subchains(subchain_data)
            results = {
                "cleaned_chains_aspects": {},
                "subchains_aspects": subchain_aspects,
                "stats": {
                    "cleaned_chains_processed": 0,
                    "subchains_processed": len(subchain_aspects),
                    "total_aspects_extracted": len(subchain_aspects)
                }
            }
        else:
            # Process both (default)
            logger.info("Processing both cleaned chains and subchains...")
            results = extractor.process_all_aspects(cleaned_chains, subchain_data)
        
        # Get token usage statistics
        token_usage = extractor.get_token_usage_stats()
        logger.info(f"Token usage: {token_usage}")
        
        # Log final results
        logger.info("=" * 80)
        logger.info("Aspect Extraction Results:")
        logger.info(f"  - Cleaned chains processed: {results['stats']['cleaned_chains_processed']}")
        logger.info(f"  - Subchains processed: {results['stats']['subchains_processed']}")
        logger.info(f"  - Total aspects extracted: {results['stats']['total_aspects_extracted']}")
        logger.info("=" * 80)
        
        # Save token usage stats
        token_stats_path = Path(output_dir) / "aspect_extraction" / "token_usage_stats.json"
        with open(token_stats_path, 'w') as f:
            json.dump(token_usage, f, indent=2)
        logger.info(f"Token usage stats saved to: {token_stats_path}")
        
        logger.info("Aspect extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during aspect extraction: {e}")
        raise

if __name__ == "__main__":
    main() 