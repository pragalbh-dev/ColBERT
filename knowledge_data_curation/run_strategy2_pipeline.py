#!/usr/bin/env python3
"""
Execution script for Strategy 2: Enhanced Pipeline with Synthetic Factsheet Generation
"""

import argparse
import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipelines.main_pipeline import ColBERTTrainingPipeline

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/strategy2_execution.log')
        ]
    )

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run ColBERT training data curation pipeline with Strategy 2 synthetic factsheet generation"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/strategy2_config.yaml",
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
        "--disable-synthetic",
        action="store_true",
        help="Disable synthetic factsheet generation for this run"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true", 
        help="Force regeneration of synthetic data (ignore cache)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis only, don't generate synthetic data"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Starting Strategy 2: Enhanced ColBERT Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Synthetic generation: {'Disabled' if args.disable_synthetic else 'Enabled'}")
    logger.info(f"Force regenerate: {args.force_regenerate}")
    logger.info(f"Dry run: {args.dry_run}")
    
    try:
        # Initialize the pipeline
        logger.info("Initializing pipeline...")
        pipeline = ColBERTTrainingPipeline(args.config)
        
        # Modify configuration based on command line arguments
        if args.disable_synthetic:
            pipeline.config["synthetic_generation"]["enabled"] = False
            logger.info("Synthetic generation disabled via command line")
        
        if args.force_regenerate:
            pipeline.config["synthetic_generation"]["reuse_existing"] = False
            logger.info("Forced regeneration enabled - will ignore cache")
        
        if args.dry_run:
            logger.info("DRY RUN MODE: Analysis only, no synthetic generation")
            
            # Run analysis only
            chains = pipeline.data_loader.get_unique_chains()
            company_chains = pipeline.data_loader.get_company_chains()
            company_factsheets = pipeline.data_loader.get_company_factsheets()
            
            logger.info(f"Data Overview:")
            logger.info(f"  - Total unique chains: {len(chains)}")
            logger.info(f"  - Total companies: {len(company_chains)}")
            logger.info(f"  - Companies with factsheets: {len(company_factsheets)}")
            
            # Analyze data imbalance
            analysis_stats = pipeline.imbalance_analyzer.analyze_chain_distribution(company_chains)
            
            logger.info(f"Data Balance Analysis:")
            logger.info(f"  - Underrepresented chains: {analysis_stats['underrepresented_count']}")
            logger.info(f"  - Well represented chains: {analysis_stats['well_represented_count']}")
            logger.info(f"  - Threshold used: {analysis_stats['threshold_used']}")
            
            # Generate balance report
            target_count = pipeline.config.get("synthetic_generation", {}).get("target_factsheets_per_chain", 10)
            balance_report = pipeline.imbalance_analyzer.generate_balance_report(analysis_stats, target_count)
            
            logger.info(f"Projected Impact:")
            logger.info(f"  - Current total examples: {balance_report['current_state']['total_examples']}")
            logger.info(f"  - Synthetic examples to generate: {balance_report['synthetic_generation']['total_synthetic_factsheets']}")
            logger.info(f"  - Projected total examples: {balance_report['projected_state']['total_examples']}")
            logger.info(f"  - Improvement ratio: {balance_report['projected_state']['improvement_ratio']:.2f}x")
            
            logger.info("Dry run completed successfully!")
            return 0
        
        # Run the full pipeline
        logger.info("Starting pipeline execution...")
        pipeline.run()
        
        logger.info("=" * 80)
        logger.info("🎉 Strategy 2 pipeline completed successfully!")
        logger.info("=" * 80)
        
        # Log final statistics
        output_dir = Path(pipeline.config["paths"]["output_dir"])
        
        if pipeline.config.get("synthetic_generation", {}).get("enabled", False):
            synthetic_dir = output_dir / "synthetic_data_cache"
            
            if synthetic_dir.exists():
                logger.info("Synthetic Data Generation Results:")
                
                # Check for augmented datasets
                industry_file = synthetic_dir / "augmented_industry_data.csv"
                factsheet_file = synthetic_dir / "augmented_factsheet_data.csv"
                
                if industry_file.exists() and factsheet_file.exists():
                    import pandas as pd
                    aug_industry = pd.read_csv(industry_file)
                    aug_factsheet = pd.read_csv(factsheet_file)
                    
                    logger.info(f"  - Augmented industry entries: {len(aug_industry)}")
                    logger.info(f"  - Augmented factsheet entries: {len(aug_factsheet)}")
                    
                    # Count synthetic entries
                    synthetic_companies = aug_industry[aug_industry['company_id'].str.startswith('synthetic_')]
                    logger.info(f"  - Synthetic companies added: {len(synthetic_companies)}")
                else:
                    logger.warning("Augmented dataset files not found")
        
        # Log training data results
        training_dir = output_dir / "colbert_training"
        if training_dir.exists():
            logger.info("Training Data Generation Results:")
            
            combined_file = training_dir / "combined_training.csv"
            if combined_file.exists():
                import pandas as pd
                combined_data = pd.read_csv(combined_file)
                
                logger.info(f"  - Total training examples: {len(combined_data)}")
                logger.info(f"  - Positive examples: {len(combined_data[combined_data['is_positive'] == True])}")
                logger.info(f"  - Negative examples: {len(combined_data[combined_data['is_positive'] == False])}")
                
                if 'query_type' in combined_data.columns:
                    full_chain_count = len(combined_data[combined_data['query_type'] == 'full_chain'])
                    subchain_count = len(combined_data[combined_data['query_type'] == 'subchain'])
                    logger.info(f"  - Full chain examples: {full_chain_count}")
                    logger.info(f"  - Subchain examples: {subchain_count}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 