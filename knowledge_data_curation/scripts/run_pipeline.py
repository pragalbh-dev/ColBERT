#!/usr/bin/env python3
"""
Run the ColBERT training data curation pipeline.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knowledge_data_curation.src.pipelines.main_pipeline import ColBERTTrainingPipeline
from knowledge_data_curation.src.utils.logger import Logger

def main():
    """
    Main entry point for the pipeline.
    """
    parser = argparse.ArgumentParser(description="Run ColBERT training data curation pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="knowledge_data_curation/config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to checkpoint directory to resume from"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup basic logging until we load the config
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("pipeline_runner")
    
    # Run the pipeline
    try:
        logger.info(f"Starting pipeline with config: {args.config}")
        
        if args.checkpoint:
            logger.info(f"Resuming from checkpoint: {args.checkpoint}")
            pipeline = ColBERTTrainingPipeline.load_from_checkpoint(args.config, args.checkpoint)
        else:
            pipeline = ColBERTTrainingPipeline(args.config)
            
        pipeline.run()
        
        logger.info("Pipeline completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 