from typing import Dict, List, Tuple
import logging
import json
import os
from pathlib import Path
from knowledge_data_curation.src.models.llm_client import OpenAIClient
from knowledge_data_curation.src.prompts.chain_cleaning import CHAIN_CLEANING_PROMPT, CHAIN_CLEANING_SYSTEM_PROMPT
from knowledge_data_curation.src.utils.parallel import batch_process, ordered_batch_results

logger = logging.getLogger(__name__)

class ChainCleaner:
    def __init__(self, config: Dict):
        """
        Initialize the chain cleaner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm_client = OpenAIClient(
            model=config["openai"]["model_chain_cleaner"],
            max_threads=config["openai"]["parallel_threads"],
            rate_limit_delay=1.0 / config["openai"]["rate_limit"]
        )
        
        # Setup reusability options
        self.enable_reuse = config.get("reusability", {}).get("enable_reuse", False)
        self.reuse_cleaned_chains = config.get("reusability", {}).get("reuse_cleaned_chains", False)
        self.output_dir = Path(config["paths"]["output_dir"])
        self.cleaned_chains_path = self.output_dir / "cleaned_chains" / "cleaned_chains.json"
        
        logger.info(f"Initialized ChainCleaner with model {config['openai']['model_chain_cleaner']}")
        logger.info(f"Reusability settings - Enable: {self.enable_reuse}, Reuse cleaned chains: {self.reuse_cleaned_chains}")
        
    def clean_chain(self, chain: str) -> str:
        """
        Clean a single industry chain by removing unnecessary elements
        
        Args:
            chain: Original industry chain
            
        Returns:
            Cleaned industry chain
        """
        logger.debug(f"Cleaning chain: {chain}")
        
        try:
            response = self.llm_client.complete(
                prompt=CHAIN_CLEANING_PROMPT.format(chain=chain),
                system_prompt=CHAIN_CLEANING_SYSTEM_PROMPT,
                temperature=0.1
            )
            
            # Process response to ensure it's in the correct format
            cleaned_chain = response.strip()
            logger.debug(f"Original: {chain} -> Cleaned: {cleaned_chain}")
            return cleaned_chain
        except Exception as e:
            logger.error(f"Error cleaning chain: {e}")
            raise
        
    def process_chains(self, chains: List[str]) -> Dict[str, str]:
        """
        Process multiple chains in parallel batches
        
        Args:
            chains: List of original industry chains
            
        Returns:
            Dictionary mapping original chains to cleaned chains
        """
        logger.info(f"Processing {len(chains)} chains")
        
        # Check if we can reuse existing cleaned chains
        if self.enable_reuse and self.reuse_cleaned_chains and self.cleaned_chains_path.exists():
            logger.info(f"Found existing cleaned chains at {self.cleaned_chains_path}")
            try:
                with open(self.cleaned_chains_path, 'r') as f:
                    existing_cleaned_chains = json.load(f)
                    
                # Check if we have all the chains we need
                missing_chains = [chain for chain in chains if chain not in existing_cleaned_chains]
                
                if not missing_chains:
                    logger.info("All chains found in existing cleaned chains file. Reusing them.")
                    return existing_cleaned_chains
                else:
                    logger.info(f"Found {len(missing_chains)} chains not in existing file. Will process only these.")
                    chains = missing_chains
            except Exception as e:
                logger.warning(f"Error loading existing cleaned chains: {e}. Will process all chains.")
        
        # Remove duplicates to avoid unnecessary processing
        unique_chains = list(set(chains))
        logger.info(f"Processing {len(unique_chains)} unique chains")
        
        try:
            # Process chains in parallel
            batch_results = batch_process(
                items=unique_chains,
                process_fn=self.clean_chain,
                batch_size=self.config["openai"]["batch_size"],
                max_workers=self.config["openai"]["parallel_threads"]
            )
            
            # Create mapping for all original chains
            results = {}
            for i, chain in enumerate(unique_chains):
                cleaned_chain = batch_results.get(i)
                if cleaned_chain:
                    results[chain] = cleaned_chain
                    
            # Ensure all original chains are in the results
            for chain in chains:
                if chain not in results:
                    # Use original chain as fallback
                    logger.warning(f"No cleaned chain found for: {chain}, using original")
                    results[chain] = chain
            
            # If reusability is enabled, merge with existing results
            if self.enable_reuse and self.reuse_cleaned_chains and self.cleaned_chains_path.exists():
                try:
                    with open(self.cleaned_chains_path, 'r') as f:
                        existing_cleaned_chains = json.load(f)
                    results.update(existing_cleaned_chains)
                    logger.info(f"Merged {len(existing_cleaned_chains)} existing cleaned chains with new results")
                except Exception as e:
                    logger.warning(f"Error merging with existing cleaned chains: {e}")
            
            # Save results if reusability is enabled
            if self.enable_reuse and self.reuse_cleaned_chains:
                self.cleaned_chains_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cleaned_chains_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved cleaned chains to {self.cleaned_chains_path}")
                    
            logger.info(f"Successfully processed {len(results)} chains")
            return results
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise 