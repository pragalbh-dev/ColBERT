from typing import Dict, List, Tuple
import logging
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
        logger.info(f"Initialized ChainCleaner with model {config['openai']['model_chain_cleaner']}")
        
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
                    
            logger.info(f"Successfully processed {len(results)} chains")
            return results
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise 