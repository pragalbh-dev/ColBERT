from typing import Dict, List, Any, Optional
import logging
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from src.models.llm_client import OpenAIClient
from src.models.pydantic import IndustryAspects
from src.prompts.industry_aspect_extraction import (
    INDUSTRY_CHAIN_ASPECT_EXTRACTION_SYSTEM_PROMPT,
    INDUSTRY_SUBCHAIN_ASPECT_EXTRACTION_SYSTEM_PROMPT,
    INDUSTRY_CHAIN_USER_PROMPT_TEMPLATE,
    INDUSTRY_SUBCHAIN_USER_PROMPT_TEMPLATE
)
from src.utils.io import save_json

logger = logging.getLogger(__name__)

class IndustryAspectExtractor:
    def __init__(self, config: Dict):
        """
        Initialize the industry aspect extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        aspect_config = config.get("aspect_extraction", {})
        
        # Parallelization settings
        self.parallel_threads = aspect_config.get("parallel_threads", 10)
        self.batch_size = aspect_config.get("batch_size", 50)
        
        # Initialize LLM client
        llm_config = config.get("openai", {})
        self.llm_client = OpenAIClient(
            model=llm_config.get("model_aspect_extractor", "gpt-4"),
            max_threads=self.parallel_threads,
            rate_limit_delay=1.0 / llm_config.get("rate_limit", 100)
        )
        
        # Setup output paths
        self.output_dir = Path(config["paths"]["output_dir"])
        self.aspect_dir = self.output_dir / "aspect_extraction"
        self.aspect_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized IndustryAspectExtractor:")
        logger.info(f"  - Parallel threads: {self.parallel_threads}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Output directory: {self.aspect_dir}")
    
    def extract_aspects_from_chain(self, industry_chain: str, is_subchain: bool = False) -> Optional[Dict[str, List[str]]]:
        """
        Extract structured aspects from a single industry chain
        
        Args:
            industry_chain: The industry chain string (e.g., "Tech>Software>Web>Frontend")
            is_subchain: Whether this is a subchain (affects prompt selection)
            
        Returns:
            Dictionary with extracted aspects or None if extraction failed
        """
        try:
            # Select appropriate prompts
            if is_subchain:
                system_prompt = INDUSTRY_SUBCHAIN_ASPECT_EXTRACTION_SYSTEM_PROMPT
                user_prompt = INDUSTRY_SUBCHAIN_USER_PROMPT_TEMPLATE.format(industry_subchain=industry_chain)
            else:
                system_prompt = INDUSTRY_CHAIN_ASPECT_EXTRACTION_SYSTEM_PROMPT
                user_prompt = INDUSTRY_CHAIN_USER_PROMPT_TEMPLATE.format(industry_chain=industry_chain)
            
            # Make LLM call with structured output
            import openai
            completion = openai.beta.chat.completions.parse(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=IndustryAspects,
                max_tokens=1000,
                timeout=60
            )
            
            parsed_response = completion.choices[0].message.parsed
            if not parsed_response:
                logger.warning(f"No structured response for chain: {industry_chain}")
                return None
            
            # Convert to dictionary
            aspects_dict = {
                "industry": parsed_response.industry,
                "target_audience": parsed_response.target_audience,
                "technology_used": parsed_response.technology_used,
                "products_solutions": parsed_response.products_solutions,
                "business_model": parsed_response.business_model,
                "revenue_model": parsed_response.revenue_model
            }
            
            logger.debug(f"Extracted aspects for: {industry_chain}")
            return aspects_dict
            
        except Exception as e:
            logger.error(f"Error extracting aspects for chain '{industry_chain}': {e}")
            return None
    
    def extract_aspects_batch(self, industry_chains: List[str], is_subchain: bool = False) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract aspects from multiple industry chains in parallel
        
        Args:
            industry_chains: List of industry chain strings
            is_subchain: Whether these are subchains
            
        Returns:
            Dictionary mapping industry chains to their extracted aspects
        """
        chain_type = "subchains" if is_subchain else "chains"
        logger.info(f"Extracting aspects from {len(industry_chains)} {chain_type}")
        
        results = {}
        successful_extractions = 0
        failed_extractions = 0
        
        # Process in parallel batches
        for i in range(0, len(industry_chains), self.batch_size):
            batch_chains = industry_chains[i:i + self.batch_size]
            batch_number = (i // self.batch_size) + 1
            total_batches = math.ceil(len(industry_chains) / self.batch_size)
            
            logger.info(f"Processing batch {batch_number}/{total_batches} with {len(batch_chains)} {chain_type}")
            
            # Process current batch in parallel
            with ThreadPoolExecutor(max_workers=self.parallel_threads) as executor:
                futures = {
                    executor.submit(
                        self.extract_aspects_from_chain,
                        chain,
                        is_subchain
                    ): chain
                    for chain in batch_chains
                }
                
                for future in as_completed(futures):
                    chain = futures[future]
                    try:
                        aspects = future.result()
                        if aspects:
                            results[chain] = aspects
                            successful_extractions += 1
                        else:
                            failed_extractions += 1
                            logger.warning(f"Failed to extract aspects for: {chain}")
                    except Exception as e:
                        failed_extractions += 1
                        logger.error(f"Error processing chain {chain}: {e}")
        
        logger.info(f"Aspect extraction completed for {chain_type}:")
        logger.info(f"  - Successful extractions: {successful_extractions}")
        logger.info(f"  - Failed extractions: {failed_extractions}")
        logger.info(f"  - Success rate: {successful_extractions / len(industry_chains) * 100:.1f}%")
        
        return results
    
    def process_cleaned_chains(self, cleaned_chains: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Process all cleaned chains to extract aspects
        
        Args:
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            
        Returns:
            Dictionary mapping cleaned chains to their extracted aspects
        """
        logger.info(f"Processing aspects for {len(cleaned_chains)} cleaned chains")
        
        # Extract unique cleaned chains
        unique_cleaned_chains = list(set(cleaned_chains.values()))
        logger.info(f"Processing {len(unique_cleaned_chains)} unique cleaned chains")
        
        # Extract aspects
        chain_aspects = self.extract_aspects_batch(unique_cleaned_chains, is_subchain=False)
        
        # Save results
        output_path = self.aspect_dir / "cleaned_chains_aspects.json"
        save_json(chain_aspects, str(output_path))
        logger.info(f"Saved cleaned chain aspects to: {output_path}")
        
        return chain_aspects
    
    def process_subchains(self, subchain_data: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
        """
        Process all subchains to extract aspects
        
        Args:
            subchain_data: Subchain data from SubchainGenerator
            
        Returns:
            Dictionary mapping subchains to their extracted aspects
        """
        if not subchain_data.get("enabled", False):
            logger.info("Subchain processing disabled - skipping aspect extraction")
            return {}
        
        # Get deduplicated subchains
        subchains = subchain_data.get("deduplicated_subchains", [])
        logger.info(f"Processing aspects for {len(subchains)} subchains")
        
        if not subchains:
            logger.warning("No subchains found for aspect extraction")
            return {}
        
        # Extract aspects
        subchain_aspects = self.extract_aspects_batch(subchains, is_subchain=True)
        
        # Save results
        output_path = self.aspect_dir / "subchains_aspects.json"
        save_json(subchain_aspects, str(output_path))
        logger.info(f"Saved subchain aspects to: {output_path}")
        
        return subchain_aspects
    
    def process_all_aspects(self, 
                          cleaned_chains: Dict[str, str], 
                          subchain_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process aspects for both cleaned chains and subchains
        
        Args:
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            subchain_data: Optional subchain data from SubchainGenerator
            
        Returns:
            Dictionary containing all aspect extraction results
        """
        logger.info("Starting comprehensive aspect extraction")
        
        results = {
            "cleaned_chains_aspects": {},
            "subchains_aspects": {},
            "stats": {}
        }
        
        # Process cleaned chains
        logger.info("=== Processing Cleaned Chains ===")
        results["cleaned_chains_aspects"] = self.process_cleaned_chains(cleaned_chains)
        
        # Process subchains if available
        if subchain_data:
            logger.info("=== Processing Subchains ===")
            results["subchains_aspects"] = self.process_subchains(subchain_data)
        else:
            logger.info("No subchain data provided - skipping subchain aspect extraction")
        
        # Calculate statistics
        chain_count = len(results["cleaned_chains_aspects"])
        subchain_count = len(results["subchains_aspects"])
        total_count = chain_count + subchain_count
        
        results["stats"] = {
            "cleaned_chains_processed": chain_count,
            "subchains_processed": subchain_count,
            "total_aspects_extracted": total_count,
            "subchain_enabled": subchain_data.get("enabled", False) if subchain_data else False
        }
        
        # Save comprehensive results
        comprehensive_output_path = self.aspect_dir / "comprehensive_aspects.json"
        save_json(results, str(comprehensive_output_path))
        
        logger.info("Comprehensive aspect extraction completed:")
        logger.info(f"  - Cleaned chains: {chain_count}")
        logger.info(f"  - Subchains: {subchain_count}")
        logger.info(f"  - Total extractions: {total_count}")
        logger.info(f"  - Comprehensive results saved to: {comprehensive_output_path}")
        
        return results
    
    def get_token_usage_stats(self) -> Dict[str, Any]:
        """
        Get token usage statistics from the LLM client
        
        Returns:
            Dictionary with token usage statistics
        """
        return self.llm_client.get_token_usage() 