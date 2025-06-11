from typing import Dict, List, Any, Tuple
import logging
import hashlib
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from src.models.llm_client import OpenAIClient
from src.utils.parallel import batch_process
from src.models.pydantic import SyntheticFactsheetBatch
logger = logging.getLogger(__name__)

# Enhanced prompt template for batch generation
BATCH_SYNTHETIC_FACTSHEET_PROMPT = """You are tasked with creating {batch_size} realistic synthetic company factsheets for companies operating in the industry chain: "{industry_chain}".

REQUIREMENTS:
1. Generate exactly {batch_size} distinct, realistic company factsheets
2. Each factsheet should represent a different company in the "{industry_chain}" industry
3. Ensure diversity in company size, focus areas, and business models
4. Make each factsheet unique and realistic
5. Each factsheet should be 150-300 words

FORMAT: Return exactly {batch_size} factsheets separated by "---FACTSHEET_SEPARATOR---"

SAMPLE FACTSHEETS FOR REFERENCE:
{sample_factsheets}

Generate {batch_size} synthetic factsheets now:"""

class SyntheticFactsheetGenerator:
    def __init__(self, config: Dict):
        """
        Initialize the synthetic factsheet generator with optimized parallelization
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        synthetic_config = config.get("synthetic_generation", {})
        
        # Target generation settings
        self.target_count = synthetic_config.get("target_factsheets_per_chain", 10)
        
        # Optimized parallelization settings
        self.factsheets_per_call = synthetic_config.get("factsheets_per_call", 3)
        self.parallel_chains = synthetic_config.get("parallel_chains", 4)
        self.max_parallel_calls = synthetic_config.get("max_parallel_calls", 20)
        
        # Calculate optimal batch configuration
        self._optimize_batch_configuration()
        
        # Initialize LLM client
        llm_config = config.get("openai", {})
        self.llm_client = OpenAIClient(
            model=llm_config.get("model_factsheet_generator", "gpt-4"),
            max_threads=self.max_parallel_calls,  # Use optimized parallel capacity
            rate_limit_delay=1.0 / llm_config.get("rate_limit", 100)
        )
        
        # Setup output paths
        self.output_dir = Path(config["paths"]["output_dir"])
        self.synthetic_dir = self.output_dir / "synthetic_data_cache"
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Optimized SyntheticFactsheetGenerator:")
        logger.info(f"  - Target factsheets per chain: {self.target_count}")
        logger.info(f"  - Factsheets per LLM call: {self.factsheets_per_call}")
        logger.info(f"  - Parallel chains: {self.parallel_chains}")
        logger.info(f"  - Max parallel calls: {self.max_parallel_calls}")
        logger.info(f"  - Calls per chain: {self.calls_per_chain}")
        
    def _optimize_batch_configuration(self):
        """
        Optimize batch configuration based on parallel capacity and target counts
        """
        # Calculate calls needed per chain
        self.calls_per_chain = math.ceil(self.target_count / self.factsheets_per_call)
        
        # Calculate optimal parallel chains based on available capacity
        total_calls_needed = self.calls_per_chain * self.parallel_chains
        
        if total_calls_needed > self.max_parallel_calls:
            # Reduce parallel chains to fit within capacity
            optimal_parallel_chains = self.max_parallel_calls // self.calls_per_chain
            if optimal_parallel_chains < 1:
                optimal_parallel_chains = 1
                logger.warning(f"Reducing parallel chains to {optimal_parallel_chains} due to capacity constraints")
            self.parallel_chains = optimal_parallel_chains
        
        logger.info(f"Optimized configuration: {self.calls_per_chain} calls per chain, {self.parallel_chains} parallel chains")
        
    def generate_batch_factsheets_for_chain(self, 
                                           industry_chain: str, 
                                           sample_factsheets: List[str], 
                                           batch_size: int) -> List[str]:
        """
        Generate a batch of factsheets in a single LLM call using OpenAI structured output
        Returns a list of factsheet strings (for pipeline compatibility)
        """
        if not sample_factsheets:
            logger.warning(f"No sample factsheets provided for chain: {industry_chain}")
            return []
            
        logger.debug(f"Generating batch of {batch_size} factsheets for: {industry_chain}")
        
        # Create sample context
        samples_context = "\n---\n".join(sample_factsheets[:2])  # Limit to 2 samples to control prompt size
        
        prompt = f"""Generate {batch_size} unique, realistic company factsheets for companies in this industry chain: "{industry_chain}"

Use these sample factsheets as reference for style and format:
{samples_context}

Create {batch_size} different factsheets with:
- Unique company names and details
- Realistic business descriptions matching the industry chain
- Similar length and structure to the samples
- Vary the company size, focus areas, and specific details"""

        try:
            # ✅ FIX: Use openai module directly, not self.llm_client.client
            import openai
            completion = openai.beta.chat.completions.parse(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": "Generate realistic synthetic company factsheets. Return exactly the requested number of factsheets."},
                    {"role": "user", "content": prompt}
                ],
                response_format=SyntheticFactsheetBatch,
                max_tokens=16000,  # Limit to prevent hitting 32k limit
                timeout=120  # 2 minute timeout
            )
            
            parsed_response = completion.choices[0].message.parsed
            if not parsed_response or not parsed_response.factsheets:
                logger.warning(f"No factsheets generated for chain: {industry_chain}")
                return []
            
            # Extract factsheet content from parsed response
            factsheets = [item.factsheet for item in parsed_response.factsheets if item.factsheet and item.factsheet.strip()]
            
            logger.debug(f"Generated {len(factsheets)} factsheets for: {industry_chain}")
            return factsheets
            
        except Exception as e:
            logger.error(f"Error generating batch factsheets for chain {industry_chain}: {e}")
            return []
    
    def generate_factsheets_for_chain_optimized(self, 
                                               industry_chain: str, 
                                               sample_factsheets: List[str], 
                                               target_count: int) -> List[str]:
        """
        Generate factsheets for a chain using optimized batch calls
        
        Args:
            industry_chain: The industry chain to generate factsheets for
            sample_factsheets: List of sample factsheets to use as examples
            target_count: Total number of factsheets to generate
            
        Returns:
            List of generated factsheet strings
        """
        if not sample_factsheets:
            logger.warning(f"No sample factsheets provided for chain: {industry_chain}")
            return []
            
        logger.info(f"Generating {target_count} factsheets for chain: {industry_chain} using {self.calls_per_chain} batch calls")
        
        # Calculate batch sizes for each call
        batch_calls = []
        remaining = target_count
        
        for i in range(self.calls_per_chain):
            if remaining <= 0:
                break
            batch_size = min(self.factsheets_per_call, remaining)
            batch_calls.append(batch_size)
            remaining -= batch_size
        
        # Generate batches in parallel
        all_factsheets = []
        
        with ThreadPoolExecutor(max_workers=len(batch_calls)) as executor:
            futures = [
                executor.submit(
                    self.generate_batch_factsheets_for_chain,
                    industry_chain,
                    sample_factsheets,
                    batch_size
                )
                for batch_size in batch_calls
            ]
            
            for future in as_completed(futures):
                try:
                    batch_factsheets = future.result()
                    all_factsheets.extend(batch_factsheets)
                except Exception as e:
                    logger.error(f"Error in batch generation: {e}")
        
        logger.info(f"Generated {len(all_factsheets)} factsheets for chain: {industry_chain}")
        return all_factsheets
    
    def batch_generate_factsheets_optimized(self, 
                                           chains_needing_data: List[str], 
                                           chain_samples: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Generate factsheets for multiple chains with optimized cross-chain parallelization
        
        Args:
            chains_needing_data: List of chains that need synthetic data
            chain_samples: Dictionary mapping chains to their sample factsheets
            
        Returns:
            Dictionary mapping chains to lists of (company_id, factsheet) tuples
        """
        logger.info(f"Optimized batch generation for {len(chains_needing_data)} chains")
        logger.info(f"Processing {self.parallel_chains} chains in parallel")
        
        results = {}
        total_factsheets_generated = 0
        
        # Process chains in parallel batches
        for i in range(0, len(chains_needing_data), self.parallel_chains):
            batch_chains = chains_needing_data[i:i + self.parallel_chains]
            batch_number = (i // self.parallel_chains) + 1
            total_batches = math.ceil(len(chains_needing_data) / self.parallel_chains)
            
            logger.info(f"Processing batch {batch_number}/{total_batches} with {len(batch_chains)} chains")
            
            # Process current batch in parallel
            with ThreadPoolExecutor(max_workers=len(batch_chains)) as executor:
                futures = {
                    executor.submit(
                        self._process_single_chain,
                        chain,
                        chain_samples.get(chain, [])
                    ): chain
                    for chain in batch_chains
                }
                
                for future in as_completed(futures):
                    chain = futures[future]
                    try:
                        chain_results = future.result()
                        results[chain] = chain_results
                        total_factsheets_generated += len(chain_results)
                        logger.info(f"Completed chain: {chain} ({len(chain_results)} factsheets)")
                    except Exception as e:
                        logger.error(f"Error processing chain {chain}: {e}")
                        results[chain] = []
        
        logger.info(f"Optimized batch generation completed:")
        logger.info(f"  - Processed {len(chains_needing_data)} chains")
        logger.info(f"  - Generated {total_factsheets_generated} total factsheets")
        logger.info(f"  - Average {total_factsheets_generated / len(chains_needing_data):.1f} factsheets per chain")
        
        return results
    
    def _process_single_chain(self, chain: str, samples: List[str]) -> List[Tuple[str, str]]:
        """
        Process a single chain to generate factsheets and company IDs
        
        Args:
            chain: Industry chain to process
            samples: Sample factsheets for this chain
            
        Returns:
            List of (company_id, factsheet) tuples
        """
        if not samples:
            logger.warning(f"No samples available for chain: {chain}")
            return []
        
        # Generate factsheets using optimized method
        generated_factsheets = self.generate_factsheets_for_chain_optimized(
            chain, samples, self.target_count
        )
        
        # Create (company_id, factsheet) tuples
        chain_results = []
        for factsheet in generated_factsheets:
            company_id = self.generate_company_id(factsheet)
            chain_results.append((company_id, factsheet))
        
        return chain_results

    def generate_factsheets_for_chain_simple(self, 
                                            industry_chain: str, 
                                            sample_factsheets: List[str], 
                                            target_count: int) -> List[str]:
        """
        Generate factsheets for ONE chain with exactly ONE API call
        Returns crisp, short factsheets (< 500 tokens each)
        """
        if not sample_factsheets:
            logger.warning(f"No sample factsheets provided for chain: {industry_chain}")
            return []
            
        logger.info(f"Generating {target_count} factsheets for chain: {industry_chain} (1 API call)")
        
        # Create concise sample context (only first sample to keep prompt small)
        sample_context = sample_factsheets[0][:800] if sample_factsheets else ""  # Limit sample size
        
        # Crisp, concise prompt focusing on SHORT factsheets
        prompt = f"""Generate exactly {target_count} CRISP, CONCISE company factsheets for industry: "{industry_chain}"

SAMPLE FORMAT (use as reference):
{sample_context}

REQUIREMENTS:
- Generate exactly {target_count} factsheets
- Keep each factsheet under 300 words (crisp and concise)
- Include: Company name, brief description, key products/services, target market
- Make each company unique but realistic for this industry
- Use professional, factual tone
- NO lengthy descriptions or excessive detail"""

        try:
            # Single API call per chain
            import openai
            completion = openai.beta.chat.completions.parse(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": "Generate crisp, concise company factsheets under 300 words each. Be factual and professional."},
                    {"role": "user", "content": prompt}
                ],
                response_format=SyntheticFactsheetBatch,
                max_tokens=4000,  # Reduced limit for crisp factsheets
                timeout=60  # Reduced timeout
            )
            
            parsed_response = completion.choices[0].message.parsed
            if not parsed_response or not parsed_response.factsheets:
                logger.warning(f"No factsheets generated for chain: {industry_chain}")
                return []
            
            # Extract crisp factsheets
            factsheets = [
                item.factsheet.strip() 
                for item in parsed_response.factsheets 
                if item.factsheet and item.factsheet.strip()
            ]
            
            logger.info(f"Generated {len(factsheets)} crisp factsheets for: {industry_chain}")
            return factsheets
            
        except Exception as e:
            logger.error(f"Error generating factsheets for chain {industry_chain}: {e}")
            return []

    def batch_generate_factsheets_simple(self, 
                                        chains_needing_data: List[str], 
                                        chain_samples: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Simple parallel generation: exactly 1 API call per chain
        Process multiple chains in parallel batches
        """
        logger.info(f"Simple batch generation for {len(chains_needing_data)} chains")
        logger.info(f"Processing {self.parallel_chains} chains in parallel (1 call per chain)")
        
        results = {}
        total_factsheets_generated = 0
        
        # Process chains in parallel batches
        for i in range(0, len(chains_needing_data), self.parallel_chains):
            batch_chains = chains_needing_data[i:i + self.parallel_chains]
            batch_number = (i // self.parallel_chains) + 1
            total_batches = math.ceil(len(chains_needing_data) / self.parallel_chains)
            
            logger.info(f"Processing batch {batch_number}/{total_batches} with {len(batch_chains)} chains")
            
            # Process current batch in parallel (1 call per chain)
            with ThreadPoolExecutor(max_workers=len(batch_chains)) as executor:
                futures = {
                    executor.submit(
                        self._process_single_chain_simple,
                        chain,
                        chain_samples.get(chain, [])
                    ): chain
                    for chain in batch_chains
                }
                
                for future in as_completed(futures):
                    chain = futures[future]
                    try:
                        chain_results = future.result()
                        results[chain] = chain_results
                        total_factsheets_generated += len(chain_results)
                        logger.info(f"Completed chain: {chain} ({len(chain_results)} factsheets)")
                    except Exception as e:
                        logger.error(f"Error processing chain {chain}: {e}")
                        results[chain] = []
        
        logger.info(f"Simple batch generation completed:")
        logger.info(f"  - Processed {len(chains_needing_data)} chains")
        logger.info(f"  - Generated {total_factsheets_generated} total factsheets")
        logger.info(f"  - API calls made: {len(chains_needing_data)} (1 per chain)")
        
        return results
    
    def _process_single_chain_simple(self, chain: str, samples: List[str]) -> List[Tuple[str, str]]:
        """
        Process a single chain with exactly 1 API call
        """
        if not samples:
            logger.warning(f"No samples available for chain: {chain}")
            return []
        
        # Generate factsheets with single API call
        generated_factsheets = self.generate_factsheets_for_chain_simple(
            chain, samples, self.target_count
        )
        
        # Create (company_id, factsheet) tuples
        chain_results = []
        for factsheet in generated_factsheets:
            company_id = self.generate_company_id(factsheet)
            chain_results.append((company_id, factsheet))
        
        return chain_results

    # Update main methods to use simplified approach
    def batch_generate_factsheets(self, 
                                chains_needing_data: List[str], 
                                chain_samples: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Main method - now uses simplified 1-call-per-chain approach
        """
        return self.batch_generate_factsheets_simple(chains_needing_data, chain_samples)

    def generate_factsheets_for_chain(self, 
                                    industry_chain: str, 
                                    sample_factsheets: List[str], 
                                    target_count: int) -> List[str]:
        """
        Main method - now uses simplified 1-call-per-chain approach
        """
        return self.generate_factsheets_for_chain_simple(industry_chain, sample_factsheets, target_count)

    def generate_company_id(self, factsheet: str) -> str:
        """
        Generate deterministic company ID as hash of factsheet content
        
        Args:
            factsheet: The factsheet content
            
        Returns:
            Hexadecimal hash string to use as company ID
        """
        # Create a hash of the factsheet content
        hash_object = hashlib.sha256(factsheet.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        
        # Take first 12 characters for reasonable length ID
        company_id = f"synthetic_{hash_hex[:12]}"
        
        return company_id

    def validate_generated_factsheets(self, 
                                    generated_data: Dict[str, List[Tuple[str, str]]],
                                    min_length: int = 100) -> Dict[str, Any]:
        """
        Validate the quality and consistency of generated factsheets
        
        Args:
            generated_data: Dictionary mapping chains to (company_id, factsheet) tuples
            min_length: Minimum expected length for a valid factsheet
            
        Returns:
            Dictionary with validation statistics and results
        """
        logger.info("Validating generated factsheets")
        
        total_factsheets = 0
        valid_factsheets = 0
        invalid_factsheets = []
        company_ids = set()
        duplicate_ids = []
        
        for chain, factsheet_tuples in generated_data.items():
            for company_id, factsheet in factsheet_tuples:
                total_factsheets += 1
                
                # Check for duplicate company IDs
                if company_id in company_ids:
                    duplicate_ids.append(company_id)
                else:
                    company_ids.add(company_id)
                
                # Validate factsheet content
                if len(factsheet.strip()) >= min_length:
                    valid_factsheets += 1
                else:
                    invalid_factsheets.append({
                        "chain": chain,
                        "company_id": company_id,
                        "length": len(factsheet.strip()),
                        "reason": "Too short"
                    })
        
        validation_results = {
            "total_factsheets": total_factsheets,
            "valid_factsheets": valid_factsheets,
            "invalid_factsheets": len(invalid_factsheets),
            "validation_rate": valid_factsheets / total_factsheets if total_factsheets > 0 else 0,
            "unique_company_ids": len(company_ids),
            "duplicate_ids": len(duplicate_ids),
            "invalid_details": invalid_factsheets[:10],  # First 10 invalid entries
            "duplicate_id_details": duplicate_ids[:10]   # First 10 duplicate IDs
        }
        
        logger.info(f"Validation completed:")
        logger.info(f"  - Total factsheets: {validation_results['total_factsheets']}")
        logger.info(f"  - Valid factsheets: {validation_results['valid_factsheets']}")
        logger.info(f"  - Validation rate: {validation_results['validation_rate']:.2%}")
        logger.info(f"  - Unique company IDs: {validation_results['unique_company_ids']}")
        
        if duplicate_ids:
            logger.warning(f"Found {len(duplicate_ids)} duplicate company IDs")
        
        return validation_results
    
    def save_generation_metadata(self, 
                               generated_data: Dict[str, List[Tuple[str, str]]],
                               validation_results: Dict[str, Any],
                               generation_stats: Dict[str, Any]) -> None:
        """
        Save metadata about the generation process
        
        Args:
            generated_data: The generated factsheet data
            validation_results: Results from validation
            generation_stats: Statistics from generation process
        """
        metadata = {
            "generation_config": {
                "target_factsheets_per_chain": self.target_count,
                "factsheets_per_call": self.factsheets_per_call,
                "parallel_chains": self.parallel_chains,
                "max_parallel_calls": self.max_parallel_calls,
                "calls_per_chain": self.calls_per_chain
            },
            "generation_stats": generation_stats,
            "validation_results": validation_results,
            "chains_processed": list(generated_data.keys()),
            "total_synthetic_companies": sum(len(factsheets) for factsheets in generated_data.values())
        }
        
        metadata_path = self.synthetic_dir / "generation_metadata.json"
        
        from src.utils.io import save_json
        save_json(metadata, str(metadata_path))
        
        logger.info(f"Saved generation metadata to {metadata_path}") 