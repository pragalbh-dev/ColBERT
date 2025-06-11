from typing import Dict, List, Tuple, Any, Set
import logging
import random
import re
import json
from src.models.llm_client import OpenAIClient
from src.prompts.negative_sampling import NEGATIVE_SAMPLING_PROMPT, NEGATIVE_SAMPLING_SYSTEM_PROMPT
from src.utils.elasticsearch import ESClient
from src.utils.parallel import batch_process
from src.models.pydantic import NegativeSamplingResponse
from src.utils.chain_overlap import ChainOverlapManager
import time

logger = logging.getLogger(__name__)

class NegativeSampleGenerator:
    def __init__(self, config: Dict):
        """
        Initialize the negative sample generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm_client = OpenAIClient(
            model=config["openai"]["model_negative_generator"],
            max_threads=config["openai"]["parallel_threads"],
            rate_limit_delay=1.0 / config["openai"]["rate_limit"]
        )
        # Initialize ESClient in indexing mode for indexing operations
        self.indexing_es_client = ESClient(config["elasticsearch"], is_indexing=True)
        # Initialize ESClient in non-indexing mode for search operations
        self.search_es_client = ESClient(config["elasticsearch"], is_indexing=False)
        
        self.overlap_manager = ChainOverlapManager()
        
        # Store all documents in memory
        self._all_documents = None
        
        # Setup reusability options
        self.enable_reuse = config.get("reusability", {}).get("enable_reuse", False)
        self.reuse_elasticsearch_index = config.get("reusability", {}).get("reuse_elasticsearch_index", False)
        
        # Sampling configuration
        self.nearest_neighbors = config["sampling"]["nearest_neighbors"]
        self.window_size = config["sampling"]["window_size"]
        self.samples_per_window = config["sampling"]["samples_per_window"]
        self.soft_negative_count = config["sampling"]["soft_negative_count"]
        
        logger.info(f"Initialized NegativeSampleGenerator with model {config['openai']['model_negative_generator']}")
        logger.info(f"Reusability settings - Enable: {self.enable_reuse}, Reuse elasticsearch index: {self.reuse_elasticsearch_index}")
        
    def initialize(self, company_chains: Dict[str, List[str]], min_overlap: int = 1) -> None:
        """
        Initialize the generator with company chain data
        
        Args:
            company_chains: Dictionary mapping company IDs to their chains
            min_overlap: Minimum number of shared companies to consider chains as overlapping
        """
        logger.info("Initializing negative sample generator")
        
        # Build overlap indices
        self.overlap_manager.build_overlap_index(company_chains, min_overlap)
        
        # Index chains in Elasticsearch if needed
        unique_chains = list(set(
            chain for chains in company_chains.values() 
            for chain in chains
        ))
        
        if self.enable_reuse and self.reuse_elasticsearch_index:
            # Check if index exists and has the same number of documents
            if self.indexing_es_client.index_exists():
                current_doc_count = self.indexing_es_client.get_document_count()
                if current_doc_count == len(unique_chains):
                    logger.info(f"Found existing Elasticsearch index with {current_doc_count} documents. Skipping indexing.")
                    # Load all documents into memory
                    self._all_documents = self.search_es_client.get_all_documents()
                    return
                else:
                    logger.info(f"Existing index has {current_doc_count} documents but we need {len(unique_chains)}. Will reindex.")
            else:
                logger.info(f"Index does not exist. Will create and index documents.")
        
        # If we reach here, we need to index the documents
        self.index_chains(unique_chains)
        # Load all documents into memory after indexing
        self._all_documents = self.search_es_client.get_all_documents()
        
    def index_chains(self, chains: List[str]) -> None:
        """
        Index all chains in Elasticsearch
        
        Args:
            chains: List of industry chains to index
        """
        logger.info(f"Indexing {len(chains)} chains in Elasticsearch")
        
        # Delete existing index if it exists
        if self.indexing_es_client.index_exists():
            logger.info(f"Deleting existing index")
            self.indexing_es_client.delete_index()
        
        # Create new index and index documents
        self.indexing_es_client.create_index()
        self.indexing_es_client.index_documents(chains)
        logger.info(f"Successfully indexed {len(chains)} chains in Elasticsearch")
        
    def get_candidate_negatives(self, chain: str) -> Tuple[List[str], List[str]]:
        """
        Get candidate negative chains for a given chain
        
        Args:
            chain: The industry chain to find negatives for
            
        Returns:
            Tuple of (hard negative candidates, soft negative candidates)
        """
        start_time = time.time()
        logger.debug(f"Getting candidate negatives for: {chain}")
        
        # Use cached documents instead of fetching again
        valid_candidates = self.overlap_manager.get_valid_negative_candidates(chain, self._all_documents)
        
        if not valid_candidates:
            logger.warning(f"No valid negative candidates found for chain: {chain}")
            return [], []
            
        # Get nearest neighbors from valid candidates using search client
        nearest = []
        size = min(self.nearest_neighbors * 2, len(valid_candidates))
        search_start = time.time()
        
        while len(nearest) < self.nearest_neighbors and size <= len(valid_candidates):
            candidates = self.search_es_client.search(query=chain, size=size)
            # Filter candidates to only include valid ones
            nearest = [c for c in candidates if c in valid_candidates]
            size *= 2
            
            if size > len(valid_candidates):
                break
            break
        
        search_time = time.time() - search_start
        logger.debug(
            f"Found {len(nearest)} nearest neighbors for chain: {chain} "
            f"(search time: {search_time:.2f}s)"
        )
        
        # Sample hard negatives from windows
        hard_candidates = []
        for i in range(0, len(nearest), self.window_size):
            window = nearest[i:i+self.window_size]
            if window:  # Only sample if window has candidates
                samples = random.sample(
                    window,
                    min(len(window), self.samples_per_window)
                )
                hard_candidates.extend(samples)
            
        # Sample soft negatives from remaining valid candidates
        non_nearest = [c for c in valid_candidates if c not in nearest]
        if non_nearest:
            soft_candidates = random.sample(
                non_nearest,
                min(len(non_nearest), self.soft_negative_count)
            )
        else:
            logger.warning(f"No soft negative candidates available for chain: {chain}")
            soft_candidates = []
        
        total_time = time.time() - start_time
        logger.debug(
            f"Generated candidates for chain: {chain} in {total_time:.2f}s. "
            f"Found {len(hard_candidates)} hard candidates and "
            f"{len(soft_candidates)} soft candidates"
        )
        return hard_candidates, soft_candidates
        
    def select_negatives(
        self, 
        chain: str, 
        hard_candidates: List[str], 
        soft_candidates: List[str]
    ) -> dict:
        """
        Use LLM to select definite negatives from candidates
        Returns a dict with hard_negatives and soft_negatives (for pipeline compatibility)
        """
        logger.debug(f"Selecting negatives for: {chain}")
        
        # Format candidates for prompt
        hard_candidates_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(hard_candidates)])
        soft_candidates_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(soft_candidates)])
        
        try:
            from openai import OpenAI
            from src.models.pydantic import NegativeSamplingResponse
            client = OpenAI()
            completion = client.beta.chat.completions.parse(
                model=self.llm_client.model,
                messages=[
                    {"role": "system", "content": NEGATIVE_SAMPLING_SYSTEM_PROMPT},
                    {"role": "user", "content": NEGATIVE_SAMPLING_PROMPT.format(
                        chain=chain,
                        hard_candidates=hard_candidates_str,
                        soft_candidates=soft_candidates_str
                    )}
                ],
                response_format=NegativeSamplingResponse,
                temperature=0.1
            )
            parsed = completion.choices[0].message.parsed
            # Return as dict for compatibility
            return parsed.model_dump() if hasattr(parsed, 'model_dump') else dict(parsed)

            # return parsed
        except Exception as e:
            logger.error(f"Error selecting negatives: {e}")
            raise
            
    def _parse_response(
        self, 
        response: str, 
        hard_candidates: List[str], 
        soft_candidates: List[str]
    ) -> Dict[str, List[str]]:
        """
        Parse LLM response to extract selected negatives
        
        Args:
            response: LLM response
            hard_candidates: Original hard candidates
            soft_candidates: Original soft candidates
            
        Returns:
            Dictionary with selected hard and soft negatives
        """
        hard_negatives = []
        soft_negatives = []
        
        # Extract hard negatives
        hard_section_match = re.search(r"HARD NEGATIVES:(.*?)(?:SOFT NEGATIVES:|$)", response, re.DOTALL)
        if hard_section_match:
            hard_section = hard_section_match.group(1).strip()
            if hard_section != "None":
                # Extract numbered items
                items = re.findall(r"\d+\.\s*(.*?)(?:\n|$)", hard_section)
                for item in items:
                    item = item.strip()
                    if item in hard_candidates:
                        hard_negatives.append(item)
                    else:
                        # Try to match closely
                        for candidate in hard_candidates:
                            if item.lower() == candidate.lower():
                                hard_negatives.append(candidate)
                                break
        
        # Extract soft negatives
        soft_section_match = re.search(r"SOFT NEGATIVES:(.*?)$", response, re.DOTALL)
        if soft_section_match:
            soft_section = soft_section_match.group(1).strip()
            if soft_section != "None":
                # Extract numbered items
                items = re.findall(r"\d+\.\s*(.*?)(?:\n|$)", soft_section)
                for item in items:
                    item = item.strip()
                    if item in soft_candidates:
                        soft_negatives.append(item)
                    else:
                        # Try to match closely
                        for candidate in soft_candidates:
                            if item.lower() == candidate.lower():
                                soft_negatives.append(candidate)
                                break
        
        logger.debug(f"Selected {len(hard_negatives)} hard negatives and {len(soft_negatives)} soft negatives")
        return {
            "hard_negatives": hard_negatives,
            "soft_negatives": soft_negatives
        }
        
    def generate_negatives_for_chain(self, chain: str) -> Dict[str, List[str]]:
        """
        Generate negative chains for a single input chain
        
        Args:
            chain: The industry chain to find negatives for
            
        Returns:
            Dictionary with selected hard and soft negatives
        """
        logger.debug(f"Generating negatives for chain: {chain}")
        
        hard_candidates, soft_candidates = self.get_candidate_negatives(chain)
        negatives = self.select_negatives(chain, hard_candidates, soft_candidates)
        
        return negatives
        
    def generate_negatives(self, chains: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Generate negative chains for multiple input chains in parallel
        
        Args:
            chains: List of industry chains to find negatives for
            
        Returns:
            Dictionary mapping chains to their negative chains
        """
        start_time = time.time()
        logger.info(f"Generating negatives for {len(chains)} chains using parallel processing")
        
        def process_chain(chain: str) -> Tuple[str, Dict[str, List[str]]]:
            try:
                chain_start = time.time()
                hard_candidates, soft_candidates = self.get_candidate_negatives(chain)
                negatives = self.select_negatives(chain, hard_candidates, soft_candidates)
                chain_time = time.time() - chain_start
                logger.debug(f"Processed chain in {chain_time:.2f}s: {chain[:50]}...")
                # return chain, negatives.model_dump()

                return chain,negatives
            except Exception as e:
                logger.error(f"Error generating negatives for {chain}: {e}")
                return chain, {"hard_negatives": [], "soft_negatives": []}
                
        # Process chains in parallel
        batch_results = batch_process(
            items=chains,
            process_fn=process_chain,
            batch_size=self.config["openai"].get("batch_size", 10),
            max_workers=self.config["openai"]["parallel_threads"]
        )
        
        # Convert batch results to dictionary
        results = {}
        for idx, result in batch_results.items():
            chain, negatives = result
            results[chain] = negatives
            
        total_time = time.time() - start_time
        rate = len(chains) / total_time if total_time > 0 else 0
        logger.info(
            f"Generated negatives for {len(results)} chains in {total_time:.2f}s "
            f"({rate:.1f} chains/s)"
        )
        return results
    
    def get_all_negatives_for_chain(self, chain: str, negative_results: Dict[str, Dict[str, List[str]]]) -> List[str]:
        """
        Get all negative chains for a given chain
        
        Args:
            chain: The industry chain
            negative_results: The results from generate_negatives
            
        Returns:
            List of all negative chains
        """
        negatives = set()
        
        if chain in negative_results:
            hard_negatives = negative_results[chain].get("hard_negatives", [])
            soft_negatives = negative_results[chain].get("soft_negatives", [])
            negatives.update(hard_negatives)
            negatives.update(soft_negatives)
            
        return list(negatives)
        
 