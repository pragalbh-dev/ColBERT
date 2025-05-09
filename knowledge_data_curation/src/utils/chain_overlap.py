from typing import Dict, Set, List
import logging
from collections import defaultdict
import json
logger = logging.getLogger(__name__)

class ChainOverlapManager:
    def __init__(self):
        """Initialize the chain overlap manager"""
        self.chain_to_companies: Dict[str, Set[str]] = {}
        self.overlapping_chains: Dict[str, Set[str]] = {}
        self.is_initialized: bool = False
        
    def build_overlap_index(self, company_chains: Dict[str, List[str]], min_overlap: int = 1) -> None:
        """
        Build indices for chain overlaps
        
        Args:
            company_chains: Dictionary mapping company IDs to their chains
            min_overlap: Minimum number of shared companies to consider chains as overlapping
        """
        logger.info(f"Building chain overlap indices with minimum overlap of {min_overlap}")
        
        # First pass: Build chain -> companies mapping
        chain_to_companies = defaultdict(set)
        for company_id, chains in company_chains.items():
            for chain in chains:
                chain_to_companies[chain].add(company_id)
                
        self.chain_to_companies = dict(chain_to_companies)
        
        # Second pass: Build overlapping chains mapping
        # For each chain, find other chains that share any companies
        overlapping_chains = defaultdict(set)
        chains = list(self.chain_to_companies.keys())
        total_comparisons = len(chains) * (len(chains) - 1) // 2
        
        logger.info(f"Computing overlaps for {len(chains)} chains ({total_comparisons} comparisons)")
        
        processed = 0
        for i, chain1 in enumerate(chains):
            companies1 = self.chain_to_companies[chain1]
            
            # Compare with all other chains
            for chain2 in chains[i+1:]:
                companies2 = self.chain_to_companies[chain2]
                
                # Check overlap size
                overlap_size = len(companies1 & companies2)
                if overlap_size >= min_overlap:
                    overlapping_chains[chain1].add(chain2)
                    overlapping_chains[chain2].add(chain1)
                
                processed += 1
                if processed % 1000000 == 0:
                    logger.info(f"Processed {processed}/{total_comparisons} comparisons")

        self.overlapping_chains = dict(overlapping_chains)
        for key, value in self.overlapping_chains.items():
            self.overlapping_chains[key] = list(value)
        self.is_initialized = True
        
        # Log statistics
        total_overlapping = sum(1 for chains in overlapping_chains.values() if chains)
        avg_overlaps = sum(len(chains) for chains in overlapping_chains.values()) / len(chains)
        logger.info(f"Built overlap indices for {len(self.chain_to_companies)} chains:")
        logger.info(f"- {total_overlapping} chains have overlaps")
        logger.info(f"- Average {avg_overlaps:.2f} overlapping chains per chain")
        
    def get_valid_negative_candidates(self, chain: str, all_chains: List[str]) -> List[str]:
        """
        Get list of chains that can be negative samples for the given chain
        
        Args:
            chain: The chain to find negative candidates for
            all_chains: List of all possible chains
            
        Returns:
            List of valid negative candidate chains
        """
        if not self.is_initialized:
            raise RuntimeError("ChainOverlapManager not initialized. Call build_overlap_index first.")
            
        # Get overlapping chains
        overlapping = self.overlapping_chains.get(chain, set())
        
        # Filter out the chain itself and its overlapping chains
        valid_candidates = [c for c in all_chains if c != chain and c not in overlapping]
        
        logger.debug(f"Found {len(valid_candidates)} valid candidates for chain: {chain}")
        return valid_candidates
        
    def get_overlap_size(self, chain1: str, chain2: str) -> int:
        """
        Get the number of companies shared between two chains
        
        Args:
            chain1: First chain
            chain2: Second chain
            
        Returns:
            Number of shared companies
        """
        if not self.is_initialized:
            raise RuntimeError("ChainOverlapManager not initialized. Call build_overlap_index first.")
            
        companies1 = self.chain_to_companies.get(chain1, set())
        companies2 = self.chain_to_companies.get(chain2, set())
        return len(companies1 & companies2)
        
    def get_overlap_stats(self) -> Dict:
        """
        Get statistics about chain overlaps
        
        Returns:
            Dictionary with overlap statistics
        """
        if not self.is_initialized:
            raise RuntimeError("ChainOverlapManager not initialized. Call build_overlap_index first.")
            
        total_chains = len(self.chain_to_companies)
        chains_with_overlaps = sum(1 for overlaps in self.overlapping_chains.values() if overlaps)
        total_overlaps = sum(len(overlaps) for overlaps in self.overlapping_chains.values())
        
        return {
            "total_chains": total_chains,
            "chains_with_overlaps": chains_with_overlaps,
            "average_overlaps_per_chain": total_overlaps / total_chains if total_chains > 0 else 0,
            "max_overlaps": max((len(overlaps) for overlaps in self.overlapping_chains.values()), default=0)
        } 