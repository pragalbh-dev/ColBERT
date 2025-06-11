from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict, Counter
import random

logger = logging.getLogger(__name__)

class DataImbalanceAnalyzer:
    def __init__(self, config: Dict):
        """
        Initialize the data imbalance analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_threshold = config.get("synthetic_generation", {}).get("min_companies_threshold", 5)
        self.max_sample_factsheets = config.get("synthetic_generation", {}).get("max_sample_factsheets", 2)
        
        logger.info(f"Initialized DataImbalanceAnalyzer with min_threshold={self.min_threshold}")
    
    def analyze_chain_distribution(self, company_chains: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze distribution of companies across industry chains
        
        Args:
            company_chains: Dictionary mapping company IDs to their chains
            
        Returns:
            Dictionary with distribution statistics and analysis
        """
        logger.info("Analyzing chain distribution across companies")
        
        # Count companies per chain
        chain_company_counts = defaultdict(int)
        for company_id, chains in company_chains.items():
            for chain in chains:
                chain_company_counts[chain] += 1
        
        # Convert to regular dict and sort by count
        chain_counts = dict(chain_company_counts)
        sorted_chains = sorted(chain_counts.items(), key=lambda x: x[1])
        
        # Calculate statistics
        counts = list(chain_counts.values())
        total_chains = len(chain_counts)
        total_companies = len(company_chains)
        
        underrepresented_chains = [
            chain for chain, count in chain_counts.items() 
            if count < self.min_threshold
        ]
        
        well_represented_chains = [
            chain for chain, count in chain_counts.items() 
            if count >= self.min_threshold
        ]
        
        stats = {
            "total_chains": total_chains,
            "total_companies": total_companies,
            "chain_counts": chain_counts,
            "underrepresented_chains": underrepresented_chains,
            "well_represented_chains": well_represented_chains,
            "underrepresented_count": len(underrepresented_chains),
            "well_represented_count": len(well_represented_chains),
            "min_companies_per_chain": min(counts) if counts else 0,
            "max_companies_per_chain": max(counts) if counts else 0,
            "avg_companies_per_chain": sum(counts) / len(counts) if counts else 0,
            "median_companies_per_chain": sorted(counts)[len(counts)//2] if counts else 0,
            "threshold_used": self.min_threshold
        }
        
        logger.info(f"Chain distribution analysis completed:")
        logger.info(f"  - Total chains: {stats['total_chains']}")
        logger.info(f"  - Underrepresented chains: {stats['underrepresented_count']} (< {self.min_threshold} companies)")
        logger.info(f"  - Well represented chains: {stats['well_represented_count']} (>= {self.min_threshold} companies)")
        logger.info(f"  - Average companies per chain: {stats['avg_companies_per_chain']:.2f}")
        
        return stats
    
    def identify_underrepresented_chains(self, cleaned_chains: Dict[str, str], company_chains: Dict[str, List[str]]) -> List[str]:
        """
        Identify chains with less than threshold companies
        
        Args:
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            company_chains: Dictionary mapping company IDs to their chains
            
        Returns:
            List of cleaned chains needing synthetic data
        """
        logger.info(f"Identifying chains with less than {self.min_threshold} companies")
        
        # Count companies per original chain first
        chain_company_counts = defaultdict(int)
        for company_id, chains in company_chains.items():
            for chain in chains:
                chain_company_counts[chain] += 1
        
        # Map to cleaned chains and identify underrepresented ones
        underrepresented_cleaned_chains = []
        
        for original_chain, cleaned_chain in cleaned_chains.items():
            company_count = chain_company_counts.get(original_chain, 0)
            if company_count < self.min_threshold:
                if cleaned_chain not in underrepresented_cleaned_chains:
                    underrepresented_cleaned_chains.append(cleaned_chain)
        
        logger.info(f"Found {len(underrepresented_cleaned_chains)} underrepresented cleaned chains")
        
        return underrepresented_cleaned_chains
    
    def get_sample_factsheets(self, 
                            target_chain: str, 
                            cleaned_chains: Dict[str, str],
                            company_chains: Dict[str, List[str]], 
                            company_factsheets: Dict[str, str], 
                            max_samples: int = None) -> List[str]:
        """
        Get 1-2 sample factsheets for a given cleaned chain to use as examples
        
        Args:
            target_chain: The cleaned chain to get samples for
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            company_chains: Dictionary mapping company IDs to their chains
            company_factsheets: Dictionary mapping company IDs to factsheets
            max_samples: Maximum number of samples to return
            
        Returns:
            List of sample factsheets (up to max_samples)
        """
        if max_samples is None:
            max_samples = self.max_sample_factsheets
            
        logger.debug(f"Getting sample factsheets for chain: {target_chain}")
        
        # Find original chains that map to this cleaned chain
        original_chains_for_target = [
            original for original, cleaned in cleaned_chains.items() 
            if cleaned == target_chain
        ]
        
        # Find companies that belong to these original chains
        sample_companies = []
        for company_id, chains in company_chains.items():
            for chain in chains:
                if chain in original_chains_for_target:
                    if company_id in company_factsheets:
                        sample_companies.append(company_id)
                    break
        
        # Get factsheets for these companies
        sample_factsheets = []
        for company_id in sample_companies[:max_samples]:
            factsheet = company_factsheets.get(company_id)
            if factsheet:
                sample_factsheets.append(factsheet)
        
        logger.debug(f"Found {len(sample_factsheets)} sample factsheets for chain: {target_chain}")
        
        return sample_factsheets
    
    def get_chain_sample_mapping(self, 
                               underrepresented_chains: List[str],
                               cleaned_chains: Dict[str, str],
                               company_chains: Dict[str, List[str]], 
                               company_factsheets: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Get sample factsheets for all underrepresented chains
        
        Args:
            underrepresented_chains: List of cleaned chains that need synthetic data
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            company_chains: Dictionary mapping company IDs to their chains
            company_factsheets: Dictionary mapping company IDs to factsheets
            
        Returns:
            Dictionary mapping cleaned chains to their sample factsheets
        """
        logger.info(f"Getting sample factsheets for {len(underrepresented_chains)} underrepresented chains")
        
        chain_samples = {}
        chains_without_samples = []
        
        for chain in underrepresented_chains:
            samples = self.get_sample_factsheets(
                chain, cleaned_chains, company_chains, company_factsheets
            )
            
            if samples:
                chain_samples[chain] = samples
            else:
                chains_without_samples.append(chain)
        
        if chains_without_samples:
            logger.warning(f"Could not find sample factsheets for {len(chains_without_samples)} chains:")
            for chain in chains_without_samples[:5]:  # Log first 5
                logger.warning(f"  - {chain}")
            if len(chains_without_samples) > 5:
                logger.warning(f"  ... and {len(chains_without_samples) - 5} more")
        
        logger.info(f"Successfully found samples for {len(chain_samples)} chains")
        
        return chain_samples
    
    def generate_balance_report(self, 
                              analysis_stats: Dict[str, Any],
                              target_factsheets_per_chain: int) -> Dict[str, Any]:
        """
        Generate a report on what synthetic data generation will accomplish
        
        Args:
            analysis_stats: Results from analyze_chain_distribution
            target_factsheets_per_chain: Target number of factsheets to generate per chain
            
        Returns:
            Dictionary with balance improvement projections
        """
        underrepresented_count = analysis_stats["underrepresented_count"]
        current_total_examples = sum(analysis_stats["chain_counts"].values())
        
        # Calculate synthetic data to be generated
        synthetic_factsheets_total = underrepresented_count * target_factsheets_per_chain
        
        # Project new distribution
        projected_total_examples = current_total_examples + synthetic_factsheets_total
        projected_min_per_chain = min(
            target_factsheets_per_chain,
            analysis_stats["min_companies_per_chain"]
        )
        
        balance_report = {
            "current_state": {
                "total_examples": current_total_examples,
                "underrepresented_chains": underrepresented_count,
                "min_examples_per_chain": analysis_stats["min_companies_per_chain"],
                "avg_examples_per_chain": analysis_stats["avg_companies_per_chain"]
            },
            "synthetic_generation": {
                "chains_to_augment": underrepresented_count,
                "factsheets_per_chain": target_factsheets_per_chain,
                "total_synthetic_factsheets": synthetic_factsheets_total
            },
            "projected_state": {
                "total_examples": projected_total_examples,
                "improvement_ratio": projected_total_examples / current_total_examples if current_total_examples > 0 else 0,
                "projected_min_per_chain": projected_min_per_chain,
                "balance_improvement": True if projected_min_per_chain >= self.min_threshold else False
            }
        }
        
        logger.info("Balance improvement projection:")
        logger.info(f"  - Current total examples: {current_total_examples}")
        logger.info(f"  - Synthetic examples to generate: {synthetic_factsheets_total}")
        logger.info(f"  - Projected total examples: {projected_total_examples}")
        logger.info(f"  - Improvement ratio: {balance_report['projected_state']['improvement_ratio']:.2f}x")
        
        return balance_report 