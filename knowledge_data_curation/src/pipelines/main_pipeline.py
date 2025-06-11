from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import logging
import os
import yaml
from pathlib import Path
import pandas as pd

from src.data_processors.chain_cleaner import ChainCleaner
from src.data_processors.negative_generator import NegativeSampleGenerator
from src.data_processors.data_loader import DataLoader
from src.data_processors.subchain_generator import SubchainGenerator
from src.data_processors.imbalance_analyzer import DataImbalanceAnalyzer
from src.data_processors.synthetic_factsheet_generator import SyntheticFactsheetGenerator
from src.data_processors.data_normalizer import DataNormalizer
from src.utils.cache_manager import SyntheticDataCacheManager
from src.pipelines.triplet_generator import TripletGenerator
from src.utils.io import save_json, save_csv
from src.utils.logger import Logger

logger = logging.getLogger(__name__)

class ColBERTTrainingPipeline:
    def __init__(self, config_path: str):
        """
        Initialize the ColBERT training pipeline
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        
        # Setup logging
        Logger.setup_logger("colbert_pipeline", self.config, "main_pipeline")
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.chain_cleaner = ChainCleaner(self.config)
        self.subchain_generator = SubchainGenerator(self.config)
        self.negative_generator = NegativeSampleGenerator(self.config)
        self.triplet_generator = TripletGenerator(self.config)
        
        # Initialize synthetic data generation components
        self.imbalance_analyzer = DataImbalanceAnalyzer(self.config)
        self.synthetic_generator = SyntheticFactsheetGenerator(self.config)
        self.data_normalizer = DataNormalizer(self.config)
        self.cache_manager = SyntheticDataCacheManager(self.config)
        
        # Initialize paths
        self.output_dir = Path(self.config["paths"]["output_dir"])
        self.cleaned_chains_dir = self.output_dir / "cleaned_chains"
        self.negative_chains_dir = self.output_dir / "negative_chains"
        self.colbert_training_dir = self.output_dir / "colbert_training"
        
        # Create directories if they don't exist
        self.cleaned_chains_dir.mkdir(parents=True, exist_ok=True)
        self.negative_chains_dir.mkdir(parents=True, exist_ok=True)
        self.colbert_training_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ColBERTTrainingPipeline with config from {config_path}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load and merge configurations
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Merged configuration dictionary
        """
        # Default config path
        default_config_path = os.path.join(
            os.path.dirname(config_path), 
            "default.yaml"
        )
        
        # Load default config
        try:
            with open(default_config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading default config: {e}")
            config = {}
            
        # Load custom config and merge
        try:
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                
            # Merge configs (custom overrides default)
            if custom_config:
                for section, values in custom_config.items():
                    if section in config:
                        config[section].update(values)
                    else:
                        config[section] = values
        except Exception as e:
            print(f"Error loading custom config: {e}")
            
        return config
        
    def run(self) -> None:
        """
        Run the complete pipeline
        """
        logger.info("Starting ColBERT training data curation pipeline")
        
        # Step 1: Load data and extract unique chains
        logger.info("Step 1: Loading data")
        chains = self.data_loader.get_unique_chains()
        company_chains = self.data_loader.get_company_chains()
        
        # Step 2: Clean chains
        logger.info("Step 2: Cleaning chains")
        cleaned_chains = self.chain_cleaner.process_chains(chains)
        cleaned_chains_path = self.cleaned_chains_dir / "cleaned_chains.json"
        save_json(cleaned_chains, str(cleaned_chains_path))
        
        # Log token usage after chain cleaning
        chain_cleaning_usage = self.chain_cleaner.llm_client.get_token_usage()
        logger.info(f"Chain cleaning token usage: {chain_cleaning_usage}")
        
        # Update company_chains with cleaned chains
        cleaned_company_chains = {}
        for company_id, chains in company_chains.items():
            cleaned_company_chains[company_id] = [
                cleaned_chains[chain] for chain in chains 
                if chain in cleaned_chains
            ]
        
        # Step 2.5: Synthetic Data Generation (if enabled)
        if self.config.get("synthetic_generation", {}).get("enabled", False):
            logger.info("Step 2.5: Generating synthetic factsheets for data balancing")
            
            # Check cache first
            if self.cache_manager.reuse_existing and self.cache_manager.is_cached_data_available():
                logger.info("Using cached synthetic data")
                cached_data = self.cache_manager.load_cached_data()
                augmented_industry_df = cached_data["industry_df"]
                augmented_factsheet_df = cached_data["factsheet_df"]
            else:
                # Generate new synthetic data
                augmented_industry_df, augmented_factsheet_df = self._generate_synthetic_data_step(
                    self.data_loader.load_industry_data(),
                    self.data_loader.load_factsheet_data(),
                    cleaned_chains,
                    company_chains
                )
                
                # Cache the results if caching is enabled
                if self.cache_manager.cache_enabled:
                    # Note: Individual components handle their own caching
                    pass
            
            # Update data loader to use augmented data
            self.data_loader.override_data(augmented_industry_df, augmented_factsheet_df)
            
            # Update company mappings with augmented data
            company_chains = self.data_loader.get_company_chains()
            cleaned_company_chains = {}
            for company_id, chains in company_chains.items():
                cleaned_company_chains[company_id] = [
                    cleaned_chains[chain] for chain in chains 
                    if chain in cleaned_chains
                ]
        
        # Step 3: Generate subchains from cleaned chains
        logger.info("Step 3: Generating subchains from cleaned chains")
        subchain_data = self.subchain_generator.generate_all_subchains(cleaned_chains)
        
        # Step 4: Initialize negative generator for chains
        logger.info("Step 4: Initializing negative generator for chains")
        self.negative_generator.initialize(cleaned_company_chains)
        
        # Step 5: Generate negative chains
        logger.info("Step 5: Generating negative chains")
        negative_chains = self.negative_generator.generate_negatives(list(cleaned_chains.values()))
        negative_chains_path = self.negative_chains_dir / "negative_chains.json"
        save_json(negative_chains, str(negative_chains_path))
        
        # Step 6: Generate negative subchains (if enabled) - separate process
        negative_subchains = {}
        if subchain_data.get("enabled", False):
            logger.info("Step 6: Processing subchains separately")
            
            # Create company-subchains mapping for proper overlap calculation
            company_subchains = self.subchain_generator.create_company_subchains_mapping(
                subchain_data, company_chains, cleaned_chains
            )
            
            # Create a separate NegativeSampleGenerator instance for subchains
            logger.info("Step 6a: Initializing separate negative generator for subchains")
            subchain_negative_generator = NegativeSampleGenerator(self.config)
            subchain_negative_generator.initialize(company_subchains)
            
            # Generate negatives for subchains
            logger.info("Step 6b: Generating negative subchains")
            unique_subchains = subchain_data.get("deduplicated_subchains", [])
            negative_subchains = subchain_negative_generator.generate_negatives(unique_subchains)
            negative_subchains_path = self.negative_chains_dir / "negative_subchains.json"
            save_json(negative_subchains, str(negative_subchains_path))
        
        # Log token usage after negative generation
        negative_gen_usage = self.negative_generator.llm_client.get_token_usage()
        logger.info(f"Chain negative generation token usage: {negative_gen_usage}")
        
        # Log subchain negative generation token usage if applicable
        subchain_negative_gen_usage = {}
        if subchain_data.get("enabled", False) and 'subchain_negative_generator' in locals():
            subchain_negative_gen_usage = subchain_negative_generator.llm_client.get_token_usage()
            logger.info(f"Subchain negative generation token usage: {subchain_negative_gen_usage}")
        
        # Step 7: Generate query-factsheet positives
        logger.info("Step 7: Generating query-factsheet positives for full chains")
        query_factsheet_positives = self._generate_query_factsheet_positives(cleaned_chains)
        
        # Step 8: Generate query-factsheet positives for subchains (if enabled)
        subchain_query_factsheet_positives = {}
        if subchain_data.get("enabled", False):
            logger.info("Step 8: Generating query-factsheet positives for subchains")
            subchain_query_factsheet_positives = self._generate_subchain_factsheet_positives(subchain_data, cleaned_chains)
        
        # Step 9: Generate query-factsheet negatives for full chains
        logger.info("Step 9: Generating query-factsheet negatives for full chains")
        query_factsheet_negatives = self._generate_query_factsheet_negatives(cleaned_chains, negative_chains)
        
        # Step 10: Generate query-factsheet negatives for subchains (if enabled)
        subchain_query_factsheet_negatives = {}
        if subchain_data.get("enabled", False):
            logger.info("Step 10: Generating query-factsheet negatives for subchains")
            subchain_query_factsheet_negatives = self._generate_subchain_factsheet_negatives(subchain_data, negative_subchains, cleaned_chains)
        
        # Step 11: Generate final triplets
        logger.info("Step 11: Generating final triplets")
        triplets = self.triplet_generator.generate_triplets(
            query_factsheet_positives, 
            query_factsheet_negatives
        )
        
        # Generate subchain triplets (if enabled)
        subchain_triplets = {}
        if subchain_data.get("enabled", False):
            logger.info("Step 11b: Generating subchain triplets")
            subchain_triplets = self.triplet_generator.generate_triplets(
                subchain_query_factsheet_positives,
                subchain_query_factsheet_negatives
            )
        
        # Step 12: Save results
        logger.info("Step 12: Saving results")
        self._save_results(triplets, subchain_triplets, subchain_data)
        
        # Log final token usage statistics
        total_usage = {
            "chain_cleaning": chain_cleaning_usage,
            "chain_negative_generation": negative_gen_usage,
            "subchain_negative_generation": subchain_negative_gen_usage
        }
        logger.info("Final token usage statistics:")
        logger.info(f"Chain cleaning: {chain_cleaning_usage}")
        logger.info(f"Chain negative generation: {negative_gen_usage}")
        if subchain_negative_gen_usage:
            logger.info(f"Subchain negative generation: {subchain_negative_gen_usage}")
        
        # Save token usage statistics
        token_stats_path = self.output_dir / "token_usage_stats.json"
        save_json(total_usage, str(token_stats_path))
        
        logger.info("Pipeline completed successfully!")
        
    def _generate_query_factsheet_positives(
        self, 
        cleaned_chains: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """
        Generate mappings from queries to positive factsheets
        
        Args:
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            
        Returns:
            Dictionary mapping queries to lists of positive factsheets
        """
        logger.info("Generating query-factsheet positives")
        
        # Create reverse mapping from original chains to cleaned chains
        original_to_cleaned = {k: v for k, v in cleaned_chains.items()}
        
        # Get mappings of companies to chains and factsheets
        company_chains = self.data_loader.get_company_chains()
        company_factsheets = self.data_loader.get_company_factsheets()
        
        # Create mapping from cleaned chains to factsheets
        query_factsheet_positives = {}
        
        for company_id, chains in company_chains.items():
            # Skip if company doesn't have a factsheet
            if company_id not in company_factsheets:
                continue
                
            factsheet = company_factsheets[company_id]
            
            # Map each chain to the factsheet
            for chain in chains:
                if chain in original_to_cleaned:
                    cleaned_chain = original_to_cleaned[chain]
                    
                    if cleaned_chain not in query_factsheet_positives:
                        query_factsheet_positives[cleaned_chain] = []
                        
                    query_factsheet_positives[cleaned_chain].append(factsheet)
        
        # Remove duplicates
        for query, factsheets in query_factsheet_positives.items():
            query_factsheet_positives[query] = list(set(factsheets))
            
        logger.info(f"Generated positives for {len(query_factsheet_positives)} queries")
        return query_factsheet_positives
        
    def _generate_query_factsheet_negatives(
        self,
        cleaned_chains: Dict[str, str],
        negative_chains: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, List[str]]:
        """
        Generate mappings from queries to negative factsheets
        
        Args:
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            negative_chains: Dictionary mapping cleaned chains to their negative chains
            
        Returns:
            Dictionary mapping queries to lists of negative factsheets
        """
        logger.info("Generating query-factsheet negatives")
        
        # Create reverse mapping from cleaned chains to original chains
        cleaned_to_original = {}
        for original, cleaned in cleaned_chains.items():
            if cleaned not in cleaned_to_original:
                cleaned_to_original[cleaned] = []
            cleaned_to_original[cleaned].append(original)
            
        # Get mappings of companies to chains and factsheets
        company_chains = self.data_loader.get_company_chains()
        company_factsheets = self.data_loader.get_company_factsheets()
        
        # Create mapping from cleaned chains to negative factsheets
        query_factsheet_negatives = {}
        
        for cleaned_chain, negatives_dict in negative_chains.items():
            # Combine hard and soft negatives
            all_negatives = []
            all_negatives.extend(negatives_dict.get("hard_negatives", []))
            all_negatives.extend(negatives_dict.get("soft_negatives", []))
            
            if not all_negatives:
                continue
                
            # Find original chains for these negative cleaned chains
            negative_original_chains = []
            for negative_cleaned in all_negatives:
                if negative_cleaned in cleaned_to_original:
                    negative_original_chains.extend(cleaned_to_original[negative_cleaned])
            
            if not negative_original_chains:
                continue
                
            # Find companies in these negative industries
            negative_companies = set()
            for company_id, chains in company_chains.items():
                for chain in chains:
                    if chain in negative_original_chains:
                        negative_companies.add(company_id)
                        break
            
            # Find factsheets for these negative companies
            negative_factsheets = []
            for company_id in negative_companies:
                if company_id in company_factsheets:
                    negative_factsheets.append(company_factsheets[company_id])
            
            # Store negative factsheets for this query
            if negative_factsheets:
                query_factsheet_negatives[cleaned_chain] = negative_factsheets
        
        logger.info(f"Generated negatives for {len(query_factsheet_negatives)} queries")
        return query_factsheet_negatives
        
    def _generate_subchain_factsheet_positives(
        self,
        subchain_data: Dict[str, Any],
        cleaned_chains: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """
        Generate mappings from subchains to positive factsheets
        
        Args:
            subchain_data: Complete subchain data from SubchainGenerator
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            
        Returns:
            Dictionary mapping subchains to lists of positive factsheets
        """
        logger.info("Generating subchain-factsheet positives")
        
        if not subchain_data.get("enabled", False):
            return {}
        
        # Get the reverse mappings from subchains to original chains
        reverse_mappings = subchain_data.get("reverse_mappings", {})
        subchain_to_original = reverse_mappings.get("subchain_to_original_chains", {})
        
        # Get mappings of companies to chains and factsheets
        company_chains = self.data_loader.get_company_chains()
        company_factsheets = self.data_loader.get_company_factsheets()
        
        # Create mapping from subchains to factsheets
        subchain_factsheet_positives = {}
        
        for subchain, original_chains in subchain_to_original.items():
            factsheets = []
            
            # For each original chain that contains this subchain
            for original_chain in original_chains:
                # Find companies that belong to this original chain
                for company_id, company_chains_list in company_chains.items():
                    if original_chain in company_chains_list:
                        # Add the company's factsheet if it exists
                        if company_id in company_factsheets:
                            factsheets.append(company_factsheets[company_id])
            
            # Remove duplicates and store
            if factsheets:
                subchain_factsheet_positives[subchain] = list(set(factsheets))
        
        logger.info(f"Generated subchain positives for {len(subchain_factsheet_positives)} subchains")
        return subchain_factsheet_positives
        
    def _generate_subchain_factsheet_negatives(
        self,
        subchain_data: Dict[str, Any],
        negative_subchains: Dict[str, Dict[str, List[str]]],
        cleaned_chains: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """
        Generate mappings from subchains to negative factsheets
        
        Args:
            subchain_data: Complete subchain data from SubchainGenerator  
            negative_subchains: Dictionary mapping subchains to their negative subchains
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            
        Returns:
            Dictionary mapping subchains to lists of negative factsheets
        """
        logger.info("Generating subchain-factsheet negatives")
        
        if not subchain_data.get("enabled", False):
            return {}
        
        # Get reverse mappings
        reverse_mappings = subchain_data.get("reverse_mappings", {})
        subchain_to_original = reverse_mappings.get("subchain_to_original_chains", {})
        
        # Get mappings of companies to chains and factsheets
        company_chains = self.data_loader.get_company_chains()
        company_factsheets = self.data_loader.get_company_factsheets()
        
        # Create mapping from subchains to negative factsheets
        subchain_factsheet_negatives = {}
        
        for subchain, negatives_dict in negative_subchains.items():
            # Combine hard and soft negatives
            all_negative_subchains = []
            all_negative_subchains.extend(negatives_dict.get("hard_negatives", []))
            all_negative_subchains.extend(negatives_dict.get("soft_negatives", []))
            
            if not all_negative_subchains:
                continue
            
            # Find original chains for these negative subchains
            negative_original_chains = []
            for negative_subchain in all_negative_subchains:
                if negative_subchain in subchain_to_original:
                    negative_original_chains.extend(subchain_to_original[negative_subchain])
            
            if not negative_original_chains:
                continue
            
            # Find companies in these negative industries
            negative_companies = set()
            for company_id, chains in company_chains.items():
                for chain in chains:
                    if chain in negative_original_chains:
                        negative_companies.add(company_id)
                        break
            
            # Find factsheets for these negative companies
            negative_factsheets = []
            for company_id in negative_companies:
                if company_id in company_factsheets:
                    negative_factsheets.append(company_factsheets[company_id])
            
            # Store negative factsheets for this subchain
            if negative_factsheets:
                subchain_factsheet_negatives[subchain] = negative_factsheets
        
        logger.info(f"Generated subchain negatives for {len(subchain_factsheet_negatives)} subchains")
        return subchain_factsheet_negatives
        
    def _save_results(self, triplets: Dict[str, Dict[str, Any]], subchain_triplets: Dict[str, Dict[str, Any]] = None, subchain_data: Dict[str, Any] = None) -> None:
        """
        Save the final triplets to files
        
        Args:
            triplets: Dictionary of full chain triplets with scores
            subchain_triplets: Dictionary of subchain triplets with scores (optional)
            subchain_data: Subchain configuration and stats (optional)
        """
        logger.info("Saving final triplets")
        
        # Save full chain triplets
        colbert_data = []
        
        for query, data in triplets.items():
            for factsheet, score in data["positives"]:
                colbert_data.append({
                    "query": query,
                    "factsheet": factsheet,
                    "score": score,
                    "is_positive": True,
                    "query_type": "full_chain"
                })
                
            for factsheet, score in data["negatives"]:
                colbert_data.append({
                    "query": query,
                    "factsheet": factsheet,
                    "score": score,
                    "is_positive": False,
                    "query_type": "full_chain"
                })
        
        # Save full chain data
        full_chains_training_path = self.colbert_training_dir / "full_chains_training.csv"
        save_csv(colbert_data, str(full_chains_training_path))
        
        # Save subchain triplets if enabled
        subchain_colbert_data = []
        if subchain_triplets and subchain_data and subchain_data.get("enabled", False):
            logger.info("Saving subchain triplets")
            
            for query, data in subchain_triplets.items():
                for factsheet, score in data["positives"]:
                    subchain_colbert_data.append({
                        "query": query,
                        "factsheet": factsheet,
                        "score": score,
                        "is_positive": True,
                        "query_type": "subchain"
                    })
                    
                for factsheet, score in data["negatives"]:
                    subchain_colbert_data.append({
                        "query": query,
                        "factsheet": factsheet,
                        "score": score,
                        "is_positive": False,
                        "query_type": "subchain"
                    })
            
            # Save subchain data
            subchains_training_path = self.colbert_training_dir / "subchains_training.csv"
            save_csv(subchain_colbert_data, str(subchains_training_path))
        
        # Save combined data
        combined_data = colbert_data + subchain_colbert_data
        if combined_data:
            combined_training_path = self.colbert_training_dir / "combined_training.csv"
            save_csv(combined_data, str(combined_training_path))
        
        # Get overlap statistics
        overlap_stats = self.negative_generator.overlap_manager.get_overlap_stats()
        
        # Calculate comprehensive statistics
        full_chain_positives = sum(len(data["positives"]) for data in triplets.values())
        full_chain_negatives = sum(len(data["negatives"]) for data in triplets.values())
        
        subchain_positives = sum(len(data["positives"]) for data in subchain_triplets.values()) if subchain_triplets else 0
        subchain_negatives = sum(len(data["negatives"]) for data in subchain_triplets.values()) if subchain_triplets else 0
        
        stats = {
            "full_chains": {
                "total_queries": len(triplets),
                "total_positives": full_chain_positives,
                "total_negatives": full_chain_negatives,
                "total_triplets": full_chain_positives + full_chain_negatives,
                "queries_with_negatives": sum(1 for data in triplets.values() if data["negatives"]),
                "queries_with_positives": sum(1 for data in triplets.values() if data["positives"])
            },
            "subchains": {
                "enabled": subchain_data.get("enabled", False) if subchain_data else False,
                "total_queries": len(subchain_triplets) if subchain_triplets else 0,
                "total_positives": subchain_positives,
                "total_negatives": subchain_negatives,
                "total_triplets": subchain_positives + subchain_negatives,
                "queries_with_negatives": sum(1 for data in subchain_triplets.values() if data["negatives"]) if subchain_triplets else 0,
                "queries_with_positives": sum(1 for data in subchain_triplets.values() if data["positives"]) if subchain_triplets else 0,
                "config_stats": subchain_data.get("stats", {}) if subchain_data else {}
            },
            "combined": {
                "total_queries": len(triplets) + (len(subchain_triplets) if subchain_triplets else 0),
                "total_positives": full_chain_positives + subchain_positives,
                "total_negatives": full_chain_negatives + subchain_negatives,
                "total_triplets": full_chain_positives + full_chain_negatives + subchain_positives + subchain_negatives
            },
            "overlap_stats": overlap_stats
        }
        
        stats_path = self.colbert_training_dir / "stats.json"
        save_json(stats, str(stats_path))
        
        logger.info(f"Saved full chain triplets: {stats['full_chains']['total_triplets']} for {stats['full_chains']['total_queries']} queries")
        if subchain_data and subchain_data.get("enabled", False):
            logger.info(f"Saved subchain triplets: {stats['subchains']['total_triplets']} for {stats['subchains']['total_queries']} queries")
        logger.info(f"Total combined triplets: {stats['combined']['total_triplets']} for {stats['combined']['total_queries']} queries")
        logger.info(f"Statistics: {stats}")
    
    def _generate_synthetic_data_step(self, 
                                    industry_df: pd.DataFrame, 
                                    factsheet_df: pd.DataFrame,
                                    cleaned_chains: Dict[str, str],
                                    company_chains: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic factsheets for underrepresented chains
        
        Args:
            industry_df: Original industry data
            factsheet_df: Original factsheet data
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            company_chains: Dictionary mapping company IDs to their chains
            
        Returns:
            Tuple of (augmented_industry_df, augmented_factsheet_df)
        """
        logger.info("Starting synthetic data generation step")
        
        # Step 1: Analyze data imbalance
        logger.info("Step 1: Analyzing data imbalance")
        company_factsheets = self.data_loader.get_company_factsheets()
        analysis_stats = self.imbalance_analyzer.analyze_chain_distribution(company_chains)
        
        # Step 2: Identify underrepresented chains
        logger.info("Step 2: Identifying underrepresented chains")
        underrepresented_chains = self.imbalance_analyzer.identify_underrepresented_chains(
            cleaned_chains, company_chains
        )
        
        if not underrepresented_chains:
            logger.info("No underrepresented chains found - skipping synthetic generation")
            # Return normalized original data
            normalized_industry_df = self.data_normalizer.normalize_company_ids(industry_df)
            normalized_factsheet_df = self.data_normalizer.normalize_company_ids(factsheet_df)
            return normalized_industry_df, normalized_factsheet_df
        
        # Step 3: Get sample factsheets for underrepresented chains
        logger.info("Step 3: Getting sample factsheets for underrepresented chains")
        chain_samples = self.imbalance_analyzer.get_chain_sample_mapping(
            underrepresented_chains, cleaned_chains, company_chains, company_factsheets
        )
        
        # Filter out chains without samples
        chains_with_samples = [chain for chain in underrepresented_chains if chain in chain_samples]
        if len(chains_with_samples) < len(underrepresented_chains):
            logger.warning(f"Could only find samples for {len(chains_with_samples)} out of {len(underrepresented_chains)} underrepresented chains")
        
        if not chains_with_samples:
            logger.warning("No chains with sample factsheets found - skipping synthetic generation")
            # Return normalized original data
            normalized_industry_df = self.data_normalizer.normalize_company_ids(industry_df)
            normalized_factsheet_df = self.data_normalizer.normalize_company_ids(factsheet_df)
            return normalized_industry_df, normalized_factsheet_df
        
        # Step 4: Generate balance report
        target_count = self.config.get("synthetic_generation", {}).get("target_factsheets_per_chain", 10)
        balance_report = self.imbalance_analyzer.generate_balance_report(analysis_stats, target_count)
        
        # Step 5: Generate synthetic factsheets
        logger.info("Step 5: Generating synthetic factsheets")
        synthetic_data = self.synthetic_generator.batch_generate_factsheets(
            chains_with_samples, chain_samples
        )
        
        # Step 6: Validate generated factsheets
        logger.info("Step 6: Validating generated factsheets")
        validation_results = self.synthetic_generator.validate_generated_factsheets(synthetic_data)
        
        # Step 7: Save generation metadata
        generation_stats = {
            "underrepresented_chains": len(underrepresented_chains),
            "chains_with_samples": len(chains_with_samples),
            "chains_processed": len(synthetic_data),
            "balance_report": balance_report
        }
        self.synthetic_generator.save_generation_metadata(
            synthetic_data, validation_results, generation_stats
        )
        
        # Step 8: Create augmented datasets
        logger.info("Step 8: Creating augmented datasets with synthetic data")
        augmented_industry_df, augmented_factsheet_df = self.data_normalizer.create_augmented_datasets(
            industry_df, factsheet_df, synthetic_data, cleaned_chains
        )
        
        # Step 9: Cache the synthetic data if enabled
        if self.cache_manager.cache_enabled:
            logger.info("Step 9: Caching synthetic data")
            cache_data = {
                "synthetic_data": synthetic_data,
                "generation_config": {
                    "target_count": target_count,
                    "chains_processed": list(synthetic_data.keys()),
                    "validation_results": validation_results
                }
            }
            self.cache_manager.save_cached_data(cache_data)
        
        logger.info("Synthetic data generation step completed successfully")
        return augmented_industry_df, augmented_factsheet_df
    
    @classmethod
    def load_from_checkpoint(cls, config_path: str, checkpoint_dir: str) -> 'ColBERTTrainingPipeline':
        """
        Load pipeline from a checkpoint
        
        Args:
            config_path: Path to the configuration file
            checkpoint_dir: Path to the checkpoint directory
            
        Returns:
            Initialized pipeline with checkpoint data
        """
        pipeline = cls(config_path)
        # Load intermediate data as needed
        return pipeline 