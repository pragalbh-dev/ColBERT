from typing import Dict, List, Tuple, Any, Optional
import logging
import os
import yaml
from pathlib import Path
import pandas as pd

from knowledge_data_curation.src.data_processors.chain_cleaner import ChainCleaner
from knowledge_data_curation.src.data_processors.negative_generator import NegativeSampleGenerator
from knowledge_data_curation.src.data_processors.data_loader import DataLoader
from knowledge_data_curation.src.pipelines.triplet_generator import TripletGenerator
from knowledge_data_curation.src.utils.io import save_json, save_csv
from knowledge_data_curation.src.utils.logger import Logger

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
        self.negative_generator = NegativeSampleGenerator(self.config)
        self.triplet_generator = TripletGenerator(self.config)
        
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
        
        # Step 3: Initialize negative generator
        logger.info("Step 3: Initializing negative generator")
        self.negative_generator.initialize(cleaned_company_chains)
        
        # Step 4: Generate negative chains
        logger.info("Step 4: Generating negative chains")
        negative_chains = self.negative_generator.generate_negatives(list(cleaned_chains.values()))
        negative_chains_path = self.negative_chains_dir / "negative_chains.json"
        save_json(negative_chains, str(negative_chains_path))
        
        # Log token usage after negative generation
        negative_gen_usage = self.negative_generator.llm_client.get_token_usage()
        logger.info(f"Negative generation token usage: {negative_gen_usage}")
        
        # Step 5: Generate query-factsheet positives
        logger.info("Step 5: Generating query-factsheet positives")
        query_factsheet_positives = self._generate_query_factsheet_positives(cleaned_chains)
        
        # Step 6: Generate query-factsheet negatives
        logger.info("Step 6: Generating query-factsheet negatives")
        query_factsheet_negatives = self._generate_query_factsheet_negatives(cleaned_chains, negative_chains)
        
        # Step 7: Generate final triplets
        logger.info("Step 7: Generating final triplets")
        triplets = self.triplet_generator.generate_triplets(
            query_factsheet_positives, 
            query_factsheet_negatives
        )
        
        # Step 8: Save results
        logger.info("Step 8: Saving results")
        self._save_results(triplets)
        
        # Log final token usage statistics
        total_usage = {
            "chain_cleaning": chain_cleaning_usage,
            "negative_generation": negative_gen_usage
        }
        logger.info("Final token usage statistics:")
        logger.info(f"Chain cleaning: {chain_cleaning_usage}")
        logger.info(f"Negative generation: {negative_gen_usage}")
        
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
        
    def _save_results(self, triplets: Dict[str, Dict[str, Any]]) -> None:
        """
        Save the final triplets to files
        
        Args:
            triplets: Dictionary of triplets with scores
        """
        logger.info("Saving final triplets")
        
        # Save in ColBERT format (query, positive, negative)
        colbert_data = []
        
        for query, data in triplets.items():
            for factsheet, score in data["positives"]:
                colbert_data.append({
                    "query": query,
                    "factsheet": factsheet,
                    "score": score,
                    "is_positive": True
                })
                
            for factsheet, score in data["negatives"]:
                colbert_data.append({
                    "query": query,
                    "factsheet": factsheet,
                    "score": score,
                    "is_positive": False
                })
        
        # Save as CSV for ColBERT training
        colbert_training_path = self.colbert_training_dir / "colbert_training_data.csv"
        save_csv(colbert_data, str(colbert_training_path))
        
        # Get overlap statistics
        overlap_stats = self.negative_generator.overlap_manager.get_overlap_stats()
        
        # Save summary statistics
        total_positives = sum(len(data["positives"]) for data in triplets.values())
        total_negatives = sum(len(data["negatives"]) for data in triplets.values())
        
        stats = {
            "total_queries": len(triplets),
            "total_positives": total_positives,
            "total_negatives": total_negatives,
            "total_triplets": total_positives + total_negatives,
            "queries_with_negatives": sum(1 for data in triplets.values() if data["negatives"]),
            "queries_with_positives": sum(1 for data in triplets.values() if data["positives"]),
            "overlap_stats": overlap_stats
        }
        
        stats_path = self.colbert_training_dir / "stats.json"
        save_json(stats, str(stats_path))
        
        logger.info(f"Saved {stats['total_triplets']} triplets for {stats['total_queries']} queries")
        logger.info(f"Statistics: {stats}")
    
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