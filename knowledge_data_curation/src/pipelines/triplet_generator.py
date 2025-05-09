from typing import Dict, List, Tuple, Any
import logging
from knowledge_data_curation.src.models.reranker import Reranker

logger = logging.getLogger(__name__)

class TripletGenerator:
    def __init__(self, config: Dict):
        """
        Initialize the triplet generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scoring_strategy = config["scoring"]["strategy"]
        
        if self.scoring_strategy == "reranker":
            self.reranker = Reranker(config["scoring"]["reranker_model"])
            logger.info(f"Using reranker strategy with model {config['scoring']['reranker_model']}")
        else:
            logger.info(f"Using binary scoring strategy")
    
    def generate_binary_triplets(
        self,
        query_factsheet_positives: Dict[str, List[str]],
        query_factsheet_negatives: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate binary triplets (query, factsheet, score)
        where score is 1 for positives and 0 for negatives
        
        Args:
            query_factsheet_positives: Dictionary mapping queries to positive factsheets
            query_factsheet_negatives: Dictionary mapping queries to negative factsheets
            
        Returns:
            Dictionary of triplets with binary scores
        """
        logger.info("Generating binary triplets")
        results = {}
        
        for query, positive_factsheets in query_factsheet_positives.items():
            results[query] = {
                "positives": [(factsheet, 1.0) for factsheet in positive_factsheets],
                "negatives": [(factsheet, 0.0) for factsheet in query_factsheet_negatives.get(query, [])]
            }
            
        logger.info(f"Generated binary triplets for {len(results)} queries")
        return results
        
    def generate_reranker_triplets(
        self,
        query_factsheet_positives: Dict[str, List[str]],
        query_factsheet_negatives: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate scored triplets (query, factsheet, score)
        where score is determined by a reranker model
        
        Args:
            query_factsheet_positives: Dictionary mapping queries to positive factsheets
            query_factsheet_negatives: Dictionary mapping queries to negative factsheets
            
        Returns:
            Dictionary of triplets with reranker scores
        """
        logger.info("Generating reranker triplets")
        results = {}
        
        for query, positive_factsheets in query_factsheet_positives.items():
            logger.debug(f"Scoring positives for query: {query}")
            
            # Score positives
            positive_scores = []
            if positive_factsheets:
                positive_scores = self.reranker.score_batch(
                    queries=[query] * len(positive_factsheets),
                    documents=positive_factsheets
                )
            
            # Score negatives
            negatives = query_factsheet_negatives.get(query, [])
            negative_scores = []
            if negatives:
                logger.debug(f"Scoring negatives for query: {query}")
                negative_scores = self.reranker.score_batch(
                    queries=[query] * len(negatives),
                    documents=negatives
                )
            
            results[query] = {
                "positives": list(zip(positive_factsheets, positive_scores)),
                "negatives": list(zip(negatives, negative_scores))
            }
            
        logger.info(f"Generated reranker triplets for {len(results)} queries")
        return results
        
    def generate_triplets(
        self,
        query_factsheet_positives: Dict[str, List[str]],
        query_factsheet_negatives: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate triplets using the configured scoring strategy
        
        Args:
            query_factsheet_positives: Dictionary mapping queries to positive factsheets
            query_factsheet_negatives: Dictionary mapping queries to negative factsheets
            
        Returns:
            Dictionary of triplets with scores
        """
        logger.info(f"Generating triplets using {self.scoring_strategy} strategy")
        
        if self.scoring_strategy == "binary":
            return self.generate_binary_triplets(
                query_factsheet_positives, 
                query_factsheet_negatives
            )
        elif self.scoring_strategy == "reranker":
            return self.generate_reranker_triplets(
                query_factsheet_positives,
                query_factsheet_negatives
            )
        else:
            error_msg = f"Unknown scoring strategy: {self.scoring_strategy}"
            logger.error(error_msg)
            raise ValueError(error_msg) 