from typing import List, Dict, Any
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str):
        """
        Initialize the reranker model
        
        Args:
            model_name: HuggingFace model name for the reranker
        """
        self.model_name = model_name
        logger.info(f"Initializing reranker model: {model_name}")
        
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            logger.info(f"Successfully loaded model and tokenizer")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def score(self, query: str, document: str) -> float:
        """
        Score a single query-document pair
        
        Args:
            query: The query string
            document: The document string
            
        Returns:
            Score between 0 and 1
        """
        logger.debug(f"Scoring query: {query[:30]}... with document: {document[:30]}...")
        
        try:
            inputs = self.tokenizer(
                [query], 
                [document], 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                scores = self.model(**inputs).logits
                
            # For binary classification models, apply sigmoid to convert to [0, 1]
            if scores.shape[1] == 1:
                score = torch.sigmoid(scores).item()
            # For models with (negative, positive) outputs
            else:
                # Apply softmax and take the score for the positive class
                score = torch.softmax(scores, dim=1)[0, 1].item()
                
            logger.debug(f"Score: {score}")
            return score
        except Exception as e:
            logger.error(f"Error scoring: {e}")
            raise
            
    def score_batch(self, queries: List[str], documents: List[str]) -> List[float]:
        """
        Score multiple query-document pairs
        
        Args:
            queries: List of query strings
            documents: List of document strings
            
        Returns:
            List of scores between 0 and 1
        """
        logger.info(f"Batch scoring {len(queries)} query-document pairs")
        
        if len(queries) != len(documents):
            error_msg = f"Number of queries ({len(queries)}) must match number of documents ({len(documents)})"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            inputs = self.tokenizer(
                queries, 
                documents, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            batch_size = 32
            all_scores = []
            
            # Process in batches to avoid OOM errors
            for i in range(0, len(queries), batch_size):
                batch_inputs = {
                    k: v[i:i+batch_size] for k, v in inputs.items()
                }
                
                with torch.no_grad():
                    batch_scores = self.model(**batch_inputs).logits
                    
                # For binary classification models
                if batch_scores.shape[1] == 1:
                    batch_scores = torch.sigmoid(batch_scores).squeeze().tolist()
                # For models with (negative, positive) outputs
                else:
                    batch_scores = torch.softmax(batch_scores, dim=1)[:, 1].tolist()
                
                # Ensure the result is always a list
                if not isinstance(batch_scores, list):
                    batch_scores = [batch_scores]
                    
                all_scores.extend(batch_scores)
            
            logger.info(f"Successfully scored {len(all_scores)} pairs")
            return all_scores
        except Exception as e:
            logger.error(f"Error batch scoring: {e}")
            raise 