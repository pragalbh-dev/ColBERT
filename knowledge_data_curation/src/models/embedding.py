from typing import List, Dict, Any
import logging
import openai
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the OpenAI embedding model
        """
        self.model_name = model_name
        logger.info(f"Initializing embedding model: {model_name}")
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        logger.debug(f"Getting embedding for text: {text[:50]}...")
        
        try:
            response = openai.embeddings.create(
                model=self.model_name,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Successfully generated embedding with {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
            
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Getting batch embeddings for {len(texts)} texts")
        
        try:
            response = openai.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise 