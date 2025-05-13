from typing import List, Dict, Any
import logging
import openai
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str, max_threads: int = 8, batch_size: int = 100, rate_limit_delay: float = 0.001, enable_parallel: bool = False):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the OpenAI embedding model
            max_threads: Maximum number of parallel threads
            batch_size: Size of batches for parallel processing
            rate_limit_delay: Delay between API calls to avoid rate limits
            enable_parallel: Whether to enable parallel processing
        """
        self.model_name = model_name
        self.max_threads = max_threads
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.enable_parallel = enable_parallel
        self.total_embeddings = 0
        logger.info(f"Initializing embedding model: {model_name} with parallel={enable_parallel}, max_threads={max_threads}, batch_size={batch_size}")
        
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
            
    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """
        Process a batch of texts to get embeddings
        
        Args:
            batch: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Add rate limiting delay
            time.sleep(self.rate_limit_delay)
            
            response = openai.embeddings.create(
                model=self.model_name,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Successfully processed batch of {len(batch)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
            
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts, optionally using parallel processing
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        mode = "parallel" if self.enable_parallel else "sequential"
        logger.info(f"[{mode}] Getting batch embeddings for {len(texts)} texts")
        start_time = time.time()
        
        try:
            if not self.enable_parallel:
                # Process sequentially in batches
                all_embeddings = []
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    batch_start = time.time()
                    embeddings = self._process_batch(batch)
                    all_embeddings.extend(embeddings)
                    batch_time = time.time() - batch_start
                    self.total_embeddings += len(batch)
                    if (i + 1) % (self.batch_size * 10) == 0:
                        logger.info(f"[{mode}] Processed {i + 1}/{len(texts)} texts. Batch time: {batch_time:.2f}s. Total embeddings: {self.total_embeddings}")
                total_time = time.time() - start_time
                logger.info(f"[{mode}] Completed sequential processing of {len(texts)} texts in {total_time:.2f}s")
                return all_embeddings
            
            # Process in parallel if enabled
            batches = []
            current_batch = []
            
            for text in texts:
                current_batch.append(text)
                if len(current_batch) >= self.batch_size:
                    batches.append(current_batch)
                    current_batch = []
            
            if current_batch:
                batches.append(current_batch)
                
            logger.info(f"[{mode}] Created {len(batches)} batches for parallel processing")
            
            # Process batches in parallel
            all_embeddings = []
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_to_batch = {
                    executor.submit(self._process_batch, batch): i 
                    for i, batch in enumerate(batches)
                }
                
                completed_batches = 0
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_start = time.time()
                        batch_embeddings = future.result()
                        batch_time = time.time() - batch_start
                        all_embeddings.extend(batch_embeddings)
                        completed_batches += 1
                        self.total_embeddings += len(batch_embeddings)
                        if completed_batches % 10 == 0:
                            logger.info(f"[{mode}] Completed {completed_batches}/{len(batches)} batches. Batch time: {batch_time:.2f}s. Total embeddings: {self.total_embeddings}")
                    except Exception as e:
                        logger.error(f"[{mode}] Batch {batch_idx} failed: {e}")
                        raise
            
            total_time = time.time() - start_time
            logger.info(f"[{mode}] Successfully generated {len(all_embeddings)} embeddings in {total_time:.2f}s")
            return all_embeddings
        except Exception as e:
            logger.error(f"[{mode}] Error in embedding generation: {e}")
            raise 