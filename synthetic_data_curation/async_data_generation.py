import os
import json
import random
import asyncio
import numpy as np
from typing import List, Dict, Tuple, Set, Any
from tqdm.asyncio import tqdm as async_tqdm
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
import aiohttp
from functools import partial

# Assume old_search.py exists in the same directory
import old_search

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
ASPECTS = [
    "Industry", 
    "Customer Segment", 
    # "Products & Services", 
    "Business Model", 
    "Technology Used", 
    "Revenue Model"
]

# Set up rate limiting
MAX_CONCURRENT_REQUESTS = 10  # Adjust based on your OpenAI rate limits
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class AsyncSyntheticDataGenerator:
    def __init__(self, model_name='e5_production_model', data_dir='data'):
        """Initialize the data generator with the embedding model."""
        self.embedding_model = SentenceTransformer(model_name)
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_openai_request(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> Any:
        """Make an asynchronous request to OpenAI API."""
        async with request_semaphore:
            # Use the OpenAI client with async support
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response
    
    async def generate_aspect_queries(self, aspect: str, num_queries: int = 100) -> List[str]:
        """Generate top queries for a specific aspect using GPT model asynchronously."""
        prompt = f"""
        Generate {num_queries} realistic search queries that users might use when looking for information 
        about a company's {aspect}. These queries will be used for training a retrieval model.
        
        Return the queries as a JSON array of strings.
        """
        
        response = await self._make_openai_request(
            model="gpt-4o", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        
        queries = json.loads(response.choices[0].message.content)
        return queries
    
    async def fetch_search_results(self, query: str, top_k: int = 30) -> List[str]:
        """Fetch top search results from current search engine (made async-compatible)."""
        # Assuming the old_search.search function is not async-aware, wrap it
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(old_search.search, query, top_k=top_k))
    
    async def filter_single_passage(self, query: str, passage: str) -> bool:
        """Filter a single passage to determine if it's positive or negative."""
        prompt = f"""
        Query: {query}
        Passage: {passage}
        
        Is this passage relevant to the query? The passage should directly answer or provide 
        information related to the query to be considered relevant.
        
        Answer only with 'YES' or 'NO'.
        """
        
        response = await self._make_openai_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        
        answer = response.choices[0].message.content.strip().upper()
        return answer == "YES"
    
    async def filter_negative_passages(self, query: str, passages: List[str]) -> Tuple[List[str], List[str]]:
        """Use LLM to classify passages as positive or negative for the query asynchronously."""
        tasks = [self.filter_single_passage(query, passage) for passage in passages]
        results = await async_tqdm.gather(*tasks, desc=f"Filtering passages for '{query}'")
        
        positives = [passage for passage, is_positive in zip(passages, results) if is_positive]
        negatives = [passage for passage, is_positive in zip(passages, results) if not is_positive]
        
        return positives, negatives
    
    async def generate_nearest_queries(self, query: str, aspect_queries: Dict[str, List[str]], k: int = 10) -> Dict[str, List[str]]:
        """Find k nearest queries from each aspect using embedding similarity."""
        # Run embedding computation in a threadpool as it's CPU-bound
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(None, 
                                                     lambda: self.embedding_model.encode(query))
        nearest_queries = {}
        
        for aspect, queries in aspect_queries.items():
            if not queries:
                continue
            
            # Run embedding computation in a threadpool
            query_embeddings = await loop.run_in_executor(None, 
                                                         lambda: self.embedding_model.encode(queries))
            
            similarities = np.dot(query_embeddings, query_embedding) / (
                np.linalg.norm(query_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get indices of top k most similar queries
            top_indices = np.argsort(similarities)[-k:]
            nearest_queries[aspect] = [queries[i] for i in top_indices]
            
        return nearest_queries
    
    async def filter_single_negative_query(self, query: str, candidate_query: str) -> bool:
        """Filter a single candidate negative query."""
        prompt = f"""
        Original Query: {query}
        Candidate Query: {candidate_query}
        
        Are these queries asking for fundamentally different information? If they would likely 
        retrieve the same type of content, they're similar. If they would retrieve different 
        content, they're different.
        
        Answer only with 'DIFFERENT' or 'SIMILAR'.
        """
        
        response = await self._make_openai_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        
        answer = response.choices[0].message.content.strip().upper()
        return answer == "DIFFERENT"
    
    async def filter_negative_queries(self, query: str, candidate_queries: List[str]) -> List[str]:
        """Use LLM to filter candidate negative queries asynchronously."""
        tasks = [self.filter_single_negative_query(query, candidate) for candidate in candidate_queries]
        results = await asyncio.gather(*tasks)
        
        negatives = [candidate for candidate, is_negative in zip(candidate_queries, results) if is_negative]
        return negatives
    
    def create_triplets(self, 
                        query: str, 
                        positives: List[str], 
                        negatives: List[str],
                        negative_queries: List[str],
                        negative_passages_for_negative_queries: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
        """Create triplets from the query, positives, and negatives."""
        triplets = []
        
        # If we have no positives, we can't create triplets
        if not positives:
            return triplets
            
        # Create triplets with negative passages for original query
        for positive in positives:
            for negative in negatives:
                triplets.append((query, positive, negative))
                
        # Create triplets with passages from negative queries
        for negative_query in negative_queries:
            negative_passages = negative_passages_for_negative_queries.get(negative_query, [])
            for positive in positives:
                for negative_passage in negative_passages:
                    triplets.append((query, positive, negative_passage))
                    
        return triplets
    
    async def process_query(self, aspect: str, query: str, aspect_queries: Dict[str, List[str]], top_k_search: int) -> List[Tuple[str, str, str]]:
        """Process a single query to generate triplets."""
        # Fetch search results
        search_results = await self.fetch_search_results(query, top_k_search)
        
        # Filter into positive and negative passages
        positives, negatives = await self.filter_negative_passages(query, search_results)
        
        # Get nearest queries from each aspect
        nearest_queries_by_aspect = await self.generate_nearest_queries(query, aspect_queries)
        
        # Get intra-aspect negatives (from same aspect)
        intra_aspect_candidates = nearest_queries_by_aspect.get(aspect, [])
        intra_aspect_negatives = await self.filter_negative_queries(query, intra_aspect_candidates)
        
        # Get inter-aspect negatives (from other aspects)
        inter_aspect_candidates = []
        for other_aspect, queries_list in nearest_queries_by_aspect.items():
            if other_aspect != aspect:
                inter_aspect_candidates.extend(queries_list)
                
        inter_aspect_negatives = await self.filter_negative_queries(query, inter_aspect_candidates)
        
        # Combine all negative queries
        negative_queries = intra_aspect_negatives + inter_aspect_negatives
        
        # Get passages for negative queries (in parallel)
        negative_passages_tasks = [self.fetch_search_results(neg_query, top_k=5) for neg_query in negative_queries]
        negative_passages_results = await asyncio.gather(*negative_passages_tasks)
        
        negative_passages_for_negative_queries = {
            neg_query: passages for neg_query, passages in zip(negative_queries, negative_passages_results)
        }
        
        # Create triplets
        triplets = self.create_triplets(
            query, 
            positives, 
            negatives, 
            negative_queries, 
            negative_passages_for_negative_queries
        )
        
        return triplets
    
    async def curate_dataset(self, queries_per_aspect: int = 100, top_k_search: int = 30):
        """Generate the complete dataset of triplets asynchronously."""
        aspect_queries = {}
        all_triplets = []
        
        # Step 1: Generate queries for each aspect (can be parallelized)
        aspect_query_tasks = [self.generate_aspect_queries(aspect, queries_per_aspect) for aspect in ASPECTS]
        aspect_query_results = await asyncio.gather(*aspect_query_tasks)
        
        # Combine results
        for aspect, queries in zip(ASPECTS, aspect_query_results):
            aspect_queries[aspect] = queries
            print(f"Generated {len(queries)} queries for aspect: {aspect}")
            
            # Save queries for each aspect
            with open(f"{self.data_dir}/{aspect.lower().replace(' ', '_')}_queries.json", 'w') as f:
                json.dump(queries, f)
        
        # Step 2: Process each aspect
        for aspect, queries in aspect_queries.items():
            aspect_triplets = []
            
            # Process queries in parallel (with batching to control concurrency)
            batch_size = 5  # Adjust based on your rate limits and resources
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i+batch_size]
                
                # Process this batch of queries in parallel
                query_tasks = [
                    self.process_query(aspect, query, aspect_queries, top_k_search) 
                    for query in batch_queries
                ]
                batch_results = await asyncio.gather(*query_tasks)
                
                # Collect triplets from all queries in this batch
                for triplets in batch_results:
                    aspect_triplets.extend(triplets)
                
                print(f"Processed {min(i+batch_size, len(queries))}/{len(queries)} queries for {aspect}")
            
            # Save triplets for this aspect
            print(f"Generated {len(aspect_triplets)} triplets for {aspect}")
            with open(f"{self.data_dir}/{aspect.lower().replace(' ', '_')}_triplets.json", 'w') as f:
                json.dump(aspect_triplets, f)
                
            all_triplets.extend(aspect_triplets)
        
        # Save all triplets
        print(f"Total triplets generated: {len(all_triplets)}")
        with open(f"{self.data_dir}/all_triplets.json", 'w') as f:
            json.dump(all_triplets, f)
            
        return all_triplets


# Script to run the async data generation
async def main(queries_per_aspect=100, top_k_search=30):
    async with AsyncSyntheticDataGenerator() as generator:
        print("Starting asynchronous synthetic data generation process...")
        triplets = await generator.curate_dataset(
            queries_per_aspect=queries_per_aspect,
            top_k_search=top_k_search
        )
        print(f"Data generation complete! Generated {len(triplets)} triplets.")
        print(f"Data saved to the 'data' directory.")


if __name__ == "__main__":
    # Parse command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic data for aspect-based retrieval')
    parser.add_argument('--queries', type=int, default=100, help='Number of queries per aspect')
    parser.add_argument('--results', type=int, default=30, help='Number of search results per query')
    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(main(queries_per_aspect=args.queries, top_k_search=args.results)) 