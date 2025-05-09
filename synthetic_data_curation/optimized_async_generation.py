import os
import json
import random
import asyncio
import numpy as np
from typing import List, Dict, Tuple, Set, Any
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
import aiohttp

# Assume old_search.py exists in the same directory
import old_search

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
ASPECTS = [
    "Industry", 
    "Customer Segment", 
    "Products & Services", 
    "Business Model", 
    "Technology Used", 
    "Revenue Model"
]

# Set up rate limiting
MAX_CONCURRENT_REQUESTS = 10  # Adjust based on your OpenAI rate limits
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class OptimizedAsyncDataGenerator:
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
    
    def fetch_search_results_batch(self, queries: List[str], top_k: int = 30) -> Dict[str, List[str]]:
        """Fetch top search results for multiple queries at once.
        
        Returns a dictionary mapping each query to its search results.
        """
        results = {}
        for query in tqdm(queries, desc="Fetching search results"):
            results[query] = old_search.search(query, top_k=top_k)
        return results
    
    async def filter_passages_batch(self, query_passage_pairs: List[Tuple[str, str]]) -> List[bool]:
        """Filter multiple query-passage pairs in batches."""
        # Group by batches to optimize API calls (we can send multiple passage assessments in one prompt)
        BATCH_SIZE = 5  # Number of passages to evaluate in one API call
        all_results = []
        
        for i in range(0, len(query_passage_pairs), BATCH_SIZE):
            batch = query_passage_pairs[i:i+BATCH_SIZE]
            
            # Create a prompt that evaluates multiple passages at once
            prompt = "For each query-passage pair, determine if the passage is relevant to the query. Answer with YES or NO for each pair.\n\n"
            
            for idx, (query, passage) in enumerate(batch):
                prompt += f"Pair {idx+1}:\nQuery: {query}\nPassage: {passage}\n\n"
            
            prompt += "Provide your answers as a JSON array of 'YES' or 'NO' values, one for each pair."
            
            try:
                response = await self._make_openai_request(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                
                # Parse the response
                answer_text = response.choices[0].message.content.strip()
                
                # Handle different response formats
                if answer_text.startswith('[') and answer_text.endswith(']'):
                    # JSON array format
                    answers = json.loads(answer_text)
                else:
                    # Extract answers from text
                    answers = []
                    for line in answer_text.split('\n'):
                        if 'YES' in line.upper():
                            answers.append('YES')
                        elif 'NO' in line.upper():
                            answers.append('NO')
                
                # Ensure we have the right number of answers
                if len(answers) != len(batch):
                    print(f"Warning: Expected {len(batch)} answers but got {len(answers)}. Assuming all NO.")
                    answers = ['NO'] * len(batch)
                
                batch_results = [answer.upper() == 'YES' for answer in answers]
                all_results.extend(batch_results)
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Assume all negative in case of error
                all_results.extend([False] * len(batch))
        
        return all_results
    
    async def filter_multiple_queries(self, query_pairs: List[Tuple[str, str]]) -> List[bool]:
        """Filter multiple original-candidate query pairs in batches."""
        BATCH_SIZE = 5  # Number of query pairs to evaluate in one API call
        all_results = []
        
        for i in range(0, len(query_pairs), BATCH_SIZE):
            batch = query_pairs[i:i+BATCH_SIZE]
            
            prompt = "For each pair of queries, determine if they are asking for fundamentally different information. Answer with DIFFERENT or SIMILAR for each pair.\n\n"
            
            for idx, (original, candidate) in enumerate(batch):
                prompt += f"Pair {idx+1}:\nOriginal Query: {original}\nCandidate Query: {candidate}\n\n"
            
            prompt += "Provide your answers as a JSON array of 'DIFFERENT' or 'SIMILAR' values, one for each pair."
            
            try:
                response = await self._make_openai_request(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                
                # Parse the response
                answer_text = response.choices[0].message.content.strip()
                
                # Handle different response formats
                if answer_text.startswith('[') and answer_text.endswith(']'):
                    # JSON array format
                    answers = json.loads(answer_text)
                else:
                    # Extract answers from text
                    answers = []
                    for line in answer_text.split('\n'):
                        if 'DIFFERENT' in line.upper():
                            answers.append('DIFFERENT')
                        elif 'SIMILAR' in line.upper():
                            answers.append('SIMILAR')
                
                # Ensure we have the right number of answers
                if len(answers) != len(batch):
                    print(f"Warning: Expected {len(batch)} answers but got {len(answers)}. Assuming all SIMILAR.")
                    answers = ['SIMILAR'] * len(batch)
                
                batch_results = [answer.upper() == 'DIFFERENT' for answer in answers]
                all_results.extend(batch_results)
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Assume all similar in case of error
                all_results.extend([False] * len(batch))
        
        return all_results
    
    def generate_nearest_queries(self, query: str, aspect_queries: Dict[str, List[str]], k: int = 10) -> Dict[str, List[str]]:
        """Find k nearest queries from each aspect using embedding similarity - using batch processing."""
        query_embedding = self.embedding_model.encode(query)
        nearest_queries = {}
        
        for aspect, queries in aspect_queries.items():
            if not queries:
                continue
            
            # Batch encode all queries at once
            query_embeddings = self.embedding_model.encode(queries)
            
            similarities = np.dot(query_embeddings, query_embedding) / (
                np.linalg.norm(query_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get indices of top k most similar queries
            top_indices = np.argsort(similarities)[-k:]
            nearest_queries[aspect] = [queries[i] for i in top_indices]
            
        return nearest_queries
    
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
    
    async def process_query_batch(self, aspect: str, queries: List[str], aspect_queries: Dict[str, List[str]], top_k_search: int) -> List[Tuple[str, str, str]]:
        """Process a batch of queries to generate triplets."""
        all_triplets = []
        
        # 1. Fetch search results for all queries at once
        print(f"Fetching search results for {len(queries)} queries...")
        search_results_dict = self.fetch_search_results_batch(queries, top_k_search)
        
        # 2. Prepare query-passage pairs for filtering
        all_query_passage_pairs = []
        query_passage_map = {}  # To track which pairs belong to which query
        
        for query, passages in search_results_dict.items():
            query_passage_pairs = [(query, passage) for passage in passages]
            all_query_passage_pairs.extend(query_passage_pairs)
            query_passage_map[query] = (len(all_query_passage_pairs) - len(passages), len(all_query_passage_pairs))
        
        # 3. Filter all passages in batches
        print(f"Filtering {len(all_query_passage_pairs)} passages...")
        all_passage_results = await self.filter_passages_batch(all_query_passage_pairs)
        
        # 4. Organize passages by query
        query_positives = {}
        query_negatives = {}
        
        for query, (start_idx, end_idx) in query_passage_map.items():
            passages = [all_query_passage_pairs[i][1] for i in range(start_idx, end_idx)]
            results = all_passage_results[start_idx:end_idx]
            
            positives = [passage for passage, is_positive in zip(passages, results) if is_positive]
            negatives = [passage for passage, is_positive in zip(passages, results) if not is_positive]
            
            query_positives[query] = positives
            query_negatives[query] = negatives
        
        # 5. For each query, find nearest queries from other aspects
        print(f"Finding nearest queries for {len(queries)} queries...")
        query_nearest = {}
        
        for query in queries:
            # Get nearest queries using batch processing
            query_nearest[query] = self.generate_nearest_queries(query, aspect_queries)
        
        # 6. Prepare query pairs for filtering
        all_query_pairs = []
        query_pair_map = {}  # To track which pairs belong to which query
        
        for query in queries:
            # Get intra-aspect candidates
            intra_aspect_candidates = query_nearest[query].get(aspect, [])
            
            # Get inter-aspect candidates
            inter_aspect_candidates = []
            for other_aspect, queries_list in query_nearest[query].items():
                if other_aspect != aspect:
                    inter_aspect_candidates.extend(queries_list)
            
            # Create all query pairs
            query_pairs = [(query, candidate) for candidate in intra_aspect_candidates + inter_aspect_candidates]
            start_idx = len(all_query_pairs)
            all_query_pairs.extend(query_pairs)
            query_pair_map[query] = (start_idx, len(all_query_pairs))
        
        # 7. Filter all query pairs in batches
        print(f"Filtering {len(all_query_pairs)} query pairs...")
        all_query_pair_results = await self.filter_multiple_queries(all_query_pairs)
        
        # 8. Organize negative queries by query
        query_negative_queries = {}
        
        for query, (start_idx, end_idx) in query_pair_map.items():
            candidates = [all_query_pairs[i][1] for i in range(start_idx, end_idx)]
            results = all_query_pair_results[start_idx:end_idx]
            
            negative_queries = [candidate for candidate, is_negative in zip(candidates, results) if is_negative]
            query_negative_queries[query] = negative_queries
        
        # 9. Fetch passages for all negative queries
        all_negative_queries = []
        for negatives in query_negative_queries.values():
            all_negative_queries.extend(negatives)
        
        # Remove duplicates while preserving order
        all_negative_queries = list(dict.fromkeys(all_negative_queries))
        
        print(f"Fetching passages for {len(all_negative_queries)} negative queries...")
        negative_query_passages = self.fetch_search_results_batch(all_negative_queries, top_k=5)
        
        # 10. Create triplets for each query
        for query in queries:
            positives = query_positives.get(query, [])
            negatives = query_negatives.get(query, [])
            negative_queries = query_negative_queries.get(query, [])
            
            negative_passages_for_negative_queries = {
                neg_query: negative_query_passages.get(neg_query, [])
                for neg_query in negative_queries
            }
            
            triplets = self.create_triplets(
                query,
                positives,
                negatives,
                negative_queries,
                negative_passages_for_negative_queries
            )
            
            all_triplets.extend(triplets)
        
        return all_triplets
    
    async def curate_dataset(self, queries_per_aspect: int = 100, top_k_search: int = 30):
        """Generate the complete dataset of triplets using batch processing."""
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
            
            # Process queries in batches
            batch_size = 10  # Adjust based on your resources
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(queries) + batch_size - 1)//batch_size} for {aspect}")
                
                # Process this batch of queries
                batch_triplets = await self.process_query_batch(aspect, batch_queries, aspect_queries, top_k_search)
                aspect_triplets.extend(batch_triplets)
                
                print(f"Generated {len(batch_triplets)} triplets from this batch")
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


# Script to run the optimized async data generation
async def main(queries_per_aspect=100, top_k_search=30):
    async with OptimizedAsyncDataGenerator() as generator:
        print("Starting optimized asynchronous synthetic data generation process...")
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