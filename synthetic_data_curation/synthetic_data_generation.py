import os
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

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

class SyntheticDataGenerator:
    def __init__(self, model_name='e5_production_model', data_dir='data'):
        """Initialize the data generator with the embedding model."""
        self.embedding_model = SentenceTransformer(model_name)
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def generate_aspect_queries(self, aspect: str, num_queries: int = 100) -> List[str]:
        """Generate top queries for a specific aspect using GPT model."""
        prompt = f"""
        Generate {num_queries} realistic search queries that users might use when looking for information 
        about a company's {aspect}. These queries will be used for training a retrieval model.
        
        Return the queries as a JSON array of strings.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4o", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        
        queries = json.loads(response.choices[0].message.content)
        return queries
    
    def fetch_search_results(self, query: str, top_k: int = 30) -> List[str]:
        """Fetch top search results from current search engine."""
        return old_search.search(query, top_k=top_k)
    
    def filter_negative_passages(self, query: str, passages: List[str]) -> Tuple[List[str], List[str]]:
        """Use LLM to classify passages as positive or negative for the query."""
        positives = []
        negatives = []
        
        for passage in tqdm(passages, desc=f"Filtering passages for '{query}'"):
            prompt = f"""
            Query: {query}
            Passage: {passage}
            
            Is this passage relevant to the query? The passage should directly answer or provide 
            information related to the query to be considered relevant.
            
            Answer only with 'YES' or 'NO'.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            answer = response.choices[0].message.content.strip().upper()
            if answer == "YES":
                positives.append(passage)
            else:
                negatives.append(passage)
                
        return positives, negatives
    
    def generate_nearest_queries(self, query: str, aspect_queries: Dict[str, List[str]], k: int = 10) -> Dict[str, List[str]]:
        """Find k nearest queries from each aspect using embedding similarity."""
        query_embedding = self.embedding_model.encode(query)
        nearest_queries = {}
        
        for aspect, queries in aspect_queries.items():
            if not queries:
                continue
                
            query_embeddings = self.embedding_model.encode(queries)
            similarities = np.dot(query_embeddings, query_embedding) / (
                np.linalg.norm(query_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get indices of top k most similar queries
            top_indices = np.argsort(similarities)[-k:]
            nearest_queries[aspect] = [queries[i] for i in top_indices]
            
        return nearest_queries
    
    def filter_negative_queries(self, query: str, candidate_queries: List[str]) -> List[str]:
        """Use LLM to filter candidate negative queries."""
        negatives = []
        
        for candidate in candidate_queries:
            prompt = f"""
            Original Query: {query}
            Candidate Query: {candidate}
            
            Are these queries asking for fundamentally different information? If they would likely 
            retrieve the same type of content, they're similar. If they would retrieve different 
            content, they're different.
            
            Answer only with 'DIFFERENT' or 'SIMILAR'.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            answer = response.choices[0].message.content.strip().upper()
            if answer == "DIFFERENT":
                negatives.append(candidate)
                
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
    
    def curate_dataset(self, queries_per_aspect: int = 100, top_k_search: int = 30):
        """Generate the complete dataset of triplets."""
        aspect_queries = {}
        all_triplets = []
        
        # Step 1: Generate queries for each aspect
        for aspect in ASPECTS:
            print(f"Generating queries for aspect: {aspect}")
            aspect_queries[aspect] = self.generate_aspect_queries(aspect, queries_per_aspect)
            
            # Save queries for each aspect
            with open(f"{self.data_dir}/{aspect.lower().replace(' ', '_')}_queries.json", 'w') as f:
                json.dump(aspect_queries[aspect], f)
                
        # Step 2: Process each aspect
        for aspect, queries in aspect_queries.items():
            aspect_triplets = []
            
            for query in tqdm(queries, desc=f"Processing {aspect} queries"):
                # Fetch search results
                search_results = self.fetch_search_results(query, top_k_search)
                
                # Filter into positive and negative passages
                positives, negatives = self.filter_negative_passages(query, search_results)
                
                # Get nearest queries from each aspect
                nearest_queries_by_aspect = self.generate_nearest_queries(query, aspect_queries)
                
                # Get intra-aspect negatives (from same aspect)
                intra_aspect_candidates = nearest_queries_by_aspect.get(aspect, [])
                intra_aspect_negatives = self.filter_negative_queries(query, intra_aspect_candidates)
                
                # Get inter-aspect negatives (from other aspects)
                inter_aspect_candidates = []
                for other_aspect, queries_list in nearest_queries_by_aspect.items():
                    if other_aspect != aspect:
                        inter_aspect_candidates.extend(queries_list)
                        
                inter_aspect_negatives = self.filter_negative_queries(query, inter_aspect_candidates)
                
                # Combine all negative queries
                negative_queries = intra_aspect_negatives + inter_aspect_negatives
                
                # Get passages for negative queries
                negative_passages_for_negative_queries = {}
                for neg_query in negative_queries:
                    results = self.fetch_search_results(neg_query, top_k=5)  # Just get a few results
                    negative_passages_for_negative_queries[neg_query] = results
                
                # Create triplets
                triplets = self.create_triplets(
                    query, 
                    positives, 
                    negatives, 
                    negative_queries, 
                    negative_passages_for_negative_queries
                )
                
                aspect_triplets.extend(triplets)
            
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

# Usage example
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    triplets = generator.curate_dataset(queries_per_aspect=10, top_k_search=5)  # Small sample for testing
    print(f"Generated {len(triplets)} triplets") 