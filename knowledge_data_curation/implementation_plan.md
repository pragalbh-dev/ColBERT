# ColBERT Training Data Curation Pipeline

## Overview

This pipeline transforms industry hierarchy data and company factsheets into training data for ColBERT in the format of (query, positive factsheets) and (query, negative factsheets).

## Directory Structure

```
knowledge_data_curation/
├── config/
│   ├── default.yaml         # Default configuration values
│   └── custom.yaml          # Custom overrides
├── src/
│   ├── data_processors/
│   │   ├── __init__.py
│   │   ├── chain_cleaner.py      # Cleans industry chains using LLM
│   │   ├── negative_generator.py  # Generates negative samples
│   │   └── data_loader.py         # Loads and processes data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embedding.py      # Embedding models for ES indexing
│   │   ├── reranker.py       # Reranker models for scoring
│   │   └── llm_client.py     # Interface to OpenAI models
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── chain_cleaning.py  # Prompts for chain cleaning
│   │   └── negative_sampling.py  # Prompts for negative sampling
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── parallel.py       # Parallel processing utilities
│   │   ├── elasticsearch.py  # ES index utilities
│   │   └── io.py             # I/O utilities
│   └── pipelines/
│       ├── __init__.py
│       ├── main_pipeline.py  # Main orchestration pipeline
│       ├── query_processor.py # Chain cleaning pipeline
│       ├── negative_sampler.py # Negative sampling pipeline
│       └── triplet_generator.py # Generating final triplets
├── output/
│   ├── cleaned_chains/       # Cleaned chain outputs
│   ├── negative_chains/      # Negative chain mappings
│   └── colbert_training/     # Final training data
├── tests/                    # Unit tests
└── scripts/
    ├── run_pipeline.py       # Main execution script
    └── evaluate_results.py   # Evaluation utilities
```

## Configuration System

Create a YAML-based configuration system:

```yaml
# default.yaml example
paths:
  industry_data: "data/industry_df.csv"
  factsheet_data: "data/company_factsheets_df.csv"
  output_dir: "output/"

openai:
  model_chain_cleaner: "gpt-4.1-mini"
  model_negative_generator: "gpt-4.1-mini"
  parallel_threads: 8
  batch_size: 20
  rate_limit: 100

elasticsearch:
  host: "localhost"
  port: 9200
  index_name: "industry_chains"
  embedding_model: "text-embedding-3-small"

sampling:
  hard_negative_count: 5
  soft_negative_count: 5
  nearest_neighbors: 50
  window_size: 10
  samples_per_window: 2

scoring:
  strategy: "binary"  # Options: "binary", "reranker"
  reranker_model: "BAAI/bge-reranker-large"
```

## Component Implementation Details

### 1. Chain Processor (Chain Cleaner)

```python
# Pseudocode for chain_cleaner.py

from typing import Dict, List, Tuple
from src.models.llm_client import OpenAIClient
from src.prompts.chain_cleaning import CHAIN_CLEANING_PROMPT
from src.utils.parallel import batch_process

class ChainCleaner:
    def __init__(self, config: Dict):
        self.config = config
        self.llm_client = OpenAIClient(
            model=config["openai"]["model_chain_cleaner"],
            max_threads=config["openai"]["parallel_threads"]
        )
        
    def clean_chain(self, chain: str) -> str:
        """Clean a single industry chain by removing unnecessary elements."""
        response = self.llm_client.complete(
            prompt=CHAIN_CLEANING_PROMPT.format(chain=chain),
            temperature=0.1
        )
        return self._parse_response(response)
        
    def _parse_response(self, response: str) -> str:
        # Parse LLM response to extract cleaned chain
        pass
        
    def process_chains(self, chains: List[str]) -> Dict[str, str]:
        """Process multiple chains in parallel batches."""
        return batch_process(
            items=chains,
            process_fn=self.clean_chain,
            batch_size=self.config["openai"]["batch_size"]
        )
```

### 2. Negative Sample Generator

```python
# Pseudocode for negative_generator.py

from typing import Dict, List, Tuple
from src.models.llm_client import OpenAIClient
from src.prompts.negative_sampling import NEGATIVE_SAMPLING_PROMPT
from src.utils.elasticsearch import ESClient
from src.utils.parallel import batch_process

class NegativeSampleGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.llm_client = OpenAIClient(
            model=config["openai"]["model_negative_generator"],
            max_threads=config["openai"]["parallel_threads"]
        )
        self.es_client = ESClient(config["elasticsearch"])
        
    def index_chains(self, chains: List[str]) -> None:
        """Index all chains in Elasticsearch."""
        self.es_client.index_documents(chains)
        
    def get_candidate_negatives(self, chain: str) -> Tuple[List[str], List[str]]:
        """
        Get candidate negative chains for a given chain:
        - Hard negative candidates: semantically close but different
        - Soft negative candidates: random chains
        """
        # Get nearest neighbors from ES
        nearest = self.es_client.search(
            query=chain, 
            size=self.config["sampling"]["nearest_neighbors"]
        )
        
        # Sample hard negatives from windows
        hard_candidates = []
        window_size = self.config["sampling"]["window_size"]
        for i in range(0, len(nearest), window_size):
            window = nearest[i:i+window_size]
            samples = random.sample(
                window, 
                min(len(window), self.config["sampling"]["samples_per_window"])
            )
            hard_candidates.extend(samples)
        
        # Sample soft negatives (random chains not in nearest)
        all_chains = self.es_client.get_all_documents()
        non_nearest = [c for c in all_chains if c not in nearest]
        soft_candidates = random.sample(
            non_nearest,
            min(len(non_nearest), self.config["sampling"]["soft_negative_count"])
        )
        
        return hard_candidates, soft_candidates
        
    def select_negatives(
        self, 
        chain: str, 
        hard_candidates: List[str], 
        soft_candidates: List[str]
    ) -> Dict[str, List[str]]:
        """Use LLM to select definite negatives from candidates."""
        response = self.llm_client.complete(
            prompt=NEGATIVE_SAMPLING_PROMPT.format(
                chain=chain,
                hard_candidates=hard_candidates,
                soft_candidates=soft_candidates
            ),
            temperature=0.1
        )
        return self._parse_response(response)
        
    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        # Parse LLM response to extract selected negatives
        pass
        
    def generate_negatives(self, chains: List[str]) -> Dict[str, List[str]]:
        """
        Generate negative chains for each input chain:
        1. Get candidate negatives
        2. Select definite negatives using LLM
        """
        results = {}
        
        for chain in chains:
            hard_candidates, soft_candidates = self.get_candidate_negatives(chain)
            negatives = self.select_negatives(chain, hard_candidates, soft_candidates)
            results[chain] = negatives
            
        return results
```

### 3. Triplet Generator

```python
# Pseudocode for triplet_generator.py

from typing import Dict, List, Tuple, Any
from src.models.reranker import Reranker

class TripletGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.scoring_strategy = config["scoring"]["strategy"]
        
        if self.scoring_strategy == "reranker":
            self.reranker = Reranker(config["scoring"]["reranker_model"])
    
    def generate_binary_triplets(
        self,
        query_factsheet_positives: Dict[str, List[str]],
        query_factsheet_negatives: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate binary triplets (query, factsheet, score)
        where score is 1 for positives and 0 for negatives
        """
        results = {}
        
        for query, positive_factsheets in query_factsheet_positives.items():
            results[query] = {
                "positives": [(factsheet, 1.0) for factsheet in positive_factsheets],
                "negatives": [(factsheet, 0.0) for factsheet in query_factsheet_negatives.get(query, [])]
            }
            
        return results
        
    def generate_reranker_triplets(
        self,
        query_factsheet_positives: Dict[str, List[str]],
        query_factsheet_negatives: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate scored triplets (query, factsheet, score)
        where score is determined by a reranker model
        """
        results = {}
        
        for query, positive_factsheets in query_factsheet_positives.items():
            # Score positives
            positive_scores = self.reranker.score_batch(
                queries=[query] * len(positive_factsheets),
                documents=positive_factsheets
            )
            
            # Score negatives
            negatives = query_factsheet_negatives.get(query, [])
            negative_scores = []
            if negatives:
                negative_scores = self.reranker.score_batch(
                    queries=[query] * len(negatives),
                    documents=negatives
                )
            
            results[query] = {
                "positives": list(zip(positive_factsheets, positive_scores)),
                "negatives": list(zip(negatives, negative_scores))
            }
            
        return results
        
    def generate_triplets(
        self,
        query_factsheet_positives: Dict[str, List[str]],
        query_factsheet_negatives: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate triplets using the configured scoring strategy."""
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
            raise ValueError(f"Unknown scoring strategy: {self.scoring_strategy}")
```

### 4. Main Pipeline

```python
# Pseudocode for main_pipeline.py

from typing import Dict, Any
import pandas as pd
from src.data_processors.chain_cleaner import ChainCleaner
from src.data_processors.negative_generator import NegativeSampleGenerator
from src.pipelines.triplet_generator import TripletGenerator
from src.utils.io import load_dataframe, save_json, save_csv

class ColBERTTrainingPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.chain_cleaner = ChainCleaner(self.config)
        self.negative_generator = NegativeSampleGenerator(self.config)
        self.triplet_generator = TripletGenerator(self.config)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        # Load and merge configs
        pass
        
    def run(self) -> None:
        """Run the complete pipeline."""
        print("Loading data...")
        industry_df = load_dataframe(self.config["paths"]["industry_data"])
        factsheet_df = load_dataframe(self.config["paths"]["factsheet_data"])
        
        # Extract unique chains
        chains = industry_df["industry_chain_hierarchy"].unique().tolist()
        
        print("Cleaning chains...")
        cleaned_chains = self.chain_cleaner.process_chains(chains)
        save_json(cleaned_chains, f"{self.config['paths']['output_dir']}/cleaned_chains.json")
        
        print("Indexing chains for negative sampling...")
        self.negative_generator.index_chains(list(cleaned_chains.values()))
        
        print("Generating negative chains...")
        negative_chains = self.negative_generator.generate_negatives(list(cleaned_chains.values()))
        save_json(negative_chains, f"{self.config['paths']['output_dir']}/negative_chains.json")
        
        print("Generating query-factsheet positives...")
        query_factsheet_positives = self._generate_query_factsheet_positives(
            industry_df, factsheet_df, cleaned_chains
        )
        
        print("Generating query-factsheet negatives...")
        query_factsheet_negatives = self._generate_query_factsheet_negatives(
            industry_df, factsheet_df, cleaned_chains, negative_chains
        )
        
        print("Generating final triplets...")
        triplets = self.triplet_generator.generate_triplets(
            query_factsheet_positives, query_factsheet_negatives
        )
        
        print("Saving results...")
        self._save_results(triplets)
        
        print("Pipeline completed successfully!")
        
    def _generate_query_factsheet_positives(
        self, 
        industry_df: pd.DataFrame, 
        factsheet_df: pd.DataFrame,
        cleaned_chains: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Generate mappings from queries to positive factsheets."""
        query_factsheet_positives = {}
        
        # Merge industry data with factsheet data
        merged = pd.merge(industry_df, factsheet_df, on="company_id")
        
        # For each chain, find all factsheets from companies in that industry
        for original_chain, cleaned_chain in cleaned_chains.items():
            companies = industry_df[industry_df["industry_chain_hierarchy"] == original_chain]["company_id"].unique()
            factsheets = factsheet_df[factsheet_df["company_id"].isin(companies)]["factsheet"].tolist()
            
            query_factsheet_positives[cleaned_chain] = factsheets
            
        return query_factsheet_positives
        
    def _generate_query_factsheet_negatives(
        self,
        industry_df: pd.DataFrame,
        factsheet_df: pd.DataFrame,
        cleaned_chains: Dict[str, str],
        negative_chains: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Generate mappings from queries to negative factsheets."""
        query_factsheet_negatives = {}
        
        # For each chain and its negatives, find factsheets from negative industries
        for original_chain, cleaned_chain in cleaned_chains.items():
            negative_cleaned_chains = negative_chains.get(cleaned_chain, [])
            
            # Find original chains corresponding to negative cleaned chains
            negative_original_chains = [
                oc for oc, cc in cleaned_chains.items() 
                if cc in negative_cleaned_chains
            ]
            
            # Find companies in negative industries
            negative_companies = industry_df[
                industry_df["industry_chain_hierarchy"].isin(negative_original_chains)
            ]["company_id"].unique()
            
            # Find factsheets for negative companies
            negative_factsheets = factsheet_df[
                factsheet_df["company_id"].isin(negative_companies)
            ]["factsheet"].tolist()
            
            query_factsheet_negatives[cleaned_chain] = negative_factsheets
            
        return query_factsheet_negatives
        
    def _save_results(self, triplets: Dict[str, Dict[str, Any]]) -> None:
        """Save the final triplets to files."""
        output_dir = self.config["paths"]["output_dir"]
        
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
        save_csv(colbert_data, f"{output_dir}/colbert_training_data.csv")
```

## Utilities

### Parallel Processing Utility

```python
# Pseudocode for parallel.py

from typing import List, Callable, Any, TypeVar
from concurrent.futures import ThreadPoolExecutor

T = TypeVar('T')
U = TypeVar('U')

def batch_process(
    items: List[T], 
    process_fn: Callable[[T], U], 
    batch_size: int = 10,
    max_workers: int = 8
) -> List[U]:
    """
    Process items in parallel batches with a specified batch size.
    
    Args:
        items: List of items to process
        process_fn: Function to apply to each item
        batch_size: Size of batches to process
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of results
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            futures = {
                executor.submit(process_fn, item): idx 
                for idx, item in enumerate(batch, start=i)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
    
    # Convert dict to list in original order
    return [results[i] for i in range(len(items))]
```

### OpenAI Client

```python
# Pseudocode for llm_client.py

from typing import Dict, List, Any
import time
import openai
from concurrent.futures import ThreadPoolExecutor

class OpenAIClient:
    def __init__(self, model: str, max_threads: int = 8):
        self.model = model
        self.max_threads = max_threads
        self.rate_limit_delay = 0.1  # Seconds between API calls
        
    def complete(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Get a completion from OpenAI API.
        
        Args:
            prompt: The prompt to send
            temperature: Temperature for generation
            
        Returns:
            The model's response text
        """
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        
        return response.choices[0].message.content
        
    def batch_complete(
        self, 
        prompts: List[str], 
        temperature: float = 0.7
    ) -> List[str]:
        """
        Process multiple prompts in parallel with rate limiting.
        
        Args:
            prompts: List of prompts to process
            temperature: Temperature for generation
            
        Returns:
            List of responses in the same order as prompts
        """
        results = {}
        
        def process_prompt(idx, prompt):
            if idx > 0:  # Rate limiting
                time.sleep(self.rate_limit_delay)
            return idx, self.complete(prompt, temperature)
            
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(process_prompt, i, prompt)
                for i, prompt in enumerate(prompts)
            ]
            
            for future in futures:
                idx, result = future.result()
                results[idx] = result
                
        # Return results in correct order
        return [results[i] for i in range(len(prompts))]
```

## Execution Script

```python
# Pseudocode for run_pipeline.py

import argparse
from src.pipelines.main_pipeline import ColBERTTrainingPipeline

def main():
    parser = argparse.ArgumentParser(description="Run ColBERT training data curation pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Run the pipeline
    pipeline = ColBERTTrainingPipeline(args.config)
    pipeline.run()

if __name__ == "__main__":
    main()
```

## Deployment and Scaling

1. **Processing Large Datasets**:
   - The pipeline uses batched processing and parallel execution for efficiency
   - Elasticsearch for efficient similarity search for negative sampling
   - Configurable batch sizes and worker counts for different hardware

2. **Monitoring and Logging**:
   - Add comprehensive logging to track progress
   - Save intermediate results to allow restarts from checkpoints

3. **Extension Points**:
   - All components are decoupled with clear interfaces
   - New scoring strategies can be easily added
   - Different LLM models can be swapped in without changing the pipeline

4. **Testing Strategy**:
   - Unit tests for each component
   - Integration tests for the full pipeline
   - Evaluation scripts to assess the quality of the generated training data 