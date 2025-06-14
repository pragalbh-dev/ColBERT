import pandas as pd
import os
import json
import random
from typing import Dict, List, Tuple, Any
import numpy as np
from collections import defaultdict

class TripletGenerator:
    def __init__(self, root_dir='output'):
        self.root_dir = root_dir
        self.aspect_dir = os.path.join(root_dir, 'aspect_extraction')
        self.cleaned_chains_dir = os.path.join(root_dir, 'cleaned_chains')
        self.negative_chains_dir = os.path.join(root_dir, 'negative_chains')
        self.subchains_dir = os.path.join(root_dir, 'subchains')
        self.synthetic_data_dir = os.path.join(root_dir, 'synthetic_data_cache')
        
        # Configuration parameters from strategy
        self.query_to_positive_ratio = 5  # Each query maps to 5 positive factsheets
        self.negative_to_positive_ratio = 32  # 32 negatives for each positive
        self.hard_to_soft_ratio = 5  # 5:1 ratio of hard to soft negatives
        
        # Data containers
        self.cleaned_chains = {}
        self.subchains = []
        self.negative_chains = {}
        self.negative_subchains = {}
        self.reverse_mappings = {}
        self.all_factsheets = {}  # Unified factsheet dataset
        self.industry_mappings = {}  # company_id -> industry chains
        self.aspect_data = {}
        self.chain_to_companies = {}
        
    def load_data(self):
        """Load all required data files"""
        print("Loading data files...")
        
        # Load cleaned chains, raw chain : cleaned chain 
        with open(os.path.join(self.cleaned_chains_dir, 'cleaned_chains.json'), 'r') as f:
            self.cleaned_chains = json.load(f)
        print(f"Loaded {len(self.cleaned_chains)} cleaned chains")
        
        # Load subchains [subchains]
        with open(os.path.join(self.subchains_dir, 'deduplicated_subchains.json'), 'r') as f:
            self.subchains = json.load(f)
        print(f"Loaded {len(self.subchains)} subchains")
        
        # Load negative chains {cleaned_chain: {hard_negatives:[], soft_negatives:[]}}
        with open(os.path.join(self.negative_chains_dir, 'negative_chains.json'), 'r') as f:
            self.negative_chains = json.load(f)
        print(f"Loaded negative mappings for {len(self.negative_chains)} chains")
        
        # Load negative subchains {subchain: {hard_negatives:[], soft_negatives:[]}}
        with open(os.path.join(self.negative_chains_dir, 'negative_subchains.json'), 'r') as f:
            self.negative_subchains = json.load(f)
        print(f"Loaded negative mappings for {len(self.negative_subchains)} subchains")
        
        # Load unified factsheet dataset (original + synthetic)
        factsheet_path = os.path.join(self.synthetic_data_dir, 'integrated_augmented_factsheet_data.csv')
        df_factsheets = pd.read_csv(factsheet_path)
        self.all_factsheets = dict(zip(df_factsheets['company_id'].astype(str), 
                                     df_factsheets['factsheet']))
        print(f"Loaded {len(self.all_factsheets)} unified factsheets (original + synthetic)")
        
        # Create direct mapping: industry_chain_hierarchy (including both subchains and full cleaned chains) -> [company_ids]
        industry_path = os.path.join(self.synthetic_data_dir, 'integrated_augmented_industry_data.csv')
        df_industry = pd.read_csv(industry_path)
        
        self.chain_to_companies = {}
        for _, row in df_industry.iterrows():
            company_id = str(row['company_id'])
            industry_chain = row['industry_chain_hierarchy']
            
            if industry_chain not in self.chain_to_companies:
                self.chain_to_companies[industry_chain] = []
            self.chain_to_companies[industry_chain].append(company_id)
        
        print(f"Created direct mappings for {len(self.chain_to_companies)} industry chains")
        
        # Load aspect data
        with open(os.path.join(self.aspect_dir, 'cleaned_chains_aspects.json'), 'r') as f:
            self.aspect_data = json.load(f)
        print(f"Loaded aspect data for {len(self.aspect_data)} chains")
        
    def get_positive_factsheets_for_query(self, query: str) -> List[str]:
        """Get positive factsheets for a given query (chain or subchain)"""
        # Direct lookup from the chain_to_companies mapping
        company_ids = self.chain_to_companies.get(query, [])
        
        # Filter to only those with factsheets
        positive_company_ids = [cid for cid in company_ids if cid in self.all_factsheets]
        
        return positive_company_ids
    
    def get_negative_factsheets_for_query(self, query: str, exclude_company_ids: List[str] = None) -> List[str]:
        """Get negative factsheets for a given query"""
        if exclude_company_ids is None:
            exclude_company_ids = []
            
        # Check if query is a subchain or full chain for negative mapping
        if query in self.negative_subchains:
            negative_mapping = self.negative_subchains[query]
        elif query in self.negative_chains:
            negative_mapping = self.negative_chains[query]
        else:
            # If no specific negatives, return empty list
            # all_company_ids = [cid for cid in self.all_factsheets.keys() if cid not in exclude_company_ids]
            return []
        
        hard_negative_queries = negative_mapping.get('hard_negatives', [])
        soft_negative_queries = negative_mapping.get('soft_negatives', [])
        
        # Calculate how many hard and soft negatives to sample
        total_negatives = self.negative_to_positive_ratio
        hard_count = int(total_negatives * self.hard_to_soft_ratio / (self.hard_to_soft_ratio + 1))
        soft_count = total_negatives - hard_count
        
        # Get hard negative company IDs
        hard_negative_company_ids = []
        for neg_query in hard_negative_queries:
            company_ids = self.get_positive_factsheets_for_query(neg_query)
            hard_negative_company_ids.extend(company_ids)
        
        # Get soft negative company IDs
        soft_negative_company_ids = []
        for neg_query in soft_negative_queries:
            company_ids = self.get_positive_factsheets_for_query(neg_query)
            soft_negative_company_ids.extend(company_ids)
        
        # Remove duplicates and excluded IDs
        hard_negative_company_ids = [cid for cid in set(hard_negative_company_ids) 
                                   if cid not in exclude_company_ids and cid in self.all_factsheets]
        soft_negative_company_ids = [cid for cid in set(soft_negative_company_ids) 
                                   if cid not in exclude_company_ids and cid in self.all_factsheets]
        
        # Sample negatives
        sampled_hard = random.sample(hard_negative_company_ids, 
                                   min(hard_count, len(hard_negative_company_ids)))
        
        # Calculate remaining slots after hard negatives
        remaining_slots = total_negatives - len(sampled_hard)
        
        # Fill remaining slots with soft negatives
        sampled_soft = random.sample(soft_negative_company_ids, 
                                   min(remaining_slots, len(soft_negative_company_ids)))
        
        # Return what we could sample (no random fallback)
        return sampled_hard + sampled_soft
    
    def generate_triplets(self, use_aspects: bool = False) -> pd.DataFrame:
        """Generate triplets dataset"""
        print(f"Generating triplets dataset (use_aspects={use_aspects})...")
        
        triplets = []
        
        # Determine which queries to use
        if use_aspects:
            # For aspect dataset, only use full chains (no subchains)
            queries = [chain for chain in self.cleaned_chains.values() 
                      if self.get_positive_factsheets_for_query(chain)]
        else:
            # For regular dataset, use both chains and subchains
            chain_queries = [chain for chain in self.cleaned_chains.values() 
                           if self.get_positive_factsheets_for_query(chain)]
            subchain_queries = [subchain for subchain in self.subchains 
                              if self.get_positive_factsheets_for_query(subchain)]
            queries = chain_queries + subchain_queries
        
        print(f"Processing {len(queries)} queries...")
        
        for query in queries:
            # Get positive factsheets for this query
            # this will work for subcchain too, since get_positive_factsheets_for_query is handling retrieval of poisitives for both 
            positive_company_ids = self.get_positive_factsheets_for_query(query)
            if not positive_company_ids:
                continue
            
            # Sample required number of positive factsheets
            sampled_positive_ids = random.sample(positive_company_ids, 
                                                min(self.query_to_positive_ratio, len(positive_company_ids)))
            
            # For each positive factsheet, create triplets with negatives
            for positive_id in sampled_positive_ids:
                positive_content = self.all_factsheets.get(positive_id, "")
                if not positive_content:
                    continue
                
                # Sample negative factsheets (exclude the positive one)
                negative_company_ids = self.get_negative_factsheets_for_query(
                    query, exclude_company_ids=[positive_id])
                
                # Create triplets
                for negative_id in negative_company_ids:
                    negative_content = self.all_factsheets.get(negative_id, "")
                    if not negative_content:
                        continue
                    
                    # For aspect dataset, replace query with aspect JSON
                    final_query = query
                    if use_aspects and query in self.aspect_data:
                        final_query = json.dumps(self.aspect_data[query])
                    
                    triplets.append({
                        'query': final_query,
                        'positive_factsheet': positive_content,
                        'negative_factsheet': negative_content
                    })
        
        print(f"Generated {len(triplets)} triplets")
        return pd.DataFrame(triplets)
    
    def save_dataset(self, df: pd.DataFrame, filename: str):
        """Save the dataset to CSV"""
        output_path = os.path.join(self.root_dir, 'colbert_training', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total triplets: {len(df)}")
        print(f"Unique queries: {df['query'].nunique()}")
        print(f"Unique positive factsheets: {df['positive_factsheet'].nunique()}")
        print(f"Unique negative factsheets: {df['negative_factsheet'].nunique()}")
    
    def generate_both_datasets(self):
        """Generate both regular and aspect-based datasets"""
        print("=== Generating Regular Dataset ===")
        regular_df = self.generate_triplets(use_aspects=False)
        self.save_dataset(regular_df, 'regular_triplets.csv')
        
        print("\n=== Generating Aspect-based Dataset ===")
        aspect_df = self.generate_triplets(use_aspects=True)
        self.save_dataset(aspect_df, 'aspect_triplets.csv')
        
        return regular_df, aspect_df

def main():
    """Main function to run the triplet generation"""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize generator
    generator = TripletGenerator()
    
    # Load all data
    generator.load_data()
    
    # Generate both datasets
    regular_df, aspect_df = generator.generate_both_datasets()
    
    print("\n=== Triplet Generation Complete ===")
    print(f"Regular dataset: {len(regular_df)} triplets")
    print(f"Aspect dataset: {len(aspect_df)} triplets")

if __name__ == "__main__":
    main()
