#!/usr/bin/env python
"""
Simple script to measure TripletGenerator performance on real data.
Takes a sample of your real dataset and runs measurements to predict full-scale processing time.
"""

import os
import time
import random
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple

# Import the TripletGenerator
from colbert.data.train_data_preprocessor import TripletGenerator

def measure_generation_time(labeled_pairs, collection, sample_size=None, 
                            max_triplets_per_query=20, max_positives=None):
    """
    Measure triplet generation time on a sample of labeled pairs.
    
    Args:
        labeled_pairs: List of (query, passage, label) tuples
        collection: List of documents
        sample_size: Sample size (use None for full dataset)
        max_triplets_per_query: Maximum triplets per query
        max_positives: Maximum positives per query
    
    Returns:
        Dictionary with timing results
    """
    # Take a sample if requested
    if sample_size and sample_size < len(labeled_pairs):
        sampled_pairs = random.sample(labeled_pairs, sample_size)
        print(f"Testing with {sample_size:,} pairs (sampled from {len(labeled_pairs):,})")
    else:
        sampled_pairs = labeled_pairs
        print(f"Testing with all {len(labeled_pairs):,} pairs")
    
    # Create a set of all documents in the labeled pairs for collection subset
    docs_in_pairs = set()
    for _, doc, _ in sampled_pairs:
        docs_in_pairs.add(doc)
    
    # Create a subcollection (only include docs in the sample)
    collection_subset = [doc for doc in collection if doc in docs_in_pairs]
    print(f"Using collection subset with {len(collection_subset):,} documents")
    
    # Initialize TripletGenerator
    print("Initializing TripletGenerator...")
    init_start = time.time()
    generator = TripletGenerator(
        labeled_pairs=sampled_pairs,
        collection=collection_subset,
        negative_miner=None,
        aspect_delimiter="||",
        pos_neg_ratio=1.0,
        negative_sampling_weights={"rule_based": 0.4, "random": 0.6, "miner": 0.0},
        seed=42,
        debug=True
    )
    init_time = time.time() - init_start
    print(f"Initialization completed in {init_time:.2f} seconds")
    
    # Generate triplets
    print("Generating triplets...")
    gen_start = time.time()
    triplets = generator.generate_triplets(
        max_triplets_per_query=max_triplets_per_query,
        max_positives=max_positives
    )
    gen_time = time.time() - gen_start
    
    # Get triplet count
    if isinstance(triplets, tuple) and triplets[0]:
        triplet_count = len(triplets[0])
    else:
        triplet_count = len(triplets)
    
    # Calculate results
    total_time = init_time + gen_time
    throughput = len(sampled_pairs) / total_time  # pairs per second
    
    # Calculate estimated time for full dataset
    full_dataset_size = len(labeled_pairs)
    est_full_time = full_dataset_size / throughput
    
    # Calculate estimated time for 50M pairs
    est_50m_time = 50_000_000 / throughput
    
    # Print results
    print("\nResults:")
    print(f"  Initialization time: {init_time:.2f}s")
    print(f"  Triplet generation time: {gen_time:.2f}s")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Generated {triplet_count:,} triplets")
    print(f"  Processing rate: {throughput:.2f} pairs/second")
    
    if sample_size and sample_size < len(labeled_pairs):
        print(f"\nEstimated time for full dataset ({full_dataset_size:,} pairs):")
        print(f"  - {est_full_time:.2f} seconds")
        print(f"  - {est_full_time/60:.2f} minutes")
        print(f"  - {est_full_time/3600:.2f} hours")
    
    print(f"\nEstimated time for 50 million pairs:")
    print(f"  - {est_50m_time:.2f} seconds")
    print(f"  - {est_50m_time/60:.2f} minutes")
    print(f"  - {est_50m_time/3600:.2f} hours")
    print(f"  - {est_50m_time/(3600*24):.2f} days")
    
    # Component-level timing
    if hasattr(generator, 'timing'):
        print("\nTiming breakdown:")
        for component, time_taken in generator.timing.items():
            if isinstance(time_taken, dict):
                print(f"  - {component}:")
                for subcomp, subtime in time_taken.items():
                    print(f"    - {subcomp}: {subtime:.2f}s")
            else:
                print(f"  - {component}: {time_taken:.2f}s")
    
    results = {
        "sample_size": len(sampled_pairs),
        "full_dataset_size": full_dataset_size,
        "init_time": init_time,
        "generation_time": gen_time,
        "total_time": total_time,
        "triplet_count": triplet_count,
        "throughput": throughput,
        "estimated_full_dataset_time": est_full_time,
        "estimated_50m_time": est_50m_time,
        "timing_breakdown": generator.timing if hasattr(generator, 'timing') else None
    }
    
    return results

def load_data_from_jsonl(file_path):
    """Load labeled pairs from a JSONL file."""
    print(f"Loading data from {file_path}...")
    labeled_pairs = []
    collection = set()
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                query = item['query']
                doc = item['document']
                label = item['label']
                labeled_pairs.append((query, doc, label))
                collection.add(doc)
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    
    print(f"Loaded {len(labeled_pairs):,} labeled pairs with {len(collection):,} unique documents")
    return labeled_pairs, list(collection)

def load_data_from_csv(file_path, delimiter='\t', query_col=0, doc_col=1, label_col=2, 
                     has_header=True):
    """Load labeled pairs from a CSV/TSV file."""
    print(f"Loading data from {file_path}...")
    labeled_pairs = []
    collection = set()
    
    with open(file_path, 'r') as f:
        # Skip header if needed
        if has_header:
            next(f)
            
        for line in f:
            try:
                parts = line.strip().split(delimiter)
                query = parts[query_col]
                doc = parts[doc_col]
                label = int(parts[label_col])
                labeled_pairs.append((query, doc, label))
                collection.add(doc)
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
    
    print(f"Loaded {len(labeled_pairs):,} labeled pairs with {len(collection):,} unique documents")
    return labeled_pairs, list(collection)

def main():
    parser = argparse.ArgumentParser(description="Measure TripletGenerator performance")
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to data file (JSONL or CSV/TSV)")
    parser.add_argument("--format", type=str, choices=["jsonl", "csv", "tsv"], default="jsonl",
                       help="Format of the data file")
    parser.add_argument("--delimiter", type=str, default="\t",
                       help="Delimiter for CSV/TSV files")
    parser.add_argument("--query-col", type=int, default=0,
                       help="Query column index for CSV/TSV files")
    parser.add_argument("--doc-col", type=int, default=1,
                       help="Document column index for CSV/TSV files")
    parser.add_argument("--label-col", type=int, default=2,
                       help="Label column index for CSV/TSV files")
    parser.add_argument("--has-header", action="store_true",
                       help="Whether CSV/TSV file has a header row")
    parser.add_argument("--sample-size", type=int, default=10000,
                       help="Number of labeled pairs to sample (use 0 for all)")
    parser.add_argument("--max-triplets", type=int, default=20,
                       help="Maximum triplets per query")
    parser.add_argument("--max-positives", type=int, default=None,
                       help="Maximum positives per query")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Path to save results (JSON)")
    
    args = parser.parse_args()
    
    # Load data
    if args.format == "jsonl":
        labeled_pairs, collection = load_data_from_jsonl(args.data_file)
    else:  # csv or tsv
        labeled_pairs, collection = load_data_from_csv(
            args.data_file, 
            delimiter=args.delimiter,
            query_col=args.query_col,
            doc_col=args.doc_col,
            label_col=args.label_col,
            has_header=args.has_header
        )
    
    # Use sample size 0 to indicate full dataset
    sample_size = None if args.sample_size == 0 else args.sample_size
    
    # Run measurement
    results = measure_generation_time(
        labeled_pairs=labeled_pairs,
        collection=collection,
        sample_size=sample_size,
        max_triplets_per_query=args.max_triplets,
        max_positives=args.max_positives
    )
    
    # Save results if requested
    if args.output_file:
        print(f"\nSaving results to {args.output_file}")
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    np.random.seed(42)
    
    main() 