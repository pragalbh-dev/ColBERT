#!/usr/bin/env python
"""
Benchmarking script for TripletGenerator to understand scaling behavior 
with increasing numbers of labeled pairs.
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import json

# Import the TripletGenerator
from colbert.data.train_data_preprocessor import TripletGenerator

def generate_synthetic_dataset(num_pairs: int, num_queries: int = None, num_docs: int = None) -> Tuple[List, List]:
    """
    Generate a synthetic dataset with the specified number of labeled pairs.
    
    Args:
        num_pairs: Number of labeled pairs to generate
        num_queries: Number of unique queries (default: num_pairs/10)
        num_docs: Number of unique documents (default: num_pairs/5)
        
    Returns:
        Tuple of (labeled_pairs, collection)
    """
    # Default values if not specified
    if num_queries is None:
        num_queries = max(10, min(num_pairs // 10, 10000))  # At most 10K queries
    if num_docs is None:
        num_docs = max(20, min(num_pairs // 5, 20000))  # At most 20K docs
    
    print(f"Generating dataset with {num_pairs} pairs, {num_queries} queries, {num_docs} docs...")
    
    # Generate queries and documents
    aspects = ["price", "quality", "service", "features", "reliability"]
    base_queries = [f"Tell me about the {i}" for i in range(num_queries)]
    queries = []
    
    # Generate unique queries with aspects
    for i in range(num_queries):
        aspect = aspects[i % len(aspects)]
        base = base_queries[i]
        queries.append(f"{aspect}||{base}")
    
    # Generate documents
    documents = []
    for i in range(num_docs):
        # Create documents with varying aspect mentions
        mentioned_aspects = random.sample(aspects, k=random.randint(1, min(3, len(aspects))))
        doc_text = f"Document {i}: "
        
        for aspect in mentioned_aspects:
            quality = random.choice(["excellent", "good", "average", "poor", "terrible"])
            doc_text += f"The {aspect} is {quality}. "
            
        documents.append(doc_text)
    
    # Generate labeled pairs
    labeled_pairs = []
    pairs_per_query = max(1, num_pairs // num_queries)
    
    for query in queries:
        # Extract the aspect from the query
        aspect = query.split("||")[0]
        
        # Find documents mentioning this aspect (potential positives)
        positive_candidates = [i for i, doc in enumerate(documents) if aspect in doc]
        negative_candidates = [i for i in range(len(documents)) if i not in positive_candidates]
        
        # Calculate how many pairs to generate for this query
        pos_count = min(len(positive_candidates), pairs_per_query // 3)  # 1/3 positives
        neg_count = min(len(negative_candidates), pairs_per_query - pos_count)  # 2/3 negatives
        
        # Sample positives and negatives
        if positive_candidates:
            sampled_positives = random.sample(positive_candidates, k=pos_count)
            for doc_idx in sampled_positives:
                labeled_pairs.append((query, documents[doc_idx], 1))
        
        if negative_candidates:
            sampled_negatives = random.sample(negative_candidates, k=neg_count)
            for doc_idx in sampled_negatives:
                labeled_pairs.append((query, documents[doc_idx], 0))
    
    # If we need more pairs, add random ones
    while len(labeled_pairs) < num_pairs:
        query = random.choice(queries)
        doc = random.choice(documents)
        label = random.choice([0, 0, 1])  # 2/3 chance of negative
        labeled_pairs.append((query, doc, label))
    
    # Shuffle and trim to exactly num_pairs
    random.shuffle(labeled_pairs)
    labeled_pairs = labeled_pairs[:num_pairs]
    
    print(f"Generated {len(labeled_pairs)} labeled pairs with {len(queries)} queries and {len(documents)} documents")
    
    return labeled_pairs, documents

def benchmark_triplet_generator(sizes: List[int], 
                               max_triplets_per_query: int = 20,
                               max_positives: int = None,
                               pos_neg_ratio: float = 1.0,
                               negative_sampling_weights: Dict[str, float] = None,
                               output_dir: str = "benchmark_results",
                               export_results: bool = False) -> Dict:
    """
    Benchmark TripletGenerator with datasets of different sizes.
    
    Args:
        sizes: List of dataset sizes (number of labeled pairs) to test
        max_triplets_per_query: Maximum triplets per query
        max_positives: Maximum positives per query
        pos_neg_ratio: Ratio of positives to negatives
        negative_sampling_weights: Weights for different negative sampling strategies
        output_dir: Directory to save benchmark results
        export_results: Whether to export generated triplets
        
    Returns:
        Dictionary with benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "sizes": sizes,
        "times": [],
        "init_times": [],
        "generate_times": [],
        "throughputs": [],
        "triplet_counts": [],
        "config": {
            "max_triplets_per_query": max_triplets_per_query,
            "max_positives": max_positives,
            "pos_neg_ratio": pos_neg_ratio,
            "negative_sampling_weights": negative_sampling_weights or {"rule_based": 0.4, "random": 0.6, "miner": 0.0}
        }
    }
    
    for size in sizes:
        print(f"\n{'='*80}\nBenchmarking dataset with {size} labeled pairs...")
        
        # Generate dataset
        dataset_start = time.time()
        labeled_pairs, collection = generate_synthetic_dataset(size)
        dataset_time = time.time() - dataset_start
        print(f"Dataset generation took {dataset_time:.2f} seconds")
        
        # Create TripletGenerator
        init_start = time.time()
        generator = TripletGenerator(
            labeled_pairs=labeled_pairs,
            collection=collection,
            negative_miner=None,
            aspect_delimiter="||",
            pos_neg_ratio=pos_neg_ratio,
            negative_sampling_weights=negative_sampling_weights,
            seed=42,
            debug=True
        )
        init_time = time.time() - init_start
        
        # Generate triplets
        gen_start = time.time()
        if export_results:
            export_path = os.path.join(output_dir, f"triplets_{size}")
            triplets = generator.generate_triplets(
                max_triplets_per_query=max_triplets_per_query,
                max_positives=max_positives,
                export_path=export_path
            )
        else:
            triplets = generator.generate_triplets(
                max_triplets_per_query=max_triplets_per_query,
                max_positives=max_positives
            )
        
        gen_time = time.time() - gen_start
        total_time = init_time + gen_time
        
        # Get triplet count
        if isinstance(triplets, tuple) and triplets[0]:
            triplet_count = len(triplets[0])
        else:
            triplet_count = len(triplets)
        
        # Calculate throughput metrics
        throughput = size / total_time  # labeled pairs per second
        
        # Record results
        results["times"].append(total_time)
        results["init_times"].append(init_time)
        results["generate_times"].append(gen_time)
        results["throughputs"].append(throughput)
        results["triplet_counts"].append(triplet_count)
        
        # Print results for this size
        print(f"\nResults for {size} labeled pairs:")
        print(f"  - Initialization time: {init_time:.2f}s")
        print(f"  - Triplet generation time: {gen_time:.2f}s")
        print(f"  - Total processing time: {total_time:.2f}s")
        print(f"  - Generated {triplet_count} triplets")
        print(f"  - Throughput: {throughput:.2f} pairs/second")
        
        # Component-level timing from generator
        if hasattr(generator, 'timing'):
            print("\nTiming breakdown:")
            for component, time_taken in generator.timing.items():
                if isinstance(time_taken, dict):
                    print(f"  - {component}:")
                    for subcomp, subtime in time_taken.items():
                        print(f"    - {subcomp}: {subtime:.2f}s")
                else:
                    print(f"  - {component}: {time_taken:.2f}s")
    
    # Save results
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_results(results: Dict, output_dir: str = "benchmark_results"):
    """Plot benchmark results and extrapolate to larger dataset sizes."""
    sizes = results["sizes"]
    times = results["times"]
    init_times = results["init_times"]
    generate_times = results["generate_times"]
    throughputs = results["throughputs"]
    
    # Create figure
    plt.figure(figsize=(20, 15))
    
    # 1. Processing time vs dataset size
    plt.subplot(2, 2, 1)
    plt.plot(sizes, times, 'o-', label='Total time')
    plt.plot(sizes, init_times, 's--', label='Initialization time')
    plt.plot(sizes, generate_times, '^--', label='Generation time')
    plt.xlabel('Number of labeled pairs')
    plt.ylabel('Time (seconds)')
    plt.title('Processing Time vs Dataset Size')
    plt.grid(True)
    plt.legend()
    
    # 2. Throughput vs dataset size
    plt.subplot(2, 2, 2)
    plt.plot(sizes, throughputs, 'o-')
    plt.xlabel('Number of labeled pairs')
    plt.ylabel('Throughput (pairs/second)')
    plt.title('Throughput vs Dataset Size')
    plt.grid(True)
    
    # 3. Extrapolation to larger dataset sizes
    plt.subplot(2, 2, 3)
    
    # Use polynomial regression to predict processing time for larger datasets
    if len(sizes) >= 3:
        # Use polynomial regression (degree 2) for extrapolation
        coeffs = np.polyfit(sizes, times, 2)
        poly = np.poly1d(coeffs)
        
        # Create an extended range of sizes
        max_size = max(sizes)
        extended_sizes = np.array(list(sizes) + [max_size*2, max_size*5, max_size*10, 50_000_000])
        predicted_times = poly(extended_sizes)
        
        # Plot original data and extrapolation
        plt.plot(sizes, times, 'o', label='Measured')
        plt.plot(extended_sizes, predicted_times, '--', label='Extrapolated')
        
        # Add annotations for large sizes
        for i, size in enumerate(extended_sizes[len(sizes):]):
            time_hrs = predicted_times[len(sizes) + i] / 3600  # convert to hours
            plt.annotate(
                f"{size/1e6:.1f}M pairs: {time_hrs:.1f} hours",
                xy=(size, predicted_times[len(sizes) + i]),
                xytext=(0, 20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
            )
        
    plt.xlabel('Number of labeled pairs')
    plt.ylabel('Predicted time (seconds)')
    plt.title('Extrapolated Processing Time for Larger Datasets')
    plt.grid(True)
    plt.legend()
    
    # 4. Log-log plot for scaling behavior
    plt.subplot(2, 2, 4)
    plt.loglog(sizes, times, 'o-', base=10, label='Total time')
    plt.loglog(sizes, init_times, 's--', base=10, label='Initialization')
    plt.loglog(sizes, generate_times, '^--', base=10, label='Generation')
    plt.xlabel('Number of labeled pairs (log scale)')
    plt.ylabel('Time in seconds (log scale)')
    plt.title('Scaling Behavior (Log-Log Plot)')
    plt.grid(True, which="both")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_results.png"))
    plt.show()
    
    # Calculate and display time estimates for 50M pairs
    if len(sizes) >= 3:
        time_50m = poly(50_000_000)
        print(f"\nExtrapolated processing time for 50 million pairs:")
        print(f"  - Seconds: {time_50m:.1f}s")
        print(f"  - Minutes: {time_50m/60:.1f}m")
        print(f"  - Hours: {time_50m/3600:.1f}h")
        print(f"  - Days: {time_50m/(3600*24):.1f}d")

def main():
    parser = argparse.ArgumentParser(description="Benchmark TripletGenerator scaling")
    parser.add_argument("--sizes", type=str, default="1000,5000,10000,50000,100000,500000",
                       help="Comma-separated list of dataset sizes to test")
    parser.add_argument("--max-triplets", type=int, default=10000,
                       help="Maximum triplets per query")
    parser.add_argument("--max-positives", type=int, default=100,
                       help="Maximum positives per query")
    parser.add_argument("--pos-neg-ratio", type=float, default=0.1,
                       help="Ratio of positives to negatives")
    parser.add_argument("--rule-based-weight", type=float, default=0.1,
                       help="Weight for rule-based negative sampling")
    parser.add_argument("--random-weight", type=float, default=0.9,
                       help="Weight for random negative sampling")
    parser.add_argument("--output-dir", type=str, default="data/test/benchmark_results",
                       help="Directory to save benchmark results")
    parser.add_argument("--export-triplets", action="store_true",
                       help="Whether to export generated triplets")
    
    args = parser.parse_args()
    
    # Parse sizes
    sizes = [int(s) for s in args.sizes.split(",")]
    sizes.sort()  # Ensure sizes are in ascending order
    
    # Set up negative sampling weights
    negative_sampling_weights = {
        "rule_based": args.rule_based_weight,
        "random": args.random_weight,
        "miner": 0.0  # No miner in this benchmark
    }
    
    # Normalize weights
    total_weight = sum(negative_sampling_weights.values())
    for k in negative_sampling_weights:
        negative_sampling_weights[k] /= total_weight
    
    # Run benchmark
    print(f"Starting benchmark with sizes: {sizes}")
    results = benchmark_triplet_generator(
        sizes=sizes,
        max_triplets_per_query=args.max_triplets,
        max_positives=args.max_positives,
        pos_neg_ratio=args.pos_neg_ratio,
        negative_sampling_weights=negative_sampling_weights,
        output_dir=args.output_dir,
        export_results=args.export_triplets
    )
    
    # Plot results
    plot_results(results, args.output_dir)

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    main() 