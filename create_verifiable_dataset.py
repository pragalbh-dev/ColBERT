import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import time

# Import our data generator
from create_synthetic_dataset import AspectDatasetGenerator, load_dataset_for_processor

# Import the processor and splitter from colbert
from colbert.data.train_data_preprocessor import TripletGenerator, TripletDatasetSplitter

def create_and_analyze_dataset(
    output_dir: str = "data/human_verification",
    num_aspects: int = 3,
    num_queries_per_aspect: int = 5,
    num_docs: int = 50,
    seed: int = 42,
    train_val_test_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    aspect_delimiter: str = "||"
):
    """
    Create a small verifiable dataset, process it with TripletDatasetSplitter, and analyze results.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate synthetic dataset
    print("Generating synthetic dataset...")
    start_time = time.time()
    generator = AspectDatasetGenerator(
        seed=seed,
        num_aspects=num_aspects,
        num_queries_per_aspect=num_queries_per_aspect,
        num_docs=num_docs,
        aspect_delimiter=aspect_delimiter,
        output_dir=output_dir
    )
    
    # Export the dataset
    generator.export_dataset() 
    
    # Print initial statistics
    generator.print_statistics()
    
    # Get dataset in processor-compatible format
    labeled_pairs, collection = generator.get_dataset_for_processor()
    data_gen_time = time.time() - start_time
    print(f"⏱️ Synthetic data generation completed in {data_gen_time:.2f} seconds")
    print(f"📊 Generated {len(labeled_pairs)} labeled pairs, {len(collection)} documents")
    
    # 2. Split dataset using TripletDatasetSplitter
    print(f"\n2. Creating splits using TripletDatasetSplitter...")
    split_start_time = time.time()
    splitter = TripletDatasetSplitter(
        labeled_pairs=labeled_pairs,
        collection=collection,
        negative_miner=None,  # No miner for this demo
        aspect_delimiter=aspect_delimiter,
        train_val_test_ratio=train_val_test_ratio,
        pos_neg_ratio=1.0,  # 1:1 ratio of positives to negatives
        seed=seed,
        debug=True  # Enable debug mode for more detailed output
    )
    
    # Split the dataset into train/val/test
    splitter.split_dataset()
    split_time = time.time() - split_start_time
    print(f"⏱️ Dataset splitting completed in {split_time:.2f} seconds")
    print(f"📊 Split sizes - Train: {len(splitter.train_pairs)}, Val: {len(splitter.val_pairs)}, Test: {len(splitter.test_pairs)}")
    
    # 3. Generate triplets for each split
    print(f"\n3. Generating triplets for each split...")
    triplet_start_time = time.time()
    results = splitter.process_data(
        output_dir=output_dir,
        max_triplets_per_query=10,  # Small number for verification
        max_positives=5,
        # Use default negative sampling weights
        train_negative_sampling_weights={"rule_based": 0.5, "random": 0.5, "miner": 0.0},
        val_negative_sampling_weights={"rule_based": 0.5, "random": 0.5, "miner": 0.0},
        test_negative_sampling_weights={"rule_based": 0.5, "random": 0.5, "miner": 0.0}
    )
    triplet_gen_time = time.time() - triplet_start_time
    total_triplets = sum(len(results[split][0]) if isinstance(results[split], tuple) and results[split][0] else 0 
                         for split in ["train", "val", "test", "test_rule_based"])
    print(f"⏱️ Triplet generation completed in {triplet_gen_time:.2f} seconds")
    print(f"📊 Generated {total_triplets} total triplets")
    
    # Per-split timing breakdown
    train_triplets = results["train"]
    val_triplets = results["val"]
    test_triplets = results["test"]
    rule_test_triplets = results["test_rule_based"]
    
    train_count = len(train_triplets[0]) if isinstance(train_triplets, tuple) and train_triplets[0] else 0
    val_count = len(val_triplets[0]) if isinstance(val_triplets, tuple) and val_triplets[0] else 0
    test_count = len(test_triplets[0]) if isinstance(test_triplets, tuple) and test_triplets[0] else 0
    rule_test_count = len(rule_test_triplets[0]) if isinstance(rule_test_triplets, tuple) and rule_test_triplets[0] else 0
    
    # Calculate rate (triplets/second) for each split
    total_pairs = len(labeled_pairs)
    print(f"\n⏱️ Performance Metrics:")
    print(f"  - Overall: {total_pairs / (data_gen_time + split_time + triplet_gen_time):.2f} labeled pairs/second")
    print(f"  - Data Generation: {total_pairs / data_gen_time:.2f} labeled pairs/second")
    print(f"  - Dataset Splitting: {total_pairs / split_time:.2f} labeled pairs/second")
    print(f"  - Triplet Generation: {total_triplets / triplet_gen_time:.2f} triplets/second")
    
    # Print triplet counts per split
    print(f"\n📊 Triplet Counts per Split:")
    print(f"  - Train: {train_count} triplets")
    print(f"  - Val: {val_count} triplets")
    print(f"  - Test: {test_count} triplets")
    print(f"  - Test (rule-based): {rule_test_count} triplets")
    
    # 4. Analyze the processed data
    analysis_start_time = time.time()
    create_analysis(
        splitter=splitter,
        generator=generator,
        results=results,
        output_dir=output_dir,
        train_val_test_ratio=train_val_test_ratio
    )
    analysis_time = time.time() - analysis_start_time
    
    # Final timing summary
    total_time = data_gen_time + split_time + triplet_gen_time + analysis_time
    print(f"\n⏱️ Timing Summary:")
    print(f"  - Data Generation: {data_gen_time:.2f}s ({data_gen_time/total_time*100:.1f}%)")
    print(f"  - Dataset Splitting: {split_time:.2f}s ({split_time/total_time*100:.1f}%)")
    print(f"  - Triplet Generation: {triplet_gen_time:.2f}s ({triplet_gen_time/total_time*100:.1f}%)")
    print(f"  - Analysis: {analysis_time:.2f}s ({analysis_time/total_time*100:.1f}%)")
    print(f"  - Total Time: {total_time:.2f}s")
    
    print(f"\nVerification dataset created and processed successfully!")
    print(f"All data and statistics are available in: {output_dir}")
    print(f"\nTo use with ColBERT training, you can refer to the processed triplets in:")
    print(f"  - Train: {output_dir}/train")
    print(f"  - Val: {output_dir}/val")
    print(f"  - Test: {output_dir}/test")
    print(f"  - Test (rule-based): {output_dir}/test_rule_based")

def create_analysis(splitter, generator, results, output_dir, train_val_test_ratio):
    """Create detailed analysis of the dataset splits"""
    
    # Gather statistics about splits
    stats = {}
    
    # Original data statistics
    stats["original"] = {
        "num_queries": len(generator.queries),
        "num_docs": len(generator.documents),
        "num_pairs": len(generator.labeled_pairs),
        "num_positives": sum(1 for _, _, label in generator.labeled_pairs if label == 1),
        "num_negatives": sum(1 for _, _, label in generator.labeled_pairs if label == 0),
        "aspects": generator.aspects
    }
    
    # Split statistics
    stats["splits"] = {
        "train": {
            "num_pairs": len(splitter.train_pairs),
            "num_docs": len(splitter.train_documents),
            "num_positives": sum(1 for _, _, label in splitter.train_pairs if label == 1),
            "num_negatives": sum(1 for _, _, label in splitter.train_pairs if label == 0)
        },
        "val": {
            "num_pairs": len(splitter.val_pairs),
            "num_docs": len(splitter.val_documents),
            "num_positives": sum(1 for _, _, label in splitter.val_pairs if label == 1),
            "num_negatives": sum(1 for _, _, label in splitter.val_pairs if label == 0)
        },
        "test": {
            "num_pairs": len(splitter.test_pairs),
            "num_docs": len(splitter.test_documents),
            "num_positives": sum(1 for _, _, label in splitter.test_pairs if label == 1),
            "num_negatives": sum(1 for _, _, label in splitter.test_pairs if label == 0)
        }
    }
    
    # Processed data statistics
    stats["processed"] = {
        "train": {
            "num_triplets": len(results["train"][0]) if results["train"][0] else 0
        },
        "val": {
            "num_triplets": len(results["val"][0]) if results["val"][0] else 0
        },
        "test": {
            "num_triplets": len(results["test"][0]) if results["test"][0] else 0
        },
        "test_rule_based": {
            "num_triplets": len(results["test_rule_based"][0]) if results["test_rule_based"][0] else 0
        }
    }
    
    # Aspect distribution in each split
    stats["aspect_distribution"] = {split: {} for split in ["train", "val", "test"]}
    
    for split_name in ["train", "val", "test"]:
        split_pairs = getattr(splitter, f"{split_name}_pairs")
        aspect_counts = Counter()
        
        for query, _, _ in split_pairs:
            if splitter.aspect_delimiter in query:
                aspect = query.split(splitter.aspect_delimiter)[0]
                aspect_counts[aspect] += 1
        
        stats["aspect_distribution"][split_name] = dict(aspect_counts)
    
    # Configuration settings
    stats["config"] = {
        "train_val_test_ratio": train_val_test_ratio,
        "seed": splitter.seed,
        "aspect_delimiter": splitter.aspect_delimiter,
        "pos_neg_ratio": splitter.pos_neg_ratio
    }
    
    # Save statistics to file
    with open(output_dir / "dataset_analysis.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # Create visualizations
    create_visualizations(stats, output_dir)
    
    # Print summary
    print_statistics_summary(stats)

def create_visualizations(stats, output_dir):
    """Create visualizations of dataset statistics"""
    
    # 1. Split distribution
    fig = plt.figure(figsize=(15, 12))
    
    # Document distribution across splits
    ax1 = fig.add_subplot(2, 2, 1)
    split_docs = [stats["splits"][split]["num_docs"] for split in ["train", "val", "test"]]
    ax1.pie(
        split_docs,
        labels=["Train", "Val", "Test"],
        autopct='%1.1f%%',
        colors=['#5cb85c', '#f0ad4e', '#d9534f']
    )
    ax1.set_title("Document Distribution Across Splits")
    
    # Pair distribution across splits
    ax2 = fig.add_subplot(2, 2, 2)
    split_pairs = [stats["splits"][split]["num_pairs"] for split in ["train", "val", "test"]]
    ax2.pie(
        split_pairs,
        labels=["Train", "Val", "Test"],
        autopct='%1.1f%%',
        colors=['#5cb85c', '#f0ad4e', '#d9534f']
    )
    ax2.set_title("Labeled Pair Distribution Across Splits")
    
    # Aspect distribution across splits
    ax3 = fig.add_subplot(2, 2, 3)
    aspects = stats["original"]["aspects"]
    
    aspect_data = []
    for aspect in aspects:
        aspect_data.append([
            stats["aspect_distribution"]["train"].get(aspect, 0),
            stats["aspect_distribution"]["val"].get(aspect, 0),
            stats["aspect_distribution"]["test"].get(aspect, 0)
        ])
    
    x = np.arange(len(aspects))
    width = 0.25
    
    ax3.bar(x - width, [d[0] for d in aspect_data], width, label="Train", color='#5cb85c')
    ax3.bar(x, [d[1] for d in aspect_data], width, label="Val", color='#f0ad4e')
    ax3.bar(x + width, [d[2] for d in aspect_data], width, label="Test", color='#d9534f')
    ax3.set_title("Aspect Distribution Across Splits")
    ax3.set_xticks(x)
    ax3.set_xticklabels(aspects, rotation=45)
    ax3.legend()
    
    # Positive/Negative distribution across splits
    ax4 = fig.add_subplot(2, 2, 4)
    
    pos_neg_data = []
    for split in ["train", "val", "test"]:
        pos_neg_data.append([
            stats["splits"][split]["num_positives"],
            stats["splits"][split]["num_negatives"]
        ])
    
    x = np.arange(3)  # train, val, test
    width = 0.35
    
    ax4.bar(x - width/2, [d[0] for d in pos_neg_data], width, label="Positive", color='#5bc0de')
    ax4.bar(x + width/2, [d[1] for d in pos_neg_data], width, label="Negative", color='#292b2c')
    ax4.set_title("Positive/Negative Distribution Across Splits")
    ax4.set_xticks(x)
    ax4.set_xticklabels(["Train", "Val", "Test"])
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_analysis.png")
    
    
def print_statistics_summary(stats):
    """Print a summary of the dataset statistics"""
    
    print("\n=== Dataset Statistics Summary ===\n")
    
    print("Original Data:")
    print(f"  Number of queries: {stats['original']['num_queries']}")
    print(f"  Number of documents: {stats['original']['num_docs']}")
    print(f"  Number of labeled pairs: {stats['original']['num_pairs']}")
    print(f"  Positives/Negatives: {stats['original']['num_positives']}/{stats['original']['num_negatives']}")
    print(f"  Aspects: {', '.join(stats['original']['aspects'])}")
    
    print("\nSplit Distribution:")
    for split in ["train", "val", "test"]:
        print(f"  {split.capitalize()}:")
        print(f"    Documents: {stats['splits'][split]['num_docs']}")
        print(f"    Labeled pairs: {stats['splits'][split]['num_pairs']}")
        print(f"    Positives/Negatives: {stats['splits'][split]['num_positives']}/{stats['splits'][split]['num_negatives']}")
    
    print("\nAspect Distribution Across Splits:")
    aspects = stats["original"]["aspects"]
    for aspect in aspects:
        print(f"  {aspect}:")
        for split in ["train", "val", "test"]:
            count = stats["aspect_distribution"][split].get(aspect, 0)
            print(f"    {split.capitalize()}: {count}")
    
    print("\nProcessed Data:")
    for split in ["train", "val", "test", "test_rule_based"]:
        print(f"  {split.capitalize()}: {stats['processed'][split]['num_triplets']} triplets")
    
    print("\nConfiguration:")
    print(f"  Train/Val/Test Ratio: {stats['config']['train_val_test_ratio']}")
    print(f"  Seed: {stats['config']['seed']}")
    print(f"  Aspect Delimiter: '{stats['config']['aspect_delimiter']}'")
    print(f"  Positive-Negative Ratio: {stats['config']['pos_neg_ratio']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create and analyze a verifiable dataset for aspect-based retrieval")
    parser.add_argument("--output-dir", type=str, default="data/human_verification", 
                        help="Directory to store the dataset and analysis")
    parser.add_argument("--num-aspects", type=int, default=10,
                        help="Number of aspects to generate")
    parser.add_argument("--num-queries", type=int, default=200,
                        help="Number of queries per aspect")
    parser.add_argument("--num-docs", type=int, default=10000,
                        help="Number of documents to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train-ratio", type=float, default=0.6,
                        help="Proportion of data for training")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Proportion of data for validation")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Proportion of data for testing")
    
    args = parser.parse_args()
    
    # Normalize the train/val/test ratio
    total = args.train_ratio + args.val_ratio + args.test_ratio
    train_ratio = args.train_ratio / total
    val_ratio = args.val_ratio / total
    test_ratio = args.test_ratio / total
    
    create_and_analyze_dataset(
        output_dir=args.output_dir,
        num_aspects=args.num_aspects,
        num_queries_per_aspect=args.num_queries,
        num_docs=args.num_docs,
        seed=args.seed,
        train_val_test_ratio=(train_ratio, val_ratio, test_ratio)
    ) 