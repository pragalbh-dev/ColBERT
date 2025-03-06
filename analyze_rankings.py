#!/usr/bin/env python

import json
import os
import argparse
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Set, Tuple

def load_triples(triples_file: str) -> Dict[int, Set[int]]:
    """
    Load triples from jsonl file and extract query-positive document pairs.
    
    Args:
        triples_file: Path to the triples jsonl file
        
    Returns:
        Dictionary mapping query IDs to sets of positive document IDs
    """
    query_positives = defaultdict(set)
    
    with open(triples_file, 'r') as f:
        for line in f:
            # Parse the triple [qid, pos_id, neg_id]
            triple = json.loads(line.strip())
            qid, pos_id, _ = triple
            
            # Add the positive document to the set for this query
            query_positives[qid].add(pos_id)
    
    return query_positives

def load_rankings(output_file: str) -> Dict[int, List[Tuple[int, int, float]]]:
    """
    Load rankings from the output TSV file.
    
    Args:
        output_file: Path to the output TSV file
        
    Returns:
        Dictionary mapping query IDs to lists of (rank, doc_id, score) tuples
    """
    query_rankings = defaultdict(list)
    
    # Read the TSV file
    try:
        # Try reading with pandas first (handles TSV well)
        df = pd.read_csv(output_file, sep='\t', header=None)
        for _, row in df.iterrows():
            qid, doc_id, score = int(row[0]), int(row[1]), float(row[2])
            rank = len(query_rankings[qid]) + 1  # Rank is 1-indexed position in results
            query_rankings[qid].append((rank, doc_id, score))
    except:
        # Fallback to manual parsing if pandas fails
        with open(output_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:  # Ensure we have at least qid, doc_id, score
                    try:
                        qid, doc_id, score = int(parts[0]), int(parts[1]), float(parts[2])
                        rank = len(query_rankings[qid]) + 1  # Rank is 1-indexed position in results
                        query_rankings[qid].append((rank, doc_id, score))
                    except (ValueError, IndexError):
                        continue  # Skip invalid lines
    
    return query_rankings

def analyze_positive_rankings(query_positives: Dict[int, Set[int]], 
                             query_rankings: Dict[int, List[Tuple[int, int, float]]]) -> Dict:
    """
    Analyze where positive documents appear in the rankings.
    
    Args:
        query_positives: Dictionary mapping query IDs to sets of positive document IDs
        query_rankings: Dictionary mapping query IDs to lists of (rank, doc_id, score) tuples
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        "query_summaries": [],
        "overall": {
            "total_queries": 0,
            "total_positives": 0,
            "positives_found": 0,
            "positives_not_found": 0,
            "avg_rank": 0.0,
            "median_rank": 0.0,
            "rank_at_1": 0,
            "rank_at_5": 0,
            "rank_at_10": 0
        }
    }
    
    all_ranks = []
    
    # For each query
    for qid in sorted(query_positives.keys()):
        positives = query_positives[qid]
        rankings = query_rankings.get(qid, [])
        
        # Create a mapping from doc_id to rank for this query
        doc_to_rank = {doc_id: rank for rank, doc_id, _ in rankings}
        
        # Find ranks of all positives
        positive_ranks = []
        for pos_id in positives:
            if pos_id in doc_to_rank:
                positive_ranks.append(doc_to_rank[pos_id])
                all_ranks.append(doc_to_rank[pos_id])
        
        # Count found and not found
        found = len(positive_ranks)
        not_found = len(positives) - found
        
        # Calculate statistics for this query
        query_summary = {
            "query_id": qid,
            "total_positives": len(positives),
            "positives_found": found,
            "positives_not_found": not_found,
            "positive_ranks": positive_ranks,
            "avg_rank": sum(positive_ranks) / max(1, len(positive_ranks)) if positive_ranks else None,
            "has_positive_at_1": 1 in positive_ranks,
            "has_positive_at_5": any(rank <= 5 for rank in positive_ranks),
            "has_positive_at_10": any(rank <= 10 for rank in positive_ranks)
        }
        
        results["query_summaries"].append(query_summary)
        
        # Update overall statistics
        results["overall"]["total_queries"] += 1
        results["overall"]["total_positives"] += len(positives)
        results["overall"]["positives_found"] += found
        results["overall"]["positives_not_found"] += not_found
        if query_summary["has_positive_at_1"]:
            results["overall"]["rank_at_1"] += 1
        if query_summary["has_positive_at_5"]:
            results["overall"]["rank_at_5"] += 1
        if query_summary["has_positive_at_10"]:
            results["overall"]["rank_at_10"] += 1
    
    # Calculate overall averages
    if all_ranks:
        results["overall"]["avg_rank"] = sum(all_ranks) / len(all_ranks)
        results["overall"]["median_rank"] = sorted(all_ranks)[len(all_ranks) // 2]
    
    return results

def print_results(results: Dict):
    """Print analysis results in a readable format"""
    overall = results["overall"]
    
    print("\n" + "="*80)
    print("OVERALL EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Queries: {overall['total_queries']}")
    print(f"Total Positive Documents: {overall['total_positives']}")
    print(f"Positives Found in Rankings: {overall['positives_found']} ({overall['positives_found']/max(1, overall['total_positives'])*100:.2f}%)")
    print(f"Positives Not Found: {overall['positives_not_found']}")
    print(f"Average Rank of Positives: {overall['avg_rank']:.2f}")
    print(f"Median Rank of Positives: {overall['median_rank']:.2f}")
    print(f"Queries with Positive at Rank 1: {overall['rank_at_1']} ({overall['rank_at_1']/max(1, overall['total_queries'])*100:.2f}%)")
    print(f"Queries with Positive in Top 5: {overall['rank_at_5']} ({overall['rank_at_5']/max(1, overall['total_queries'])*100:.2f}%)")
    print(f"Queries with Positive in Top 10: {overall['rank_at_10']} ({overall['rank_at_10']/max(1, overall['total_queries'])*100:.2f}%)")
    
    print("\n" + "="*80)
    print("QUERY-BY-QUERY SUMMARY")
    print("="*80)
    
    for query in results["query_summaries"]:
        print(f"\nQuery {query['query_id']}:")
        print(f"  Total Positives: {query['total_positives']}")
        print(f"  Positives Found: {query['positives_found']}")
        print(f"  Positives Not Found: {query['positives_not_found']}")
        if query['positive_ranks']:
            print(f"  Ranks of Positives: {', '.join(map(str, sorted(query['positive_ranks'])))}")
            print(f"  Average Rank: {query['avg_rank']:.2f}")
        else:
            print("  No positives found in rankings")
        print(f"  Has Positive at Rank 1: {'Yes' if query['has_positive_at_1'] else 'No'}")
        print(f"  Has Positive in Top 5: {'Yes' if query['has_positive_at_5'] else 'No'}")
        print(f"  Has Positive in Top 10: {'Yes' if query['has_positive_at_10'] else 'No'}")

def main():
    parser = argparse.ArgumentParser(description='Analyze ranking positions of positive documents')
    parser.add_argument('--triples', type=str, default='data/human_curated/test/triples.train.colbert.jsonl',
                        help='Path to the triples file')
    parser.add_argument('--output', type=str, default='data/human_curated/test/eval_out/output.train.colbert.tsv',
                        help='Path to the output file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.triples):
        print(f"Error: Triples file not found at {args.triples}")
        return
    
    if not os.path.exists(args.output):
        print(f"Error: Output file not found at {args.output}")
        return
    
    # Load data
    print(f"Loading triples from {args.triples}...")
    query_positives = load_triples(args.triples)
    
    print(f"Loading rankings from {args.output}...")
    query_rankings = load_rankings(args.output)
    
    # Analyze
    print("Analyzing ranking positions...")
    results = analyze_positive_rankings(query_positives, query_rankings)
    
    # Print results
    print_results(results)

if __name__ == "__main__":
    main() 