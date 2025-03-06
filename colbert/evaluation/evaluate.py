#!/usr/bin/env python3

import os
import argparse
import torch
import json
from pathlib import Path

from colbert.utils.parser import Arguments
from colbert.utils.utils import print_message
from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT
from colbert.evaluation.evaluator import ColBERTEvaluator
from colbert.utils.tracker import ColBERTTracker
from colbert.infra.run import Run


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ColBERT model on a test set")
    
    # Required arguments
    parser.add_argument('--checkpoint', required=True, help="Path to model checkpoint")
    parser.add_argument('--queries', required=True, help="Path to queries TSV file")
    parser.add_argument('--collection', required=True, help="Path to collection TSV file")
    
    # Optional arguments
    parser.add_argument('--qrels', help="Path to qrels TSV file")
    parser.add_argument('--topk', help="Path to pre-computed top-k results")
    parser.add_argument('--triplets', help="Path to triplets TSV file (qid, pid, nid)")
    parser.add_argument('--output_dir', default='logs/evaluation', help="Directory to save evaluation results")
    parser.add_argument('--experiment_name', default='evaluation', help="Name of the experiment")
    parser.add_argument('--run_name', default=None, help="Name of the run (default: checkpoint name)")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for inference")
    parser.add_argument('--depth', type=int, default=1000, help="Maximum depth for retrieval")
    parser.add_argument('--query_maxlen', type=int, default=32, help="Maximum query length")
    parser.add_argument('--doc_maxlen', type=int, default=180, help="Maximum document length")
    parser.add_argument('--dim', type=int, default=128, help="Dimension of the ColBERT embeddings")
    parser.add_argument('--similarity', default='cosine', choices=['cosine', 'l2'], help="Similarity metric")
    parser.add_argument('--mask_punctuation', action='store_true', help="Whether to mask punctuation")
    parser.add_argument('--no_tensorboard', action='store_true', help="Disable TensorBoard logging")
    parser.add_argument('--rank', type=int, default=0, help="Process rank for distributed training")
    
    args = parser.parse_args()
    
    # Convert triplets to qrels if provided
    if args.triplets and not args.qrels:
        from colbert.evaluation.triplet_loader import convert_triplets_to_qrels
        args.qrels = convert_triplets_to_qrels(args.triplets)
    
    # Set run name if not provided
    if args.run_name is None:
        args.run_name = Path(args.checkpoint).stem
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tracker
    tracker = ColBERTTracker(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        log_dir=args.output_dir,
        enable_tensorboard=not args.no_tensorboard,
        rank=args.rank
    )
    
    # Set tracker on Run singleton
    Run().set_tracker(tracker)
    
    # Log configuration
    config_dict = vars(args)
    tracker.log_config(config_dict)
    
    # Create ColBERT configuration
    colbert_config = Arguments()
    colbert_config.query_maxlen = args.query_maxlen
    colbert_config.doc_maxlen = args.doc_maxlen
    colbert_config.dim = args.dim
    colbert_config.similarity = args.similarity
    colbert_config.mask_punctuation = args.mask_punctuation
    colbert_config.checkpoint = args.checkpoint
    colbert_config.bsize = args.batch_size
    colbert_config.rank = args.rank
    
    # Initialize evaluator
    evaluator = ColBERTEvaluator(
        checkpoint_path=args.checkpoint,
        config=colbert_config,
        tracker=tracker
    )
    
    # Run evaluation
    print_message(f"#> Evaluating model from {args.checkpoint}")
    print_message(f"#> Queries: {args.queries}")
    print_message(f"#> Collection: {args.collection}")
    print_message(f"#> Qrels: {args.qrels}")
    print_message(f"#> Top-k: {args.topk}")
    
    results = evaluator.evaluate(
        queries_path=args.queries,
        collection_path=args.collection,
        qrels_path=args.qrels,
        topk_path=args.topk,
        output_path=os.path.join(args.output_dir, "rankings.tsv"),
        batch_size=args.batch_size,
        depth=args.depth
    )
    
    # Save results to file
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print_message(f"#> Evaluation results saved to {results_path}")
    
    # Print summary
    print_message("\n#> Evaluation Summary:")
    for metric_type in results:
        print_message(f"#> {metric_type.upper()}:")
        for depth, value in results[metric_type].items():
            print_message(f"#>   @{depth}: {value:.4f}")
    
    # Close tracker
    tracker.close()


if __name__ == "__main__":
    main()