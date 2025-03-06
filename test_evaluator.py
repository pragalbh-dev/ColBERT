#!/usr/bin/env python

import os
from colbert.evaluation.evaluator import ColBERTEvaluator
from colbert.infra import ColBERTConfig

def main():
    # Setup paths
    data_dir = 'data/human_verification/test'
    queries_path = os.path.join(data_dir, 'queries.train.colbert.tsv')
    collection_path = os.path.join(data_dir, 'corpus.train.colbert.tsv')
    triplets_path = os.path.join(data_dir, 'triples.train.colbert.jsonl')
    eval_out_path = os.path.join(data_dir, 'eval_out')
    
    # Create output directory if it doesn't exist
    os.makedirs(eval_out_path, exist_ok=True)
    
    # Initialize configuration
    config = ColBERTConfig(
        bsize=32, 
        lr=1e-05, 
        warmup=20_000, 
        doc_maxlen=512, 
        dim=128, 
        attend_to_mask_tokens=False, 
        nway=64, 
        accumsteps=1, 
        similarity='cosine', 
        use_ib_negatives=True
    )
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = ColBERTEvaluator(config=config, checkpoint_path='bert-base-uncased')
    
    # Run evaluation
    print("\nRunning evaluation with FIXED metrics calculation...")
    eval_res = evaluator.evaluate(
        queries_path=queries_path,
        collection_path=collection_path,
        triplets_path=triplets_path,
        output_path=os.path.join(eval_out_path, 'output.train.fixed.tsv'),
        batch_size=128,
        depth=1000,
        step=None
    )
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print("-" * 50)
    print("MRR:")
    for depth, value in eval_res['mrr'].items():
        print(f"  @{depth}: {value:.6f}")
    
    print("\nSucccess:")
    for depth, value in eval_res['success'].items():
        print(f"  @{depth}: {value:.6f}")
    
    print("\nRecall:")
    for depth, value in eval_res['recall'].items():
        print(f"  @{depth}: {value:.6f}")
    
    print("\nPrecision:")
    for depth, value in eval_res['precision'].items():
        print(f"  @{depth}: {value:.6f}")
    
    print("\nNDCG:")
    for depth, value in eval_res['ndcg'].items():
        print(f"  @{depth}: {value:.6f}")

if __name__ == "__main__":
    main() 