import os
import sys
import torch
import random
import numpy as np
import argparse
from pathlib import Path

from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from training_workflow.tracking.training_tracker import TrainingTracker

def parse_args():
    parser = argparse.ArgumentParser(description='Train ColBERT with experiment tracking')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--job_id', type=str, required=True, help='Job ID')
    parser.add_argument('--triples', type=str, required=True, help='Path to training triples')
    parser.add_argument('--queries', type=str, required=True, help='Path to queries')
    parser.add_argument('--collection', type=str, required=True, help='Path to collection')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_colbert(config, triples, queries, collection, tracker):
    """
    Train a ColBERT model with tracking
    """
    # Initialize model
    colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    colbert = colbert.to(DEVICE)
    colbert.train()

    # Initialize data loader
    reader = LazyBatcher(config, triples, queries, collection, 0, 1)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(colbert.parameters(), lr=config.lr, eps=1e-8)
    
    # Initialize scheduler
    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=config.warmup
        )
    
    # Set up labels for loss calculation
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)
    
    # Track best model
    tracker.set_best_metric('loss', higher_better=False)
    
    # Training loop
    for batch_idx, BatchSteps in enumerate(reader):
        if batch_idx >= config.maxsteps:
            break
            
        # Log learning rate
        tracker.log_metric('lr', optimizer.param_groups[0]['lr'], batch_idx)
        
        # Process batch
        this_batch_loss = 0.0
        
        for batch in BatchSteps:
            # Get batch data
            try:
                queries, passages, target_scores = batch
                encoding = [queries, passages]
            except:
                encoding, target_scores = batch
                encoding = [encoding.to(DEVICE)]
            
            # Forward pass
            scores = colbert(*encoding)
            
            # Calculate loss
            if len(target_scores) and not config.ignore_scores:
                target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                target_scores = target_scores * config.distillation_alpha
                target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)
                
                log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
            else:
                loss = torch.nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])
            
            # Backward pass
            loss = loss / config.accumsteps
            loss.backward()
            
            this_batch_loss += loss.item()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        # Log metrics
        metrics = {
            'loss': this_batch_loss,
            'batch': batch_idx
        }
        tracker.log_epoch(batch_idx, metrics)
        
        # Save checkpoint
        if batch_idx % config.save_every == 0:
            tracker.save_checkpoint(colbert, optimizer, batch_idx)
            
        # Save best model
        if batch_idx % 10 == 0:
            tracker.save_model(colbert, f"model_{batch_idx}.pt")
    
    # Save final model
    tracker.save_model(colbert, "final_model.pt")
    
    # Log final results
    results = {
        'total_steps': batch_idx,
        'final_loss': this_batch_loss,
        'best_loss': tracker.best_metric_value
    }
    tracker.save_results(results)
    
    return tracker.best_model_path

def main():
    args = parse_args()
    
    # Load config
    config = ColBERTConfig.from_file(args.config)
    
    # Set seed for reproducibility
    set_seed(12345)
    
    # Initialize tracker
    with TrainingTracker(args.experiment, args.job_id, args.output_dir) as tracker:
        # Train model
        best_model_path = train_colbert(
            config=config,
            triples=args.triples,
            queries=args.queries,
            collection=args.collection,
            tracker=tracker
        )
        
        print(f"Training completed. Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main() 