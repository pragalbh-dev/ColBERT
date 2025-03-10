import time
import torch
import random
import torch.nn as nn
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints

import torch.distributed as dist
from colbert.infra.run import Run

def validate(colbert, val_reader, config, step_idx):
    """
    Calculate validation loss on the validation dataset.
    
    Args:
        colbert: The model to evaluate
        val_reader: LazyBatcher for validation data
        config: ColBERTConfig containing training parameters
        step_idx: Current training step index for logging
        
    Returns:
        The average validation loss
    """
    # Switch to evaluation mode
    training = colbert.training
    colbert.eval()
    
    validation_loss = torch.tensor(0.0, device=DEVICE)
    num_batches = 0
    
    # Labels tensor for CrossEntropyLoss
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)
    print("validation")
    start_batch_idx=0
    
    with torch.no_grad():
        # Match the training loop's iteration pattern
        # for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), val_reader):
        i=0
        for BatchSteps in val_reader: 
            ### FIXME: this is a hack to stop the validation loop from running forever. we need to reiterate over the same dataset every validatrion run, 
            ### but in one validation run, we want to iterate ove the entire dataset . hence shuffle=false makes it possible to have only 1 validation run whule 
            # ### shuffle =True makes one run run for inf steps
            if i>len(val_reader)//val_reader.bsize:
                break
            print("batch steps")
            # Now iterate over each batch in BatchSteps
            for val_batch in BatchSteps:
                print("val batch: {}".format(i))
                i+=1
                # Process validation batch
                try:
                    queries, passages, target_scores = val_batch
                    encoding = [queries, passages]
                except:
                    encoding, target_scores = val_batch
                    encoding = [encoding.to(DEVICE)]
                    
                scores = colbert(*encoding)
                
                if config.use_ib_negatives:
                    scores, _ = scores
                    
                scores = scores.view(-1, config.nway)
                
                # Calculate loss
                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)
                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    batch_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                else:
                    batch_loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])
                
                validation_loss += batch_loss
                num_batches += 1
                print(num_batches,'batches done')
    
    is_distributed = config.nranks > 1 and dist.is_available() and dist.is_initialized()
    
    # Wait for all processes to finish evaluation
    if is_distributed:
        dist.barrier()
        
        # Sum validation losses from all processes
        dist.all_reduce(validation_loss, op=dist.ReduceOp.SUM)
        
        # Average the validation loss (divide by world_size * num_batches)
        avg_val_loss = validation_loss / (dist.get_world_size() * max(1, num_batches))
    else:
        avg_val_loss = validation_loss / max(1, num_batches)
    
    # Apply exponential moving average (EMA) smoothing to validation loss
    # Default smoothing factor if not specified
    val_ema_alpha = 0.95 if not hasattr(config, 'val_ema_alpha') else config.val_ema_alpha
    
    # Initialize smoothed loss on first validation or update existing
    if not hasattr(config, 'smoothed_val_loss'):
        print("initializing smoothed val loss")
        config.smoothed_val_loss = avg_val_loss.item()
    else:
        config.smoothed_val_loss = val_ema_alpha * config.smoothed_val_loss + (1 - val_ema_alpha) * avg_val_loss.item()
    
    # Only rank 0 logs results
    if config.rank < 1:
        # Log both raw and smoothed validation loss
        Run().log_metric('val/loss_raw', avg_val_loss.item(), step=step_idx)
        Run().log_metric('val/loss_smooth', config.smoothed_val_loss, step=step_idx)
        print_message(f"Step {step_idx}: Val loss = {avg_val_loss.item():.4f}, Smoothed = {config.smoothed_val_loss:.4f}")
    
    # Restore the previous training mode
    # if training:
    colbert.train()
    
    # Return smoothed validation loss for model selection
    return config.smoothed_val_loss

def train(config: ColBERTConfig, triples, queries=None, collection=None,val_triples=None,val_queries=None,val_collection=None):
    config.checkpoint = config.checkpoint or 'bert-base-uncased'
    
    if config.rank < 1:
        print(f'we are in the main process: {Run().rank}')
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks,shuffle=False)
            val_reader = LazyBatcher(config, val_triples, val_queries, val_collection, (0 if config.rank == -1 else config.rank), config.nranks,shuffle=True)
    else:
        raise NotImplementedError()

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()
    is_distributed = config.nranks > 1 and dist.is_available() and dist.is_initialized()
        
    if is_distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                            output_device=config.rank,
                                                            find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.95

    start_batch_idx = 0
    
    # Configure validation settings
    val_check_interval = config.val_check_interval if hasattr(config, 'val_check_interval') else 50
    best_val_loss = float('inf')

    # if config.resume:
    #     assert config.checkpoint is not None
    #     start_batch_idx = checkpoint['batch']

    #     reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        for batch in BatchSteps: ### mini batches 
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                except:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                scores = colbert(*encoding)

                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                else:
                    loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

                if config.use_ib_negatives:
                    if config.rank < 1:
                        print('\t\t\t\t', loss.item(), ib_loss.item())

                    loss += ib_loss

                loss = loss / config.accumsteps

            if config.rank < 1:
                print_progress(scores)

            amp.backward(loss) ### accumulating gradients from all mini batches

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        amp.step(colbert, optimizer, scheduler) ### updating the model parameters
        
        # Log training metrics
        if Run().rank == 0:
            print_message(batch_idx, train_loss)
            # Add tracking of training loss
            Run().log_metric('train/loss_mav', train_loss, step=batch_idx)
            Run().log_metric('train/loss_this_batch', this_batch_loss, step=batch_idx)
        
        # Run validation at regular intervals
        if val_triples is not None and (batch_idx + 1) % val_check_interval == 0:
            val_loss = validate(colbert, val_reader, config, batch_idx)
            
            # Save best model based on validation loss
            if val_loss < best_val_loss and config.rank < 1:
                best_val_loss = val_loss
                manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, is_best=True)
        
        # Regular checkpoint saving
        if config.rank < 1:
            manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None)

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True)

        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.



def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
