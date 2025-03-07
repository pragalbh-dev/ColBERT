import os
import random
import torch
import numpy as np
import time
from pathlib import Path

from colbert.data.train_data_preprocessor import TripletDatasetSplitter
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
# from colbert.trainer import Trainer
from colbert.trainer import SingleGPUTrainer as Trainer
from colbert.utils.tracker import ColBERTTracker
def load_real_data():

    pass

def create_sample_dataset(num_queries=50, num_docs=200, aspect_delimiter="||",multiplier=None):
    """Create a small synthetic dataset for testing"""
    aspects = ["price", "quality", "service", "location", "features"]
    base_queries = [
        "how is the", "tell me about the", "what's the", "describe the", 
        "information on", "details about", "explain the", "rate the"
    ]
    
    # Create documents with more specific content
    documents = []
    for i in range(num_docs):
        # Create documents that mention specific aspects
        mentioned_aspects = random.sample(aspects, k=random.randint(1, len(aspects)))
        doc_text = f"Document {i}: "
        
        for aspect in mentioned_aspects:
            quality = random.choice(["excellent", "good", "average", "poor", "terrible"])
            doc_text += f"The {aspect} is {quality}. "
            
        documents.append(doc_text)
    
    # Create labeled pairs with more structure
    labeled_pairs = []
    for i in range(num_queries):
        # Select a random aspect for this query
        aspect = random.choice(aspects)
        base = random.choice(base_queries)
        query = f"{aspect}{aspect_delimiter}{base} {aspect}"
        
        # Find documents that mention this aspect (for positive examples)
        positive_docs = [idx for idx, doc in enumerate(documents) if aspect in doc.lower()]
        
        # If no documents mention this aspect, create one that does
        if not positive_docs:
            new_doc_idx = len(documents)
            quality = random.choice(["excellent", "good", "average", "poor", "terrible"])
            new_doc = ". ".join([f"Document {new_doc_idx}: The {aspect} is {quality}."]*100)
            documents.append(new_doc)
            positive_docs = [new_doc_idx]
        
        # Create positive examples (documents that mention the aspect)
        for _ in range(min(5, len(positive_docs))):
            doc_idx = random.choice(positive_docs)
            labeled_pairs.append((query, documents[doc_idx], 1))
        
        # Create negative examples (documents that don't mention the aspect)
        negative_docs = [idx for idx in range(len(documents)) if idx not in positive_docs]
        for _ in range(min(8, len(negative_docs))):
            doc_idx = random.choice(negative_docs)
            labeled_pairs.append((query, documents[doc_idx], 0))
    if multiplier:
        labeled_pairs=labeled_pairs*multiplier
    print(f"Created {len(labeled_pairs)} labeled pairs for {len(set([q for q, _, _ in labeled_pairs]))} unique queries")
    print(f"Collection has {len(documents)} documents")
    return labeled_pairs, documents

def train():
    nranks=1

    avoid_fork_if_possible=False
    if nranks<=1:
        avoid_fork_if_possible=True
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Create experiment folder
    experiment_name = "colbert_aspect_training"
    run_name = f"run_{int(time.time())}"

    print(run_name)
    base_dir = Path("experiments")
    experiment_dir = base_dir / experiment_name / run_name
    os.makedirs(experiment_dir, exist_ok=True)
    
    data_dir = experiment_dir / "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 2. Create a sample dataset
    print("Creating sample dataset...")
    labeled_pairs, documents = create_sample_dataset(num_queries=20, num_docs=500,multiplier=None)
    
    # 3. Split the dataset
    print("Splitting dataset...")
    splitter = TripletDatasetSplitter(
        labeled_pairs=labeled_pairs,
        collection=documents,
        negative_miner=None,  # No need for a miner in this example
        aspect_delimiter="||",
        train_val_test_ratio=(0.7, 0.1, 0.2),
        pos_neg_ratio=1.0,
        seed=42,
        debug=False
    )
    
    # Split and process the dataset
    datasets = splitter.process_data(
        output_dir=data_dir,
        max_triplets_per_query=100,  # Increased from 10
        max_positives=10  # Increased from 2
    )
    
    # Setup run context configuration
    # For single-GPU training (nranks=1), we use avoid_fork_if_possible=True
    # to prevent distributed initialization issues
    run_config = RunConfig(
        nranks=nranks,  # For single machine training
        amp=True,  # Mixed precision
        experiment=experiment_name,
        root=str(base_dir),
        name=run_name,
        avoid_fork_if_possible=avoid_fork_if_possible  # Important for single-GPU training
    )
    
    # Initialize the tracker first (outside the context)
    
    

    
    s=time.time()
    # Use the Run context for training
    with Run().context(run_config):
        # # Set the tracker in the Run context
        # if Run().rank==0 and hasattr(Run(), 'tracker') and Run().tracker is None:
        #     tracker = ColBERTTracker(
        #                 experiment_name=experiment_name,
        #                 run_name=run_name,
        #                 log_dir=str(experiment_dir / "logs"),
        #                 enable_tensorboard=True,
        #                 rank=Run().rank
        #             )
            
            # # Add debug print to verify tracker initialization
            # print(f"INIT DEBUG: Created tracker with log_dir={str(experiment_dir / 'logs')}")
            # print(f"INIT DEBUG: TensorBoard dir={tracker.tensorboard_dir}")
            # Run().set_tracker(tracker)
        
        # # Add debug print to verify tracker is set in Run()
        # if hasattr(Run(), 'tracker'):
        #     print(f"INIT DEBUG: Verified tracker is set in Run()")
        # else:
        #     print(f"INIT DEBUG: ERROR - tracker not set in Run()")
        
        # ColBERT configuration
        config = ColBERTConfig(
            bsize=2*nranks,  # Small batch size for testing
            accumsteps=1,
            lr=5e-6,
            nway=2,  # Binary pairs for simplicity  
            query_maxlen=128,  
            doc_maxlen=512,   
            dim=128,
            similarity="cosine",
            use_ib_negatives=False,
            maxsteps=100,  # Limit training steps
            warmup=10,
            nranks=nranks,
            val_check_interval=5,
            val_ema_alpha=0.9

        )
        
        # Make sure to pass the RunConfig settings to the ColBERTConfig
        if nranks<=1:
            config.rank = 0
        # config.nranks = 1
        # config.avoid_fork_if_possible = True
        
        # # Log the configuration
        # if Run().rank==0:
        #     print(Run().tracker)
        
        # Setup paths to training data
        triples = str(data_dir / "train" / "triples.train.colbert.jsonl")
        queries = str(data_dir / "train" / "queries.train.colbert.tsv")
        collection = str(data_dir / "train" / "corpus.train.colbert.tsv")
        val_triples = str(data_dir / "val" / "triples.train.colbert.jsonl")
        val_queries = str(data_dir / "val" / "queries.train.colbert.tsv")
        val_collection = str(data_dir / "val" / "corpus.train.colbert.tsv")
        # Initialize trainer and run training
        print("Starting training...")
        
        tracker_config={'experiment_name':experiment_name,
                        'run_name':run_name,
                        'log_dir':str(experiment_dir / "logs"),
                        'enable_tensorboard':True,
                        'rank':0}
        
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config,tracker_config=tracker_config,val_triples=val_triples,val_queries=val_queries,val_collection=val_collection)
        trainer.train(checkpoint='bert-base-uncased')
        
        # Get the path to the best checkpoint
        best_checkpoint = trainer.best_checkpoint_path()
        print(f"Training completed. Best checkpoint: {best_checkpoint}")
    
    # Close the tracker after the context ends
    if Run().rank==0:
        Run().close()
    print(f"Done! in {time.time()-s}")

if __name__ == "__main__":
    train()