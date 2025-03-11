import os
import random
import torch
import numpy as np
import pickle
import time
from pathlib import Path
import pandas as pd
from colbert.data.train_data_preprocessor import TripletDatasetSplitter
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
# from colbert.trainer import Trainer
from colbert.trainer import SingleGPUTrainer as Trainer
from colbert.utils.tracker import ColBERTTracker
from colbert.negative_miners.simple_miner import SimpleMiner
import os

def load_real_data(labeled_pairs_path,collections_path,aspect_delimiter='||'):
    labeled_pairs=pd.read_pickle(labeled_pairs_path)
    collections=pd.read_csv(collections_path,sep='\t')
    labeled_pairs['made_up_query']=labeled_pairs.apply(lambda x:f'{x.aspect}{aspect_delimiter}{x.actual_query}',axis=1)
    
    labeled_pairs=list(zip(labeled_pairs.made_up_query.to_list(),labeled_pairs.collection.to_list(),labeled_pairs.label.to_list()))
    collections=collections.collection.to_list()
    return labeled_pairs,collections

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

def train(labelled_pairs_path=None,collections_path=None,load_from_disk=False,triples_path=None, queries_path=None):
    try:
        nranks=4
    
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
        if triples_path is None:
            s=time.time()
            if labelled_pairs_path is None:
                print("Creating sample dataset...")
                labeled_pairs, documents = create_sample_dataset(num_queries=20, num_docs=200,multiplier=None)
            else:
                print("loading dataset...")
                labeled_pairs, documents=load_real_data(labeled_pairs_path,collections_path,aspect_delimiter='||')
        
            print(f'time for loading = {time.time()-s}')
            s=time.time()
            # 3. Split the dataset
            print("Splitting dataset...")
            negative_miner=SimpleMiner(language_code='other')
            
            splitter = TripletDatasetSplitter(
                labeled_pairs=labeled_pairs,
                collection=documents,
                negative_miner=negative_miner,  # No need for a miner in this example
                aspect_delimiter="||",
                train_val_test_ratio=(0.6, 0.1, 0.3),
                train_pos_neg_ratio=12.0,
                val_pos_neg_ratio=12.0,
                test_pos_neg_ratio=8.0,
                seed=42,
                debug=False
            )
            
            print(f'time for splitting = {time.time()-s}')
            # Split and process the dataset
            s=time.time()
            if os.path.exists('/home/ec2-user/SageMaker/data/triplets.pkl') and load_from_disk:
                try:
                    datasets=pd.read_pickle('/home/ec2-user/SageMaker/data/triplets.pkl')
                except Exception as e:
                    print(f'cannot read: {e}')
                    datasets = splitter.process_data(
                    output_dir=data_dir,
                    max_triplets_per_query=10000,  # Increased from 10
                    train_max_positives=100,  # Increased from 2
                    test_max_positives=50,
                    val_max_positives=2
                    )
                
                    with open('/home/ec2-user/SageMaker/data/triplets.pkl','wb') as f:
                        pickle.dump(datasets,f)
            else:
                datasets = splitter.process_data(
                    output_dir=data_dir,
                    max_triplets_per_query=10000,  # Increased from 10
                    train_max_positives=100,  # Increased from 2
                    test_max_positives=50,
                    val_max_positives=2
                )
            
                with open('./data/triplets.pkl','wb') as f:
                    pickle.dump(datasets,f)
        
            print()
            negative_miner.free_gpu_memory()
            del negative_miner
            del splitter
            print(f'creating triplets = {time.time()-s}')
            del labeled_pairs
            del documents
            del datasets
    
            
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
    
            # ColBERT configuration
            config = ColBERTConfig(
                bsize=64*nranks,  # Small batch size for testing
                accumsteps=2,
                lr=5e-6,
                nway=2,  # Binary pairs for simplicity  
                query_maxlen=128,  
                doc_maxlen=512,   
                dim=128,
                similarity="cosine",
                use_ib_negatives=False,
                maxsteps=10000,  # Limit training steps
                warmup=250,
                nranks=nranks,
                val_check_interval=150,
                val_ema_alpha=0.95,
                attend_to_mask_tokens=False
            )
            
            # Make sure to pass the RunConfig settings to the ColBERTConfig
            if nranks<=1:
                config.rank = 0
    
            # Setup paths to training data
            if triples_path is not None:
                print(f'triples path provided {triples_path} loading data')
                data_dirs=[str(data_dir / "train" ),str(data_dir / "val" )]
                for _data_dir in data_dirs:
                    os.makedirs(_data_dir,exist_ok=True)
                os.system(f'cp {triples_path["train"]} {os.path.join(data_dir , "train" , "triples.train.colbert.jsonl")}')
                os.system(f'cp {queries_path["train"]} {os.path.join(data_dir , "train" , "queries.train.colbert.tsv")}')
                os.system(f'cp {collections_path["train"]} {os.path.join(data_dir , "train" , "corpus.train.colbert.tsv")}')
                
                os.system(f'cp {triples_path["val"]} {os.path.join(data_dir , "val" , "triples.train.colbert.jsonl")}')
                os.system(f'cp {queries_path["val"]} {os.path.join(data_dir , "val" , "queries.train.colbert.tsv")}')
                os.system(f'cp {collections_path["val"]} {os.path.join(data_dir , "val" , "corpus.train.colbert.tsv")}')
    
                
            triples = os.path.join(data_dir , "train" , "triples.train.colbert.jsonl")
            queries = os.path.join(data_dir , "train" , "queries.train.colbert.tsv")
            collection = os.path.join(data_dir , "train" , "corpus.train.colbert.tsv")
            val_triples = os.path.join(data_dir , "val" , "triples.train.colbert.jsonl")
            val_queries = os.path.join(data_dir , "val" , "queries.train.colbert.tsv")
            val_collection = os.path.join(data_dir , "val" , "corpus.train.colbert.tsv")
            # Initialize trainer and run training
            print("Starting training...")
            
            tracker_config={'experiment_name':experiment_name,
                            'run_name':run_name,
                            'log_dir':str(experiment_dir / "logs"),
                            'enable_tensorboard':True,
                            'rank':0}
            
            trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config,
                              tracker_config=tracker_config,val_triples=val_triples,val_queries=val_queries,val_collection=val_collection)
            trainer.train(checkpoint='bert-base-uncased')
            
            # Get the path to the best checkpoint
            best_checkpoint = trainer.best_checkpoint_path()
            print(f"Training completed. Best checkpoint: {best_checkpoint}")
        
        # Close the tracker after the context ends
        if Run().rank==0:
            Run().close()
        print(f"Done! in {time.time()-s}")

    finally: 

        if nranks>1:
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # labeled_pairs_path='/home/ec2-user/SageMaker/data/labelled_pairs.all.pkl'
    # collections_path='/home/ec2-user/SageMaker/data/collections.all.tsv'
    
    triples_path={'val':'/home/ec2-user/SageMaker/ColBERT/experiments/colbert_aspect_training/run_1741558706/data/val/triples.train.colbert.oversampled.jsonl',
                 
                 'train':'/home/ec2-user/SageMaker/ColBERT/experiments/colbert_aspect_training/run_1741558706/data/train/triples.train.colbert.shuffled.jsonl'}

    
    collections_path={'val':'/home/ec2-user/SageMaker/ColBERT/experiments/colbert_aspect_training/run_1741558706/data/val/corpus.train.colbert.tsv',
                     'train':'/home/ec2-user/SageMaker/ColBERT/experiments/colbert_aspect_training/run_1741558706/data/train/corpus.train.colbert.tsv'}
    
    queries_path={'val':'/home/ec2-user/SageMaker/ColBERT/experiments/colbert_aspect_training/run_1741558706/data/val/queries.train.colbert.tsv',
                  'train':'/home/ec2-user/SageMaker/ColBERT/experiments/colbert_aspect_training/run_1741558706/data/train/queries.train.colbert.tsv'}

    train(triples_path=triples_path,queries_path=queries_path,collections_path=collections_path)
    # train(labeled_pairs_path,collections_path)
    # train()