import os
import torch
import numpy as np
from typing import Dict, List, Set, Optional, Union, Any
from collections import defaultdict

from colbert.evaluation.metrics import Metrics
from colbert.evaluation.loaders import (
    load_queries, load_qrels, load_collection, 
    load_colbert, load_topK, load_topK_pids
)
from colbert.evaluation.triplet_loader import load_triplets,convert_triplets_to_qrels
from colbert.utils.utils import print_message
from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT
from pathlib import Path
class ColBERTEvaluator:
    """
    A comprehensive evaluator for ColBERT models that supports:
    - MRR, Recall, Success metrics (from existing Metrics class)
    - NDCG and Precision@k metrics
    - Integration with tracking systems for TensorBoard visualization
    - Mid-training and final evaluation
    """
    
    def __init__(
        self,
        model=None,
        checkpoint_path=None,
        config=None,
        mrr_depths: Set[int] = {10, 100, 1000},
        recall_depths: Set[int] = {10, 100, 1000},
        success_depths: Set[int] = {10, 100, 1000},
        precision_depths: Set[int] = {30,50,100,200,500},
        ndcg_depths: Set[int] = {10, 100},
        tracker=None
    ):
        """
        Initialize the evaluator with a model or checkpoint path
        
        Args:
            model: A loaded ColBERT model (optional if checkpoint_path is provided)
            checkpoint_path: Path to a model checkpoint (optional if model is provided)
            config: Configuration for the model
            mrr_depths: Depths for Mean Reciprocal Rank calculation
            recall_depths: Depths for Recall calculation
            success_depths: Depths for Success calculation
            precision_depths: Depths for Precision calculation
            ndcg_depths: Depths for NDCG calculation
            tracker: A tracking object for logging metrics (optional)
        """
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.tracker = tracker
        
        # Evaluation depths
        self.mrr_depths = mrr_depths
        self.recall_depths = recall_depths
        self.success_depths = success_depths
        self.precision_depths = precision_depths
        self.ndcg_depths = ndcg_depths
        
        # Initialize metrics
        self.metrics = None
        self.precision_sums = {depth: 0.0 for depth in precision_depths}
        self.ndcg_sums = {depth: 0.0 for depth in ndcg_depths}
        
        # Load model if not provided
        if self.model is None and self.checkpoint_path is not None:
            self._load_model()
    
    def _load_model(self):
        """Load the model from checkpoint if not provided"""
        if self.config is None:
            raise ValueError("Config must be provided to load model from checkpoint")
        self.config.configure(checkpoint=self.checkpoint_path)
        self.model = ColBERT(name=self.config.checkpoint, colbert_config=self.config)
    
    def evaluate(
        self,
        queries_path: str,
        collection_path: str,
        qrels_path: Optional[str] = None,
        triplets_path: Optional[str] = None,
        topk_path: Optional[str] = None,
        output_path: Optional[str] = None,
        batch_size: int = 128,
        depth: int = 1000,
        step: Optional[int] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate the model on a test set
        
        Args:
            queries_path: Path to queries TSV file
            collection_path: Path to collection TSV file
            qrels_path: Path to qrels TSV file (optional)
            topk_path: Path to pre-computed top-k results (optional)
            output_path: Path to save evaluation results (optional)
            batch_size: Batch size for inference
            depth: Maximum depth for retrieval
            step: Current training step (for mid-training evaluation)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load data
        queries = load_queries(queries_path)
        collection = load_collection(collection_path)
        qrels_path = convert_triplets_to_qrels(triplets_path,output_path=os.path.join(Path(queries_path).parent,'qrels.train.colbert.tsv')) if triplets_path else None
        qrels = load_qrels(qrels_path) if qrels_path else None
        
        # Initialize metrics
        self.metrics = Metrics(
            mrr_depths=self.mrr_depths,
            recall_depths=self.recall_depths,
            success_depths=self.success_depths,
            total_queries=len(queries)
        )
        
        # Reset precision and ndcg sums
        self.precision_sums = {depth: 0.0 for depth in self.precision_depths}
        self.ndcg_sums = {depth: 0.0 for depth in self.ndcg_depths}
        
        # If topk_path is provided, evaluate using pre-computed rankings
        if topk_path:
            return self._evaluate_from_topk(topk_path, qrels, step)
        
        # Otherwise, perform retrieval and evaluate
        return self._evaluate_with_retrieval(queries, collection, qrels, batch_size, depth, output_path, step)
    
    def _evaluate_from_topk(
        self,
        topk_path: str,
        qrels: Dict[int, List[int]],
        step: Optional[int] = None
    ) -> Dict[str, Dict[int, float]]:
        """Evaluate using pre-computed top-k results"""
        topk_pids, topk_positives = load_topK_pids(topk_path, qrels)
        
        # Process each query
        for query_idx, qid in enumerate(topk_pids):
            ranking = [(0, pid, 0) for pid in topk_pids[qid]]  # Format: (score, pid, passage_idx)
            gold_positives = topk_positives[qid] if qid in topk_positives else []
            
            # Add to metrics
            self.metrics.add(query_idx, qid, ranking, gold_positives)
            
            # Calculate precision and NDCG
            self._calculate_precision(ranking, gold_positives)
            self._calculate_ndcg(ranking, gold_positives)
        
        # Log metrics
        return self._log_metrics(len(topk_pids) - 1, step)
    
    def _evaluate_with_retrieval(
        self,
        queries: Dict[int, str],
        collection: List[str],
        qrels: Dict[int, List[int]],
        batch_size: int,
        depth: int,
        output_path: Optional[str] = None,
        step: Optional[int] = None
    ) -> Dict[str, Dict[int, float]]:
        """Perform retrieval and evaluate"""
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Get tokenizers (creating them if needed)
        if not hasattr(self, 'query_tokenizer'):
            from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
            self.query_tokenizer = QueryTokenizer(self.config)
            self.doc_tokenizer = DocTokenizer(self.config)
        
        # Process each query
        rankings = {}
        queries_with_relevance = 0  # Count queries that have relevant documents
        
        for query_idx, (qid, query) in enumerate(queries.items()):
            # Skip queries without relevant documents
            gold_positives = qrels.get(qid, [])
            if not gold_positives:
                print_message(f"#> Skipping query {query_idx+1} / {len(queries)} (no relevant documents)")
                continue
                
            print_message(f"#> Processing query {query_idx+1} / {len(queries)}")
            queries_with_relevance += 1
            
            # Process query in batches
            with torch.no_grad():
                # Tokenize query
                Q_tokens, Q_mask = self.query_tokenizer.tensorize([query])
                Q_tokens, Q_mask = Q_tokens.to(DEVICE), Q_mask.to(DEVICE)
                
                # Process the query encodings - directly use model's query method
                Q_encodings = self.model.query(Q_tokens, Q_mask)
                
                # Process documents in smaller batches to avoid dimension issues
                all_scores = []
                for batch_idx in range(0, len(collection), batch_size):
                    batch_docs = collection[batch_idx:batch_idx + batch_size]
                    
                    # Process each document in the batch individually to maintain correct dimensions
                    batch_scores = []
                    for doc in batch_docs:
                        # Tokenize a single document
                        D_tokens, D_mask = self.doc_tokenizer.tensorize([doc])
                        D_tokens, D_mask = D_tokens.to(DEVICE), D_mask.to(DEVICE)
                        
                        # Process document
                        D_encodings, D_enc_mask = self.model.doc(D_tokens, D_mask, keep_dims='return_mask')
                        
                        # Score this document against the query
                        doc_score = self.model.score(Q_encodings, D_encodings, D_enc_mask)
                        batch_scores.append(doc_score.item())
                    
                    # Convert batch scores to tensor
                    batch_scores_tensor = torch.tensor(batch_scores, device=DEVICE)
                    all_scores.append(batch_scores_tensor)
                
                # Concatenate all scores
                all_scores = torch.cat(all_scores, dim=0).unsqueeze(0)  # Add batch dimension for consistent handling
                
                # Get ranking (top 'depth' documents)
                ranked_indices = torch.argsort(all_scores, dim=1, descending=True)
                ranked_indices = ranked_indices[0, :depth].tolist()
                
                ranking = [(all_scores[0, idx].item(), idx, idx) for idx in ranked_indices]
                
                # Add to metrics - Use a consecutive index for metrics
                # This ensures max_query_idx is incremented properly
                rel_query_idx = queries_with_relevance - 1  # 0-indexed
                self.metrics.add(rel_query_idx, qid, ranking, gold_positives)
                rankings[qid] = ranking
                
                # Calculate precision and NDCG
                self._calculate_precision(ranking, gold_positives)
                self._calculate_ndcg(ranking, gold_positives)
        
        # Save rankings if output_path is provided
        if output_path:
            self._save_rankings(rankings, queries, collection, output_path)
        
        # Log statistics about queries
        print_message(f"#> Total queries processed: {len(queries)}")
        print_message(f"#> Queries with relevant documents: {queries_with_relevance}")
        print_message(f"#> Queries without relevant documents: {len(queries) - queries_with_relevance}")
        
        # Log metrics with queries_with_relevance instead of total queries
        return self._log_metrics(queries_with_relevance, step)
    
    def _calculate_precision(self, ranking, gold_positives):
        """Calculate precision at different depths"""
        for depth in self.precision_depths:
            if not gold_positives:
                continue
                
            hits = sum(1 for _, pid, _ in ranking[:depth] if pid in gold_positives)
            self.precision_sums[depth] += hits / min(depth, len(ranking))
    
    def _calculate_ndcg(self, ranking, gold_positives):
        """Calculate NDCG at different depths"""
        for depth in self.ndcg_depths:
            if not gold_positives:
                continue
                
            # Calculate DCG
            dcg = 0
            for i, (_, pid, _) in enumerate(ranking[:depth]):
                if pid in gold_positives:
                    # Using log base 2 as is standard for NDCG
                    dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed and log_2(1) = 0
            
            # Calculate ideal DCG
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_positives), depth)))
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            self.ndcg_sums[depth] += ndcg
    
    def _log_metrics(self, queries_with_relevance, step=None):
        """Log metrics and return results, using only queries with relevant docs"""
        # Log metrics from the Metrics class - use queries_with_relevance-1 as the max_query_idx
        # because metrics.add() uses 0-indexed query indices
        if queries_with_relevance > 0:  # Avoid division by zero if no queries had relevant docs
            self.metrics.log(queries_with_relevance - 1)
        else:
            # Handle edge case: no queries with relevance
            print_message("#> No queries with relevant documents. Metrics will be zeros.")
            return defaultdict(dict)
        
        # Create results dictionary
        results = defaultdict(dict)
        
        # Add MRR, Success, and Recall from Metrics class
        # Using queries_with_relevance as denominator instead of total queries
        for depth in sorted(self.metrics.mrr_sums):
            score = self.metrics.mrr_sums[depth] / max(1, queries_with_relevance)
            results['mrr'][depth] = score
            
            if self.tracker and step is not None:
                self.tracker.log_metric(f"eval/MRR@{depth}", score, step)
        
        for depth in sorted(self.metrics.success_sums):
            score = self.metrics.success_sums[depth] / max(1, queries_with_relevance)
            results['success'][depth] = score
            
            if self.tracker and step is not None:
                self.tracker.log_metric(f"eval/Success@{depth}", score, step)
        
        for depth in sorted(self.metrics.recall_sums):
            score = self.metrics.recall_sums[depth] / max(1, queries_with_relevance)
            results['recall'][depth] = score
            
            if self.tracker and step is not None:
                self.tracker.log_metric(f"eval/Recall@{depth}", score, step)
        
        # Add Precision
        for depth in sorted(self.precision_sums):
            score = self.precision_sums[depth] / max(1, queries_with_relevance)
            results['precision'][depth] = score
            
            if self.tracker and step is not None:
                self.tracker.log_metric(f"eval/Precision@{depth}", score, step)
                
            print(f"Precision@{depth} = {score}")
        
        # Add NDCG
        for depth in sorted(self.ndcg_sums):
            score = self.ndcg_sums[depth] / max(1, queries_with_relevance)
            results['ndcg'][depth] = score
            
            if self.tracker and step is not None:
                self.tracker.log_metric(f"eval/NDCG@{depth}", score, step)
                
            print(f"NDCG@{depth} = {score}")
        
        return results
    
    def _save_rankings(self, rankings, queries, collection, output_path):
        """Save rankings to a file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for qid, ranking in rankings.items():
                query = queries[qid]
                
                for score, pid, _ in ranking:
                    passage = collection[pid]
                    f.write(f"{qid}\t{pid}\t{score:.6f}\t{query}\t{passage}\n") 