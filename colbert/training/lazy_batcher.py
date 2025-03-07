import os
import ujson

from functools import partial
from colbert.infra.config.config import ColBERTConfig
from colbert.utils.utils import print_message, zipstar
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.evaluation.loaders import load_collection

from colbert.data.collection import Collection
from colbert.data.queries import Queries
from colbert.data.examples import Examples

# from colbert.utils.runs import Run

import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import random

class LazyBatcher():
    def __init__(self, config: ColBERTConfig, triples, queries, collection, rank=0, nranks=1, shuffle=True, fb_protection=False,divide_into_groups=True):
        """
        Args:
            triples: List of triples (query_id, pos_id, neg_id)
            queries: Dict mapping query_id -> query text
            collection: Dict mapping passage_id -> passage text
            rank: Process rank for distributed training
            nranks: Total number of ranks for distributed training
            shuffle: Whether to shuffle the triples
            fb_protection: Whether to protect against false negatives in batch
            divide_into_groups: Whether to divide the triples into groups based on rank of process |  basically for distributed training
        """
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway
        self.divide_into_groups = divide_into_groups
        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        if self.divide_into_groups:
            self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)
        else:
            self.triples = Examples.cast(triples, nway=self.nway).tolist()
        
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)
        assert len(self.triples) > 0, "Received no triples on which to train."
        assert len(self.queries) > 0, "Received no queries on which to train."
        assert len(self.collection) > 0, "Received no collection on which to train."

        # Use config.bsize for batch size
        self.batch_size = config.bsize
        self.shuffle = shuffle
        self.fb_protection = fb_protection

        # Filter triples for this process
        # self.triples = self._filter_triples_for_rank(rank, nranks)
        
        # If false negative protection is enabled, organize triples
        if self.fb_protection:
            self.triples = self._organize_triples_for_false_negative_protection()
            
        # Ensure the number of triples is divisible by batch size
        self._ensure_triples_divisible_by_bsize()
        
    def _filter_triples_for_rank(self, rank, nranks):
        """Filter triples for this process rank."""
        if nranks > 1:
            return [t for idx, t in enumerate(self.triples) if idx % nranks == rank]
        return self.triples
    
    def _organize_triples_for_false_negative_protection(self):
        """
        Organize triples to prevent false negatives in batches.
        Returns reorganized triples list.
        
        Note: The caller must ensure the returned list size is divisible by batch size
        by calling _ensure_triples_divisible_by_bsize() after this method.
        """
        print("Building query-positive mappings...")
        # Step 1: Create query->positives mapping
        query_to_positives = defaultdict(set)
        for qid, pid, _ in tqdm(self.triples):
            query_to_positives[qid].add(pid)
        
        print(f"Found {len(query_to_positives)} queries with positives")
        
        # Step 2: Create a graph where queries are nodes and edges exist if queries share positives
        print("Creating query conflict graph...")
        query_conflicts = defaultdict(set)
        
        # Find all query pairs that share positives
        for q1 in query_to_positives:
            for q2 in query_to_positives:
                if q1 != q2 and query_to_positives[q1].intersection(query_to_positives[q2]):
                    query_conflicts[q1].add(q2)
        
        # Step 3: Use graph coloring to assign queries to batches
        print("Assigning queries to batches using graph coloring...")
        query_to_color = {}
        
        # Sort queries by degree (number of conflicts) for better coloring
        sorted_queries = sorted(
            query_to_positives.keys(),
            key=lambda q: len(query_conflicts[q]),
            reverse=True
        )
        
        # Assign colors (batch groups) to queries
        for query in sorted_queries:
            # Find the first available color not used by neighbors
            neighbor_colors = {query_to_color.get(n) for n in query_conflicts[query] if n in query_to_color}
            
            # Find the first available color
            color = 0
            while color in neighbor_colors:
                color += 1
            
            query_to_color[query] = color
        
        # Count the number of colors used
        num_colors = max(query_to_color.values()) + 1 if query_to_color else 0
        print(f"Assigned queries to {num_colors} conflict-free groups")
        
        # Step 4: Create batches based on the coloring
        # Group triples by query
        query_to_triples = defaultdict(list)
        for triple in self.triples:
            qid = triple[0]
            query_to_triples[qid].append(triple)
        
        # Shuffle triples within each query if needed
        if self.shuffle:
            for qid in query_to_triples:
                np.random.shuffle(query_to_triples[qid])
        
        # Group triples by color
        color_to_triples = defaultdict(list)
        for qid, triples in query_to_triples.items():
            color = query_to_color.get(qid, 0)  # Default to color 0 if not assigned
            color_to_triples[color].extend(triples)
        
        # Create final list of triples organized by color
        sorted_triples = []
        
        # Process each color group
        for color in range(num_colors):
            color_triples = color_to_triples.get(color, [])
            
            if self.shuffle:
                np.random.shuffle(color_triples)
            
            # Create full batches from this color group
            num_full_batches = len(color_triples) // self.batch_size
            
            for i in range(num_full_batches):
                batch_start = i * self.batch_size
                batch_end = batch_start + self.batch_size
                sorted_triples.extend(color_triples[batch_start:batch_end])
            
            # Add remaining triples to the end
            remaining_idx = num_full_batches * self.batch_size
            if remaining_idx < len(color_triples):
                sorted_triples.extend(color_triples[remaining_idx:])
        
        # Verify that our batches don't have false negatives
        print("Verifying batch organization...")
        batch_count = len(sorted_triples) // self.batch_size
        false_negative_free_batches = 0
        
        for i in range(batch_count):
            batch_start = i * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(sorted_triples))
            batch_triples = sorted_triples[batch_start:batch_end]
            
            # Check for potential false negatives in this batch
            batch_queries = {t[0] for t in batch_triples}
            has_conflicts = False
            
            for q1 in batch_queries:
                for q2 in batch_queries:
                    if q1 != q2 and q1 in query_conflicts.get(q2, set()):
                        has_conflicts = True
                        break
                if has_conflicts:
                    break
            
            if not has_conflicts:
                false_negative_free_batches += 1
        
        print(f"Created {false_negative_free_batches}/{batch_count} false-negative-free batches")
        print(f"Reorganized {len(sorted_triples)} triples to prevent false negatives")
        
        return sorted_triples
    
    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset = self.position
        
        if offset >= len(self.triples):
            raise StopIteration
        
        end_offset = min(offset + self.batch_size, len(self.triples))
        batch_triples = self.triples[offset:end_offset]
        
        # Move position forward
        self.position = end_offset
        
        # If we've issued all batches and shuffle is enabled, reshuffle
        if self.position >= len(self.triples) and self.shuffle:
            if self.fb_protection:
                # Reorganize with false negative protection
                self.triples = self._organize_triples_for_false_negative_protection()
            else:
                # Simple shuffle
                np.random.shuffle(self.triples)
            
            # Ensure divisibility by batch size is maintained after shuffling
            self._ensure_triples_divisible_by_bsize()
            
            self.position = 0
        
        # If false negative protection is enabled, ensure this batch doesn't have false negatives
        if self.fb_protection:
            batch_triples = self._ensure_batch_has_no_false_negatives(batch_triples)
            
        # Verify batch size - this should never fail now because we pad the triples at initialization
        # if len(batch_triples) < self.bsize:
            # import pdb; pdb.set_trace()
        if len(batch_triples) < self.bsize:
            print(f"Warning: Got batch of size {len(batch_triples)}, expected {self.bsize}")
            # Either pad this batch to full size:
            padding_needed = self.bsize - len(batch_triples)
            padding_triples = batch_triples[:padding_needed]  # Reuse existing triples
            batch_triples.extend(padding_triples)
            # Or raise a StopIteration if you don't want to process partial batches
        return self._prepare_batch(batch_triples)
    
    def _prepare_batch(self, batch_triples):
        """Convert triples to the required batch format."""
        queries_batch = [self.queries[qid] for qid, _, _ in batch_triples]
        passages_batch = []
        
        # For in-batch negatives, we need all passages
        for _, pos_id, neg_id in batch_triples:
            passages_batch.append(self.collection[pos_id])
            passages_batch.append(self.collection[neg_id])
        
        # Store the original query and passage IDs for testing
        self.last_batch_query_ids = [qid for qid, _, _ in batch_triples]
        self.last_batch_passage_ids = []
        for _, pos_id, neg_id in batch_triples:
            self.last_batch_passage_ids.append(pos_id)
            self.last_batch_passage_ids.append(neg_id)
        
        # If false negative protection is enabled, check for potential false negatives
        if self.fb_protection:
            # Build query->positives mapping for this batch
            batch_query_to_positives = defaultdict(set)
            
            # First add the explicit positives from this batch
            for qid, pos_id, _ in batch_triples:
                batch_query_to_positives[qid].add(pos_id)
            
            # Check for potential false negatives in this batch
            potential_false_negatives = 0
            
            # For each query, check all passages that aren't this query's own passages
            for i, qid in enumerate(self.last_batch_query_ids):
                positives = batch_query_to_positives[qid]
                
                # The query's own passages (pos+neg) are at positions i*2 and i*2+1
                own_passage_indices = {i*2, i*2+1}
                
                # Check all other passages in the batch
                for j, pid in enumerate(self.last_batch_passage_ids):
                    if j not in own_passage_indices and pid in positives:
                        potential_false_negatives += 1
            
            if potential_false_negatives > 0:
                # If we detect false negatives despite protection, we can try to fix the batch
                # by replacing problematic passages with safe ones
                if hasattr(self, 'safe_passages') and self.safe_passages:
                    # We have a list of safe passages (not positive for any query in this batch)
                    # Replace the false negatives with safe passages
                    print(f"Fixing {potential_false_negatives} potential false negatives in batch")
                    
                    # This would require modifying the batch, which is complex
                    # For now, we'll just log the warning
                    print(f"Warning: Detected {potential_false_negatives} potential false negatives in batch despite protection")
                else:
                    print(f"Warning: Detected {potential_false_negatives} potential false negatives in batch despite protection")
        
        return self.collate(queries_batch, passages_batch, [])

    def collate(self, queries, passages, scores):
        # Original strict assertions since we now ensure proper batch sizes
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize
        return self.tensorize_triples(queries, passages, scores, self.bsize // self.accumsteps, self.nway)

    def _ensure_batch_has_no_false_negatives(self, batch_triples):
        """
        Ensure the batch doesn't have false negatives by modifying it if necessary.
        """
        # Build query->positives mapping for all queries in the dataset
        if not hasattr(self, 'query_to_positives'):
            self.query_to_positives = defaultdict(set)
            for qid, pid, _ in self.triples:
                self.query_to_positives[qid].add(pid)
        
        # Extract query IDs and passage IDs from the batch
        batch_query_ids = [qid for qid, _, _ in batch_triples]
        batch_passage_ids = []
        for _, pos_id, neg_id in batch_triples:
            batch_passage_ids.append(pos_id)
            batch_passage_ids.append(neg_id)
        
        # Check for false negatives
        has_false_negatives = False
        false_negative_positions = []
        
        for i, qid in enumerate(batch_query_ids):
            positives = self.query_to_positives[qid]
            
            # The query's own passages (pos+neg) are at positions i*2 and i*2+1
            own_passage_indices = {i*2, i*2+1}
            
            # Check all other passages in the batch
            for j, pid in enumerate(batch_passage_ids):
                if j not in own_passage_indices and pid in positives:
                    has_false_negatives = True
                    false_negative_positions.append((i, j, pid))
        
        # If there are false negatives, fix the batch
        if has_false_negatives:
            print(f"Fixing {len(false_negative_positions)} false negatives in batch")
            
            # Create a set of all positive passages for any query in this batch
            all_positives = set()
            for qid in batch_query_ids:
                all_positives.update(self.query_to_positives[qid])
            
            # Find safe negative passages (not positive for any query in the batch)
            safe_negatives = []
            for pid in range(len(self.collection)):
                if pid not in all_positives:
                    safe_negatives.append(pid)
            
            if not safe_negatives:
                print("Warning: No safe negatives found, cannot fix batch")
                return batch_triples
            
            # Create a modified batch by replacing false negatives with safe negatives
            modified_batch = []
            
            for i, (qid, pos_id, neg_id) in enumerate(batch_triples):
                # Check if this triple's negative is a false negative for any other query
                is_false_negative = False
                for other_i, j, pid in false_negative_positions:
                    if other_i != i and (j == i*2 or j == i*2+1):
                        is_false_negative = True
                        break
                
                if is_false_negative:
                    # Replace with a safe negative
                    safe_neg_id = random.choice(safe_negatives)
                    modified_batch.append((qid, pos_id, safe_neg_id))
                else:
                    # Keep the original triple
                    modified_batch.append((qid, pos_id, neg_id))
            
            return modified_batch
        
        return batch_triples

    def _ensure_triples_divisible_by_bsize(self):
        """
        Ensure that the number of triples is divisible by the batch size.
        If not, add padding triples by duplicating existing ones.
        """
        remainder = len(self.triples) % self.bsize
        if remainder > 0:
            original_count = len(self.triples)
            padding_needed = self.bsize - remainder
            
            # Create padding triples by duplicating from the beginning
            # This ensures we have valid triples that the model can process
            padding_triples = self.triples[:padding_needed]
            self.triples.extend(padding_triples)
            
            print(f"Added {padding_needed} padding triples to make dataset size ({original_count}) divisible by batch size ({self.bsize})")
            print(f"New dataset size: {len(self.triples)}")

    # def skip_to_batch(self, batch_idx, intended_batch_size):
    #     Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
    #     self.position = intended_batch_size * batch_idx
