import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple, Union, Optional, Literal
import srsly

class AspectTrainingDataProcessor:
    """
    Processes training data for ColBERT with queries in the format [ASPECT]<delimiter>[Query_Phrase].
    
    FIXME: This implementation assumes exhaustive labeling - i.e., for every query-document pair,
    a label exists in the input data. This needs to be generalized for datasets where this assumption
    doesn't hold.
    """
    
    def __init__(
        self,
        labeled_pairs: List[Tuple[str, str, int]],  # (query, passage, label)
        collection: List[str],
        negative_miner = None,
        aspect_delimiter: str = "||",
        pos_neg_ratio: float = 1.0,  # 1:1 ratio by default
        negative_sampling_weights: Dict[str, float] = None,  # Weights for each strategy
        seed: int = 42,
        debug: bool = False
    ):
        self.labeled_pairs = labeled_pairs
        self.collection = collection
        self.negative_miner = negative_miner
        self.aspect_delimiter = aspect_delimiter
        self.pos_neg_ratio = pos_neg_ratio
        self.debug = debug
        
        # Default sampling weights if none provided
        self.negative_sampling_weights = negative_sampling_weights or {
            "rule_based": 0.5, 
            "random": 0.3, 
            "miner": 0.2
        }
        
        # Normalize weights
        total_weight = sum(self.negative_sampling_weights.values())
        for k in self.negative_sampling_weights:
            self.negative_sampling_weights[k] /= total_weight
            
        random.seed(seed)
        
        # Parse query aspects and build data structures
        self._parse_query_aspects()
        self._make_data_maps()
        self._build_rule_based_negatives()
        
        # Results
        self.training_triplets = []
        self.sampling_origins = {}  # Map triplet -> sampling technique when in debug mode
    
    def _parse_query_aspects(self):
        """Parse query aspects from labeled pairs."""
        self.queries = set()
        self.aspect_to_queries = defaultdict(list)
        self.query_to_aspect = {}
        self.query_to_base = {}
        
        for query, _, _ in self.labeled_pairs:
            self.queries.add(query)
            if self.aspect_delimiter in query:
                aspect, base = query.split(self.aspect_delimiter, 1)
                self.aspect_to_queries[aspect].append(query)
                self.query_to_aspect[query] = aspect
                self.query_to_base[query] = base
        
        if self.debug:
            unique_bases = set(self.query_to_base.values())
            print(f"Processed {len(self.queries)} queries with {len(unique_bases)} unique base queries")
    
    def _make_data_maps(self):
        """Create mappings for queries and passages, and separate positive/negative pairs."""
        self.query_map = {q: idx for idx, q in enumerate(self.queries)}
        self.passage_map = {p: idx for idx, p in enumerate(set(self.collection))}
        
        # Group labeled pairs by query and label
        self.query_positives = defaultdict(set)
        self.query_negatives = defaultdict(set)
        
        for query, passage, label in self.labeled_pairs:
            if label == 1:
                self.query_positives[query].add(passage)
            else:
                self.query_negatives[query].add(passage)
                
        if self.debug:
            print(f"Created mappings for {len(self.query_map)} queries and {len(self.passage_map)} passages")
    
    def _build_rule_based_negatives(self):
        """
        Build rule-based hard negatives.
        These are passages that are positive for one aspect but negative for another aspect
        of the same base query.
        """
        rule_based_negatives = {}
        
        # Get base queries (without aspect)
        base_to_queries = {}
        for query in self.queries:
            if self.aspect_delimiter in query:
                aspect, base = query.split(self.aspect_delimiter, 1)
                if base not in base_to_queries:
                    base_to_queries[base] = []
                base_to_queries[base].append(query)
        
        # For each base query, find passages that are positive for one aspect but negative for others
        for base, queries in base_to_queries.items():
            if len(queries) <= 1:  # Skip if only one aspect
                continue
                
            # For each aspect, find passages that are positive
            aspect_positives = {}
            for query in queries:
                aspect = query.split(self.aspect_delimiter, 1)[0]
                aspect_positives[aspect] = self.query_positives.get(query, set())
            
            # For each aspect's query, add other aspect's positives as hard negatives
            # if they're explicitly negative or not labeled for this aspect
            for query in queries:
                query_aspect = query.split(self.aspect_delimiter, 1)[0]
                rule_negatives = set()
                
                # Get passages positive for other aspects
                for other_aspect, positives in aspect_positives.items():
                    if other_aspect == query_aspect:
                        continue
                    
                    # For each passage positive for other aspect
                    for passage in positives:
                        # Only add if it's negative or not labeled for this query
                        if passage in self.query_negatives.get(query, set()) or passage not in self.query_positives.get(query, set()):
                            # Only add if the passage exists in our collection
                            if passage in self.passage_map:
                                rule_negatives.add(passage)
                
                if rule_negatives:
                    rule_based_negatives[query] = rule_negatives
        
        self.rule_based_negatives = rule_based_negatives
        
        if self.debug:
            total_negatives = sum(len(negs) for negs in rule_based_negatives.values())
            print(f"Found {total_negatives} rule-based hard negatives across {len(rule_based_negatives)} queries")
    
    def process_data(self, max_triplets_per_query=20, max_positives=None, export_path=None):
        """
        Process the data and generate triplets.
        
        Args:
            max_triplets_per_query: Maximum number of triplets to generate per query
            max_positives: Maximum number of positives to use per query (None = use all)
            export_path: Path to export the generated data
        
        Returns:
            List of triplets, plus debug info if debug=True
        """
        training_triplets = []
        # Track generated triplets to avoid duplicates
        generated_triplets_set = set()
        
        # Debug info
        sampling_origins = {} if self.debug else None
        
        for query in self.query_map:
            positives = list(self.query_positives[query])
            if not positives:
                continue  # Skip queries with no positives
            
            # Apply max_positives cap if specified
            if max_positives and len(positives) > max_positives:
                # Randomly sample max_positives
                positives = random.sample(positives, max_positives)
            
            # For each positive, we want pos_neg_ratio negatives
            triplets_per_positive = int(self.pos_neg_ratio)
            
            # Generate triplets for each positive
            for positive in positives:
                # Get negatives for this query from different strategies
                sampled_negatives, origins = self._sample_negatives(
                    query, 
                    positive, 
                    triplets_per_positive
                )
                
                # Create triplets
                q_id = self.query_map[query]
                p_id = self.passage_map[positive]
                
                for i, negative in enumerate(sampled_negatives):
                    n_id = self.passage_map[negative]
                    # Check for duplicates before adding
                    triplet = (q_id, p_id, n_id)
                    if triplet not in generated_triplets_set:
                        triplet_as_list = [q_id, p_id, n_id]
                        training_triplets.append(triplet_as_list)
                        generated_triplets_set.add(triplet)
                        
                        # Store origin if in debug mode
                        if self.debug:
                            # Use tuple as key (hashable and consistent)
                            sampling_origins[tuple(triplet_as_list)] = origins[i]
        
        self.training_triplets = training_triplets
        if self.debug:
            self.sampling_origins = sampling_origins
        
        print(f"Generated {len(training_triplets)} unique training triplets")
        
        # Export if path provided
        if export_path:
            self.export_training_data(export_path)
        
        # Return triplets and sampling origins if in debug mode
        if self.debug:
            return self.training_triplets, self.sampling_origins
        return self.training_triplets
    
    def _sample_negatives(self, query, positive, count):
        """Sample negatives using multiple strategies based on weights."""
        # Track sampled negatives to ensure uniqueness
        sampled_negatives = []
        sampled_negatives_origins = []  # Track where each negative came from
        sampled_negatives_set = set()
        
        # Calculate how many negatives to sample from each strategy
        strategy_counts = {}
        remaining = count
        
        for strategy, weight in self.negative_sampling_weights.items():
            if strategy != list(self.negative_sampling_weights.keys())[-1]:
                strategy_counts[strategy] = int(count * weight)
                remaining -= strategy_counts[strategy]
            else:
                # Assign remaining to last strategy to avoid rounding issues
                strategy_counts[strategy] = remaining
        
        # Sample from each strategy
        if strategy_counts.get("rule_based", 0) > 0:
            rule_based = self._sample_rule_based_negatives(
                query, strategy_counts["rule_based"], sampled_negatives_set
            )
            for neg in rule_based:
                if neg not in sampled_negatives_set:
                    sampled_negatives.append(neg)
                    sampled_negatives_set.add(neg)
                    sampled_negatives_origins.append("rule_based")
        
        if strategy_counts.get("random", 0) > 0:
            random_negs = self._sample_random_negatives(
                query, positive, strategy_counts["random"], sampled_negatives_set
            )
            for neg in random_negs:
                if neg not in sampled_negatives_set:
                    sampled_negatives.append(neg)
                    sampled_negatives_set.add(neg)
                    sampled_negatives_origins.append("random")
        
        if strategy_counts.get("miner", 0) > 0 and self.negative_miner:
            miner_negs = self._sample_miner_negatives(
                query, positive, strategy_counts["miner"], sampled_negatives_set
            )
            for neg in miner_negs:
                if neg not in sampled_negatives_set:
                    sampled_negatives.append(neg)
                    sampled_negatives_set.add(neg)
                    sampled_negatives_origins.append("miner")
        
        # Ensure we have enough negatives (fall back to random if needed)
        if len(sampled_negatives) < count:
            additional = self._sample_random_negatives(
                query, positive, count - len(sampled_negatives), sampled_negatives_set
            )
            for neg in additional:
                if neg not in sampled_negatives_set:
                    sampled_negatives.append(neg)
                    sampled_negatives_set.add(neg)
                    sampled_negatives_origins.append("random_fallback")
        
        return sampled_negatives, sampled_negatives_origins if self.debug else sampled_negatives
    
    def _sample_rule_based_negatives(self, query, count, already_sampled=None):
        """Sample from rule-based hard negatives."""
        if already_sampled is None:
            already_sampled = set()
        
        candidates = []
        for c in self.rule_based_negatives.get(query, set()):
            # Only include candidates that exist in our collection
            if c in self.passage_map and c not in already_sampled:
                candidates.append(c)
        
        return random.sample(candidates, min(count, len(candidates))) if candidates else []
    
    def _sample_random_negatives(self, query, positive, count, already_sampled=None):
        """Sample random negatives from the collection."""
        if already_sampled is None:
            already_sampled = set()
        
        # Get passages that are explicitly negative for this query
        explicit_negatives = [n for n in self.query_negatives.get(query, set()) 
                            if n in self.passage_map and n not in already_sampled]
        
        # Start with explicit negatives not already sampled
        candidates = explicit_negatives
        
        # If we need more, sample from the rest of the collection
        if len(candidates) < count:
            additional_candidates = [
                p for p in self.collection 
                if (p not in self.query_positives.get(query, set()) and 
                    p != positive and 
                    p not in already_sampled)
            ]
            
            # Avoid sampling too many if we only need a few
            sample_size = min(count * 10, len(additional_candidates))
            if sample_size > 0:  # Only sample if we have candidates
                candidates.extend(random.sample(additional_candidates, sample_size))
        
        return random.sample(candidates, min(count, len(candidates))) if candidates else []
    
    def _sample_miner_negatives(self, query, positive, count, already_sampled=None):
        """Sample negatives using the provided negative miner."""
        if already_sampled is None:
            already_sampled = set()
        
        if not self.negative_miner:
            return []
            
        # Get hard negatives from miner
        hard_negatives = self.negative_miner.mine_hard_negatives(query)
        
        # Filter out positives, already sampled, and documents not in our collection
        candidates = [
            neg for neg in hard_negatives 
            if (neg in self.passage_map and
                neg not in self.query_positives.get(query, set()) and 
                neg != positive and 
                neg not in already_sampled)
        ]
        
        return random.sample(candidates, min(count, len(candidates))) if candidates else []
    
    def export_training_data(self, path: Union[str, Path]):
        """Export training data in ColBERT format."""
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        
        # Export queries
        with open(path / "queries.train.colbert.tsv", "w") as f:
            for query, idx in self.query_map.items():
                query = query.replace("\t", " ").replace("\n", " ")
                f.write(f"{idx}\t{query}\n")
        
        # Export collection
        with open(path / "corpus.train.colbert.tsv", "w") as f:
            for document, idx in self.passage_map.items():
                document = document.replace("\t", " ").replace("\n", " ")
                f.write(f"{idx}\t{document}\n")
        
        # Export triplets
        random.shuffle(self.training_triplets)
        srsly.write_jsonl(path / "triples.train.colbert.jsonl", self.training_triplets)
        
        print(f"Exported training data to {path}")
        return path

class AspectDatasetSplitter:
    """
    Splits a dataset into training, validation, and test sets for ColBERT aspect-based retrieval.
    Ensures test documents are completely isolated from training/validation documents.
    Creates specialized test sets for evaluating different negative sampling strategies.
    """
    
    def __init__(
        self,
        labeled_pairs: List[Tuple[str, str, int]],  # (query, passage, label)
        collection: List[str],
        negative_miner = None,
        aspect_delimiter: str = "||",
        train_val_test_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2),  # Default split ratio
        pos_neg_ratio: float = 1.0,
        seed: int = 42,
        debug: bool = False
    ):
        self.labeled_pairs = labeled_pairs
        self.collection = collection
        self.negative_miner = negative_miner
        self.aspect_delimiter = aspect_delimiter
        self.train_val_test_ratio = train_val_test_ratio
        self.pos_neg_ratio = pos_neg_ratio
        self.seed = seed
        self.debug = debug
        
        # Validate split ratio
        assert sum(train_val_test_ratio) == 1.0, "Split ratios must sum to 1.0"
        
        random.seed(seed)
        
        # Set up data structures
        self.query_to_aspect = {}
        self.query_to_base = {}
        self.base_to_queries = defaultdict(list)
        self._parse_query_aspects()
        
        # Document assignments (test documents must not appear in train/val)
        self.train_documents = set()
        self.val_documents = set()
        self.test_documents = set()
        
        # Splits
        self.train_pairs = []
        self.val_pairs = []
        self.test_pairs = []
        self.rule_based_test_pairs = []
        
    def _parse_query_aspects(self):
        """Parse aspects and query phrases from queries."""
        for query, _, _ in self.labeled_pairs:
            parts = query.split(self.aspect_delimiter, 1)
            if len(parts) == 2:
                aspect, base = parts[0], parts[1]
                self.query_to_aspect[query] = aspect
                self.query_to_base[query] = base
                self.base_to_queries[base].append(query)
        
        print(f"Processed {len(self.query_to_aspect)} queries with {len(self.base_to_queries)} unique base queries")
    
    def split_dataset(self):
        """
        Split the dataset into train, validation, and test sets.
        Ensures test documents don't appear in train/validation sets.
        """
        # Get all unique documents and queries
        all_documents = set(p for _, p, _ in self.labeled_pairs)
        all_queries = set(q for q, _, _ in self.labeled_pairs)
        
        # First, split documents to ensure isolation
        doc_list = list(all_documents)
        random.shuffle(doc_list)
        
        train_ratio, val_ratio, test_ratio = self.train_val_test_ratio
        train_size = int(len(doc_list) * train_ratio)
        val_size = int(len(doc_list) * val_ratio)
        
        self.train_documents = set(doc_list[:train_size])
        self.val_documents = set(doc_list[train_size:train_size+val_size])
        self.test_documents = set(doc_list[train_size+val_size:])
        
        print(f"Document split: Train={len(self.train_documents)}, Val={len(self.val_documents)}, Test={len(self.test_documents)}")
        
        # Now split labeled pairs based on document assignment
        train_pairs = []
        val_pairs = []
        test_pairs = []
        
        # Group by query to ensure we have aspect coverage in all splits
        query_to_pairs = defaultdict(list)
        for pair in self.labeled_pairs:
            query, doc, label = pair
            query_to_pairs[query].append(pair)
        
        # Process each query's pairs
        for query, pairs in query_to_pairs.items():
            train_query_pairs = []
            val_query_pairs = []
            test_query_pairs = []
            
            for pair in pairs:
                _, doc, _ = pair
                if doc in self.test_documents:
                    test_query_pairs.append(pair)
                elif doc in self.val_documents:
                    val_query_pairs.append(pair)
                else:
                    train_query_pairs.append(pair)
            
            train_pairs.extend(train_query_pairs)
            val_pairs.extend(val_query_pairs)
            test_pairs.extend(test_query_pairs)
        
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.test_pairs = test_pairs
        
        print(f"Pair split: Train={len(self.train_pairs)}, Val={len(self.val_pairs)}, Test={len(self.test_pairs)}")
        
        # Create specialized test sets
        self._create_specialized_test_sets()
        
        return self.train_pairs, self.val_pairs, self.test_pairs, self.rule_based_test_pairs
    
    def _create_specialized_test_sets(self):
        """Create specialized test sets for evaluating different negative sampling strategies."""
        # Find rule-based hard negatives for testing
        test_query_positives = defaultdict(set)
        
        # Identify positive documents for each query in test set
        for query, doc, label in self.test_pairs:
            if label == 1:
                test_query_positives[query].add(doc)
        
        # Find rule-based hard negatives for each test query
        rule_based_test_pairs = []
        
        for base_query, queries in self.base_to_queries.items():
            if len(queries) <= 1:
                continue  # Skip if only one aspect
                
            # For each pair of different aspects with the same base query
            for query1 in queries:
                positives1 = test_query_positives.get(query1, set())
                if not positives1:
                    continue
                    
                for query2 in queries:
                    if query1 == query2:
                        continue
                        
                    # Documents positive for query1 could be hard negatives for query2
                    for doc in positives1:
                        if doc in self.test_documents:
                            # Add as negative example for query2
                            rule_based_test_pairs.append((query2, doc, 0))
        
        self.rule_based_test_pairs = rule_based_test_pairs
        print(f"Created specialized rule-based test set with {len(rule_based_test_pairs)} pairs")
    
    def process_data(self, output_dir, max_triplets_per_query=20, max_positives=None,
                     train_negative_sampling_weights={"rule_based": 0.4, "random": 0.4, "miner": 0},
                     val_negative_sampling_weights={"rule_based": 0.33, "random": 0.34, "miner": 0},
                     test_negative_sampling_weights={"rule_based": 1.0, "random": 0.0, "miner": 0}):
        """
        Process the split data and generate train/val/test sets with independent IDs.
        Each split will have its own independent ID space for queries and documents.
        
        Args:
            output_dir: Directory to export the processed data
            max_triplets_per_query: Maximum triplets per query
            max_positives: Maximum positives per query
        """ 
        self.split_dataset()
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process training data with its own document set
        train_processor = AspectTrainingDataProcessor(
            labeled_pairs=self.train_pairs,
            collection=[doc for doc in self.collection if doc in self.train_documents],
            negative_miner=self.negative_miner,
            aspect_delimiter=self.aspect_delimiter,
            pos_neg_ratio=self.pos_neg_ratio,
            negative_sampling_weights=train_negative_sampling_weights,
            seed=self.seed,
            debug=self.debug
        )
        train_triplets = train_processor.process_data(
            max_triplets_per_query=max_triplets_per_query,
            max_positives=max_positives,
            export_path=output_dir / "train"
        )
        
        # Process validation data with its own document set
        val_processor = AspectTrainingDataProcessor(
            labeled_pairs=self.val_pairs,
            collection=[doc for doc in self.collection if doc in self.val_documents],
            negative_miner=self.negative_miner,
            aspect_delimiter=self.aspect_delimiter,
            pos_neg_ratio=self.pos_neg_ratio,
            negative_sampling_weights=val_negative_sampling_weights,
            seed=self.seed,
            debug=self.debug
        )
        val_triplets = val_processor.process_data(
            max_triplets_per_query=max_triplets_per_query,
            max_positives=max_positives,
            export_path=output_dir / "val"
        )
        
        # Process standard test data with its own document set
        test_processor = AspectTrainingDataProcessor(
            labeled_pairs=self.test_pairs,
            collection=[doc for doc in self.collection if doc in self.test_documents],
            negative_miner=self.negative_miner,
            aspect_delimiter=self.aspect_delimiter,
            pos_neg_ratio=self.pos_neg_ratio,
            negative_sampling_weights=test_negative_sampling_weights,
            seed=self.seed,
            debug=self.debug
        )
        test_triplets = test_processor.process_data(
            max_triplets_per_query=max_triplets_per_query,
            max_positives=max_positives,
            export_path=output_dir / "test"
        )
        
        # Process rule-based test data with its own document set
        rule_test_processor = AspectTrainingDataProcessor(
            labeled_pairs=self.test_pairs + self.rule_based_test_pairs,
            collection=[doc for doc in self.collection if doc in self.test_documents],
            negative_miner=self.negative_miner,
            aspect_delimiter=self.aspect_delimiter,
            pos_neg_ratio=self.pos_neg_ratio,
            negative_sampling_weights={"rule_based": 1.0, "random": 0.0, "miner": 0.0},
            seed=self.seed,
            debug=self.debug
        )
        rule_test_triplets = rule_test_processor.process_data(
            max_triplets_per_query=max_triplets_per_query,
            max_positives=max_positives,
            export_path=output_dir / "test_rule_based"
        )
        
        # Process miner-based test data if miner available
        miner_test_triplets = None
        if self.negative_miner:
            miner_test_processor = AspectTrainingDataProcessor(
                labeled_pairs=self.test_pairs,
                collection=[doc for doc in self.collection if doc in self.test_documents],
                negative_miner=self.negative_miner,
                aspect_delimiter=self.aspect_delimiter,
                pos_neg_ratio=self.pos_neg_ratio,
                negative_sampling_weights={"rule_based": 0.0, "random": 0.0, "miner": 1.0},
                seed=self.seed,
                debug=self.debug
            )
            miner_test_triplets = miner_test_processor.process_data(
                max_triplets_per_query=max_triplets_per_query,
                max_positives=max_positives,
                export_path=output_dir / "test_miner"
            )
        
        print(f"Processed all data splits and exported to {output_dir}")
        return {
            "train": train_triplets,
            "val": val_triplets,
            "test": test_triplets,
            "test_rule_based": rule_test_triplets,
            "test_miner": miner_test_triplets
        }
