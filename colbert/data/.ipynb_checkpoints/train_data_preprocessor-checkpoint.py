import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple, Union, Optional, Literal
import srsly
def query_transformer(query: str) -> str:
    """
    Transform the query by removing the aspect part.
    """

    return query.replace("||", " - ")

class TripletGenerator:
    """
    Generates triplets (query, positive, negative) for training ColBERT models.
    Works with queries in the format [ASPECT]<delimiter>[Query_Phrase].
    
    This class can be used to generate triplets for any dataset split (train/val/test)
    with configurable negative sampling strategies.
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
        debug: bool = False,
        query_transformer = query_transformer  # Add query transformer function
    ):
        
        self.labeled_pairs = labeled_pairs
        self.collection = collection
        self.negative_miner = negative_miner
        self.aspect_delimiter = aspect_delimiter
        self.pos_neg_ratio = pos_neg_ratio
        self.debug = debug
        self.query_transformer = query_transformer  # Store the transformer function
        if self.negative_miner is not None:
            self.negative_miner.build_index(collection)
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
        
        # Timing information for initialization steps
        start_time = time.time()
        
        # Parse query aspects and build data structures
        parse_start = time.time()
        self._parse_query_aspects()
        parse_time = time.time() - parse_start
        
        maps_start = time.time()
        self._make_data_maps()
        maps_time = time.time() - maps_start
        
        negatives_start = time.time()
        self._build_rule_based_negatives()
        negatives_time = time.time() - negatives_start
        
        # Results
        self.triplets = []
        self.sampling_origins = {}  # Map triplet -> sampling technique when in debug mode
        
        # Store timing information
        self.timing = {
            "init_total": time.time() - start_time,
            "parse_aspects": parse_time,
            "make_data_maps": maps_time,
            "build_rule_based_negatives": negatives_time,
            "generate_triplets": 0,
            "export_triplets": 0,
            "sampling": {
                "rule_based": 0,
                "random": 0,
                "miner": 0,
                "total": 0
            }
        }
        
        if self.debug:
            print(f"⏱️ TripletGenerator initialization: {self.timing['init_total']:.2f}s")
            print(f"  - Parse aspects: {parse_time:.2f}s")
            print(f"  - Make data maps: {maps_time:.2f}s")
            print(f"  - Build rule-based negatives: {negatives_time:.2f}s")
    
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
    
    def generate_triplets(self, max_triplets_per_query=20, max_positives=None, export_path=None):
        """
        Generate triplets for the dataset.
        
        Args:
            max_triplets_per_query: Maximum number of triplets to generate per query
            max_positives: Maximum number of positives to use per query (None = use all)
            export_path: Path to export the generated data
        
        Returns:
            List of triplets, plus debug info if debug=True
        """
        start_time = time.time()
        triplets = []
        # Track generated triplets to avoid duplicates
        generated_triplets_set = set()
        
        # Debug info
        sampling_origins = {} if self.debug else None
        
        # Timing for per-query operations
        query_times = []
        positive_selection_time = 0
        triplet_creation_time = 0
        
        for query in self.query_map:
            query_start = time.time()
            
            # Get and sample positives
            pos_start = time.time()
            positives = list(self.query_positives[query])
            if not positives:
                continue  # Skip queries with no positives
            
            # Apply max_positives cap if specified
            if max_positives and len(positives) > max_positives:
                # Randomly sample max_positives
                positives = random.sample(positives, max_positives)
            positive_selection_time += time.time() - pos_start
            
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
                triplet_start = time.time()
                q_id = self.query_map[query]
                p_id = self.passage_map[positive]
                
                for i, negative in enumerate(sampled_negatives):
                    n_id = self.passage_map[negative]
                    # Check for duplicates before adding
                    triplet = (q_id, p_id, n_id)
                    if triplet not in generated_triplets_set:
                        triplet_as_list = [q_id, p_id, n_id]
                        triplets.append(triplet_as_list)
                        generated_triplets_set.add(triplet)
                        
                        # Store origin if in debug mode
                        if self.debug:
                            # Use tuple as key (hashable and consistent)
                            sampling_origins[tuple(triplet_as_list)] = origins[i]
                triplet_creation_time += time.time() - triplet_start
            
            query_times.append(time.time() - query_start)
        
        self.triplets = triplets
        if self.debug:
            self.sampling_origins = sampling_origins
        
        # Export if path provided
        export_time = 0
        if export_path:
            export_start = time.time()
            self.export_triplets(export_path)
            export_time = time.time() - export_start
            self.timing["export_triplets"] = export_time
        
        # Update timing info
        total_time = time.time() - start_time
        self.timing["generate_triplets"] = total_time
        self.timing["positive_selection"] = positive_selection_time
        self.timing["triplet_creation"] = triplet_creation_time
        
        if self.debug:
            avg_query_time = sum(query_times) / len(query_times) if query_times else 0
            print(f"\n⏱️ Triplet generation completed in {total_time:.2f}s")
            print(f"  - Generated {len(triplets)} unique triplets")
            print(f"  - Average time per query: {avg_query_time*1000:.2f}ms")
            print(f"  - Positive selection: {positive_selection_time:.2f}s")
            print(f"  - Triplet creation: {triplet_creation_time:.2f}s")
            print(f"  - Negative sampling: {self.timing['sampling']['total']:.2f}s")
            print(f"    - Rule-based: {self.timing['sampling']['rule_based']:.2f}s")
            print(f"    - Random: {self.timing['sampling']['random']:.2f}s")
            print(f"    - Miner: {self.timing['sampling']['miner']:.2f}s")
            if export_path:
                print(f"  - Export: {export_time:.2f}s")
        
        # Return triplets and sampling origins if in debug mode
        if self.debug:
            return self.triplets, self.sampling_origins
        return self.triplets
    
    def _sample_negatives(self, query, positive, count):
        """Sample negatives using multiple strategies based on weights."""
        sampling_start = time.time()
        
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
            rule_start = time.time()
            rule_based = self._sample_rule_based_negatives(
                query, strategy_counts["rule_based"], sampled_negatives_set
            )
            for neg in rule_based:
                if neg not in sampled_negatives_set:
                    sampled_negatives.append(neg)
                    sampled_negatives_set.add(neg)
                    sampled_negatives_origins.append("rule_based")
            self.timing["sampling"]["rule_based"] += time.time() - rule_start
        
        if strategy_counts.get("random", 0) > 0:
            random_start = time.time()
            random_negs = self._sample_random_negatives(
                query, positive, strategy_counts["random"], sampled_negatives_set
            )
            for neg in random_negs:
                if neg not in sampled_negatives_set:
                    sampled_negatives.append(neg)
                    sampled_negatives_set.add(neg)
                    sampled_negatives_origins.append("random")
            self.timing["sampling"]["random"] += time.time() - random_start
        
        if strategy_counts.get("miner", 0) > 0 and self.negative_miner:
            miner_start = time.time()
            miner_negs = self._sample_miner_negatives(
                query, positive, strategy_counts["miner"], sampled_negatives_set
            )
            for neg in miner_negs:
                if neg not in sampled_negatives_set:
                    sampled_negatives.append(neg)
                    sampled_negatives_set.add(neg)
                    sampled_negatives_origins.append("miner")
            self.timing["sampling"]["miner"] += time.time() - miner_start
        
        # Ensure we have enough negatives (fall back to random if needed)
        if len(sampled_negatives) < count:
            fallback_start = time.time()
            additional = self._sample_random_negatives(
                query, positive, count - len(sampled_negatives), sampled_negatives_set
            )
            for neg in additional:
                if neg not in sampled_negatives_set:
                    sampled_negatives.append(neg)
                    sampled_negatives_set.add(neg)
                    sampled_negatives_origins.append("random_fallback")
            self.timing["sampling"]["random"] += time.time() - fallback_start
        
        # Update total sampling time
        self.timing["sampling"]["total"] += time.time() - sampling_start
        
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
    
    def export_triplets(self, path: Union[str, Path]):
        """Export triplets in ColBERT format."""
        start_time = time.time()
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        
        # Export transformed queries
        with open(path / "queries.train.colbert.tsv", "w") as f:
            for query, idx in self.query_map.items():
                # Apply transformation if available
                transformed_query = query
                if self.query_transformer is not None:
                    transformed_query = self.query_transformer(query)
                
                transformed_query = transformed_query.replace("\t", " ").replace("\n", " ")
                f.write(f"{idx}\t{transformed_query}\n")
        
        # Export raw queries (new)
        with open(path / "queries.train.raw.tsv", "w") as f:
            for query, idx in self.query_map.items():
                clean_query = query.replace("\t", " ").replace("\n", " ")
                f.write(f"{idx}\t{clean_query}\n")
        
        # Export collection
        with open(path / "corpus.train.colbert.tsv", "w") as f:
            for document, idx in self.passage_map.items():
                document = document.replace("\t", " ").replace("\n", " ")
                f.write(f"{idx}\t{document}\n")
        
        # Export triplets
        random.shuffle(self.triplets)
        srsly.write_jsonl(path / "triples.train.colbert.jsonl", self.triplets)
        
        # Update timing
        self.timing["export_triplets"] = time.time() - start_time
        
        if self.debug:
            print(f"Exported data to {path} in {self.timing['export_triplets']:.2f}s")
        return path



class TripletDatasetSplitter:
    """
    Splits datasets into train/val/test sets and generates triplets for each split.
    
    The class implements a flexible splitting strategy that preserves aspect coverage
    while sampling documents according to the specified ratio for each (query, label) group.
    """
    
    def __init__(
        self,
        labeled_pairs: List[Tuple[str, str, int]],  # (query, passage, label)
        collection: List[str],
        negative_miner = None,
        aspect_delimiter: str = "||",
        train_val_test_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2),  # Default split ratio
        train_pos_neg_ratio: float = 64.0,
        val_pos_neg_ratio:float =2.0,
        test_pos_neg_ratio:float=12.0,
        seed: int = 42,
        debug: bool = False
    ):
        self.labeled_pairs = labeled_pairs
        self.collection = collection
        self.negative_miner = negative_miner
        self.aspect_delimiter = aspect_delimiter
        self.train_val_test_ratio = train_val_test_ratio
        self.train_pos_neg_ratio = train_pos_neg_ratio
        self.val_pos_neg_ratio = val_pos_neg_ratio
        self.test_pos_neg_ratio = test_pos_neg_ratio
        self.seed = seed
        self.debug = debug
        
        # Validate split ratio
        assert sum(train_val_test_ratio) == 1.0, "Split ratios must sum to 1.0"
        
        random.seed(seed)
        
        # Timing information
        start_time = time.time()
        
        # Set up data structures
        self.query_to_aspect = {}
        self.query_to_base = {}
        self.aspect_to_queries = defaultdict(list)
        self.base_to_queries = defaultdict(list)
        
        # Process queries to extract aspects
        process_start = time.time()
        self._process_queries()
        process_time = time.time() - process_start
        
        # Group data by query-label for better splitting
        group_start = time.time()
        self.query_label_groups = self._group_by_query_label()
        group_time = time.time() - group_start
        
        # Document assignments
        self.train_documents = set()
        self.val_documents = set()
        self.test_documents = set()
        
        # Splits
        self.train_pairs = []
        self.val_pairs = []
        self.test_pairs = []
        
        # Store timing information
        self.timing = {
            "init_total": time.time() - start_time,
            "process_queries": process_time,
            "group_by_query_label": group_time,
            "split_dataset": 0,
            "process_data": 0,
            "triplet_generation": {
                "train": 0,
                "val": 0,
                "test": 0,
                "test_rule_based": 0,
                "test_miner": 0
            }
        }
        
        if self.debug:
            print(f"⏱️ TripletDatasetSplitter initialization: {self.timing['init_total']:.2f}s")
            print(f"  - Process queries: {process_time:.2f}s")
            print(f"  - Group by query-label: {group_time:.2f}s")
    
    def _process_queries(self):
        """Process queries to extract aspects and base queries."""
        # Extract unique queries
        unique_queries = set(q for q, _, _ in self.labeled_pairs)
        
        # Process each unique query
        for query in unique_queries:
            if self.aspect_delimiter in query:
                aspect, base = query.split(self.aspect_delimiter, 1)
                self.query_to_aspect[query] = aspect
                self.query_to_base[query] = base
                self.aspect_to_queries[aspect].append(query)
                self.base_to_queries[base].append(query)
        
        if self.debug:
            print(f"Processed {len(unique_queries)} queries with {len(self.base_to_queries)} unique base queries")
    
    def _group_by_query_label(self):
        """Group labeled pairs by query and label."""
        groups = defaultdict(list)
        
        for query, doc, label in self.labeled_pairs:
            # Use (query, label) as the group key
            key = (query, label)
            groups[key].append((query, doc, label))
        
        return groups
    
    def split_dataset(self):
        """
        Split the dataset using the improved strategy.
        For each (query, label) group, sample documents according to the split ratio.
        """
        start_time = time.time()
        
        # Reset splits
        self.train_pairs = []
        self.val_pairs = []
        self.test_pairs = []
        
        # Reset document sets
        self.train_documents = set()
        self.val_documents = set()
        self.test_documents = set()
        
        # Get split ratios
        train_ratio, val_ratio, test_ratio = self.train_val_test_ratio
        
        # Track processing time by group type
        positive_groups_time = 0
        negative_groups_time = 0
        group_counts = {"positive": 0, "negative": 0}
        
        # Process each (query, label) group
        for (query, label), pairs in self.query_label_groups.items():
            group_start = time.time()
            
            # Calculate sizes based on ratios
            group_size = len(pairs)
            train_size = int(group_size * train_ratio)
            val_size = int(group_size * val_ratio)
            
            # Randomly shuffle pairs in the group
            random.shuffle(pairs)
            
            # Split the group
            train_group = pairs[:train_size]
            val_group = pairs[train_size:train_size + val_size]
            test_group = pairs[train_size + val_size:]
            
            # Add to respective splits
            self.train_pairs.extend(train_group)
            self.val_pairs.extend(val_group)
            self.test_pairs.extend(test_group)
            
            # Update document sets
            for _, doc, _ in train_group:
                self.train_documents.add(doc)
            for _, doc, _ in val_group:
                self.val_documents.add(doc)
            for _, doc, _ in test_group:
                self.test_documents.add(doc)
            
            # Track time by group type (positive/negative)
            group_time = time.time() - group_start
            if label == 1:
                positive_groups_time += group_time
                group_counts["positive"] += 1
            else:
                negative_groups_time += group_time
                group_counts["negative"] += 1
        
        # Update timing information
        split_time = time.time() - start_time
        self.timing["split_dataset"] = split_time
        self.timing["positive_groups"] = positive_groups_time
        self.timing["negative_groups"] = negative_groups_time
        
        if self.debug:
            print(f"⏱️ Dataset splitting completed in {split_time:.2f}s")
            print(f"  - Positive groups ({group_counts['positive']}): {positive_groups_time:.2f}s")
            print(f"  - Negative groups ({group_counts['negative']}): {negative_groups_time:.2f}s")
            print(f"  - Train: {len(self.train_pairs)} pairs, {len(self.train_documents)} documents")
            print(f"  - Val: {len(self.val_pairs)} pairs, {len(self.val_documents)} documents")
            print(f"  - Test: {len(self.test_pairs)} pairs, {len(self.test_documents)} documents")
        
        return self.train_pairs, self.val_pairs, self.test_pairs
    
    def process_data(self, output_dir, max_triplets_per_query=20, train_max_positives=None,val_max_positives=None,test_max_positives=None,
                     train_negative_sampling_weights={"rule_based": 0.05, "random": 0.4, "miner": 0.55},
                     val_negative_sampling_weights={"rule_based": 0.05, "random": 0.4, "miner": 0.55},
                     test_negative_sampling_weights={"rule_based": 0.05, "random": 0.4, "miner": 0.55},
                     include_specialized_test_sets=True):
        """
        Process the data into triplets for each split.
        
        Args:
            output_dir: Directory to export the processed data
            max_triplets_per_query: Maximum triplets per query
            max_positives: Maximum positives per query
            *_negative_sampling_weights: Weights for each negative sampling strategy
            include_specialized_test_sets: Whether to generate specialized test sets
        
        Returns:
            Dictionary with triplets for each split
        """
        start_time = time.time()
        
        # Split the dataset first if not already split
        if not self.train_pairs and not self.val_pairs and not self.test_pairs:
            self.split_dataset()
        
        # Create output directory
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        triplet_gen_times = {}
        
        # Process training data
        train_start = time.time()
        train_generator = TripletGenerator(
            labeled_pairs=self.train_pairs,
            collection=[doc for doc in self.collection if doc in self.train_documents],
            negative_miner=self.negative_miner,
            aspect_delimiter=self.aspect_delimiter,
            pos_neg_ratio=self.train_pos_neg_ratio,
            negative_sampling_weights=train_negative_sampling_weights,
            seed=self.seed,
            debug=self.debug,
            query_transformer=query_transformer
        )
        results["train"] = train_generator.generate_triplets(
            max_triplets_per_query=max_triplets_per_query,
            max_positives=train_max_positives,
            export_path=output_dir / "train"
        )
        train_time = time.time() - train_start
        triplet_gen_times["train"] = train_time
        self.timing["triplet_generation"]["train"] = train_time
        
        # Process validation data
        val_start = time.time()
        val_generator = TripletGenerator(
            labeled_pairs=self.val_pairs,
            collection=[doc for doc in self.collection if doc in self.val_documents],
            negative_miner=self.negative_miner,
            aspect_delimiter=self.aspect_delimiter,
            pos_neg_ratio=self.val_pos_neg_ratio,
            negative_sampling_weights=val_negative_sampling_weights,
            seed=self.seed,
            debug=self.debug,
            query_transformer=query_transformer
        )
        results["val"] = val_generator.generate_triplets(
            max_triplets_per_query=max_triplets_per_query,
            max_positives=val_max_positives,
            export_path=output_dir / "val"
        )
        val_time = time.time() - val_start
        triplet_gen_times["val"] = val_time
        self.timing["triplet_generation"]["val"] = val_time
        
        # Process test data
        test_start = time.time()
        test_generator = TripletGenerator(
            labeled_pairs=self.test_pairs,
            collection=[doc for doc in self.collection if doc in self.test_documents],
            negative_miner=self.negative_miner,
            aspect_delimiter=self.aspect_delimiter,
            pos_neg_ratio=self.test_pos_neg_ratio,
            negative_sampling_weights=test_negative_sampling_weights,
            seed=self.seed,
            debug=self.debug,
            query_transformer=query_transformer
        )
        results["test"] = test_generator.generate_triplets(
            max_triplets_per_query=max_triplets_per_query,
            max_positives=test_max_positives,
            export_path=output_dir / "test"
        )
        test_time = time.time() - test_start
        triplet_gen_times["test"] = test_time
        self.timing["triplet_generation"]["test"] = test_time
        
        # Generate specialized test sets if requested
        if include_specialized_test_sets:
            # Rule-based test set - using the same test data but with only rule-based sampling
            rule_test_start = time.time()
            rule_test_generator = TripletGenerator(
                labeled_pairs=self.test_pairs,
                collection=[doc for doc in self.collection if doc in self.test_documents],
                negative_miner=self.negative_miner,
                aspect_delimiter=self.aspect_delimiter,
                pos_neg_ratio=self.test_pos_neg_ratio,
                negative_sampling_weights={"rule_based": 1.0, "random": 0.0, "miner": 0.0},
                seed=self.seed,
                debug=self.debug,
                query_transformer=query_transformer
            )
            results["test_rule_based"] = rule_test_generator.generate_triplets(
                max_triplets_per_query=max_triplets_per_query,
                max_positives=test_max_positives,
                export_path=output_dir / "test_rule_based"
            )
            rule_test_time = time.time() - rule_test_start
            triplet_gen_times["test_rule_based"] = rule_test_time
            self.timing["triplet_generation"]["test_rule_based"] = rule_test_time
            
            # Miner-based test set (if miner is available)
            if self.negative_miner:
                miner_test_start = time.time()
                miner_test_generator = TripletGenerator(
                    labeled_pairs=self.test_pairs,
                    collection=[doc for doc in self.collection if doc in self.test_documents],
                    negative_miner=self.negative_miner,
                    aspect_delimiter=self.aspect_delimiter,
                    pos_neg_ratio=self.test_pos_neg_ratio,
                    negative_sampling_weights={"rule_based": 0.0, "random": 0.0, "miner": 1.0},
                    seed=self.seed,
                    debug=self.debug,
                    query_transformer=query_transformer
                )
                results["test_miner"] = miner_test_generator.generate_triplets(
                    max_triplets_per_query=max_triplets_per_query,
                    max_positives=test_max_positives,
                    export_path=output_dir / "test_miner"
                )
                miner_test_time = time.time() - miner_test_start
                triplet_gen_times["test_miner"] = miner_test_time
                self.timing["triplet_generation"]["test_miner"] = miner_test_time
        
        # Update timing information
        total_time = time.time() - start_time
        self.timing["process_data"] = total_time
        
        if self.debug:
            # Calculate triplet counts
            triplet_counts = {}
            for split, result in results.items():
                if isinstance(result, tuple) and result[0]:
                    triplet_counts[split] = len(result[0])
                elif isinstance(result, list):
                    triplet_counts[split] = len(result)
                else:
                    triplet_counts[split] = 0
            
            print(f"\n⏱️ Data processing completed in {total_time:.2f}s")
            for split, time_taken in triplet_gen_times.items():
                count = triplet_counts.get(split, 0)
                rate = count / time_taken if time_taken > 0 else 0
                print(f"  - {split.capitalize()}: {time_taken:.2f}s, {count} triplets ({rate:.1f} triplets/s)")
        
        print(f"Processed all data splits and exported to {output_dir}")
        return results
