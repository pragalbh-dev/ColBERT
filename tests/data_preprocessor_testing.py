import os
import random
import shutil
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import time

from colbert.data.train_data_preprocessor import AspectTrainingDataProcessor
from colbert.negative_miners.simple_miner import SimpleMiner

def test_aspect_processor():
    """Test AspectTrainingDataProcessor with synthetic data."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create a small collection of passages
    collection = [
        "Information about product A specifications.",
        "Customer reviews for product A are mostly positive.",
        "Price comparison for product A shows it's competitive.",
        "Information about product B specifications.",
        "Customer reviews for product B are mixed.",
        "Price comparison for product B shows it's expensive.",
        "Information about product C specifications.",
        "Customer reviews for product C are negative.",
        "Price comparison for product C shows it's the cheapest option.",
        "General comparison of products A, B, and C.",
    ]
    
    # Create labeled pairs with queries containing aspects
    labeled_pairs = [
        # Product A
        ("SPECS||product A", "Information about product A specifications.", 1),
        ("SPECS||product A", "Customer reviews for product A are mostly positive.", 0),
        ("SPECS||product A", "Price comparison for product A shows it's competitive.", 0),
        ("REVIEWS||product A", "Information about product A specifications.", 0),
        ("REVIEWS||product A", "Customer reviews for product A are mostly positive.", 1),
        ("REVIEWS||product A", "Price comparison for product A shows it's competitive.", 0),
        ("PRICE||product A", "Information about product A specifications.", 0),
        ("PRICE||product A", "Customer reviews for product A are mostly positive.", 0),
        ("PRICE||product A", "Price comparison for product A shows it's competitive.", 1),
        
        # Product B
        ("SPECS||product B", "Information about product B specifications.", 1),
        ("SPECS||product B", "Customer reviews for product B are mixed.", 0),
        ("SPECS||product B", "Price comparison for product B shows it's expensive.", 0),
        ("REVIEWS||product B", "Information about product B specifications.", 0),
        ("REVIEWS||product B", "Customer reviews for product B are mixed.", 1),
        ("REVIEWS||product B", "Price comparison for product B shows it's expensive.", 0),
        ("PRICE||product B", "Information about product B specifications.", 0),
        ("PRICE||product B", "Customer reviews for product B are mixed.", 0),
        ("PRICE||product B", "Price comparison for product B shows it's expensive.", 1),
        
        # Add some overlaps with product C
        ("SPECS||product C", "Information about product C specifications.", 1),
        ("SPECS||product C", "General comparison of products A, B, and C.", 0),
    ]
    
    # Test directory
    test_dir = Path("test_aspect_processor_output")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Initialize processor
    processor = AspectTrainingDataProcessor(
        labeled_pairs=labeled_pairs,
        collection=collection,
        negative_miner=None,  # No miner for this test
        aspect_delimiter="||",
        pos_neg_ratio=10.0,  # 2 negatives for each positive
        negative_sampling_weights={
            "rule_based": 0.7,
            "random": 0.3,
            "miner": 0.0  # No miner, so weight is 0
        }
    )
    
    # Process data
    processor.process_data(max_triplets_per_query=10, max_positives=2, export_path=test_dir)
    
    # Analyze results
    print("\n\n--- ANALYSIS OF GENERATED TRIPLETS ---")
    
    # Load generated triplets
    triplets = []
    import srsly
    triplets = list(srsly.read_jsonl(test_dir / "triples.train.colbert.jsonl"))
    
    # Load query and passage mappings
    queries = {}
    passages = {}
    
    with open(test_dir / "queries.train.colbert.tsv") as f:
        for line in f:
            idx, query = line.strip().split("\t", 1)
            queries[int(idx)] = query
            
    with open(test_dir / "corpus.train.colbert.tsv") as f:
        for line in f:
            idx, passage = line.strip().split("\t", 1)
            passages[int(idx)] = passage
    
    # Group by query
    query_triplets = defaultdict(list)
    for q_id, p_id, n_id in triplets:
        query = queries[q_id]
        positive = passages[p_id]
        negative = passages[n_id]
        query_triplets[query].append((positive, negative))
    
    # Print stats
    print(f"Total queries: {len(query_triplets)}")
    print(f"Total triplets: {len(triplets)}")
    print(f"Average triplets per query: {len(triplets) / len(query_triplets):.2f}")
    
    # Check for rule-based negatives
    for query, trips in query_triplets.items():
        if query.startswith("REVIEWS||product A"):
            print(f"\nExample for query '{query}':")
            
            for i, (pos, neg) in enumerate(trips[:3]):  # Show first 3
                print(f"  Positive: {pos[:50]}...")
                print(f"  Negative: {neg[:50]}...")
                
                # Check if this is likely a rule-based negative
                if "product A" in neg and ("specifications" in neg or "price" in neg):
                    print(f"  ✓ This appears to be a rule-based negative (relevant to another aspect)")
                print()
    
    # Verify rule-based negatives are being used
    base_queries = defaultdict(list)
    for q in query_triplets.keys():
        if "||" in q:
            aspect, base = q.split("||", 1)
            base_queries[base].append((aspect, q))
    
    # Check triplets for products with multiple aspects
    for base, aspects in base_queries.items():
        if len(aspects) > 1:
            print(f"\nAnalyzing negatives across aspects for '{base}':")
            shared_positives = set()
            
            for aspect, query in aspects:
                for pos, _ in query_triplets[query]:
                    if any(pos in [p for p, _ in query_triplets[other_q]] for _, other_q in aspects if other_q != query):
                        shared_positives.add(pos)
            
            print(f"  Found {len(shared_positives)} passages that are positive for multiple aspects")
            
            # Count how many times these appear as negatives for other aspects
            rule_based_usage = 0
            for aspect, query in aspects:
                for _, neg in query_triplets[query]:
                    if neg in shared_positives:
                        rule_based_usage += 1
            
            print(f"  These passages were used {rule_based_usage} times as negatives for other aspects")
    
    # After analyzing results, add a uniqueness check
    print(f"Checking for duplicates...")
    unique_triplets = set(tuple(t) for t in triplets)
    if len(unique_triplets) == len(triplets):
        print("✓ No duplicate triplets found")
    else:
        print(f"✗ Found {len(triplets) - len(unique_triplets)} duplicate triplets")
    
    print("\nTest completed!")
    return test_dir

def test_aspect_processor_debug():
    """Test AspectTrainingDataProcessor with debug mode enabled."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create a small collection of passages
    collection = [
        "Information about product A specifications.",
        "Customer reviews for product A are mostly positive.",
        "Price comparison for product A shows it's competitive.",
        "Information about product B specifications.",
        "Customer reviews for product B are mixed.",
        "Price comparison for product B shows it's expensive.",
        "Information about product C specifications.",
        "Customer reviews for product C are negative.",
        "Price comparison for product C shows it's the cheapest option.",
        "General comparison of products A, B, and C.",
    ]
    
    # Create labeled pairs with queries containing aspects
    labeled_pairs = [
        # Product A
        ("SPECS||product A", "Information about product A specifications.", 1),
        ("SPECS||product A", "Customer reviews for product A are mostly positive.", 0),
        ("SPECS||product A", "Price comparison for product A shows it's competitive.", 0),
        ("REVIEWS||product A", "Information about product A specifications.", 0),
        ("REVIEWS||product A", "Customer reviews for product A are mostly positive.", 1),
        ("REVIEWS||product A", "Price comparison for product A shows it's competitive.", 0),
        ("PRICE||product A", "Information about product A specifications.", 0),
        ("PRICE||product A", "Customer reviews for product A are mostly positive.", 0),
        ("PRICE||product A", "Price comparison for product A shows it's competitive.", 1),
        
        # Product B
        ("SPECS||product B", "Information about product B specifications.", 1),
        ("SPECS||product B", "Customer reviews for product B are mixed.", 0),
        ("SPECS||product B", "Price comparison for product B shows it's expensive.", 0),
        ("REVIEWS||product B", "Information about product B specifications.", 0),
        ("REVIEWS||product B", "Customer reviews for product B are mixed.", 1),
        ("REVIEWS||product B", "Price comparison for product B shows it's expensive.", 0),
        ("PRICE||product B", "Information about product B specifications.", 0),
        ("PRICE||product B", "Customer reviews for product B are mixed.", 0),
        ("PRICE||product B", "Price comparison for product B shows it's expensive.", 1),
        
        # Add some overlaps with product C
        ("SPECS||product C", "Information about product C specifications.", 1),
        ("SPECS||product C", "General comparison of products A, B, and C.", 0),
    ]
    
    # Test directory
    test_dir = Path("test_aspect_processor_debug_output")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Initialize processor with debug=True
    processor = AspectTrainingDataProcessor(
        labeled_pairs=labeled_pairs,
        collection=collection,
        negative_miner=None,
        aspect_delimiter="||",
        pos_neg_ratio=10.0,  # 2 negatives for each positive
        negative_sampling_weights={
            "rule_based": 0.7,
            "random": 0.3,
            "miner": 0.0
        },
        debug=True  # Enable debug mode
    )
    
    # Process data with debugging
    triplets, origins = processor.process_data(
        max_triplets_per_query=10, 
        max_positives=2, 
        export_path=test_dir
    )
    
    # Load query and passage mappings
    queries = {}
    passages = {}
    
    with open(test_dir / "queries.train.colbert.tsv") as f:
        for line in f:
            idx, query = line.strip().split("\t", 1)
            queries[int(idx)] = query
            
    with open(test_dir / "corpus.train.colbert.tsv") as f:
        for line in f:
            idx, passage = line.strip().split("\t", 1)
            passages[int(idx)] = passage
    
    # Count origin types
    origin_counts = {
        "rule_based": 0,
        "random": 0,
        "miner": 0,
        "random_fallback": 0
    }
    
    for origin in origins.values():
        origin_counts[origin] += 1
    
    print("\n--- NEGATIVE SAMPLING ORIGINS ---")
    for origin, count in origin_counts.items():
        if count > 0:
            print(f"{origin}: {count} ({count/len(triplets)*100:.1f}%)")
    
    # Example: Show some triplets with their origins
    print("\nSample triplets with origins:")
    sample_size = min(5, len(triplets))
    for i, triplet in enumerate(triplets[:sample_size]):
        q_id, p_id, n_id = triplet
        query = queries[q_id]
        positive = passages[p_id]
        negative = passages[n_id]
        origin = origins[tuple(triplet)]  # Use tuple instead of string
        
        print(f"Triplet {i+1}:")
        print(f"  Query: {query}")
        print(f"  Positive: {positive[:50]}...")
        print(f"  Negative: {negative[:50]}... ({origin})")
        print()
    
    print("\nTest completed!")
    return test_dir

def test_with_simple_miner():
    """Test AspectTrainingDataProcessor with SimpleMiner for hard negative mining."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create a small collection of passages
    collection = [
        "Information about product A specifications.",
        "Customer reviews for product A are mostly positive.",
        "Price comparison for product A shows it's competitive.",
        "Information about product B specifications.",
        "Customer reviews for product B are mixed.",
        "Price comparison for product B shows it's expensive.",
        "Information about product C specifications.",
        "Customer reviews for product C are negative.",
        "Price comparison for product C shows it's the cheapest option.",
        "General comparison of products A, B, and C.",
        # Add more documents to make mining meaningful
        "Product A has excellent build quality and reliability.",
        "Product B has average build quality but good features.",
        "Product C has poor build quality but is very affordable.",
        "Battery life of product A is above average.",
        "Battery life of product B is decent for its class.",
        "Battery life of product C is disappointing.",
        "User interface of product A is intuitive and responsive.",
        "User interface of product B is somewhat confusing.",
        "User interface of product C is basic but functional.",
        "Warranty period for product A is standard.",
        "Warranty for product B is shorter than competitors.",
        "Warranty for product C is surprisingly comprehensive."
    ]
    
    # Create labeled pairs with queries containing aspects
    labeled_pairs = [
        # Product A
        ("SPECS||product A", "Information about product A specifications.", 1),
        ("SPECS||product A", "Customer reviews for product A are mostly positive.", 0),
        ("SPECS||product A", "Price comparison for product A shows it's competitive.", 0),
        ("REVIEWS||product A", "Information about product A specifications.", 0),
        ("REVIEWS||product A", "Customer reviews for product A are mostly positive.", 1),
        ("REVIEWS||product A", "Price comparison for product A shows it's competitive.", 0),
        ("PRICE||product A", "Information about product A specifications.", 0),
        ("PRICE||product A", "Customer reviews for product A are mostly positive.", 0),
        ("PRICE||product A", "Price comparison for product A shows it's competitive.", 1),
        
        # Product B
        ("SPECS||product B", "Information about product B specifications.", 1),
        ("SPECS||product B", "Customer reviews for product B are mixed.", 0),
        ("SPECS||product B", "Price comparison for product B shows it's expensive.", 0),
        ("REVIEWS||product B", "Information about product B specifications.", 0),
        ("REVIEWS||product B", "Customer reviews for product B are mixed.", 1),
        ("REVIEWS||product B", "Price comparison for product B shows it's expensive.", 0),
        ("PRICE||product B", "Information about product B specifications.", 0),
        ("PRICE||product B", "Customer reviews for product B are mixed.", 0),
        ("PRICE||product B", "Price comparison for product B shows it's expensive.", 1),
        
        # Product C
        ("SPECS||product C", "Information about product C specifications.", 1),
        ("SPECS||product C", "General comparison of products A, B, and C.", 0),
    ]
    
    print("\n=== Testing with SimpleMiner ===")
    
    # Initialize the SimpleMiner (using English small model)
    miner = SimpleMiner(language_code="en", model_size="small")
    
    # Pre-build the index with the collection
    miner.build_index(collection)
    
    # First run WITHOUT miner to establish baseline time
    test_dir_no_miner = Path("test_no_miner_output")
    if test_dir_no_miner.exists():
        shutil.rmtree(test_dir_no_miner)
    
    processor_no_miner = AspectTrainingDataProcessor(
        labeled_pairs=labeled_pairs,
        collection=collection,
        negative_miner=None,
        aspect_delimiter="||",
        pos_neg_ratio=2.0,
        negative_sampling_weights={
            "rule_based": 0.7,
            "random": 0.3,
            "miner": 0.0
        },
        debug=True
    )
    
    start_time = time.time()
    triplets_no_miner, origins_no_miner = processor_no_miner.process_data(
        max_triplets_per_query=10,
        max_positives=2,
        export_path=test_dir_no_miner
    )
    time_no_miner = time.time() - start_time
    print(f"Processing WITHOUT miner took {time_no_miner:.2f} seconds")
    
    # Now run WITH miner
    test_dir_with_miner = Path("test_with_miner_output")
    if test_dir_with_miner.exists():
        shutil.rmtree(test_dir_with_miner)
    
    processor_with_miner = AspectTrainingDataProcessor(
        labeled_pairs=labeled_pairs,
        collection=collection,
        negative_miner=miner,
        aspect_delimiter="||",
        pos_neg_ratio=2.0,
        negative_sampling_weights={
            "rule_based": 0.4,
            "random": 0.2,
            "miner": 0.4  # Give significant weight to miner
        },
        debug=True
    )
    
    start_time = time.time()
    triplets_with_miner, origins_with_miner = processor_with_miner.process_data(
        max_triplets_per_query=10,
        max_positives=2,
        export_path=test_dir_with_miner
    )
    time_with_miner = time.time() - start_time
    print(f"Processing WITH miner took {time_with_miner:.2f} seconds")
    print(f"Time difference: {time_with_miner - time_no_miner:.2f} seconds")
    
    # Analyze negative sources
    miner_origins_no_miner = sum(1 for origin in origins_no_miner.values() if origin == "miner")
    miner_origins_with_miner = sum(1 for origin in origins_with_miner.values() if origin == "miner")
    
    print("\n--- Negative Sampling Origin Comparison ---")
    print("WITHOUT miner:")
    for origin, count in {k: sum(1 for o in origins_no_miner.values() if o == k) for k in set(origins_no_miner.values())}.items():
        print(f"  {origin}: {count} ({count/len(triplets_no_miner)*100:.1f}%)")
    
    print("\nWITH miner:")
    for origin, count in {k: sum(1 for o in origins_with_miner.values() if o == k) for k in set(origins_with_miner.values())}.items():
        print(f"  {origin}: {count} ({count/len(triplets_with_miner)*100:.1f}%)")
    
    # Examine similarity of mined negatives to query
    if miner_origins_with_miner > 0:
        print("\n--- Analyzing Miner-Generated Negatives ---")
        miner_triplets = [(t, origins_with_miner[t]) for t in origins_with_miner if origins_with_miner[t] == "miner"]
        
        # Get query and passage mapping
        queries = {}
        passages = {}
        with open(test_dir_with_miner / "queries.train.colbert.tsv") as f:
            for line in f:
                idx, query = line.strip().split("\t", 1)
                queries[int(idx)] = query
                
        with open(test_dir_with_miner / "corpus.train.colbert.tsv") as f:
            for line in f:
                idx, passage = line.strip().split("\t", 1)
                passages[int(idx)] = passage
        
        # Show some examples of mined negatives
        print(f"Found {len(miner_triplets)} triplets with miner-generated negatives")
        for i, (triplet, _) in enumerate(miner_triplets[:3]):  # Show first 3
            q_id, p_id, n_id = triplet
            query = queries[q_id]
            positive = passages[p_id]
            negative = passages[n_id]
            
            print(f"\nExample {i+1}:")
            print(f"  Query: {query}")
            print(f"  Positive: {positive[:50]}...")
            print(f"  Mined Negative: {negative[:50]}...")
    
    return {
        "time_no_miner": time_no_miner,
        "time_with_miner": time_with_miner,
        "triplets_no_miner": len(triplets_no_miner),
        "triplets_with_miner": len(triplets_with_miner),
        "miner_origins_no_miner": miner_origins_no_miner,
        "miner_origins_with_miner": miner_origins_with_miner
    }

def test_aspect_dataset_splitter():
    """Test AspectDatasetSplitter with synthetic data."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create a small collection of passages
    collection = [
        "Information about product A specifications.",
        "Customer reviews for product A are mostly positive.",
        "Price comparison for product A shows it's competitive.",
        "Information about product B specifications.",
        "Customer reviews for product B are mixed.",
        "Price comparison for product B shows it's expensive.",
        "Information about product C specifications.",
        "Customer reviews for product C are negative.",
        "Price comparison for product C shows it's the cheapest option.",
        "General comparison of products A, B, and C.",
        # Add more documents to make splitting meaningful
        "Product A has excellent build quality and reliability.",
        "Product B has average build quality but good features.",
        "Product C has poor build quality but is very affordable.",
        "Battery life of product A is above average.",
        "Battery life of product B is decent for its class.",
        "Battery life of product C is disappointing.",
        "User interface of product A is intuitive and responsive.",
        "User interface of product B is somewhat confusing.",
        "User interface of product C is basic but functional.",
        "Warranty period for product A is standard.",
        "Warranty for product B is shorter than competitors.",
        "Warranty for product C is surprisingly comprehensive."
    ]
    
    # Create labeled pairs with queries containing aspects
    labeled_pairs = [
        # Product A
        ("SPECS||product A", "Information about product A specifications.", 1),
        ("SPECS||product A", "Customer reviews for product A are mostly positive.", 0),
        ("SPECS||product A", "Price comparison for product A shows it's competitive.", 0),
        ("SPECS||product A", "Product A has excellent build quality and reliability.", 1),
        ("SPECS||product A", "Battery life of product A is above average.", 0),
        
        ("REVIEWS||product A", "Information about product A specifications.", 0),
        ("REVIEWS||product A", "Customer reviews for product A are mostly positive.", 1),
        ("REVIEWS||product A", "Price comparison for product A shows it's competitive.", 0),
        ("REVIEWS||product A", "User interface of product A is intuitive and responsive.", 0),
        
        ("PRICE||product A", "Information about product A specifications.", 0),
        ("PRICE||product A", "Customer reviews for product A are mostly positive.", 0),
        ("PRICE||product A", "Price comparison for product A shows it's competitive.", 1),
        ("PRICE||product A", "Warranty period for product A is standard.", 0),
        
        # Product B
        ("SPECS||product B", "Information about product B specifications.", 1),
        ("SPECS||product B", "Customer reviews for product B are mixed.", 0),
        ("SPECS||product B", "Price comparison for product B shows it's expensive.", 0),
        ("SPECS||product B", "Product B has average build quality but good features.", 1),
        
        ("REVIEWS||product B", "Information about product B specifications.", 0),
        ("REVIEWS||product B", "Customer reviews for product B are mixed.", 1),
        ("REVIEWS||product B", "Price comparison for product B shows it's expensive.", 0),
        
        ("PRICE||product B", "Information about product B specifications.", 0),
        ("PRICE||product B", "Customer reviews for product B are mixed.", 0),
        ("PRICE||product B", "Price comparison for product B shows it's expensive.", 1),
        
        # Product C
        ("SPECS||product C", "Information about product C specifications.", 1),
        ("SPECS||product C", "Customer reviews for product C are negative.", 0),
        ("SPECS||product C", "Price comparison for product C shows it's the cheapest option.", 0),
        ("SPECS||product C", "Product C has poor build quality but is very affordable.", 1),
        
        ("REVIEWS||product C", "Information about product C specifications.", 0),
        ("REVIEWS||product C", "Customer reviews for product C are negative.", 1),
        ("REVIEWS||product C", "Battery life of product C is disappointing.", 0),
        
        ("PRICE||product C", "Information about product C specifications.", 0),
        ("PRICE||product C", "Customer reviews for product C are negative.", 0),
        ("PRICE||product C", "Price comparison for product C shows it's the cheapest option.", 1),
    ]
    
    # Create a simple miner - FIXED INITIALIZATION
    miner = SimpleMiner(language_code="en", model_size="small")
    miner.build_index(collection)
    
    # Test directory
    test_dir = Path("test_aspect_dataset_splitter_output")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Initialize dataset splitter
    from colbert.data.train_data_preprocessor import AspectDatasetSplitter
    
    splitter = AspectDatasetSplitter(
        labeled_pairs=labeled_pairs,
        collection=collection,
        negative_miner=miner,
        aspect_delimiter="||",
        train_val_test_ratio=(0.6, 0.2, 0.2),  # 60/20/20 split
        pos_neg_ratio=1.0,
        seed=42,
        debug=True
    )
    
    # Split the dataset
    train_pairs, val_pairs, test_pairs, rule_based_test_pairs = splitter.split_dataset()
    
    # Process the split data
    results = splitter.process_data(
        output_dir=test_dir,
        max_triplets_per_query=10,
        max_positives=2
    )
    
    # Validate document isolation
    train_docs = set(doc for _, doc, _ in train_pairs)
    val_docs = set(doc for _, doc, _ in val_pairs)
    test_docs = set(doc for _, doc, _ in test_pairs)
    
    print("\n--- Document Isolation Check ---")
    print(f"Train documents: {len(train_docs)}")
    print(f"Validation documents: {len(val_docs)}")
    print(f"Test documents: {len(test_docs)}")
    
    train_test_overlap = train_docs.intersection(test_docs)
    val_test_overlap = val_docs.intersection(test_docs)
    
    print(f"Train-Test overlap: {len(train_test_overlap)} documents")
    print(f"Val-Test overlap: {len(val_test_overlap)} documents")
    
    if len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("✓ Test documents are properly isolated")
    else:
        print("✗ Test documents are not properly isolated")
    
    # Verify aspect coverage in all splits
    print("\n--- Aspect Coverage Check ---")
    
    def count_aspects(pairs):
        aspects = set()
        for q, _, _ in pairs:
            if "||" in q:
                aspect = q.split("||", 1)[0]
                aspects.add(aspect)
        return aspects
    
    train_aspects = count_aspects(train_pairs)
    val_aspects = count_aspects(val_pairs)
    test_aspects = count_aspects(test_pairs)
    
    print(f"Train aspects: {train_aspects}")
    print(f"Validation aspects: {val_aspects}")
    print(f"Test aspects: {test_aspects}")
    
    all_aspects = set.union(train_aspects, val_aspects, test_aspects)
    
    if test_aspects == all_aspects:
        print("✓ All aspects are represented in the test set")
    else:
        print("✗ Some aspects are missing from the test set")
    
    # Check specialized test sets
    print("\n--- Specialized Test Sets Check ---")
    print(f"Standard test pairs: {len(test_pairs)}")
    print(f"Rule-based test pairs: {len(rule_based_test_pairs)}")
    
    # Check for triplets in each split
    print("\n--- Triplet Generation Check ---")
    for split_name, split_data in results.items():
        if split_data:
            print(f"{split_name}: {len(split_data)} triplets")
    
    # NEW: Check for independent ID spaces
    print("\n--- Independent ID Spaces Check ---")
    
    # Load query and document IDs from each split
    split_paths = {
        "train": test_dir / "train",
        "val": test_dir / "val",
        "test": test_dir / "test",
        "test_rule_based": test_dir / "test_rule_based"
    }
    
    split_query_ids = {}
    split_doc_ids = {}
    split_queries = {}
    split_docs = {}
    
    for split_name, split_path in split_paths.items():
        if not (split_path / "queries.train.colbert.tsv").exists():
            continue
            
        # Load query IDs
        query_ids = []
        queries = {}
        with open(split_path / "queries.train.colbert.tsv") as f:
            for line in f:
                idx, query = line.strip().split("\t", 1)
                idx = int(idx)
                query_ids.append(idx)
                queries[idx] = query
        
        # Load doc IDs
        doc_ids = []
        docs = {}
        with open(split_path / "corpus.train.colbert.tsv") as f:
            for line in f:
                idx, doc = line.strip().split("\t", 1)
                idx = int(idx)
                doc_ids.append(idx)
                docs[idx] = doc
        
        split_query_ids[split_name] = query_ids
        split_doc_ids[split_name] = doc_ids
        split_queries[split_name] = queries
        split_docs[split_name] = docs
    
    # Check that each split's ID space starts from 0
    for split_name in split_query_ids:
        min_query_id = min(split_query_ids[split_name]) if split_query_ids[split_name] else None
        min_doc_id = min(split_doc_ids[split_name]) if split_doc_ids[split_name] else None
        
        print(f"{split_name}:")
        print(f"  - Minimum query ID: {min_query_id}")
        print(f"  - Minimum document ID: {min_doc_id}")
        print(f"  - Number of queries: {len(split_query_ids[split_name])}")
        print(f"  - Number of documents: {len(split_doc_ids[split_name])}")
        
        if min_query_id == 0 and min_doc_id == 0:
            print(f"  ✓ {split_name} split has IDs starting from 0")
        else:
            print(f"  ✗ {split_name} split does not have IDs starting from 0")
    
    # Check for document overlap using actual document content (not IDs)
    print("\n--- Document Content Isolation Check ---")
    train_doc_contents = set(split_docs["train"].values())
    val_doc_contents = set(split_docs["val"].values())
    test_doc_contents = set(split_docs["test"].values())
    
    train_test_overlap = train_doc_contents.intersection(test_doc_contents)
    val_test_overlap = val_doc_contents.intersection(test_doc_contents)
    
    print(f"Train-Test content overlap: {len(train_test_overlap)} documents")
    print(f"Val-Test content overlap: {len(val_test_overlap)} documents")
    
    if len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("✓ Test document contents are properly isolated")
    else:
        print("✗ Test document contents are not properly isolated")
    
    # Check for document ID reuse across splits
    print("\n--- Document ID Independence Check ---")
    
    # Get documents that appear in multiple splits (by content)
    shared_docs_by_content = {}
    for doc in collection:
        splits_with_doc = []
        for split_name in split_docs:
            if doc in split_docs[split_name].values():
                splits_with_doc.append(split_name)
        
        if len(splits_with_doc) > 1:
            shared_docs_by_content[doc] = splits_with_doc
    
    if shared_docs_by_content:
        print(f"Found {len(shared_docs_by_content)} documents that appear in multiple splits:")
        for doc, splits in list(shared_docs_by_content.items())[:3]:  # Show first 3 examples
            print(f"  Document: {doc[:50]}...")
            print(f"  Appears in: {', '.join(splits)}")
            
            # Check if these documents have different IDs in each split
            doc_ids_across_splits = {}
            for split in splits:
                # Find the ID(s) for this document in this split
                ids = [idx for idx, content in split_docs[split].items() if content == doc]
                if ids:
                    doc_ids_across_splits[split] = ids
            
            print(f"  IDs: {doc_ids_across_splits}")
            
            # Check if the IDs are different
            all_ids = [id for ids_list in doc_ids_across_splits.values() for id in ids_list]
            if len(all_ids) == len(set(all_ids)):
                print("  ✓ Document has different IDs in each split")
            else:
                print("  ✗ Document has the same ID in multiple splits")
    else:
        print("No documents appear in multiple splits")
    
    # Examine one of the rule-based test triplets
    if results.get("test_rule_based"):
        print("\n--- Sample Rule-Based Test Triplet ---")
        
        # Load query and passage mappings from test_rule_based output
        rule_based_dir = test_dir / "test_rule_based"
        
        queries = {}
        passages = {}
        
        with open(rule_based_dir / "queries.train.colbert.tsv") as f:
            for line in f:
                idx, query = line.strip().split("\t", 1)
                queries[int(idx)] = query
                
        with open(rule_based_dir / "corpus.train.colbert.tsv") as f:
            for line in f:
                idx, passage = line.strip().split("\t", 1)
                passages[int(idx)] = passage
        
        # Sample a triplet from the rule-based test set
        import srsly
        triplets = list(srsly.read_jsonl(rule_based_dir / "triples.train.colbert.jsonl"))
        
        if triplets:
            sample_triplet = triplets[0]
            q_id, p_id, n_id = sample_triplet
            query = queries[q_id]
            positive = passages[p_id]
            negative = passages[n_id]
            
            print(f"Query: {query}")
            print(f"Positive: {positive[:50]}...")
            print(f"Negative: {negative[:50]}...")
            
            # Check if the negative is from another aspect
            aspect, base = query.split("||", 1)
            
            other_aspects = [q for q in queries.values() 
                            if "||" in q and q.split("||", 1)[1] == base and q.split("||", 1)[0] != aspect]
            
            if other_aspects:
                print(f"This query has other aspects with the same base: {[q.split('||', 1)[0] for q in other_aspects]}")
    
    print("\nDataset splitter test completed!")
    return results

if __name__ == "__main__":
    # Run all tests
    print("\n=== Running Standard Test ===")
    test_aspect_processor()
    
    print("\n=== Running Debug Test ===")
    test_aspect_processor_debug()
    
    print("\n=== Running Miner Test ===")
    miner_test_results = test_with_simple_miner()
    
    # Summary of miner test
    print("\n=== Miner Test Summary ===")
    print(f"Processing time without miner: {miner_test_results['time_no_miner']:.2f} seconds")
    print(f"Processing time with miner: {miner_test_results['time_with_miner']:.2f} seconds")
    print(f"Time overhead of using miner: {miner_test_results['time_with_miner'] - miner_test_results['time_no_miner']:.2f} seconds")
    print(f"Triplets generated without miner: {miner_test_results['triplets_no_miner']}")
    print(f"Triplets generated with miner: {miner_test_results['triplets_with_miner']}")
    print(f"Miner-generated negatives: {miner_test_results['miner_origins_with_miner']} ({miner_test_results['miner_origins_with_miner']/miner_test_results['triplets_with_miner']*100:.1f}%)")
    
    print("\n=== Running Dataset Splitter Test ===")
    test_aspect_dataset_splitter()