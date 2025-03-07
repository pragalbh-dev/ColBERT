import os
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import argparse
import time

class AspectDatasetGenerator:
    """
    Generates a synthetic dataset with aspects, queries, documents, and labeled pairs,
    compatible with TripletGenerator.
    """
    
    def __init__(
        self,
        seed: int = 42,
        num_aspects: int = 5,
        num_queries_per_aspect: int = 20,
        num_docs: int = 500,
        aspect_delimiter: str = "||",
        output_dir: Optional[str] = None
    ):
        start_time = time.time()
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.aspect_delimiter = aspect_delimiter
        self.output_dir = Path(output_dir or "data/test")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define aspects and sentiment values
        self.aspects = [
            "price", "quality", "service", "location", "amenities", 
            "cleanliness", "food", "staff", "ambiance", "comfort"
        ][:num_aspects]
        
        self.sentiments = ["excellent", "good", "average", "poor", "terrible"]
        self.sentiment_scores = {
            "excellent": 2,
            "good": 1,
            "average": 0,
            "poor": -1,
            "terrible": -2
        }
        
        # Create query templates WITHOUT the aspect term in them
        self.query_templates = [
            "how is it",
            "tell me about it",
            "what's it like",
            "describe it",
            "information on it",
            "details about it",
            "review of it",
            "rating for it"
        ]
        
        # Timing information
        self.timing = {
            "init": 0,
            "generate_queries": 0,
            "generate_documents": 0,
            "generate_labeled_pairs": 0,
            "export_dataset": 0,
            "total": 0
        }
        
        # Generate the dataset components
        query_start = time.time()
        self.queries = self._generate_queries(num_queries_per_aspect)  # actual text, not IDs
        self.timing["generate_queries"] = time.time() - query_start
        
        doc_start = time.time()
        self.documents = self._generate_documents(num_docs)  # actual text, not IDs
        self.timing["generate_documents"] = time.time() - doc_start
        
        pairs_start = time.time()
        self.labeled_pairs = self._generate_labeled_pairs()  # (query_text, doc_text, label)
        self.timing["generate_labeled_pairs"] = time.time() - pairs_start
        
        # Statistics
        self.stats = self._calculate_statistics()
        
        # Total initialization time
        self.timing["init"] = time.time() - start_time
        self.timing["total"] += self.timing["init"]
        
        # Debug output
        print(f"⏱️ Dataset generator initialized in {self.timing['init']:.2f}s")
        print(f"  - Generate queries: {self.timing['generate_queries']:.2f}s")
        print(f"  - Generate documents: {self.timing['generate_documents']:.2f}s")
        print(f"  - Generate labeled pairs: {self.timing['generate_labeled_pairs']:.2f}s")
        print(f"📊 Generated {len(self.queries)} queries, {len(self.documents)} documents, {len(self.labeled_pairs)} labeled pairs")
    
    def _generate_queries(self, num_per_aspect: int) -> List[str]:
        """Generate queries for each aspect"""
        queries = []
        
        for aspect in self.aspects:
            # Choose unique templates if possible, otherwise allow repeats
            if num_per_aspect <= len(self.query_templates):
                templates = random.sample(self.query_templates, k=num_per_aspect)
            else:
                # If we need more queries than templates, we'll have duplicates
                templates = random.choices(self.query_templates, k=num_per_aspect)
            
            for template in templates:
                # The aspect is only in the prefix, not in the query phrase
                prefixed_query = f"{aspect}{self.aspect_delimiter}{template}"
                queries.append(prefixed_query)
        
        return queries
    
    def _generate_documents(self, num_docs: int) -> List[str]:
        """
        Generate documents with controlled aspect coverage.
        Each document mentions 1-3 aspects with assigned sentiment.
        """
        documents = []
        
        # Create a distribution of how many aspects each document will mention
        aspect_counts = np.random.choice([1, 2, 3], size=num_docs, p=[0.3, 0.5, 0.2])
        
        for doc_id in range(num_docs):
            doc_text = f"Document {doc_id}: "
            
            # Randomly select which aspects this document will mention
            num_aspects = aspect_counts[doc_id]
            mentioned_aspects = random.sample(self.aspects, k=min(num_aspects, len(self.aspects)))
            
            for aspect in mentioned_aspects:
                # Assign sentiment to this aspect
                sentiment = random.choice(self.sentiments)
                doc_text += f"The {aspect} is {sentiment}. "
            
            documents.append(doc_text)
        
        return documents
    
    def _generate_labeled_pairs(self) -> List[Tuple[str, str, int]]:
        """
        Generate labeled query-document pairs with controlled positive/negative distribution.
        
        Returns:
            List of tuples (query_text, doc_text, label) - exactly what TripletGenerator needs
        """
        labeled_pairs = []
        
        # Create an index of which aspects each document mentions
        doc_to_aspects = defaultdict(set)
        for doc_idx, doc_text in enumerate(self.documents):
            for aspect in self.aspects:
                if aspect in doc_text.lower():
                    doc_to_aspects[doc_idx].add(aspect)
        
        # For each query, create positive and negative examples
        for query_idx, query_text in enumerate(self.queries):
            # Extract the aspect from the query
            aspect = query_text.split(self.aspect_delimiter)[0]
            
            # Find documents that mention this aspect (candidates for positives)
            positive_candidates = [doc_idx for doc_idx, aspects in doc_to_aspects.items() if aspect in aspects]
            
            # Find documents that don't mention this aspect (definite negatives)
            negative_candidates = [doc_idx for doc_idx in range(len(self.documents)) if doc_idx not in positive_candidates]
            
            # Create a balanced set of examples (approximately 30% positive, 70% negative)
            num_positives = max(1, round(5 * 0.3))  # At least 1 positive
            num_negatives = 5 - num_positives
            
            # Sample positive examples
            if positive_candidates:
                sampled_positives = random.sample(positive_candidates, k=min(num_positives, len(positive_candidates)))
                for doc_idx in sampled_positives:
                    # Use actual text values instead of IDs
                    labeled_pairs.append((query_text, self.documents[doc_idx], 1))
            
            # Sample negative examples
            if negative_candidates:
                sampled_negatives = random.sample(negative_candidates, k=min(num_negatives, len(negative_candidates)))
                for doc_idx in sampled_negatives:
                    # Use actual text values instead of IDs
                    labeled_pairs.append((query_text, self.documents[doc_idx], 0))
        
        return labeled_pairs
    
    def _calculate_statistics(self) -> Dict:
        """Calculate statistics about the dataset"""
        stats = {}
        
        # Extract aspects from queries
        query_aspects = {}
        for query in self.queries:
            aspect = query.split(self.aspect_delimiter)[0]
            query_aspects[query] = aspect
        
        # Count positives and negatives overall
        total_pairs = len(self.labeled_pairs)
        total_positives = sum(1 for _, _, label in self.labeled_pairs if label == 1)
        total_negatives = total_pairs - total_positives
        
        stats["total"] = {
            "queries": len(self.queries),
            "documents": len(self.documents),
            "labeled_pairs": total_pairs,
            "positives": total_positives,
            "negatives": total_negatives,
            "positive_ratio": total_positives / total_pairs if total_pairs > 0 else 0
        }
        
        # Count by aspect
        aspect_stats = defaultdict(lambda: {"queries": 0, "positives": 0, "negatives": 0})
        
        for aspect in self.aspects:
            aspect_stats[aspect]["queries"] = sum(1 for a in query_aspects.values() if a == aspect)
        
        for query, _, label in self.labeled_pairs:
            aspect = query_aspects[query]
            if label == 1:
                aspect_stats[aspect]["positives"] += 1
            else:
                aspect_stats[aspect]["negatives"] += 1
        
        # Calculate ratios
        for aspect, counts in aspect_stats.items():
            total = counts["positives"] + counts["negatives"]
            counts["total_pairs"] = total
            counts["positive_ratio"] = counts["positives"] / total if total > 0 else 0
        
        stats["by_aspect"] = dict(aspect_stats)
        
        return stats
    
    def export_dataset(self):
        """
        Export the dataset in a format compatible with TripletGenerator.
        
        1. Raw data in format expected by TripletGenerator
        2. TSV files for easier inspection
        """
        export_start = time.time()
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Export queries (simple line-by-line format)
        with open(self.output_dir / "queries.tsv", "w") as f:
            for idx, query in enumerate(self.queries):
                f.write(f"{idx}\t{query}\n")
        
        # Export documents (simple line-by-line format)
        with open(self.output_dir / "documents.tsv", "w") as f:
            for idx, doc in enumerate(self.documents):
                f.write(f"{idx}\t{doc}\n")
        
        # Export labeled pairs - format compatible with TripletGenerator
        with open(self.output_dir / "labeled_pairs.tsv", "w") as f:
            f.write("query\tdocument\tlabel\n")
            for query, doc, label in self.labeled_pairs:
                # Replace tabs/newlines to ensure TSV format integrity
                query_clean = query.replace("\t", " ").replace("\n", " ")
                doc_clean = doc.replace("\t", " ").replace("\n", " ")
                f.write(f"{query_clean}\t{doc_clean}\t{label}\n")
        
        # Export JSON version for programmatic access
        with open(self.output_dir / "data.json", "w") as f:
            json.dump({
                "queries": self.queries,
                "documents": self.documents,
                "labeled_pairs": self.labeled_pairs
            }, f, indent=2)
        
        # Export statistics
        with open(self.output_dir / "statistics.json", "w") as f:
            json.dump(self.stats, f, indent=2)
        
        export_time = time.time() - export_start
        self.timing["export_dataset"] = export_time
        self.timing["total"] += export_time
        
        print(f"⏱️ Dataset exported to {self.output_dir} in {export_time:.2f}s")
        print(f"  - {len(self.queries)} queries")
        print(f"  - {len(self.documents)} documents")
        print(f"  - {len(self.labeled_pairs)} labeled pairs")
    
    def get_dataset_for_processor(self):
        """
        Return data in format directly usable by TripletGenerator
        
        Returns:
            labeled_pairs: List of (query, document, label) tuples
            collection: List of document texts
        """
        return self.labeled_pairs, self.documents
    
    def print_statistics(self):
        """Print statistics about the dataset"""
        print("\n=== Dataset Statistics ===\n")
        
        print(f"Total queries: {self.stats['total']['queries']}")
        print(f"Total documents: {self.stats['total']['documents']}")
        print(f"Total labeled pairs: {self.stats['total']['labeled_pairs']}")
        print(f"Positive pairs: {self.stats['total']['positives']} ({self.stats['total']['positive_ratio']:.2%})")
        print(f"Negative pairs: {self.stats['total']['negatives']} ({1-self.stats['total']['positive_ratio']:.2%})")
        
        print("\nAspect distribution:")
        for aspect, stats in self.stats["by_aspect"].items():
            print(f"  {aspect}:")
            print(f"    Queries: {stats['queries']}")
            print(f"    Positive pairs: {stats['positives']}")
            print(f"    Negative pairs: {stats['negatives']}")
            print(f"    Positive ratio: {stats['positive_ratio']:.2%}")
            
        print("\nGeneration timing:")
        print(f"  - Query generation: {self.timing['generate_queries']:.2f}s")
        print(f"  - Document generation: {self.timing['generate_documents']:.2f}s")
        print(f"  - Labeled pairs generation: {self.timing['generate_labeled_pairs']:.2f}s")
        if self.timing['export_dataset'] > 0:
            print(f"  - Export dataset: {self.timing['export_dataset']:.2f}s")
        print(f"  - Total time: {self.timing['total']:.2f}s")
        
        print("\nQuality analysis:")
        aspect_coverage = sum(1 for _, stats in self.stats["by_aspect"].items() if stats["positives"] > 0) / len(self.aspects)
        print(f"  - Aspect coverage: {aspect_coverage:.2%} of aspects have positive examples")
        
        # Calculate query-to-document ratio
        if len(self.documents) > 0:
            query_doc_ratio = len(self.queries) / len(self.documents)
            print(f"  - Query-to-document ratio: {query_doc_ratio:.2f}")
        
        # Calculate labeled pairs per query
        if len(self.queries) > 0:
            pairs_per_query = len(self.labeled_pairs) / len(self.queries)
            print(f"  - Labeled pairs per query: {pairs_per_query:.2f}")
    
    def visualize_statistics(self, save_path=None):
        """Create visualizations of dataset statistics"""
        plt.figure(figsize=(15, 10))
        
        # 1. Positive/Negative distribution
        plt.subplot(2, 2, 1)
        labels = ['Positive', 'Negative']
        sizes = [self.stats['total']['positives'], self.stats['total']['negatives']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#5cb85c', '#d9534f'])
        plt.title('Positive vs Negative Labeled Pairs')
        
        # 2. Aspect distribution
        plt.subplot(2, 2, 2)
        aspects = list(self.stats['by_aspect'].keys())
        query_counts = [stats['queries'] for stats in self.stats['by_aspect'].values()]
        plt.bar(aspects, query_counts)
        plt.title('Queries per Aspect')
        plt.xticks(rotation=45)
        
        # 3. Positive examples per aspect
        plt.subplot(2, 2, 3)
        positive_counts = [stats['positives'] for stats in self.stats['by_aspect'].values()]
        negative_counts = [stats['negatives'] for stats in self.stats['by_aspect'].values()]
        
        x = range(len(aspects))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], positive_counts, width, label='Positive', color='#5cb85c')
        plt.bar([i + width/2 for i in x], negative_counts, width, label='Negative', color='#d9534f')
        
        plt.title('Labeled Pairs per Aspect')
        plt.xticks(x, aspects, rotation=45)
        plt.legend()
        
        # 4. Positive ratio per aspect
        plt.subplot(2, 2, 4)
        positive_ratios = [stats['positive_ratio'] for stats in self.stats['by_aspect'].values()]
        plt.bar(aspects, positive_ratios, color='#5bc0de')
        plt.title('Positive Ratio per Aspect')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()


def load_dataset_for_processor(data_dir):
    """
    Load dataset from disk in format suitable for TripletGenerator
    
    Args:
        data_dir: Directory containing dataset files
    
    Returns:
        labeled_pairs: List of (query, document, label) tuples
        collection: List of document texts
    """
    data_dir = Path(data_dir)
    
    # Load labeled pairs
    labeled_pairs = []
    with open(data_dir / "labeled_pairs.tsv", "r") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            query, document, label = line.strip().split("\t")
            labeled_pairs.append((query, document, int(label)))
    
    # Load documents as collection
    collection = []
    with open(data_dir / "documents.tsv", "r") as f:
        for line in f:
            _, document = line.strip().split("\t", 1)
            collection.append(document)
    
    return labeled_pairs, collection


def main(output_dir="data/test"):
    # Create a synthetic dataset
    generator = AspectDatasetGenerator(
        seed=42,
        num_aspects=10,
        num_queries_per_aspect=20,
        num_docs=5000,
        aspect_delimiter="||",
        output_dir=output_dir
    )
    
    # Export the dataset
    generator.export_dataset()
    
    # Print statistics
    generator.print_statistics()
    
    # Visualize statistics
    generator.visualize_statistics(save_path=Path(output_dir) / "statistics.png")
    
    # Example of how to use with TripletGenerator
    print("\n=== Example Code for TripletGenerator ===\n")
    print("from colbert.data.train_data_preprocessor import TripletGenerator")
    print("# Option 1: Load directly from generator")
    print("labeled_pairs, collection = generator.get_dataset_for_processor()")
    print("# Option 2: Load from files")
    print(f"labeled_pairs, collection = load_dataset_for_processor('{output_dir}')")
    print("\n# Initialize generator")
    print("generator = TripletGenerator(")
    print("    labeled_pairs=labeled_pairs,")
    print("    collection=collection,")
    print("    aspect_delimiter='||'")
    print(")")
    print("# Generate triplets")
    print("triplets = generator.generate_triplets(")
    print("    max_triplets_per_query=20,")
    print("    export_path='processed_data'")
    print(")")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic aspect-based dataset")
    parser.add_argument("--output-dir", type=str, default="data/test", 
                        help="Directory where the dataset will be saved (default: data/test)")
    args = parser.parse_args()
    
    main(output_dir=args.output_dir) 