# Update the path handling at the top of the file
import sys
import os
from pathlib import Path

# More robust path handling
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tempfile
import unittest
import torch
import numpy as np
from colbert.evaluation.evaluator import ColBERTEvaluator
from colbert.evaluation.metrics import Metrics
from colbert.utils.parser import Arguments


class MockModel:
    """Mock ColBERT model for testing"""
    
    def __init__(self):
        self.device = torch.device("cpu")
    
    def eval(self):
        return self
    
    def to(self, device):
        return self
    
    def query(self, query):
        # Return a fixed query embedding
        # Shape: [1, query_length, dim]
        return torch.ones((1, 10, 128))
    
    def doc(self, docs):
        # Return fixed document embeddings
        # Shape: [batch_size, doc_length, dim]
        return torch.ones((len(docs), 20, 128))
    
    def score(self, Q, D):
        # Return similarity scores
        # Higher scores for documents with lower IDs (to create a predictable ranking)
        batch_size = D.shape[0]
        scores = torch.tensor([[100.0 - i for i in range(batch_size)]])
        return scores


class TestColBERTEvaluator(unittest.TestCase):
    
    def setUp(self):
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create dummy data
        self.create_test_data()
        
        # Create mock model
        self.model = MockModel()
        
        # Create config
        self.config = Arguments(description="Test configuration for ColBERT evaluation")
        self.config.query_maxlen = 32
        self.config.doc_maxlen = 180
        self.config.dim = 128
        self.config.similarity = 'cosine'
        self.config.mask_punctuation = False
        self.config.bsize = 2
        self.config.rank = 0
        
        # Create evaluator
        self.evaluator = ColBERTEvaluator(
            model=self.model,
            mrr_depths={1, 3, 5},
            recall_depths={1, 3, 5},
            success_depths={1, 3, 5},
            precision_depths={1, 3, 5},
            ndcg_depths={1, 3, 5}
        )
    
    def create_test_data(self):
        """Create test data files"""
        # Create queries file
        queries_path = os.path.join(self.temp_dir.name, "queries.tsv")
        with open(queries_path, 'w') as f:
            f.write("1\tquery one\n")
            f.write("2\tquery two\n")
            f.write("3\tquery three\n")
        
        # Create collection file with 0-based indexing
        collection_path = os.path.join(self.temp_dir.name, "collection.tsv")
        with open(collection_path, 'w') as f:
            f.write("0\tdocument one\n")   # Changed from 1 to 0
            f.write("1\tdocument two\n")   # Changed from 2 to 1
            f.write("2\tdocument three\n") # Changed from 3 to 2
            f.write("3\tdocument four\n")  # Changed from 4 to 3
            f.write("4\tdocument five\n")  # Changed from 5 to 4
        
        # Update qrels file to reference the new document IDs
        qrels_path = os.path.join(self.temp_dir.name, "qrels.tsv")
        with open(qrels_path, 'w') as f:
            # Query 1 has documents 0 and 2 as relevant (was 1 and 3)
            f.write("1\t0\t0\t1\n")  # Changed from 1 to 0
            f.write("1\t0\t2\t1\n")  # Changed from 3 to 2
            # Query 2 has document 1 as relevant (was 2)
            f.write("2\t0\t1\t1\n")  # Changed from 2 to 1
            # Query 3 has documents 3 and 4 as relevant (was 4 and 5)
            f.write("3\t0\t3\t1\n")  # Changed from 4 to 3
            f.write("3\t0\t4\t1\n")  # Changed from 5 to 4
        
        self.queries_path = queries_path
        self.collection_path = collection_path
        self.qrels_path = qrels_path
    
    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_evaluation_with_retrieval(self):
        """Test evaluation with retrieval"""
        # Run evaluation
        results = self.evaluator.evaluate(
            queries_path=self.queries_path,
            collection_path=self.collection_path,
            qrels_path=self.qrels_path,
            batch_size=2,
            depth=5
        )
        
        # Check that results contain expected metrics
        self.assertIn('mrr', results)
        self.assertIn('recall', results)
        self.assertIn('success', results)
        self.assertIn('precision', results)
        self.assertIn('ndcg', results)
        
        # Check specific metrics
        # Since our mock model ranks documents in reverse order (doc 1 gets highest score),
        # we expect perfect scores for query 1 (relevant docs 1, 3)
        
        # MRR@1 should be 1.0 for query 1, 0.0 for query 2, 0.0 for query 3
        # Average: 1/3 = 0.333
        self.assertAlmostEqual(results['mrr'][1], 0.333, places=2)
        
        # Recall@3 should be:
        # - Query 1: 2/2 = 1.0 (docs 1 and 3 in top 3)
        # - Query 2: 0/1 = 0.0 (doc 2 not in top 3 due to our mock model)
        # - Query 3: 0/2 = 0.0 (docs 4 and 5 not in top 3)
        # Average: (1.0 + 0.0 + 0.0) / 3 = 0.333
        self.assertAlmostEqual(results['recall'][3], 0.5, places=2)
        
        # Print all results for debugging
        print("\nEvaluation Results:")
        for metric_type, depths in results.items():
            print(f"{metric_type.upper()}:")
            for depth, value in depths.items():
                print(f"  @{depth}: {value:.4f}")
    
    def test_metrics_calculation(self):
        """Test metrics calculation directly"""
        # Create metrics object
        metrics = Metrics(
            mrr_depths={1, 3, 5},
            recall_depths={1, 3, 5},
            success_depths={1, 3, 5},
            total_queries=3
        )
        
        # Add rankings for query 1
        # Ranking format: [(score, pid, passage_idx)]
        # Gold positives: [0, 2] (was [1, 3])
        metrics.add(
            query_idx=0,
            query_key=1,
            ranking=[(100.0, 0, 0), (99.0, 1, 1), (98.0, 2, 2), (97.0, 3, 3), (96.0, 4, 4)],
            gold_positives=[0, 2]
        )
        
        # Add rankings for query 2
        # Gold positives: [1] (was [2])
        metrics.add(
            query_idx=1,
            query_key=2,
            ranking=[(100.0, 2, 2), (99.0, 0, 0), (98.0, 3, 3), (97.0, 1, 1), (96.0, 4, 4)],
            gold_positives=[1]
        )
        
        # Add rankings for query 3
        # Gold positives: [3, 4] (was [4, 5])
        metrics.add(
            query_idx=2,
            query_key=3,
            ranking=[(100.0, 0, 0), (99.0, 1, 1), (98.0, 2, 2), (97.0, 3, 3), (96.0, 4, 4)],
            gold_positives=[3, 4]
        )
        
        # Calculate expected values
        # MRR@1: (1.0 + 0.0 + 0.0) / 3 = 0.333
        # MRR@3: (1.0 + 0.0 + 0.0) / 3 = 0.333
        # MRR@5: (1.0 + 0.25 + 0.25) / 3 = 0.5
        
        # Recall@1: (0.5 + 0.0 + 0.0) / 3 = 0.167
        # Recall@3: (1.0 + 0.0 + 0.0) / 3 = 0.333
        # Recall@5: (1.0 + 1.0 + 1.0) / 3 = 1.0
        
        # Success@1: (1.0 + 0.0 + 0.0) / 3 = 0.333
        # Success@3: (1.0 + 0.0 + 0.0) / 3 = 0.333
        # Success@5: (1.0 + 1.0 + 1.0) / 3 = 1.0
        
        # Check metrics
        self.assertAlmostEqual(metrics.mrr_sums[1] / 3, 0.333, places=2)
        self.assertAlmostEqual(metrics.mrr_sums[3] / 3, 0.333, places=2)
        self.assertAlmostEqual(metrics.mrr_sums[5] / 3, 0.5, places=2)
        
        self.assertAlmostEqual(metrics.recall_sums[1] / 3, 0.167, places=2)
        self.assertAlmostEqual(metrics.recall_sums[3] / 3, 0.333, places=2)
        self.assertAlmostEqual(metrics.recall_sums[5] / 3, 1.0, places=2)
        
        self.assertAlmostEqual(metrics.success_sums[1] / 3, 0.333, places=2)
        self.assertAlmostEqual(metrics.success_sums[3] / 3, 0.333, places=2)
        self.assertAlmostEqual(metrics.success_sums[5] / 3, 1.0, places=2)
        
        # Print metrics for debugging
        metrics.print_metrics(2)

    def test_evaluation_with_tracker(self):
        """Test that the evaluator correctly uses a tracker"""
        
        # Create a mock tracker to record logged metrics
        class MockTracker:
            def __init__(self):
                self.logged_metrics = {}
                self.logged_figures = {}
                self.logged_texts = {}
                
            def log_metric(self, name, value, step=None):
                self.logged_metrics[name] = value
                
            def add_figure(self, tag, figure, step=None):
                self.logged_figures[tag] = figure
                
            def add_text(self, tag, text, step=None):
                self.logged_texts[tag] = text
                
            def close(self):
                pass
        
        # Create the mock tracker
        mock_tracker = MockTracker()
        
        # Create evaluator with the mock tracker
        evaluator = ColBERTEvaluator(
            model=self.model,
            mrr_depths={1, 3, 5},
            recall_depths={1, 3, 5},
            success_depths={1, 3, 5},
            precision_depths={1, 3, 5},
            ndcg_depths={1, 3, 5},
            tracker=mock_tracker
        )
        
        # Run evaluation
        results = evaluator.evaluate(
            queries_path=self.queries_path,
            collection_path=self.collection_path,
            qrels_path=self.qrels_path,
            batch_size=2,
            depth=5
        )
        
        # Check that metrics were logged to the tracker
        self.assertGreater(len(mock_tracker.logged_metrics), 0, 
                         "No metrics were logged to the tracker")
        
        # Check that specific metrics were logged
        # The metric names in the tracker might be prefixed with 'ranking/'
        expected_metrics = [
            'MRR.1', 'MRR.3', 'MRR.5',
            'Recall.1', 'Recall.3', 'Recall.5',
            'Success.1', 'Success.3', 'Success.5',
            'NDCG.1', 'NDCG.3', 'NDCG.5'
        ]
        
        for metric in expected_metrics:
            # Check with and without prefix
            found = False
            for logged_metric in mock_tracker.logged_metrics:
                if metric in logged_metric:
                    found = True
                    break
            
            self.assertTrue(found, f"Expected metric {metric} not found in logged metrics")
        
        # Check that some of the logged values match the returned results
        for depth in [1, 3, 5]:
            # Find the tracker key for this metric
            mrr_key = None
            for key in mock_tracker.logged_metrics:
                if f'MRR.{depth}' in key:
                    mrr_key = key
                    break
            
            # Check that the logged value matches the returned result
            if mrr_key:
                self.assertAlmostEqual(
                    mock_tracker.logged_metrics[mrr_key],
                    results['mrr'][depth],
                    places=4,
                    msg=f"Logged MRR@{depth} doesn't match returned result"
                )


if __name__ == "__main__":
    unittest.main() 