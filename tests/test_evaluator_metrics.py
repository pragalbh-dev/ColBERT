import unittest
import torch
from collections import defaultdict

from colbert.evaluation.metrics import Metrics

class TestEvaluatorMetrics(unittest.TestCase):
    """Test the metric calculation logic directly with controlled inputs"""
    
    def test_mrr_calculation(self):
        """Test MRR calculation with controlled rankings"""
        metrics = Metrics(
            mrr_depths={1, 3, 5, 10},
            recall_depths={1, 3, 5, 10},
            success_depths={1, 3, 5, 10},
            total_queries=3
        )
        
        # Query 1: First relevant doc at position 0
        metrics.add(
            query_idx=0,
            query_key="q1",
            ranking=[(10.0, "d1", 0), (9.0, "d2", 1), (8.0, "d3", 2)],
            gold_positives=["d1", "d3"]
        )
        
        # Query 2: First relevant doc at position 2
        metrics.add(
            query_idx=1,
            query_key="q2",
            ranking=[(10.0, "d4", 0), (9.0, "d5", 1), (8.0, "d6", 2)],
            gold_positives=["d6"]
        )
        
        # Query 3: First relevant doc at position 4
        metrics.add(
            query_idx=2,
            query_key="q3",
            ranking=[(10.0, "d7", 0), (9.0, "d8", 1), (8.0, "d9", 2), 
                     (7.0, "d10", 3), (6.0, "d11", 4), (5.0, "d12", 5)],
            gold_positives=["d11", "d12"]
        )
        
        # Manual calculation:
        # Query 1: 1/1 = 1.0
        # Query 2: 1/3 = 0.333
        # Query 3: 1/5 = 0.2
        # MRR@1: (1.0 + 0.0 + 0.0) / 3 = 0.333
        # MRR@3: (1.0 + 0.333 + 0.0) / 3 = 0.444
        # MRR@5: (1.0 + 0.333 + 0.2) / 3 = 0.511
        # MRR@10: (1.0 + 0.333 + 0.2) / 3 = 0.511
        
        self.assertAlmostEqual(metrics.mrr_sums[1] / 3, 0.333, places=3)
        self.assertAlmostEqual(metrics.mrr_sums[3] / 3, 0.444, places=3)
        self.assertAlmostEqual(metrics.mrr_sums[5] / 3, 0.511, places=3)
        self.assertAlmostEqual(metrics.mrr_sums[10] / 3, 0.511, places=3)
    
    def test_recall_calculation(self):
        """Test recall calculation with controlled rankings"""
        metrics = Metrics(
            mrr_depths={1, 3, 5, 10},
            recall_depths={1, 3, 5, 10},
            success_depths={1, 3, 5, 10},
            total_queries=3
        )
        
        # Query 1: 2 relevant docs, 1 in top-1, 2 in top-3
        metrics.add(
            query_idx=0,
            query_key="q1",
            ranking=[(10.0, "d1", 0), (9.0, "d2", 1), (8.0, "d3", 2)],
            gold_positives=["d1", "d3"]
        )
        
        # Query 2: 1 relevant doc, 0 in top-1, 1 in top-3
        metrics.add(
            query_idx=1,
            query_key="q2",
            ranking=[(10.0, "d4", 0), (9.0, "d5", 1), (8.0, "d6", 2)],
            gold_positives=["d6"]
        )
        
        # Query 3: 2 relevant docs, 0 in top-3, 1 in top-5, 2 in top-10
        metrics.add(
            query_idx=2,
            query_key="q3",
            ranking=[(10.0, "d7", 0), (9.0, "d8", 1), (8.0, "d9", 2), 
                     (7.0, "d10", 3), (6.0, "d11", 4), (5.0, "d12", 5)],
            gold_positives=["d11", "d12"]
        )
        
        # Manual calculation:
        # Recall@1:
        # - Query 1: 1/2 = 0.5
        # - Query 2: 0/1 = 0.0
        # - Query 3: 0/2 = 0.0
        # Average: (0.5 + 0.0 + 0.0) / 3 = 0.167
        
        # Recall@3:
        # - Query 1: 2/2 = 1.0
        # - Query 2: 1/1 = 1.0
        # - Query 3: 0/2 = 0.0
        # Average: (1.0 + 1.0 + 0.0) / 3 = 0.667
        
        # Recall@5:
        # - Query 1: 2/2 = 1.0
        # - Query 2: 1/1 = 1.0
        # - Query 3: 1/2 = 0.5
        # Average: (1.0 + 1.0 + 0.5) / 3 = 0.833
        
        # Recall@10:
        # - Query 1: 2/2 = 1.0
        # - Query 2: 1/1 = 1.0
        # - Query 3: 2/2 = 1.0
        # Average: (1.0 + 1.0 + 1.0) / 3 = 1.0
        
        self.assertAlmostEqual(metrics.recall_sums[1] / 3, 0.167, places=3)
        self.assertAlmostEqual(metrics.recall_sums[3] / 3, 0.667, places=3)
        self.assertAlmostEqual(metrics.recall_sums[5] / 3, 0.833, places=3)
        self.assertAlmostEqual(metrics.recall_sums[10] / 3, 1.0, places=3)
    
    def test_ndcg_calculation(self):
        """Test NDCG calculation directly"""
        # Create some example relevance scores (binary relevance)
        ranking_rel = [1, 0, 1, 0, 0]  # Relevance of docs at each position
        ideal_rel = [1, 1, 0, 0, 0]    # Ideal ordering of relevance
        
        # Calculate DCG and IDCG
        dcg = sum((2**rel - 1) / torch.log2(torch.tensor(pos + 2)) 
                  for pos, rel in enumerate(ranking_rel))
        idcg = sum((2**rel - 1) / torch.log2(torch.tensor(pos + 2)) 
                   for pos, rel in enumerate(ideal_rel))
        
        # Calculate NDCG
        ndcg = dcg / idcg
        
        # Expected NDCG value
        expected_ndcg = 0.9197  # Calculated by hand
        
        self.assertAlmostEqual(ndcg.item(), expected_ndcg, places=4)


if __name__ == "__main__":
    unittest.main() 