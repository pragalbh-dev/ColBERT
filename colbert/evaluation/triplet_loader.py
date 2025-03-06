import os
import json
from typing import Dict, List, Set, Tuple
from collections import defaultdict

def load_triplets(triplets_path: str) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Load triplets file and extract positives and negatives for each query
    
    Args:
        triplets_path: Path to triplets file (qid, pid, nid)
        Can handle two formats:
        - Tab-separated values: qid\\tpid\\tnid
        - JSONL arrays: [qid,pid,nid]
        
    Returns:
        Tuple of (positives, negatives) where:
            positives: Dict mapping qid to list of positive pids
            negatives: Dict mapping qid to list of negative pids
    """
    positives = defaultdict(set)
    negatives = defaultdict(set)
    
    print(f"#> Loading triplets from {triplets_path}")
    
    # Determine file format based on extension
    is_jsonl = triplets_path.endswith('.jsonl')
    
    with open(triplets_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse the line based on format
            if is_jsonl or line.startswith('['):
                try:
                    # Try to parse as JSON array
                    triplet = json.loads(line)
                    if len(triplet) >= 3:
                        qid, pid, nid = triplet[:3]
                    else:
                        continue
                except json.JSONDecodeError:
                    # Fall back to tab-separated parsing if JSON parsing fails
                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue
                    qid, pid, nid = map(int, parts[:3])
            else:
                # Parse as tab-separated
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                qid, pid, nid = map(int, parts[:3])
            
            # Add positive and negative examples
            positives[qid].add(pid)
            negatives[qid].add(nid)
    
    # Convert sets to lists for consistency with other loaders
    positives_list = {qid: list(pids) for qid, pids in positives.items()}
    negatives_list = {qid: list(nids) for qid, nids in negatives.items()}
    
    print(f"#> Loaded {len(positives)} queries with positives")
    print(f"#> Average positives per query: {sum(len(pids) for pids in positives.values()) / max(1, len(positives)):.2f}")
    print(f"#> Average negatives per query: {sum(len(nids) for nids in negatives.values()) / max(1, len(negatives)):.2f}")
    
    return positives_list, negatives_list

def convert_triplets_to_qrels(triplets_path: str, output_path: str = None) -> str:
    """
    Convert triplets file to qrels format
    
    Args:
        triplets_path: Path to triplets file (qid, pid, nid)
        output_path: Path to save qrels file (optional)
        
    Returns:
        Path to the qrels file
    """
    positives, _ = load_triplets(triplets_path)
    
    if output_path is None:
        output_path = os.path.splitext(triplets_path)[0] + '.qrels'
    
    with open(output_path, 'w') as f:
        for qid, pids in positives.items():
            for pid in pids:
                f.write(f"{qid}\t0\t{pid}\t1\n")
    
    print(f"#> Converted triplets to qrels format: {output_path}")
    return output_path 