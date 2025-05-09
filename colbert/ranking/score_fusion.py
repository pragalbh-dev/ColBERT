"""
Score fusion techniques for combining multiple ranked lists.

This module implements various score fusion methods for combining scores from multiple retrieval systems:
- Relative Score Fusion (RSF): Normalizes scores based on their relative values within each system
- Distribution-based Score Fusion (DBSF): Transforms scores based on statistical properties of their distributions
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
from math import inf
import itertools
from typing import List, Dict, Any, Union, Optional
from ranx import Run as rank_run
from ranx import fuse

# Define a simple Document class for users who don't have a document class


def ranx_fusion(dfs, method='bordafuse', norm=None, doc_id_col='Document ID', score_col="Score"):
    """
    Apply fusion using the ranx library to combine multiple ranked lists.
    
    Args:
        dfs (List[pd.DataFrame]): List of dataframes to fuse
        method (str): Fusion method to use ('bordafuse', 'combsum', 'combmnz', 'rrf', 'logistic')
        norm (str): Normalization strategy (None, 'min-max', 'sum', 'max', 'std', 'rank')
        doc_id_col (str): Name of the document ID column
        score_col (str): Name of the score column
    
    Returns:
        pd.DataFrame: Combined dataframe with fused scores
    """
    runs = []
    
    for i, df in enumerate(dfs):
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check and rename columns if needed
        if doc_id_col not in df_copy.columns:
            raise ValueError(f"Document ID column '{doc_id_col}' not found in dataframe {i}. "
                             f"Available columns: {df_copy.columns.tolist()}")
        
        if score_col not in df_copy.columns:
            raise ValueError(f"Score column '{score_col}' not found in dataframe {i}. "
                             f"Available columns: {df_copy.columns.tolist()}")
        
        # Add query ID column if not present
        if 'q_id' not in df_copy.columns:
            df_copy['q_id'] = 'q1'
        
        # Ensure score column is float type
        df_copy[score_col] = df_copy[score_col].astype(float)
        
        # Create ranx Run
        try:
            run = rank_run.from_df(
                df=df_copy,
                q_id_col="q_id",
                doc_id_col=doc_id_col,
                score_col=score_col,
                name=f"run_{i}",
            )
            runs.append(run)
        except Exception as e:
            raise ValueError(f"Error creating ranx Run for dataframe {i}: {str(e)}")
    
    # Apply fusion
    try:
        if norm is not None:
            combined_run = fuse(
                runs=runs,
                norm=norm,
                method=method,
            )
        else:
            combined_run = fuse(
                runs=runs,
                method=method,
            )
        
        # Convert to dataframe
        combined_run_df = combined_run.to_dataframe()
        return combined_run_df
    except Exception as e:
        raise ValueError(f"Error during ranx fusion: {str(e)}")


# Define a simple Document class for users who don't have a document class
class Document:
    """
    Simple document class for use with multi_dbsf if users don't have their own document class.
    """
    def __init__(self, id: str, score: Optional[float] = None, content: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.score = score
        self.content = content
        self.metadata = metadata or {}

def multiquery_dfs(dfs: List[pd.DataFrame],logical_operation:str,not_clause=(False, False),score_fusion_params=Dict[str, Any]):
    """
    Perform multi-query score fusion on a list of dataframes using a logical operation.
    
    Args:
        dfs (List[pd.DataFrame]): List of dataframes, each with 'doc_id' and 'score' columns
        logical_operation (str): Logical operation to apply between dataframes, one of 'AND', 'OR', 'XOR'
        not_clause (Tuple[bool, bool]): Tuple of two booleans, where the first is True if the query is a NOT clause, and the second is True if the query is a NOT clause    
    """
    # Validate inputs
    if not dfs:
        raise ValueError("At least one dataframe is required")
    
    if logical_operation not in ['AND', 'OR']:
        raise ValueError("Logical operation must be one of 'AND', 'OR'")
    
    # Create a copy of the first dataframe
    result = dfs[0].copy()
    
    # Iterate through the remaining dataframes
    for df in dfs[1:]:
        # Merge on doc_id
        result = pd.merge(result, df, on='doc_id', how='outer')
        
        # Apply logical operation
        if logical_operation == 'AND':
            result['score'] = result.apply(lambda row: row['score_x'] and row['score_y'] if not not_clause[0] else not (row['score_x'] and row['score_y']), axis=1)
        elif logical_operation == 'OR':
            result['score'] = result.apply(lambda row: row['score_x'] or row['score_y'] if not not_clause[0] else not (row['score_x'] or row['score_y']), axis=1)
    
    # Sort by score in descending order
    result = result.sort_values('score', ascending=False)
    
    return result


def relative_score_fusion(df1, df2, method='minmax', alpha=0.5):
    """
    Implements Relative Score Fusion (RSF) to combine scores from two retrieval systems.
    
    RSF normalizes scores from different systems relative to their score range before combining them,
    addressing the problem of different score scales and distributions.
    
    Args:
        df1 (pd.DataFrame): First dataframe with columns 'doc_id' and 'score'
        df2 (pd.DataFrame): Second dataframe with columns 'doc_id' and 'score'
        method (str): Normalization method, one of 'minmax', 'rank', or 'softmax'
        alpha (float): Weight for the first system's score (1-alpha for the second)
                      in range [0,1], default 0.5 (equal weighting)
    
    Returns:
        pd.DataFrame: Combined dataframe with columns 'doc_id' and 'fused_score', sorted by 'fused_score'
    """
    # Validate inputs
    for df in [df1, df2]:
        if not {'doc_id', 'score'}.issubset(df.columns):
            raise ValueError("Both dataframes must contain 'doc_id' and 'score' columns")
    
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be in range [0,1]")
    
    # Create copies to avoid modifying originals
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    
    # Normalize scores based on the chosen method
    if method == 'minmax':
        # Min-max normalization scales scores to [0,1] range
        df1_copy['norm_score'] = _minmax_normalize(df1_copy['score'])
        df2_copy['norm_score'] = _minmax_normalize(df2_copy['score'])
    
    elif method == 'rank':
        # Rank-based normalization (higher rank = higher normalized score)
        df1_copy['norm_score'] = _rank_normalize(df1_copy['score'])
        df2_copy['norm_score'] = _rank_normalize(df2_copy['score'])
    
    elif method == 'softmax':
        # Softmax normalization gives a probability distribution
        df1_copy['norm_score'] = _softmax_normalize(df1_copy['score'])
        df2_copy['norm_score'] = _softmax_normalize(df2_copy['score'])
    
    else:
        raise ValueError("Method must be one of 'minmax', 'rank', or 'softmax'")
    
    # Merge dataframes on doc_id
    merged_df = pd.merge(
        df1_copy[['doc_id', 'norm_score']], 
        df2_copy[['doc_id', 'norm_score']], 
        on='doc_id', 
        how='outer',
        suffixes=('_1', '_2')
    )
    
    # Fill NaN values with 0 (for documents that appear in only one system)
    merged_df.fillna(0, inplace=True)
    
    # Compute weighted fusion score
    merged_df['fused_score'] = alpha * merged_df['norm_score_1'] + (1 - alpha) * merged_df['norm_score_2']
    
    # Return sorted results with only doc_id and fused_score
    result = merged_df[['doc_id', 'fused_score']].sort_values('fused_score', ascending=False)
    
    return result


def relative_score_fusion_multiple(dataframes: List[pd.DataFrame], method='minmax', weights: Optional[List[float]] = None):
    """
    Implements Relative Score Fusion (RSF) for an arbitrary number of dataframes.
    
    This function extends the original relative_score_fusion function to handle multiple dataframes,
    normalizing each one according to the specified method and then computing a weighted fusion score.
    
    Args:
        dataframes (List[pd.DataFrame]): List of dataframes, each with 'doc_id' and 'score' columns
        method (str): Normalization method, one of 'minmax', 'rank', or 'softmax'
        weights (List[float], optional): Weights for each dataframe's scores. If None, equal weights are used.
                                       Must sum to 1.0 if provided.
    
    Returns:
        pd.DataFrame: Combined dataframe with columns 'doc_id' and 'fused_score', sorted by 'fused_score'
    """
    # Validate inputs
    if not dataframes:
        return pd.DataFrame(columns=['doc_id', 'fused_score'])
    
    for df in dataframes:
        if not {'doc_id', 'score'}.issubset(df.columns):
            raise ValueError("All dataframes must contain 'doc_id' and 'score' columns")
    
    # Set default weights if none provided (equal weighting)
    if weights is None:
        weights = [1.0 / len(dataframes)] * len(dataframes)
    else:
        if len(weights) != len(dataframes):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of dataframes ({len(dataframes)})")
        
        if abs(sum(weights) - 1.0) > 1e-6:  # Allow for small floating-point errors
            raise ValueError("Weights must sum to 1.0")
            
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")
    
    # Validate normalization method
    if method not in ['minmax', 'rank', 'softmax']:
        raise ValueError("Method must be one of 'minmax', 'rank', or 'softmax'")
    
    # Create copies to avoid modifying originals
    df_copies = [df.copy() for df in dataframes]
    
    # Normalize each dataframe independently using the specified method
    for i, df in enumerate(df_copies):
        if method == 'minmax':
            df['norm_score'] = _minmax_normalize(df['score'])
        elif method == 'rank':
            df['norm_score'] = _rank_normalize(df['score'])
        elif method == 'softmax':
            df['norm_score'] = _softmax_normalize(df['score'])
        
        # Add system identifier
        df['system_id'] = i
    
    # Combine all normalized dataframes
    combined_df = pd.concat(df_copies)[['doc_id', 'norm_score', 'system_id']]
    
    # Create a pivot table with doc_id as index and system_id as columns
    pivot_df = combined_df.pivot_table(
        index='doc_id', 
        columns='system_id', 
        values='norm_score',
        fill_value=0  # Documents not present in a system get score 0
    )
    
    # Compute weighted fusion score
    pivot_df['fused_score'] = 0
    for i, weight in enumerate(weights):
        if i in pivot_df.columns:  # Check if this system_id exists in columns
            pivot_df['fused_score'] += weight * pivot_df[i]
    
    # Convert back to regular dataframe format
    result = pivot_df.reset_index()[['doc_id', 'fused_score']]
    
    # Sort by fused_score in descending order
    result = result.sort_values('fused_score', ascending=False)
    
    return result


def distribution_based_score_fusion(df1, df2, method='z-score', alpha=0.5):
    """
    Implements Distribution-based Score Fusion (DBSF) to combine scores from two retrieval systems.
    
    DBSF transforms scores based on statistical properties of their distributions,
    making them more comparable before combination.
    
    Args:
        df1 (pd.DataFrame): First dataframe with columns 'doc_id' and 'score'
        df2 (pd.DataFrame): Second dataframe with columns 'doc_id' and 'score'
        method (str): Transformation method, one of 'z-score', 'quantile', or 'kernel'
        alpha (float): Weight for the first system's score (1-alpha for the second)
                      in range [0,1], default 0.5 (equal weighting)
    
    Returns:
        pd.DataFrame: Combined dataframe with columns 'doc_id' and 'fused_score', sorted by 'fused_score'
    """
    # Validate inputs
    for df in [df1, df2]:
        if not {'doc_id', 'score'}.issubset(df.columns):
            raise ValueError("Both dataframes must contain 'doc_id' and 'score' columns")
    
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be in range [0,1]")
    
    # Create copies to avoid modifying originals
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    
    # Transform scores based on the chosen method
    if method == 'z-score':
        # Z-score normalization (standard score)
        df1_copy['norm_score'] = _zscore_normalize(df1_copy['score'])
        df2_copy['norm_score'] = _zscore_normalize(df2_copy['score'])
    
    elif method == 'quantile':
        # Quantile-based transformation using cumulative distribution function
        df1_copy['norm_score'] = _quantile_normalize(df1_copy['score'])
        df2_copy['norm_score'] = _quantile_normalize(df2_copy['score'])
    
    elif method == 'kernel':
        # Kernel density estimation-based transformation
        df1_copy['norm_score'] = _kernel_normalize(df1_copy['score'])
        df2_copy['norm_score'] = _kernel_normalize(df2_copy['score'])
    
    else:
        raise ValueError("Method must be one of 'z-score', 'quantile', or 'kernel'")
    
    # Merge dataframes on doc_id
    merged_df = pd.merge(
        df1_copy[['doc_id', 'norm_score']], 
        df2_copy[['doc_id', 'norm_score']], 
        on='doc_id', 
        how='outer',
        suffixes=('_1', '_2')
    )
    
    # Fill NaN values with 0 (for documents that appear in only one system)
    merged_df.fillna(0, inplace=True)
    
    # Compute weighted fusion score
    merged_df['fused_score'] = alpha * merged_df['norm_score_1'] + (1 - alpha) * merged_df['norm_score_2']
    
    # Return sorted results with only doc_id and fused_score
    result = merged_df[['doc_id', 'fused_score']].sort_values('fused_score', ascending=False)
    
    return result


def dbsf(df1, df2, std_dev=3, alpha=0.5):
    """
    Implements Distribution-Based Score Fusion (DBSF) as described by Mazzeschi.
    
    DBSF normalizes scores using the distribution tails of each system, defined as
    the range [mean - std_dev * std, mean + std_dev * std], before combining them.
    This addresses the issue of embedding models having different score distributions
    and means, making scores more comparable across different retrieval models.
    
    Reference: https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18
    
    Args:
        df1 (pd.DataFrame): First dataframe with columns 'doc_id' and 'score'
        df2 (pd.DataFrame): Second dataframe with columns 'doc_id' and 'score'
        std_dev (float): Number of standard deviations to define distribution tails (default: 3)
        alpha (float): Weight for the first system's score (1-alpha for the second)
                      in range [0,1], default 0.5 (equal weighting)
    
    Returns:
        pd.DataFrame: Combined dataframe with columns 'doc_id' and 'fused_score', sorted by 'fused_score'
    """
    # Validate inputs
    for df in [df1, df2]:
        if not {'doc_id', 'score'}.issubset(df.columns):
            raise ValueError("Both dataframes must contain 'doc_id' and 'score' columns")
    
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be in range [0,1]")
        
    # Create copies to avoid modifying originals
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    
    # Calculate statistical properties for each distribution
    mean1, std1 = df1_copy['score'].mean(), df1_copy['score'].std()
    mean2, std2 = df2_copy['score'].mean(), df2_copy['score'].std()
    
    # Define the distribution tails for each system (using 3 std deviations by default)
    min1 = max(mean1 - std_dev * std1, df1_copy['score'].min())
    max1 = min(mean1 + std_dev * std1, df1_copy['score'].max())
    
    min2 = max(mean2 - std_dev * std2, df2_copy['score'].min())
    max2 = min(mean2 + std_dev * std2, df2_copy['score'].max())
    
    # Apply minmax scaling using the distribution tails
    df1_copy['norm_score'] = _custom_minmax_normalize(df1_copy['score'], min1, max1)
    df2_copy['norm_score'] = _custom_minmax_normalize(df2_copy['score'], min2, max2)
    
    # Merge dataframes on doc_id
    merged_df = pd.merge(
        df1_copy[['doc_id', 'norm_score']], 
        df2_copy[['doc_id', 'norm_score']], 
        on='doc_id', 
        how='outer',
        suffixes=('_1', '_2')
    )
    
    # Fill NaN values with 0 (for documents that appear in only one system)
    merged_df.fillna(0, inplace=True)
    
    # Compute weighted fusion score
    merged_df['fused_score'] = alpha * merged_df['norm_score_1'] + (1 - alpha) * merged_df['norm_score_2']
    
    # Return sorted results with only doc_id and fused_score
    result = merged_df[['doc_id', 'fused_score']].sort_values('fused_score', ascending=False)
    
    return result


def dbsf_multiple(dataframes: List[pd.DataFrame], weights: Optional[List[float]] = None, std_dev=3):
    """
    Implements Distribution-Based Score Fusion (DBSF) for an arbitrary number of dataframes.
    
    This function extends the original dbsf function to handle multiple dataframes,
    applying DBSF normalization to each dataframe and then computing a weighted fusion score.
    
    Reference: https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18
    
    Args:
        dataframes (List[pd.DataFrame]): List of dataframes, each with 'doc_id' and 'score' columns
        weights (List[float], optional): Weights for each dataframe's scores. If None, equal weights are used.
                                       Must sum to 1.0 if provided.
        std_dev (float): Number of standard deviations to define distribution tails (default: 3)
    
    Returns:
        pd.DataFrame: Combined dataframe with columns 'doc_id' and 'fused_score', sorted by 'fused_score'
    """
    # Validate inputs
    if not dataframes:
        return pd.DataFrame(columns=['doc_id', 'fused_score'])
    
    for df in dataframes:
        if not {'doc_id', 'score'}.issubset(df.columns):
            raise ValueError("All dataframes must contain 'doc_id' and 'score' columns")
    
    # Set default weights if none provided (equal weighting)
    if weights is None:
        weights = [1.0 / len(dataframes)] * len(dataframes)
    else:
        if len(weights) != len(dataframes):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of dataframes ({len(dataframes)})")
        
        if abs(sum(weights) - 1.0) > 1e-6:  # Allow for small floating-point errors
            raise ValueError("Weights must sum to 1.0")
            
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")
    
    # Create copies to avoid modifying originals
    df_copies = [df.copy() for df in dataframes]
    
    # Normalize each dataframe independently using DBSF approach
    for i, df in enumerate(df_copies):
        # Calculate distribution statistics
        mean_score = df['score'].mean()
        std_score = df['score'].std()
        
        # Define distribution tails
        min_score = max(mean_score - std_dev * std_score, df['score'].min())
        max_score = min(mean_score + std_dev * std_score, df['score'].max())
        
        # Apply minmax scaling using distribution tails
        df['norm_score'] = _custom_minmax_normalize(df['score'], min_score, max_score)
        
        # Add system identifier
        df['system_id'] = i
    
    # Combine all normalized dataframes
    combined_df = pd.concat(df_copies)[['doc_id', 'norm_score', 'system_id']]
    
    # Create a pivot table with doc_id as index and system_id as columns
    pivot_df = combined_df.pivot_table(
        index='doc_id', 
        columns='system_id', 
        values='norm_score',
        fill_value=0  # Documents not present in a system get score 0
    )
    
    # Compute weighted fusion score
    pivot_df['fused_score'] = 0
    for i, weight in enumerate(weights):
        if i in pivot_df.columns:  # Check if this system_id exists in columns
            pivot_df['fused_score'] += weight * pivot_df[i]
    
    # Convert back to regular dataframe format
    result = pivot_df.reset_index()[['doc_id', 'fused_score']]
    
    # Sort by fused_score in descending order
    result = result.sort_values('fused_score', ascending=False)
    
    return result


def multi_dbsf_df(dataframes: List[pd.DataFrame], std_dev=3):
    """
    Implements Distribution-Based Score Fusion (DBSF) for multiple dataframes.
    
    This function normalizes scores for each dataframe based on its own distribution,
    then combines them keeping the highest score for each document ID.
    
    This is a dataframe-based version of the multi_dbsf function, designed to work
    with the same input/output format as the other score fusion functions.
    
    Reference: https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18
    
    Args:
        dataframes: List of pandas DataFrames, each with 'doc_id' and 'score' columns
        std_dev (float): Number of standard deviations to define distribution tails (default: 3)
    
    Returns:
        pd.DataFrame: Combined dataframe with columns 'doc_id' and 'fused_score', sorted by 'fused_score'
    """
    # Validate inputs
    for df in dataframes:
        if not {'doc_id', 'score'}.issubset(df.columns):
            raise ValueError("All dataframes must contain 'doc_id' and 'score' columns")
    
    if not dataframes:
        return pd.DataFrame(columns=['doc_id', 'fused_score'])
    
    # Create copies to avoid modifying originals
    df_copies = [df.copy() for df in dataframes]
    
    # Normalize each dataframe's scores independently
    for df in df_copies:
        # Calculate distribution statistics
        mean_score = df['score'].mean()
        std_score = df['score'].std()
        
        # Define distribution tails (3 standard deviations from mean by default)
        min_score = max(mean_score - std_dev * std_score, df['score'].min())
        max_score = min(mean_score + std_dev * std_score, df['score'].max())
        delta_score = max_score - min_score
        
        # Apply min-max normalization using distribution tails
        if delta_score != 0:
            df['norm_score'] = df['score'].apply(
                lambda s: (s - min_score) / delta_score if min_score <= s <= max_score
                else 0.0 if s < min_score else 1.0
            )
        else:
            # If all scores are the same, set normalized score to 0
            df['norm_score'] = 0.0
    
    # Combine all dataframes and select the highest score for each doc_id
    all_docs = pd.concat(df_copies)[['doc_id', 'norm_score']]
    
    # Group by doc_id and take the maximum score
    result = all_docs.groupby('doc_id', as_index=False)['norm_score'].max()
    
    # Rename column for consistency with other functions
    result = result.rename(columns={'norm_score': 'fused_score'})
    
    # Sort by fused_score in descending order
    result = result.sort_values('fused_score', ascending=False)
    
    return result


def multi_dbsf(document_lists: List[List[Union[Document, Dict[str, Any]]]]) -> List[Union[Document, Dict[str, Any]]]:
    """
    Alternative implementation of Distribution-Based Score Fusion (DBSF) that works with lists of documents.
    
    This implementation:
    1. Normalizes each document list's scores independently using distribution statistics
    2. Combines all lists, keeping only the highest-scoring version of each document
    
    This matches the implementation pattern from the provided code sample and can process
    an arbitrary number of document lists.
    
    Reference: https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18
    
    Args:
        document_lists: List of lists containing Document objects or dictionaries with 'id' and 'score' keys
                       Each list represents results from a different retrieval system
    
    Returns:
        List of documents with normalized and fused scores, duplicates removed
    """
    # Process each document list independently
    for documents in document_lists:
        if len(documents) == 0:
            continue

        # Extract scores from documents
        scores_list = []
        for doc in documents:
            # Handle both Document objects and dictionaries
            if isinstance(doc, dict):
                score = doc.get('score', 0)
            else:  # Assume Document object
                score = doc.score if doc.score is not None else 0
            scores_list.append(score)

        # Calculate distribution statistics
        mean_score = sum(scores_list) / len(scores_list)
        std_dev = (sum((x - mean_score) ** 2 for x in scores_list) / len(scores_list)) ** 0.5
        
        # Define distribution tails (3 standard deviations from mean)
        min_score = mean_score - 3 * std_dev
        max_score = mean_score + 3 * std_dev
        delta_score = max_score - min_score

        # Normalize scores in-place
        for doc in documents:
            if isinstance(doc, dict):
                original_score = doc.get('score', 0)
                doc['score'] = (original_score - min_score) / delta_score if delta_score != 0.0 else 0.0
            else:  # Assume Document object
                original_score = doc.score if doc.score is not None else 0
                doc.score = (original_score - min_score) / delta_score if delta_score != 0.0 else 0.0

    # Combine documents, keeping highest score for duplicates
    return _concatenate_documents(document_lists)


def _concatenate_documents(document_lists: List[List[Union[Document, Dict[str, Any]]]]) -> List[Union[Document, Dict[str, Any]]]:
    """
    Concatenate multiple lists of documents and return only the document with the highest score for duplicates.
    
    Args:
        document_lists: List of lists containing Document objects or dictionaries with 'id' and 'score' keys
    
    Returns:
        List of documents with duplicates removed (keeping highest score)
    """
    output = []
    docs_per_id = defaultdict(list)
    
    # Flatten lists and group by document ID
    for doc in itertools.chain.from_iterable(document_lists):
        if isinstance(doc, dict):
            doc_id = doc.get('id')
        else:  # Assume Document object
            doc_id = doc.id
        docs_per_id[doc_id].append(doc)
    
    # For each document ID, keep only the version with highest score
    for docs in docs_per_id.values():
        if isinstance(docs[0], dict):
            # For dictionaries, use get() to safely access score
            doc_with_best_score = max(docs, key=lambda d: d.get('score', -inf))
        else:
            # For Document objects
            doc_with_best_score = max(docs, key=lambda d: d.score if d.score is not None else -inf)
        output.append(doc_with_best_score)
    
    # Sort by score in descending order
    if len(output) > 0:
        if isinstance(output[0], dict):
            output.sort(key=lambda d: d.get('score', 0), reverse=True)
        else:
            output.sort(key=lambda d: d.score if d.score is not None else 0, reverse=True)
    
    return output


# Helper functions for normalization methods

def _minmax_normalize(scores):
    """Min-max normalization to scale scores to [0,1] range"""
    min_score = scores.min()
    max_score = scores.max()
    
    # Handle edge case of all scores being the same
    if max_score == min_score:
        return np.ones(len(scores))
    
    return (scores - min_score) / (max_score - min_score)


def _custom_minmax_normalize(scores, min_val, max_val):
    """
    Custom min-max normalization using provided min and max values
    instead of the actual min and max of the data
    """
    # Handle edge case of min_val == max_val
    if max_val == min_val:
        return np.ones(len(scores))
    
    # Clip values to the provided range
    clipped_scores = np.clip(scores, min_val, max_val)
    
    # Apply min-max normalization
    return (clipped_scores - min_val) / (max_val - min_val)


def _rank_normalize(scores):
    """Rank-based normalization"""
    # Convert to ranks (higher score = higher rank)
    ranks = scores.rank(method='dense', ascending=False)
    
    # Normalize ranks to [0,1]
    return (len(scores) - ranks + 1) / len(scores)


def _softmax_normalize(scores):
    """Softmax normalization for probability distribution"""
    # Apply temperature scaling to prevent numerical instability
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)


def _zscore_normalize(scores):
    """Z-score normalization (standard score)"""
    mean = scores.mean()
    std = scores.std()
    
    # Handle edge case of zero standard deviation
    if std == 0:
        return np.zeros(len(scores))
    
    z_scores = (scores - mean) / std
    
    # Convert to [0,1] range for better fusion compatibility
    # Using sigmoid function: 1/(1+e^(-z))
    return 1 / (1 + np.exp(-z_scores))


def _quantile_normalize(scores):
    """Quantile-based normalization using empirical CDF"""
    # Create empirical CDF
    ecdf = stats.ecdf(scores)
    
    # Apply CDF to get quantiles (probability scores)
    return np.array([ecdf(score) for score in scores])


def _kernel_normalize(scores):
    """Kernel density estimation-based normalization"""
    # Use Gaussian KDE to estimate probability density
    if len(scores) < 2:
        # Need at least 2 points for KDE
        return np.ones(len(scores)) / len(scores)
    
    try:
        kde = stats.gaussian_kde(scores)
        density = kde(scores)
        
        # Normalize to [0,1]
        return density / np.max(density)
    except np.linalg.LinAlgError:
        # Fallback to simple normalization if KDE fails
        return _minmax_normalize(scores)


def or_operation_fusion(df1, df2):
    """
    Implements OR operation between two result dataframes.
    Takes the maximum score for overlapping documents and preserves 
    individual scores for non-overlapping documents.
    
    Args:
        df1 (pd.DataFrame): First dataframe with columns 'doc_id' and 'score'
        df2 (pd.DataFrame): Second dataframe with columns 'doc_id' and 'score'
    
    Returns:
        pd.DataFrame: Combined dataframe with columns 'doc_id' and 'score', sorted by 'score'
    """
    # Validate inputs
    for df in [df1, df2]:
        if not {'doc_id', 'score'}.issubset(df.columns):
            raise ValueError("Both dataframes must contain 'doc_id' and 'score' columns")
    
    # Merge dataframes on doc_id
    merged_df = pd.merge(
        df1[['doc_id', 'score']], 
        df2[['doc_id', 'score']], 
        on='doc_id', 
        how='outer',
        suffixes=('_1', '_2')
    )
    
    # Fill NaN values with 0 (for documents that appear in only one system)
    merged_df.fillna(0, inplace=True)
    
    # Take the maximum score for each document
    merged_df['score'] = merged_df.apply(lambda row: max(row['score_1'], row['score_2']), axis=1)
    
    # Return sorted results with only doc_id and score
    result = merged_df[['doc_id', 'score']].sort_values('score', ascending=False)
    
    return result


def rrf_or_fusion(df1, df2, k=60):
    """
    Implements OR operation between two result dataframes using Reciprocal Rank Fusion.
    
    RRF computes a score for each document as the sum of 1/(k + r_i) where r_i is the rank 
    of the document in the i-th system, and k is a constant.
    
    Args:
        df1 (pd.DataFrame): First dataframe with columns 'doc_id' and 'score'
        df2 (pd.DataFrame): Second dataframe with columns 'doc_id' and 'score'
        k (int): Constant that controls the relative impact of lower-ranked documents
    
    Returns:
        pd.DataFrame: Combined dataframe with columns 'doc_id' and 'score', sorted by 'score'
    """
    # Validate inputs
    for df in [df1, df2]:
        if not {'doc_id', 'score'}.issubset(df.columns):
            raise ValueError("Both dataframes must contain 'doc_id' and 'score' columns")
    
    # Convert scores to ranks within each dataframe
    df1_ranked = df1.copy()
    df2_ranked = df2.copy()
    
    df1_ranked['rank'] = df1_ranked['score'].rank(method='min', ascending=False)
    df2_ranked['rank'] = df2_ranked['score'].rank(method='min', ascending=False)
    
    # Calculate RRF scores
    df1_ranked['rrf_score'] = 1.0 / (k + df1_ranked['rank'])
    df2_ranked['rrf_score'] = 1.0 / (k + df2_ranked['rank'])
    
    # Combine the sets (union of document IDs)
    all_docs = pd.concat([
        df1_ranked[['doc_id', 'rrf_score']],
        df2_ranked[['doc_id', 'rrf_score']]
    ])
    
    # Group by doc_id and sum the RRF scores
    combined = all_docs.groupby('doc_id')['rrf_score'].sum().reset_index()
    
    # Rename rrf_score to score for consistency
    combined = combined.rename(columns={'rrf_score': 'score'})
    
    # Sort by score in descending order
    result = combined.sort_values('score', ascending=False)
    
    # Add metadata about the fusion method used
    result.attrs['fusion_method'] = 'rrf_or'
    
    return result


def logical_query_fusion(dfs, operations, fusion_method='dbsf', std_dev=3, alpha=0.5, rsf_method='minmax', weights=None, ranx_method='bordafuse', ranx_norm=None):
    """
    Executes a series of logical operations between dataframes using specified score fusion methods.
    
    Args:
        dfs (List[pd.DataFrame]): List of dataframes, each with columns 'doc_id' and 'score'
        operations (List[str]): List of operations ['AND', 'OR', 'NOT'] to apply between consecutive dataframes
        fusion_method (str): Method to use for combining scores in AND operations:
                           'dbsf': Distribution-based Score Fusion (weighted sum)
                           'multi_dbsf': DBSF keeping max score for each document
                           'rsf': Relative Score Fusion
                           'direct': No fusion, just filter by common documents
                           'ranx': Use ranx library for fusion
        std_dev (float): Standard deviation parameter for DBSF
        alpha (float): Weight parameter for score fusion methods
        rsf_method (str): Method for RSF ('minmax', 'rank', or 'softmax')
        weights (List[float]): Weights for multiple dataframes if using RSF or DBSF with more than 2 dataframes
        ranx_method (str): Fusion method for ranx ('bordafuse', 'combsum', 'rrf', etc.)
        ranx_norm (str): Normalization strategy for ranx fusion (None, 'min-max', 'sum', 'max')
    
    Returns:
        pd.DataFrame: Final result dataframe after applying all operations
    """
    if len(dfs) != len(operations) + 1:
        raise ValueError("Number of dataframes must be one more than the number of operations")
    
    if not all(op in ['AND', 'OR', 'NOT'] for op in operations):
        raise ValueError("Operations must be 'AND', 'OR', or 'NOT'")
    
    # Start with the first dataframe
    result = dfs[0].copy()
    
    # Initialize fusion method tracking
    result.attrs['fusion_method'] = 'raw'  # Initial dataframe has raw scores
    
    # Ensure consistent column naming for the initial dataframe
    if 'doc_id' not in result.columns and 'Document ID' in result.columns:
        result.rename(columns={'Document ID': 'doc_id'}, inplace=True)
    if 'score' not in result.columns and 'Score' in result.columns:
        result.rename(columns={'Score': 'score'}, inplace=True)
    
    # Process each operation and corresponding dataframe
    for i, operation in enumerate(operations):
        next_df = dfs[i+1].copy()
        
        # Initialize next_df fusion method tracking if not present
        if not hasattr(next_df, 'attrs') or 'fusion_method' not in next_df.attrs:
            next_df.attrs['fusion_method'] = 'raw'
        
        # Ensure consistent column naming for the next dataframe
        if 'doc_id' not in next_df.columns and 'Document ID' in next_df.columns:
            next_df.rename(columns={'Document ID': 'doc_id'}, inplace=True)
        if 'score' not in next_df.columns and 'Score' in next_df.columns:
            next_df.rename(columns={'Score': 'score'}, inplace=True)
        
        if operation == 'AND':
            # For AND, use the specified fusion method
            if fusion_method == 'dbsf':
                result = dbsf(result, next_df, std_dev=std_dev, alpha=alpha)
                # Ensure consistent column naming after fusion
                if 'fused_score' in result.columns:
                    result.rename(columns={'fused_score': 'score'}, inplace=True)
                # Track fusion method used
                result.attrs['fusion_method'] = 'dbsf'
                
            elif fusion_method == 'multi_dbsf':
                # Use the multi_dbsf_df function that takes maximum score
                result = multi_dbsf_df([result, next_df], std_dev=std_dev)
                # Ensure consistent column naming
                if 'fused_score' in result.columns:
                    result.rename(columns={'fused_score': 'score'}, inplace=True)
                # Track fusion method used
                result.attrs['fusion_method'] = 'multi_dbsf'
                
            elif fusion_method == 'rsf':
                result = relative_score_fusion(result, next_df, method=rsf_method, alpha=alpha)
                # Ensure consistent column naming after fusion
                if 'fused_score' in result.columns:
                    result.rename(columns={'fused_score': 'score'}, inplace=True)
                # Track fusion method used
                result.attrs['fusion_method'] = 'rsf'
                
            elif fusion_method == 'ranx':
                # Special case for ranx fusion with multiple dataframes
                # If we have more than 2 dataframes, we process them all at once
                if i == 0 and len(dfs) > 2 and all(op == 'AND' for op in operations):
                    # Prepare all dataframes for ranx fusion
                    dfs_for_ranx = []
                    for df in dfs:
                        df_copy = df.copy()
                        # Add query ID column
                        df_copy['q_id'] = 'q1'
                        # Ensure consistent column naming
                        if 'doc_id' in df_copy.columns:
                            df_copy.rename(columns={'doc_id': 'Document ID'}, inplace=True)
                        if 'score' in df_copy.columns:
                            df_copy.rename(columns={'score': 'Score'}, inplace=True)
                        dfs_for_ranx.append(df_copy)
                    
                    # Apply ranx fusion to all dataframes at once
                    ranx_result = ranx_fusion(dfs_for_ranx, method=ranx_method, norm=ranx_norm)
                    
                    # Convert back to our expected format with consistent naming
                    result = ranx_result.rename(columns={'docid': 'doc_id', 'score': 'score'})
                    if 'q_id' in result.columns:
                        result = result.drop(columns=['q_id'])
                    
                    # Track fusion method used
                    result.attrs['fusion_method'] = 'ranx'
                    
                    # Skip the rest of the operations since we've processed all dataframes
                    break
                else:
                    # Standard case for ranx fusion with two dataframes
                    result_for_ranx = result.copy()
                    next_df_for_ranx = next_df.copy()
                    
                    result_for_ranx['q_id'] = 'q1'
                    next_df_for_ranx['q_id'] = 'q1'
                    
                    # Rename columns to match ranx expectations
                    result_for_ranx.rename(columns={'doc_id': 'Document ID', 'score': 'Score'}, inplace=True)
                    next_df_for_ranx.rename(columns={'doc_id': 'Document ID', 'score': 'Score'}, inplace=True)
                    
                    # Apply ranx fusion
                    ranx_result = ranx_fusion([result_for_ranx, next_df_for_ranx], method=ranx_method, norm=ranx_norm)
                    
                    # Convert back to our expected format with consistent naming
                    result = ranx_result.rename(columns={'docid': 'doc_id', 'score': 'score'})
                    if 'q_id' in result.columns:
                        result = result.drop(columns=['q_id'])
                    
                    # Track fusion method used
                    result.attrs['fusion_method'] = 'ranx'
            elif fusion_method == 'direct':
                # Only keep documents that appear in both
                common_docs = set(result['doc_id']).intersection(set(next_df['doc_id']))
                result = result[result['doc_id'].isin(common_docs)]
                
                # Track fusion method used
                result.attrs['fusion_method'] = 'direct'
            else:
                raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        elif operation == 'OR':
            # Check if we're combining results from two AND operations with incompatible fusion methods
            if (hasattr(result, 'attrs') and result.attrs.get('fusion_method') not in ['raw', 'dbsf', 'rsf', 'multi_dbsf'] and 
                hasattr(next_df, 'attrs') and next_df.attrs.get('fusion_method') not in ['raw', 'dbsf', 'rsf', 'multi_dbsf']):
                raise ValueError(
                    "Invalid combination: Cannot perform OR between results of AND operations that use methods other than "
                    "'dbsf', 'multi_dbsf', or 'rsf'. Current methods are: "
                    f"'{result.attrs.get('fusion_method')}' and '{next_df.attrs.get('fusion_method')}'."
                )
            
            # For OR, use RRF fusion instead of max score
            result = rrf_or_fusion(result, next_df)
        
        elif operation == 'NOT':
            # For NOT, exclude documents from the next dataframe
            # Assuming we want to exclude top 5000 documents from next_df
            exclude_docs = set(next_df.nlargest(5000, 'score')['doc_id'])
            result = result[~result['doc_id'].isin(exclude_docs)]
            
            # NOT doesn't change the fusion method type
    
    # Sort by score (use the appropriate score column that exists)
    if 'score' in result.columns:
        return result.sort_values('score', ascending=False)
    elif 'fused_score' in result.columns:
        return result.sort_values('fused_score', ascending=False)
    else:
        return result
