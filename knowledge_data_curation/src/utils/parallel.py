from typing import List, Callable, Any, TypeVar, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')

def batch_process(
    items: List[T], 
    process_fn: Callable[[T], U], 
    batch_size: int = 10,
    max_workers: int = 8
) -> Dict[int, U]:
    """
    Process items in parallel batches with a specified batch size.
    
    Args:
        items: List of items to process
        process_fn: Function to apply to each item
        batch_size: Size of batches to process
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary mapping original indices to results
    """
    logger.info(f"Batch processing {len(items)} items with batch size {batch_size} and {max_workers} workers")
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1} with {len(batch)} items")
            
            futures = {
                executor.submit(process_fn, item): idx 
                for idx, item in enumerate(batch, start=i)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    logger.debug(f"Completed processing for item {idx}")
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    # Re-raise to handle at caller level
                    raise
    
    logger.info(f"Completed batch processing {len(items)} items")
    return results

def ordered_batch_results(batch_results: Dict[int, U], total_items: int) -> List[U]:
    """
    Convert batch processing results to an ordered list.
    
    Args:
        batch_results: Dictionary mapping indices to results
        total_items: Total number of items processed
        
    Returns:
        List of results in original order
    """
    logger.debug(f"Converting batch results to ordered list for {total_items} items")
    return [batch_results.get(i) for i in range(total_items)] 