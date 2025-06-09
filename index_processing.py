import os
import shutil
import pickle
import csv
from datetime import datetime
from pylate import indexes, models, retrieve

def backup_index(index_folder):
    """Create a backup of the index folder"""
    backup_folder = f"{index_folder}_backup"
    if os.path.exists(index_folder):
        if os.path.exists(backup_folder):
            shutil.rmtree(backup_folder)
        shutil.copytree(index_folder, backup_folder)
    return backup_folder

def restore_index(backup_folder, index_folder):
    """Restore index from backup"""
    if os.path.exists(backup_folder):
        if os.path.exists(index_folder):
            shutil.rmtree(index_folder)
        shutil.copytree(backup_folder, index_folder)

def log_failed_range(start_idx, end_idx, error_msg, log_file='failed_ranges.csv'):
    """Log failed document ranges to CSV"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(log_file) == 0:
            writer.writerow(['Timestamp', 'Start Index', 'End Index', 'Error'])
        writer.writerow([timestamp, start_idx, end_idx, str(error_msg)])

def process_documents_with_backup(documents_ids, documents_embeddings, batch_size=512, 
                                index_folder="pylate-index", index_name="modernbert-colbert-index"):
    """Process documents with backup and error handling"""
    
    def initialize_index():
        return indexes.Voyager(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
        )
    
    # Initialize index
    index = initialize_index()
    
    # Create log file if it doesn't exist
    if not os.path.exists('failed_ranges.csv'):
        with open('failed_ranges.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Start Index', 'End Index', 'Error'])
    
    total_docs = len(documents_ids)
    for start_idx in range(0, total_docs, batch_size):
        end_idx = min(start_idx + batch_size, total_docs)
        
        # Create backup before processing batch
        backup_folder = backup_index(index_folder)
        
        try:
            print(f"Processing documents {start_idx} to {end_idx}")
            index.add_documents(
                documents_ids=documents_ids[start_idx:end_idx],
                documents_embeddings=documents_embeddings[start_idx:end_idx],
                batch_size=batch_size
            )
            print(f"Successfully processed batch {start_idx} to {end_idx}")
            
        except Exception as e:
            print(f"Error processing documents {start_idx} to {end_idx}: {str(e)}")
            # Restore from backup
            restore_index(backup_folder, index_folder)
            # Reinitialize index with restored data
            index = initialize_index()
            # Log the failed range
            log_failed_range(start_idx, end_idx, str(e))
            continue
        
        finally:
            # Clean up backup after successful processing or restoration
            if os.path.exists(backup_folder):
                shutil.rmtree(backup_folder)

# Main execution
if __name__ == "__main__":
    # Load your documents
    with open('/home/sagemaker-user/SageMaker/ColBERT/colbert_embeddings.pkl','rb') as f:
        documents_ids, documents_embeddings, documents = pickle.load(f)
    
    _min = 8192 * 6
    _max = 8192 * 7
    
    print(f"Total documents to process: {len(documents_ids[_min:_max])}")
    
    process_documents_with_backup(
        documents_ids=documents_ids[_min:_max],
        documents_embeddings=documents_embeddings[_min:_max],
        batch_size=512,
        index_folder="pylate-index",
        index_name="modernbert-colbert-index"
    ) 