import json
import os
import pandas as pd
from typing import Dict, List, Any, Union
from pathlib import Path
import logging
from src.utils.logger import Logger

logger = logging.getLogger(__name__)

def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        The loaded DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.ParserError: If the file couldn't be parsed as CSV
    """
    logger.info(f"Loading DataFrame from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded DataFrame with shape {df.shape}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading DataFrame: {e}")
        raise

def save_json(data: Union[Dict, List], file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
    """
    logger.info(f"Saving JSON data to {file_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully saved JSON data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON data: {e}")
        raise

def load_json(file_path: str) -> Union[Dict, List]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        The loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file couldn't be parsed as JSON
    """
    logger.info(f"Loading JSON data from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON data from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        raise

def save_csv(data: List[Dict], file_path: str) -> None:
    """
    Save data to a CSV file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the CSV file
    """
    logger.info(f"Saving CSV data to {file_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved CSV data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving CSV data: {e}")
        raise 