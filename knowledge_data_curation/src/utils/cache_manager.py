from typing import Dict, Any, List, Tuple
import logging
import json
import pandas as pd
from pathlib import Path
import hashlib
import time

logger = logging.getLogger(__name__)

class SyntheticDataCacheManager:
    def __init__(self, config: Dict):
        """
        Initialize the synthetic data cache manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cache_dir = Path(config["paths"]["output_dir"]) / "synthetic_data_cache"
        self.cache_enabled = config.get("synthetic_generation", {}).get("enable_cache", True)
        self.reuse_existing = config.get("synthetic_generation", {}).get("reuse_existing", True)
        
        # Cache file configurations
        cache_filename = config.get("synthetic_generation", {}).get("cache_filename", "synthetic_factsheets.json")
        self.cache_paths = {
            "synthetic_data": self.cache_dir / cache_filename,
            "generation_config": self.cache_dir / "generation_config.json",
            "cache_metadata": self.cache_dir / "cache_metadata.json"
        }
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SyntheticDataCacheManager:")
        logger.info(f"  - Cache enabled: {self.cache_enabled}")
        logger.info(f"  - Reuse existing: {self.reuse_existing}")
        logger.info(f"  - Cache directory: {self.cache_dir}")
        
    def get_cache_paths(self) -> Dict[str, Path]:
        """
        Return paths for cached synthetic data files
        
        Returns:
            Dictionary mapping cache types to their file paths
        """
        return self.cache_paths.copy()
    
    def _generate_config_hash(self) -> str:
        """
        Generate a hash of the current configuration to detect changes
        
        Returns:
            Hexadecimal hash string representing the configuration
        """
        # Extract relevant config for hashing
        relevant_config = {
            "min_companies_threshold": self.config.get("synthetic_generation", {}).get("min_companies_threshold", 5),
            "target_factsheets_per_chain": self.config.get("synthetic_generation", {}).get("target_factsheets_per_chain", 10),
            "max_sample_factsheets": self.config.get("synthetic_generation", {}).get("max_sample_factsheets", 2),
            "model_factsheet_generator": self.config.get("openai", {}).get("model_factsheet_generator", "gpt-4")
        }
        
        # Create hash
        config_str = json.dumps(relevant_config, sort_keys=True)
        hash_object = hashlib.sha256(config_str.encode('utf-8'))
        return hash_object.hexdigest()[:16]  # First 16 characters
    
    def is_cached_data_available(self) -> bool:
        """
        Check if cached synthetic data exists and is valid
        
        Returns:
            True if valid cached data is available, False otherwise
        """
        if not self.cache_enabled or not self.reuse_existing:
            logger.debug("Cache disabled or reuse disabled")
            return False
        
        try:
            # Check if all required cache files exist
            required_files = ["synthetic_data", "generation_config", "cache_metadata"]
            for file_type in required_files:
                if not self.cache_paths[file_type].exists():
                    logger.debug(f"Cache file missing: {self.cache_paths[file_type]}")
                    return False
            
            # Load and validate cache metadata
            with open(self.cache_paths["cache_metadata"], 'r') as f:
                metadata = json.load(f)
            
            # Check if configuration has changed
            current_config_hash = self._generate_config_hash()
            cached_config_hash = metadata.get("config_hash", "")
            
            if current_config_hash != cached_config_hash:
                logger.info(f"Configuration changed - cache invalid")
                logger.debug(f"Current hash: {current_config_hash}, Cached hash: {cached_config_hash}")
                return False
            
            # Check cache age (optional - could add expiration logic here)
            cache_timestamp = metadata.get("timestamp", 0)
            cache_age_hours = (time.time() - cache_timestamp) / 3600
            logger.debug(f"Cache age: {cache_age_hours:.1f} hours")
            
            logger.info("Valid cached synthetic data found")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking cache availability: {e}")
            return False
    
    def load_cached_data(self) -> Dict[str, Any]:
        """
        Load existing cached synthetic data
        
        Returns:
            Dictionary containing cached synthetic data and metadata
        """
        if not self.is_cached_data_available():
            raise ValueError("No valid cached data available")
        
        logger.info("Loading cached synthetic data")
        
        try:
            # Load synthetic data
            with open(self.cache_paths["synthetic_data"], 'r') as f:
                synthetic_data = json.load(f)
            
            # Load generation config
            with open(self.cache_paths["generation_config"], 'r') as f:
                generation_config = json.load(f)
            
            # Load metadata
            with open(self.cache_paths["cache_metadata"], 'r') as f:
                metadata = json.load(f)
            
            # Load augmented datasets if they exist
            normalizer_config = self.config.get("synthetic_generation", {}).get("output_files", {})
            industry_filename = normalizer_config.get("augmented_industry_data", "augmented_industry_data.csv")
            factsheet_filename = normalizer_config.get("augmented_factsheet_data", "augmented_factsheet_data.csv")
            
            industry_path = self.cache_dir / industry_filename
            factsheet_path = self.cache_dir / factsheet_filename
            
            industry_df = None
            factsheet_df = None
            
            if industry_path.exists() and factsheet_path.exists():
                logger.info("Loading cached augmented datasets")
                industry_df = pd.read_csv(industry_path)
                factsheet_df = pd.read_csv(factsheet_path)
            else:
                logger.warning("Cached augmented datasets not found")
            
            cached_data = {
                "synthetic_data": synthetic_data,
                "generation_config": generation_config,
                "metadata": metadata,
                "industry_df": industry_df,
                "factsheet_df": factsheet_df
            }
            
            logger.info(f"Successfully loaded cached data:")
            logger.info(f"  - Chains: {len(synthetic_data)}")
            logger.info(f"  - Total factsheets: {sum(len(tuples) for tuples in synthetic_data.values())}")
            logger.info(f"  - Cache timestamp: {metadata.get('timestamp', 'unknown')}")
            
            return cached_data
            
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            raise
    
    def save_cached_data(self, data_to_cache: Dict[str, Any]) -> None:
        """
        Save synthetic data to cache
        
        Args:
            data_to_cache: Dictionary containing data to cache
        """
        if not self.cache_enabled:
            logger.debug("Cache disabled - not saving")
            return
        
        logger.info("Saving synthetic data to cache")
        
        try:
            # Prepare metadata
            metadata = {
                "timestamp": time.time(),
                "config_hash": self._generate_config_hash(),
                "cache_version": "1.0",
                "description": "Cached synthetic factsheet data for ColBERT training"
            }
            
            # Save synthetic data
            synthetic_data = data_to_cache.get("synthetic_data", {})
            with open(self.cache_paths["synthetic_data"], 'w') as f:
                json.dump(synthetic_data, f, indent=2)
            logger.info(f"Saved synthetic data to: {self.cache_paths['synthetic_data']}")
            
            # Save generation config
            generation_config = data_to_cache.get("generation_config", {})
            with open(self.cache_paths["generation_config"], 'w') as f:
                json.dump(generation_config, f, indent=2)
            logger.info(f"Saved generation config to: {self.cache_paths['generation_config']}")
            
            # Save metadata
            with open(self.cache_paths["cache_metadata"], 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved cache metadata to: {self.cache_paths['cache_metadata']}")
            
            # Note: Augmented datasets are saved separately by DataNormalizer
            
            logger.info("Cached data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving cached data: {e}")
            raise
    
    def invalidate_cache(self) -> None:
        """
        Invalidate (delete) all cached data
        """
        logger.info("Invalidating cached data")
        
        files_deleted = 0
        for cache_type, cache_path in self.cache_paths.items():
            try:
                if cache_path.exists():
                    cache_path.unlink()
                    files_deleted += 1
                    logger.debug(f"Deleted cache file: {cache_path}")
            except Exception as e:
                logger.warning(f"Error deleting cache file {cache_path}: {e}")
        
        # Also delete augmented dataset files
        try:
            normalizer_config = self.config.get("synthetic_generation", {}).get("output_files", {})
            industry_filename = normalizer_config.get("augmented_industry_data", "augmented_industry_data.csv")
            factsheet_filename = normalizer_config.get("augmented_factsheet_data", "augmented_factsheet_data.csv")
            
            for filename in [industry_filename, factsheet_filename]:
                file_path = self.cache_dir / filename
                if file_path.exists():
                    file_path.unlink()
                    files_deleted += 1
                    logger.debug(f"Deleted augmented dataset file: {file_path}")
                    
        except Exception as e:
            logger.warning(f"Error deleting augmented dataset files: {e}")
        
        logger.info(f"Cache invalidation completed - deleted {files_deleted} files")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state
        
        Returns:
            Dictionary with cache information
        """
        cache_info = {
            "cache_enabled": self.cache_enabled,
            "reuse_existing": self.reuse_existing,
            "cache_directory": str(self.cache_dir),
            "cache_files": {},
            "cache_valid": False,
            "total_size_bytes": 0
        }
        
        # Check each cache file
        for cache_type, cache_path in self.cache_paths.items():
            file_info = {
                "exists": cache_path.exists(),
                "path": str(cache_path),
                "size_bytes": 0,
                "modified_time": None
            }
            
            if cache_path.exists():
                try:
                    stat = cache_path.stat()
                    file_info["size_bytes"] = stat.st_size
                    file_info["modified_time"] = stat.st_mtime
                    cache_info["total_size_bytes"] += stat.st_size
                except Exception as e:
                    logger.warning(f"Error getting file stats for {cache_path}: {e}")
            
            cache_info["cache_files"][cache_type] = file_info
        
        # Check if cache is valid
        cache_info["cache_valid"] = self.is_cached_data_available()
        
        # Load metadata if available
        if cache_info["cache_files"]["cache_metadata"]["exists"]:
            try:
                with open(self.cache_paths["cache_metadata"], 'r') as f:
                    metadata = json.load(f)
                cache_info["metadata"] = metadata
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")
        
        return cache_info
    
    def cleanup_old_cache_files(self, max_age_days: int = 7) -> None:
        """
        Clean up old cache files
        
        Args:
            max_age_days: Maximum age in days for cache files
        """
        logger.info(f"Cleaning up cache files older than {max_age_days} days")
        
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        files_deleted = 0
        
        try:
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    if stat.st_mtime < cutoff_time:
                        file_path.unlink()
                        files_deleted += 1
                        logger.debug(f"Deleted old cache file: {file_path}")
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")
        
        logger.info(f"Cache cleanup completed - deleted {files_deleted} old files") 