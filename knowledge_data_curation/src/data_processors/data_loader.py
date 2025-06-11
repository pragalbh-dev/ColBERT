from typing import Dict, List, Tuple, Any
import logging
import pandas as pd
from pathlib import Path
from src.utils.io import load_dataframe

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: Dict):
        """
        Initialize the data loader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.industry_data_path = Path(config["paths"]["industry_data"])
        self.factsheet_data_path = Path(config["paths"]["factsheet_data"])
        logger.info(f"Initialized DataLoader with industry_data={self.industry_data_path}, factsheet_data={self.factsheet_data_path}")
        
            
    def get_unique_chains(self) -> List[str]:
        """
        Get list of unique industry chains
        
        Returns:
            List of unique industry chains
        """
        logger.info("Getting unique industry chains")
        try:
            df = self.load_industry_data()
            chains = df["industry_chain_hierarchy"].unique().tolist()
            logger.info(f"Found {len(chains)} unique industry chains")
            return chains
        except Exception as e:
            logger.error(f"Error getting unique chains: {e}")
            raise
            
    def get_company_chains(self) -> Dict[str, List[str]]:
        """
        Get mapping from company IDs to their industry chains
        
        Returns:
            Dictionary mapping company IDs to lists of industry chains
        """
        logger.info("Getting company to chains mapping")
        try:
            df = self.load_industry_data()
            company_chains = {}
            
            for _, row in df.iterrows():
                company_id = row["company_id"]
                chain = row["industry_chain_hierarchy"]
                
                if company_id not in company_chains:
                    company_chains[company_id] = []
                    
                company_chains[company_id].append(chain)
                
            logger.info(f"Created mapping for {len(company_chains)} companies")
            return company_chains
        except Exception as e:
            logger.error(f"Error getting company chains: {e}")
            raise
            
    def get_company_factsheets(self) -> Dict[str, str]:
        """
        Get mapping from company IDs to their factsheets
        
        Returns:
            Dictionary mapping company IDs to factsheets
        """
        logger.info("Getting company to factsheet mapping")
        try:
            df = self.load_factsheet_data()
            company_factsheets = dict(zip(df["company_id"], df["factsheet"]))
            logger.info(f"Created mapping for {len(company_factsheets)} companies")
            return company_factsheets
        except Exception as e:
            logger.error(f"Error getting company factsheets: {e}")
            raise
    
    def override_data(self, industry_df: pd.DataFrame, factsheet_df: pd.DataFrame) -> None:
        """
        Override the loaded data with augmented datasets
        
        Args:
            industry_df: Augmented industry DataFrame
            factsheet_df: Augmented factsheet DataFrame
        """
        logger.info("Overriding loaded data with augmented datasets")
        logger.info(f"  - New industry data shape: {industry_df.shape}")
        logger.info(f"  - New factsheet data shape: {factsheet_df.shape}")
        
        # Store the augmented data for future use
        self._cached_industry_df = industry_df
        self._cached_factsheet_df = factsheet_df
        self._using_augmented_data = True
        
        logger.info("Data override completed - pipeline will now use augmented datasets")
    
    def load_industry_data(self) -> pd.DataFrame:
        """
        Load industry data from CSV or return cached augmented data
        
        Returns:
            DataFrame with industry data
        """
        # If we have cached augmented data, use that instead
        if hasattr(self, '_cached_industry_df') and self._cached_industry_df is not None:
            logger.info("Using cached augmented industry data")
            return self._cached_industry_df
            
        logger.info(f"Loading industry data from {self.industry_data_path}")
        try:
            df = load_dataframe(str(self.industry_data_path))
            logger.info(f"Loaded industry data with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading industry data: {e}")
            raise
            
    def load_factsheet_data(self) -> pd.DataFrame:
        """
        Load factsheet data from CSV or return cached augmented data
        
        Returns:
            DataFrame with factsheet data
        """
        # If we have cached augmented data, use that instead
        if hasattr(self, '_cached_factsheet_df') and self._cached_factsheet_df is not None:
            logger.info("Using cached augmented factsheet data")
            return self._cached_factsheet_df
            
        logger.info(f"Loading factsheet data from {self.factsheet_data_path}")
        try:
            df = load_dataframe(str(self.factsheet_data_path))
            logger.info(f"Loaded factsheet data with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading factsheet data: {e}")
            raise 