from typing import Dict, List, Any, Tuple
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class DataNormalizer:
    def __init__(self, config: Dict):
        """
        Initialize the data normalizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config["paths"]["output_dir"])
        self.synthetic_dir = self.output_dir / "synthetic_data_cache"
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized DataNormalizer")
    
    def normalize_company_ids(self, df: pd.DataFrame, id_column: str = "company_id") -> pd.DataFrame:
        """
        Convert all company IDs to strings for type consistency
        
        Args:
            df: DataFrame with company IDs
            id_column: Name of the company ID column
            
        Returns:
            DataFrame with normalized company IDs
        """
        if id_column not in df.columns:
            logger.warning(f"Column '{id_column}' not found in DataFrame")
            return df
            
        logger.info(f"Normalizing company IDs in column '{id_column}' to string type")
        
        # Create a copy to avoid modifying original
        df_normalized = df.copy()
        
        # Check original types
        original_type = df_normalized[id_column].dtype
        logger.debug(f"Original {id_column} dtype: {original_type}")
        
        # Convert to string
        df_normalized[id_column] = df_normalized[id_column].astype(str)
        
        # Log conversion stats
        unique_before = df[id_column].nunique()
        unique_after = df_normalized[id_column].nunique()
        
        logger.info(f"Company ID normalization completed:")
        logger.info(f"  - Original dtype: {original_type}")
        logger.info(f"  - New dtype: {df_normalized[id_column].dtype}")
        logger.info(f"  - Unique IDs before: {unique_before}")
        logger.info(f"  - Unique IDs after: {unique_after}")
        
        if unique_before != unique_after:
            logger.warning(f"Unique ID count changed during normalization!")
        
        return df_normalized
    
    def create_synthetic_industry_entries(self, 
                                        synthetic_data: Dict[str, List[Tuple[str, str]]],
                                        cleaned_chains: Dict[str, str]) -> pd.DataFrame:
        """
        Create industry data entries for synthetic companies
        
        Args:
            synthetic_data: Dictionary mapping chains to (company_id, factsheet) tuples
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            
        Returns:
            DataFrame with synthetic industry entries
        """
        logger.info("Creating synthetic industry data entries")
        
        # Create reverse mapping from cleaned chains to original chains
        cleaned_to_original = {}
        for original, cleaned in cleaned_chains.items():
            if cleaned not in cleaned_to_original:
                cleaned_to_original[cleaned] = []
            cleaned_to_original[cleaned].append(original)
        
        synthetic_industry_rows = []
        
        for cleaned_chain, factsheet_tuples in synthetic_data.items():
            # Get an original chain for this cleaned chain (use first one)
            original_chains = cleaned_to_original.get(cleaned_chain, [])
            if not original_chains:
                logger.warning(f"No original chain found for cleaned chain: {cleaned_chain}")
                continue
                
            # Use the first original chain as template
            original_chain = original_chains[0]
            
            # Create industry entries for all synthetic companies in this chain
            for company_id, factsheet in factsheet_tuples:
                synthetic_industry_rows.append({
                    "company_id": company_id,
                    "industry_chain_hierarchy": original_chain
                })
        
        synthetic_industry_df = pd.DataFrame(synthetic_industry_rows)
        
        logger.info(f"Created {len(synthetic_industry_rows)} synthetic industry entries")
        logger.info(f"  - Chains represented: {len(synthetic_data)}")
        
        # ✅ FIX: Check if dataframe is empty before accessing columns
        if len(synthetic_industry_df) > 0:
            logger.info(f"  - Unique companies: {synthetic_industry_df['company_id'].nunique()}")
        else:
            logger.warning("  - No synthetic companies created (empty dataframe)")
        
        return synthetic_industry_df
    
    def create_synthetic_factsheet_entries(self, 
                                         synthetic_data: Dict[str, List[Tuple[str, str]]]) -> pd.DataFrame:
        """
        Create factsheet data entries for synthetic companies
        
        Args:
            synthetic_data: Dictionary mapping chains to (company_id, factsheet) tuples
            
        Returns:
            DataFrame with synthetic factsheet entries
        """
        logger.info("Creating synthetic factsheet data entries")
        
        synthetic_factsheet_rows = []
        
        for chain, factsheet_tuples in synthetic_data.items():
            for company_id, factsheet in factsheet_tuples:
                synthetic_factsheet_rows.append({
                    "company_id": company_id,
                    "factsheet": factsheet
                })
        
        synthetic_factsheet_df = pd.DataFrame(synthetic_factsheet_rows)
        
        logger.info(f"Created {len(synthetic_factsheet_rows)} synthetic factsheet entries")
        
        # ✅ FIX: Check if dataframe is empty before accessing columns
        if len(synthetic_factsheet_df) > 0:
            logger.info(f"  - Unique companies: {synthetic_factsheet_df['company_id'].nunique()}")
            logger.info(f"  - Average factsheet length: {synthetic_factsheet_df['factsheet'].str.len().mean():.0f} characters")
        else:
            logger.warning("  - No synthetic factsheets created (empty dataframe)")
        
        return synthetic_factsheet_df
    
    def create_augmented_datasets(self, 
                                original_industry_df: pd.DataFrame, 
                                original_factsheet_df: pd.DataFrame,
                                synthetic_data: Dict[str, List[Tuple[str, str]]],
                                cleaned_chains: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create new datasets with synthetic data appended
        
        Args:
            original_industry_df: Original industry data
            original_factsheet_df: Original factsheet data
            synthetic_data: Dictionary mapping chains to (company_id, factsheet) tuples
            cleaned_chains: Dictionary mapping original chains to cleaned chains
            
        Returns:
            Tuple of (augmented_industry_df, augmented_factsheet_df)
        """
        logger.info("Creating augmented datasets with synthetic data")
        
        # Step 1: Normalize original data company IDs
        logger.info("Step 1: Normalizing original company IDs")
        normalized_industry_df = self.normalize_company_ids(original_industry_df)
        normalized_factsheet_df = self.normalize_company_ids(original_factsheet_df)
        
        # Step 2: Create synthetic data entries
        logger.info("Step 2: Creating synthetic data entries")
        synthetic_industry_df = self.create_synthetic_industry_entries(synthetic_data, cleaned_chains)
        synthetic_factsheet_df = self.create_synthetic_factsheet_entries(synthetic_data)
        
        # Step 3: Combine original and synthetic data
        logger.info("Step 3: Combining original and synthetic data")
        
        # Combine industry data
        augmented_industry_df = pd.concat([
            normalized_industry_df,
            synthetic_industry_df
        ], ignore_index=True)
        
        # Combine factsheet data
        augmented_factsheet_df = pd.concat([
            normalized_factsheet_df,
            synthetic_factsheet_df
        ], ignore_index=True)
        
        # Step 4: Validate combined data
        logger.info("Step 4: Validating combined data")
        self._validate_augmented_data(
            normalized_industry_df, normalized_factsheet_df,
            augmented_industry_df, augmented_factsheet_df,
            synthetic_data
        )
        
        # Step 5: Save augmented datasets
        logger.info("Step 5: Saving augmented datasets")
        self._save_augmented_datasets(augmented_industry_df, augmented_factsheet_df)
        
        logger.info("Augmented datasets created successfully")
        return augmented_industry_df, augmented_factsheet_df
    
    def _validate_augmented_data(self,
                               original_industry_df: pd.DataFrame,
                               original_factsheet_df: pd.DataFrame,
                               augmented_industry_df: pd.DataFrame,
                               augmented_factsheet_df: pd.DataFrame,
                               synthetic_data: Dict[str, List[Tuple[str, str]]]) -> None:
        """
        Validate the augmented datasets for consistency and correctness
        
        Args:
            original_industry_df: Original industry data
            original_factsheet_df: Original factsheet data
            augmented_industry_df: Augmented industry data
            augmented_factsheet_df: Augmented factsheet data
            synthetic_data: Synthetic data used for augmentation
        """
        logger.info("Validating augmented datasets")
        
        # Calculate expected counts
        expected_synthetic_factsheets = sum(len(tuples) for tuples in synthetic_data.values())
        
        # Validate industry data
        original_industry_count = len(original_industry_df)
        augmented_industry_count = len(augmented_industry_df)
        industry_increase = augmented_industry_count - original_industry_count
        
        logger.info(f"Industry data validation:")
        logger.info(f"  - Original entries: {original_industry_count}")
        logger.info(f"  - Augmented entries: {augmented_industry_count}")
        logger.info(f"  - Synthetic entries added: {industry_increase}")
        logger.info(f"  - Expected synthetic entries: {expected_synthetic_factsheets}")
        
        # Validate factsheet data
        original_factsheet_count = len(original_factsheet_df)
        augmented_factsheet_count = len(augmented_factsheet_df)
        factsheet_increase = augmented_factsheet_count - original_factsheet_count
        
        logger.info(f"Factsheet data validation:")
        logger.info(f"  - Original entries: {original_factsheet_count}")
        logger.info(f"  - Augmented entries: {augmented_factsheet_count}")
        logger.info(f"  - Synthetic entries added: {factsheet_increase}")
        logger.info(f"  - Expected synthetic entries: {expected_synthetic_factsheets}")
        
        # Check for consistency
        if industry_increase != expected_synthetic_factsheets:
            logger.warning(f"Industry data increase ({industry_increase}) doesn't match expected ({expected_synthetic_factsheets})")
        
        if factsheet_increase != expected_synthetic_factsheets:
            logger.warning(f"Factsheet data increase ({factsheet_increase}) doesn't match expected ({expected_synthetic_factsheets})")
        
        # Check for overlapping company IDs
        original_industry_ids = set(original_industry_df["company_id"].unique())
        synthetic_industry_ids = set()
        for tuples in synthetic_data.values():
            synthetic_industry_ids.update(company_id for company_id, _ in tuples)
        
        overlapping_ids = original_industry_ids.intersection(synthetic_industry_ids)
        if overlapping_ids:
            logger.warning(f"Found {len(overlapping_ids)} overlapping company IDs between original and synthetic data")
            logger.warning(f"First few overlapping IDs: {list(overlapping_ids)[:5]}")
        else:
            logger.info("No overlapping company IDs found - good!")
        
        # Validate data types
        logger.info("Data type validation:")
        logger.info(f"  - Industry company_id dtype: {augmented_industry_df['company_id'].dtype}")
        logger.info(f"  - Factsheet company_id dtype: {augmented_factsheet_df['company_id'].dtype}")
        
        # Check for missing values
        industry_nulls = augmented_industry_df.isnull().sum().sum()
        factsheet_nulls = augmented_factsheet_df.isnull().sum().sum()
        
        if industry_nulls > 0:
            logger.warning(f"Found {industry_nulls} null values in augmented industry data")
        if factsheet_nulls > 0:
            logger.warning(f"Found {factsheet_nulls} null values in augmented factsheet data")
        
        logger.info("Validation completed")
    
    def _save_augmented_datasets(self, 
                               augmented_industry_df: pd.DataFrame, 
                               augmented_factsheet_df: pd.DataFrame) -> None:
        """
        Save the augmented datasets to files
        
        Args:
            augmented_industry_df: Augmented industry data
            augmented_factsheet_df: Augmented factsheet data
        """
        # Get output filenames from config
        output_files = self.config.get("synthetic_generation", {}).get("output_files", {})
        industry_filename = output_files.get("augmented_industry_data", "augmented_industry_data.csv")
        factsheet_filename = output_files.get("augmented_factsheet_data", "augmented_factsheet_data.csv")
        
        # Save to synthetic data cache directory
        industry_path = self.synthetic_dir / industry_filename
        factsheet_path = self.synthetic_dir / factsheet_filename
        
        logger.info(f"Saving augmented industry data to: {industry_path}")
        augmented_industry_df.to_csv(industry_path, index=False)
        
        logger.info(f"Saving augmented factsheet data to: {factsheet_path}")
        augmented_factsheet_df.to_csv(factsheet_path, index=False)
        
        logger.info("Augmented datasets saved successfully")
    
    def load_augmented_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load previously saved augmented datasets
        
        Returns:
            Tuple of (augmented_industry_df, augmented_factsheet_df)
        """
        # Get output filenames from config
        output_files = self.config.get("synthetic_generation", {}).get("output_files", {})
        industry_filename = output_files.get("augmented_industry_data", "augmented_industry_data.csv")
        factsheet_filename = output_files.get("augmented_factsheet_data", "augmented_factsheet_data.csv")
        
        industry_path = self.synthetic_dir / industry_filename
        factsheet_path = self.synthetic_dir / factsheet_filename
        
        logger.info(f"Loading augmented industry data from: {industry_path}")
        augmented_industry_df = pd.read_csv(industry_path)
        
        logger.info(f"Loading augmented factsheet data from: {factsheet_path}")
        augmented_factsheet_df = pd.read_csv(factsheet_path)
        
        logger.info(f"Loaded augmented datasets:")
        logger.info(f"  - Industry data: {len(augmented_industry_df)} rows")
        logger.info(f"  - Factsheet data: {len(augmented_factsheet_df)} rows")
        
        return augmented_industry_df, augmented_factsheet_df 