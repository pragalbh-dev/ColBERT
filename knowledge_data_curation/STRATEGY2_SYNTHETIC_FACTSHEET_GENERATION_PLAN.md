# Strategy 2: Synthetic Factsheet Generation for Data Balancing

## Overview

This strategy addresses the data imbalance issue where some industry chains have very few positive companies mapped to them. We'll manufacture realistic company factsheets for under-represented industry chains to create a more balanced training dataset.

## Problem Statement

- Some industry chains have less than a threshold number of positive companies
- This causes imbalance in the final training data
- Need to generate synthetic but realistic factsheets for these chains
- Must maintain data verifiability and type consistency

## Solution Architecture

### 1. Data Analysis & Threshold Detection
Identify industry chains with insufficient positive examples and determine which need synthetic data generation.

### 2. Synthetic Factsheet Generation
Create a  factsheet generation module to create realistic company factsheets for under-represented chains.

### 3. Data Integration & Type Normalization
Integrate synthetic data with original datasets while maintaining consistency and verifiability.

### 4. Pipeline Integration
Seamlessly integrate synthetic data into the existing triplet generation pipeline.

## Implementation Plan

### Phase 1: Data Imbalance Analyzer

**File**: `src/data_processors/imbalance_analyzer.py`

```python
class DataImbalanceAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.min_threshold = config.get("synthetic_generation", {}).get("min_companies_threshold", 5)
    
    def analyze_chain_distribution(self, company_chains: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Analyze distribution of companies across industry chains
        Returns statistics and chains needing synthetic data
        """
        
    def identify_underrepresented_chains(self, cleaned_chains: Dict[str, str], company_chains: Dict[str, List[str]]) -> List[str]:
        """
        Identify chains with less than threshold companies
        Returns list of cleaned chains needing synthetic data
        """
        
    def get_sample_factsheets(self, chain: str, company_chains: Dict[str, List[str]], 
                            company_factsheets: Dict[str, str], max_samples: int = 2) -> List[str]:
        """
        Get 1-2 sample factsheets for a given chain to use as examples
        """
```

### Phase 2: Synthetic Factsheet Generator Integration

**File**: `src/data_processors/synthetic_factsheet_generator.py`

```python
class SyntheticFactsheetGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.existing_module = self._initialize_existing_module()
        self.target_count = config.get("synthetic_generation", {}).get("target_factsheets_per_chain", 10)
    
    def generate_factsheets_for_chain(self, industry_chain: str, sample_factsheets: List[str], 
                                    target_count: int) -> List[str]:
        """
        Generate synthetic factsheets for a given industry chain
        Uses the existing module that accepts: chain, 1-2 samples, target count
        """
        
    def generate_company_id(self, factsheet: str) -> str:
        """
        Generate deterministic company ID as hash of factsheet content
        For verifiability and consistency
        """
        
    def batch_generate_factsheets(self, chains_needing_data: List[str], 
                                chain_samples: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
        """
        Generate factsheets for multiple chains in batch
        Returns: {chain: [(company_id, factsheet), ...]}
        """
```

### Phase 3: Data Type Normalizer

**File**: `src/data_processors/data_normalizer.py`

```python
class DataNormalizer:
    def __init__(self, config: Dict):
        self.config = config
    
    def normalize_company_ids(self, df: pd.DataFrame, id_column: str = "company_id") -> pd.DataFrame:
        """
        Convert all company IDs to strings for type consistency
        """
        
    def create_augmented_datasets(self, original_industry_df: pd.DataFrame, 
                                original_factsheet_df: pd.DataFrame,
                                synthetic_data: Dict[str, List[Tuple[str, str]]],
                                cleaned_chains: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create new datasets with synthetic data appended
        Returns: (augmented_industry_df, augmented_factsheet_df)
        """
```

### Phase 4: Caching & Reusability Manager

**File**: `src/utils/cache_manager.py`

```python
class SyntheticDataCacheManager:
    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = Path(config["paths"]["output_dir"]) / "synthetic_data_cache"
        self.cache_enabled = config.get("synthetic_generation", {}).get("enable_cache", True)
        self.reuse_existing = config.get("synthetic_generation", {}).get("reuse_existing", True)
    
    def get_cache_paths(self) -> Dict[str, Path]:
        """Return paths for cached synthetic data files"""
        
    def is_cached_data_available(self) -> bool:
        """Check if cached synthetic data exists and is valid"""
        
    def load_cached_data(self) -> Dict[str, Any]:
        """Load existing cached synthetic data"""
        
    def save_cached_data(self, synthetic_data: Dict[str, Any]) -> None:
        """Save synthetic data to cache"""
```

### Phase 5: Pipeline Integration

**File**: `src/pipelines/main_pipeline.py` (Enhanced)

```python
class ColBERTTrainingPipeline:
    def __init__(self, config_path: str):
        # ... existing initialization ...
        self.imbalance_analyzer = DataImbalanceAnalyzer(self.config)
        self.synthetic_generator = SyntheticFactsheetGenerator(self.config)
        self.data_normalizer = DataNormalizer(self.config)
        self.cache_manager = SyntheticDataCacheManager(self.config)
    
    def run(self) -> None:
        """Enhanced pipeline with synthetic data generation"""
        
        # Steps 1-2: Existing data loading and chain cleaning
        # ... existing code ...
        
        # NEW Step 2.5: Synthetic Data Generation (if enabled)
        if self.config.get("synthetic_generation", {}).get("enabled", False):
            logger.info("Step 2.5: Generating synthetic factsheets for data balancing")
            
            # Check cache first
            if self.cache_manager.reuse_existing and self.cache_manager.is_cached_data_available():
                logger.info("Using cached synthetic data")
                cached_data = self.cache_manager.load_cached_data()
                augmented_industry_df = cached_data["industry_df"]
                augmented_factsheet_df = cached_data["factsheet_df"]
            else:
                # Generate new synthetic data
                augmented_industry_df, augmented_factsheet_df = self._generate_synthetic_data_step(
                    self.data_loader.load_industry_data(),
                    self.data_loader.load_factsheet_data(),
                    cleaned_chains
                )
                
                # Cache the results
                if self.cache_manager.cache_enabled:
                    self.cache_manager.save_cached_data({
                        "industry_df": augmented_industry_df,
                        "factsheet_df": augmented_factsheet_df
                    })
            
            # Update data loader to use augmented data
            self.data_loader.override_data(augmented_industry_df, augmented_factsheet_df)
        
        # Continue with existing pipeline steps...
        # ... rest of existing pipeline ...
    
    def _generate_synthetic_data_step(self, industry_df: pd.DataFrame, 
                                    factsheet_df: pd.DataFrame,
                                    cleaned_chains: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic factsheets for underrepresented chains"""
```

## Configuration Schema

Add to `default.yaml`:

```yaml
synthetic_generation:
  enabled: true
  min_companies_threshold: 5        # Minimum companies per chain
  target_factsheets_per_chain: 10   # Target number of factsheets to generate
  max_sample_factsheets: 2          # Max sample factsheets to use as examples
  enable_cache: true                # Enable caching of generated data
  reuse_existing: true              # Reuse existing cached data if available
  cache_filename: "synthetic_factsheets.json"
  
  # Output file configurations
  output_files:
    augmented_industry_data: "augmented_industry_data.csv"
    augmented_factsheet_data: "augmented_factsheet_data.csv"
    synthetic_generation_stats: "synthetic_generation_stats.json"
```

## Enhanced Directory Structure

```
output/
├── cleaned_chains/
│   └── cleaned_chains.json
├── synthetic_data_cache/                    # NEW
│   ├── synthetic_factsheets.json          # Cached synthetic data
│   ├── augmented_industry_data.csv        # Industry data with synthetic entries
│   ├── augmented_factsheet_data.csv       # Factsheet data with synthetic entries
│   └── synthetic_generation_stats.json    # Generation statistics
├── subchains/
├── negative_chains/
└── colbert_training/
```

## Integration Touch Points

### 1. DataLoader Enhancement
- Add method `override_data()` to use augmented datasets
- Modify `get_company_chains()` and `get_company_factsheets()` to work with string IDs

### 2. Main Pipeline Integration
- Add synthetic data generation step between chain cleaning and subchain generation
- Ensure all downstream components work with augmented data

### 3. Statistics & Monitoring
- Track synthetic data generation metrics
- Log token usage for synthetic factsheet generation
- Report data balance improvements

### 4. Type Safety & Consistency
- Ensure all company IDs are consistently string type
- Validate synthetic company ID uniqueness
- Maintain referential integrity between datasets

## Implementation Sequence

1. **Phase 1**: Implement `DataImbalanceAnalyzer` with threshold detection
2. **Phase 2**: Integrate existing synthetic factsheet generation module
3. **Phase 3**: Implement `DataNormalizer` for type consistency
4. **Phase 4**: Add caching mechanism with `SyntheticDataCacheManager`
5. **Phase 5**: Integrate into main pipeline with configuration support
6. **Phase 6**: Update `DataLoader` for augmented data support
7. **Phase 7**: Add comprehensive testing and validation

## Benefits

- **Balanced Training Data**: Ensures all industry chains have sufficient positive examples
- **Configurable Thresholds**: Flexible control over when synthetic data is generated
- **Caching & Reusability**: Avoid regenerating synthetic data unnecessarily
- **Type Consistency**: Uniform string-based company IDs throughout pipeline
- **Verifiable Synthetic Data**: Hash-based company IDs for traceability
- **Seamless Integration**: Minimal changes to existing pipeline components

## Testing Strategy

- Unit tests for each new component
- Integration tests for synthetic data generation pipeline
- Validation of synthetic factsheet quality and realism
- Performance testing with large datasets
- Cache invalidation and reuse testing 