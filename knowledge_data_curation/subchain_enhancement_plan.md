# Subchain Enhancement Plan for ColBERT Training Data Curation

## Overview

This plan outlines the implementation of a configurable subchain generation feature that will:
1. Break industry chains into configurable subchains using sliding window approach
2. Generate negative samples for individual categories and subchains 
3. Significantly increase training data volume while making it simpler for the model to learn
4. Maintain backward compatibility with existing full-chain processing

## Current Understanding

The existing pipeline:
- Takes industry chains like `cat1>cat2>cat3>cat4>...>catN`
- Cleans these chains using LLM
- Generates hard/soft negatives using ES similarity search + LLM selection
- Maps chains to positive factsheets based on company associations
- Creates triplets for ColBERT training

## Proposed Enhancement

### Configuration Structure

Add new configuration section in `default.yaml`:

```yaml
subchain_generation:
  enabled: true
  configurations:
    - window_size: 1     # Individual categories  
      shift: 1
      enabled: true
    - window_size: 3     # 3-category subchains
      shift: 2           # Step by 2
      enabled: true
    - window_size: 5     # 5-category subchains  
      shift: 3           # Step by 3
      enabled: false
  deduplicate_subchains: true
```

### Implementation Plan

## 1. Create Subchain Generator Component

**File**: `src/data_processors/subchain_generator.py`

```python
class SubchainGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.subchain_configs = config.get("subchain_generation", {}).get("configurations", [])
        self.deduplicate = config.get("subchain_generation", {}).get("deduplicate_subchains", True)
        
    def generate_subchains(self, chain: str) -> Dict[str, List[str]]:
        """Generate subchains for a given chain using all enabled configurations"""
        
    def _extract_subchains_with_config(self, chain: str, window_size: int, shift: int) -> List[str]:
        """Extract subchains using sliding window approach"""
        
    def generate_all_subchains(self, chains: List[str]) -> Dict[str, Any]:
        """Generate subchains for all chains and create mappings"""
```

**Key Methods:**
- `generate_subchains()`: Apply sliding window with different configs
- `create_subchain_mappings()`: Map original chains to subchains  
- `deduplicate_subchains()`: Remove duplicate subchains across all chains
- `create_reverse_mappings()`: Map subchains back to original chains

## 2. Chain Cleaner (No Changes Required)

**File**: `src/data_processors/chain_cleaner.py`

**Changes:** None - existing functionality sufficient

**Rationale:** 
- Subchains will be generated from already cleaned chains
- No need for separate subchain cleaning step
- Saves API costs and processing time

## 3. Enhance Negative Generator  

**File**: `src/data_processors/negative_generator.py`

**Changes:**
- Modify `initialize()` to handle subchains alongside full chains
- Add `generate_subchain_negatives()` method
- Update ES indexing to include subchains
- Modify candidate selection to work with subchains

```python
def initialize_with_subchains(self, company_chains: Dict[str, List[str]], subchain_data: Dict[str, Any]):
    """Initialize with both chains and subchains"""
    
def generate_subchain_negatives(self, subchains: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Generate negatives specifically for subchains"""
```

## 4. Update Main Pipeline

**File**: `src/pipelines/main_pipeline.py`

**Changes:**
- Add subchain generation step after chain cleaning
- Modify positive/negative generation to handle subchains
- Update result saving to include subchain data

```python
def _generate_subchains_step(self, cleaned_chains: Dict[str, str]) -> Dict[str, Any]:
    """Generate subchains from already cleaned chains"""
    
def _generate_subchain_factsheet_positives(self, subchain_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate query-factsheet positives for subchains"""
    
def _generate_subchain_factsheet_negatives(self, subchain_data: Dict[str, Any], subchain_negatives: Dict) -> Dict[str, List[str]]:
    """Generate query-factsheet negatives for subchains"""
```

## 5. Enhanced Pipeline Flow

```
1. Load Data → Extract Chains
2. Clean Full Chains (existing)
3. NEW: Generate Subchains from Cleaned Chains
4. Initialize Negative Generator (cleaned chains + subchains)
5. Generate Negatives (cleaned chains + subchains)
6. Generate Positives (cleaned chains + subchains)  
7. Generate Triplets (combined dataset)
8. Save Results (separate files for chains vs subchains)
```

## 6. Output Structure

```
output/
├── cleaned_chains/
│   └── cleaned_chains.json           # Original cleaned full chains
├── subchains/
│   ├── subchain_mappings.json        # Cleaned chain → subchains mapping
│   ├── subchain_configs.json         # Config used for generation
│   └── deduplicated_subchains.json   # Unique subchains (already clean)
├── negative_chains/
│   ├── negative_chains.json          # Negatives for cleaned full chains  
│   └── negative_subchains.json       # Negatives for subchains
└── colbert_training/
    ├── full_chains_training.csv       # Training data from cleaned full chains
    ├── subchains_training.csv         # Training data from subchains  
    └── combined_training.csv          # Combined dataset
```

## 7. Backward Compatibility

- All existing functionality remains unchanged
- Subchain generation is controlled by `subchain_generation.enabled` flag
- When disabled, pipeline works exactly as before
- Existing configuration files will work without modification

## 8. Implementation Order

1. **Phase 1**: Create `SubchainGenerator` class with basic functionality
2. **Phase 2**: Modify `NegativeGenerator` for subchain support 
3. **Phase 3**: Update `MainPipeline` orchestration 
4. **Phase 4**: Add configuration and testing  
5. **Phase 5**: Documentation and validation

## 9. Key Design Decisions

### Minimal Changes Approach
- New components are separate classes, not modifications to existing ones
- Existing methods remain unchanged, new methods added alongside
- Configuration-driven feature enablement

### Decoupled Architecture  
- `SubchainGenerator` is independent and reusable
- Clear separation between full-chain and subchain processing
- Modular components that can be tested independently

### Flexible Configuration
- Multiple window/shift configurations supported simultaneously  
- Easy to disable/enable specific configurations
- Runtime configuration without code changes

## 10. Testing Strategy

- Unit tests for `SubchainGenerator` with various window/shift combinations
- Integration tests for pipeline with subchains enabled/disabled
- Validation of data volume increase and quality
- Performance benchmarking with large datasets

## 11. Expected Benefits

- **Data Volume**: 10-50x increase in training examples depending on configurations
- **Model Learning**: Simpler patterns for the model to learn (shorter sequences)
- **Coverage**: Better representation of industry categories at different granularities  
- **Flexibility**: Configurable generation based on specific use case needs

This enhancement maintains the existing architecture while adding powerful new capabilities for generating more comprehensive training data. 