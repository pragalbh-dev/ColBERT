# ColBERT Training Data Curation Pipeline

This pipeline transforms industry hierarchy data and company factsheets into training data for ColBERT in the format of (query, positive factsheets) and (query, negative factsheets).

## Overview

The pipeline consists of several steps:

1. **Chain Cleaning**: Process industry chains to remove unnecessary elements using LLM
2. **Negative Sample Generation**: Generate negative samples using semantic search and LLM judgment
3. **Triplet Generation**: Create training triplets in the format (query, factsheet, score)

## Requirements

- Python 3.8+
- OpenAI API key
- Elasticsearch (for negative sample generation)
- PyTorch (for reranker scoring)

## Installation

```bash
# Clone the repository
git clone <repository_url>

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY=your_api_key

# Set Elasticsearch password (if authentication is required)
export ELASTIC_PASSWORD=your_elastic_password
```

## Configuration

The pipeline can be configured using YAML files:

- `config/default.yaml`: Default configuration values
- `config/custom.yaml`: Custom overrides for specific runs

You can customize various aspects of the pipeline, including:

- Data paths
- OpenAI model settings
- Elasticsearch settings
- Sampling parameters
- Scoring strategy

Example custom configuration:

```yaml
paths:
  industry_data: "/path/to/industry_df.csv"
  factsheet_data: "/path/to/company_factsheets_df.csv"
  output_dir: "knowledge_data_curation/output/"

openai:
  parallel_threads: 4
  rate_limit: 40

elasticsearch:
  host: "localhost"
  port: 9200
  username: "elastic"  # Set custom username if needed
  # Note: Password is taken from ELASTIC_PASSWORD environment variable

scoring:
  strategy: "reranker"
```

## Usage

```bash
# Run with default configuration
python scripts/run_pipeline.py

# Run with custom configuration
python scripts/run_pipeline.py --config config/custom.yaml

# Enable debug logging
python scripts/run_pipeline.py --debug

# Resume from checkpoint
python scripts/run_pipeline.py --checkpoint output/checkpoint_20230101
```

## Elasticsearch Setup

The pipeline requires Elasticsearch for negative sample generation. To set up:

1. Install and start Elasticsearch
2. Set the `ELASTIC_PASSWORD` environment variable with your Elasticsearch password
3. Configure the host, port, and username in the configuration file if needed

If running Elasticsearch with security enabled (default in newer versions), make sure to:

```bash
# Set the password environment variable
export ELASTIC_PASSWORD=your_password

# Or set it temporarily for just one command
ELASTIC_PASSWORD=your_password python scripts/run_pipeline.py
```

## Data Format

### Input

- **Industry Data**: CSV file with columns `company_id`, `company_name`, `industry_chain_hierarchy`
- **Factsheet Data**: CSV file with columns `company_id`, `factsheet`

### Output

- **Cleaned Chains**: JSON mapping from original chains to cleaned chains
- **Negative Chains**: JSON mapping from chains to negative chains
- **ColBERT Training Data**: CSV with columns `query`, `factsheet`, `score`, `is_positive`

## Directory Structure

```
knowledge_data_curation/
├── config/
│   ├── default.yaml         # Default configuration values
│   └── custom.yaml          # Custom overrides
├── src/
│   ├── data_processors/     # Data processing components
│   ├── models/              # Model implementations
│   ├── prompts/             # LLM prompts
│   ├── utils/               # Utility functions
│   └── pipelines/           # Pipeline orchestration
├── output/
│   ├── cleaned_chains/      # Cleaned chain outputs
│   ├── negative_chains/     # Negative chain mappings
│   └── colbert_training/    # Final training data
├── tests/                   # Unit tests
├── scripts/                 # Execution scripts
└── logs/                    # Log files
```

## Extending the Pipeline

The pipeline is designed to be modular and extensible:

- **Custom Scoring**: Implement new scoring strategies by modifying the `TripletGenerator` class
- **Different Models**: Change the models used for embeddings or reranking in the configuration
- **Custom Processing**: Replace any component with your own implementation

## Logging

The pipeline includes comprehensive logging:

- Console logs with configurable level
- Module-specific log files
- Unified log file with all messages

Log levels and paths can be configured in the configuration files. 