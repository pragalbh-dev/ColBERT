# ColBERT Search Demo

This is a Streamlit demo for searching across multiple ColBERT indexes using different model checkpoints.

## Setup

1. Make sure you have installed all the requirements:
   ```bash
   pip install -r demo_requirements.txt
   pip install -e .  # Install the local ColBERT package
   ```

2. Configure your checkpoint paths:
   - Open `comparison_demo.py` or `comparison_demo_multiquey.py` and modify the `checkpoints` dictionary to match your actual ColBERT checkpoint paths and indexes.
   - Make sure the checkpoint paths and index names are correct.

## Running the Demo

Run the standard Streamlit app with:

```bash
streamlit run comparison_demo.py
```

Or run the MultiQuery demo with:

```bash
streamlit run comparison_demo_multiquey.py
```

## Features

### Standard Demo (comparison_demo.py)

- **Query Input**: Enter your search query in the text input box.
- **Model Selection**: Choose from different ColBERT checkpoints using the dropdown in the sidebar.
- **Search Settings**:
  - Control the number of results to display with the slider.
  - Toggle the display of document content if available.
- **Advanced Settings**:
  - Adjust the number of ranks, bits, and cells for the search.
- **Results Display**: 
  - View search results in a tabular format, including rank, document ID, and score.
  - See search time performance metrics.
- **Model Comparison**: 
  - Compare search results across all checkpoints using the "Compare All Models" button.
  - View results in separate tabs for easy comparison.

### MultiQuery Demo (comparison_demo_multiquey.py)

- **Query Builder**: Build complex logical queries with multiple components:
  - Add multiple query components with the "Add Query" button
  - Connect queries with logical operations (AND, OR, NOT)
  - Remove query components with the ❌ button
- **Logical Operations**:
  - **AND**: Returns documents that match both queries, using score fusion
  - **OR**: Returns documents that match either query, taking max scores
  - **NOT**: Excludes documents from the second query
- **Score Fusion Settings**:
  - Select fusion method (DBSF, RSF, ranx, or direct filtering)
  - Adjust parameters like standard deviation and alpha weight
  - Configure normalization method for RSF
  - Select ranx fusion algorithm and normalization strategy when using ranx
- **Parallel Execution**: Executes individual queries in parallel for faster performance

## Customization

To add your own checkpoint paths and indexes, modify the `checkpoints` dictionary in the demos:

```python
checkpoints = {
    "Your Model Name": {
        "root": "./path/to/your/model/checkpoint",
        "index": "your_index_name.nbits=2"
    },
    # Add more models here
}
```

## Troubleshooting

- If you encounter errors related to missing checkpoints or indexes, verify that the paths in the `checkpoints` dictionary are correct.
- Ensure that your ColBERT installation is properly set up and that the indexes have been created.
- Check that all dependencies are installed correctly.
- If document content is not displayed, ensure that the searcher's `get_text` method is implemented or available. 