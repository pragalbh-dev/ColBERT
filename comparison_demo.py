import streamlit as st
import pandas as pd
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import os
import time
import json

# Set wide layout for the entire app
st.set_page_config(layout="wide", page_title="ColBERT Search Demo")

# Title and description
st.title("ColBERT Search Demo")
st.write("Search across multiple ColBERT indexes with different model checkpoints")

# Define the checkpoint paths and their corresponding indexes
# You should modify these paths according to your actual checkpoints and indexes
checkpoints = {
    "ColBERT v2 Vanilla": {
        "root": "./experiments/colbert_aspect_training/colbertv2-vanilla",
        "index": "colbert_v2.tuned=false.nbits=2.aspects=all.form=NA"
    },
    # "ColBERT NLQ": {
    #     "root": "./experiments/colbert_aspect_training/none/run_1742739659/checkpoints/colbert-final",
    #     "index": "colbert_v2.tuned=true.nbits=2.aspects=all.form=nlq"
    # },
    "ColBERT Aspect Token": {
        "root": "./experiments/colbert_aspect_training/none/run_1742874896/checkpoints/colbert-75",
        "index": "colbert_v2.tuned=true.nbits=2.aspects=all.form=nlq.mask=False.customer_mod=True.iter=75.exp=run_1742874896"
    }
}

# Create sidebar for model selection and search settings
st.sidebar.header("Model Selection")
selected_checkpoint = st.sidebar.selectbox(
    "Select ColBERT checkpoint:",
    list(checkpoints.keys())
)

st.sidebar.header("Search Settings")
k_results = st.sidebar.slider("Number of results to display", min_value=1, max_value=1000, value=1000)
display_content = st.sidebar.checkbox("Display document content (if available)", value=True)

# Advanced settings collapsible section
with st.sidebar.expander("Advanced Settings"):
    nranks = st.number_input("Number of ranks", min_value=1, max_value=8, value=1)
    nbits = st.number_input("Number of bits", min_value=1, max_value=8, value=2)
    ncells = st.number_input("Number of cells", min_value=1, max_value=32, value=4)

# Caching mechanism for searchers to avoid reloading
@st.cache_resource
def get_searcher(root, index_name, nranks, nbits, ncells):
    """Create and cache a searcher instance for faster repeated searches"""
    try:
        with Run().context(RunConfig(nranks=nranks, experiment="colbert_aspect_training")):
            config = ColBERTConfig(
                root=root,
                nbits=nbits,
                ncells=ncells,
                kmeans_niters=32,
                ndocs=4096,
                attend_to_mask_tokens=False,
            )
            
            searcher = Searcher(index=index_name, config=config)
            return searcher
    except Exception as e:
        st.error(f"Failed to initialize searcher: {str(e)}")
        return None
@st.cache_data
def get_collections():
    all_collection=pd.read_csv('./data/all_collection.tsv',sep='\t',header=None,index_col=None)
    all_collection.columns=['Document ID','Document']
    return all_collection
all_collection=get_collections()
# Main search interface
st.header("Search Query")
query = st.text_input("Enter your search query:")

# Process the search when a query is entered
if query:
    # Get the selected checkpoint configuration
    checkpoint_config = checkpoints[selected_checkpoint]
    
    # Display a spinner while loading the searcher and performing the search
    with st.spinner(f"Searching with {selected_checkpoint}..."):
        # Get or create the searcher
        searcher = get_searcher(
            checkpoint_config["root"], 
            checkpoint_config["index"],
            nranks,
            nbits,
            ncells
        )
        
        if searcher:
            try:
                # Perform the search
                start_time = time.time()
                pids, ranks, scores = searcher.search(query, k=k_results)

                ranks=ranks[:len(pids)]
                # import pdb;pdb.set_trace()
                end_time = time.time()
                search_time = end_time - start_time
                
                # Display the results
                st.header("Search Results")
                st.write(f"Found {len(pids)} results in {search_time:.3f} seconds")
                
                if len(pids) > 0:
                    # Create a results DataFrame

                    
                    
                    # Create a results DataFrame
                    results_data = {
                        "Rank": ranks,
                        "Document ID": pids,
                        "Score": scores,
                    }
                    
                    # Add document content if available and requested
                    
                    
                    results_df = pd.DataFrame(results_data)
                    results_df=results_df.merge(all_collection,how='left')
                    # Display results as a table
                    st.dataframe(results_df, use_container_width=True)
                else:
                    st.info("No results found for your query.")
            
            except Exception as e:
                st.error(f"An error occurred during search: {str(e)}")
        else:
            st.error(f"Could not initialize searcher for {selected_checkpoint}. Please check your configuration.")

# Additional information
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.markdown(
    """
    This demo allows you to search across multiple ColBERT indexes 
    using different model checkpoints. Select a checkpoint from the dropdown 
    and enter your query to see the search results.
    
    **Configure the checkpoint paths in the code to match your actual model paths.**
    """
)

# Add a divider and comparison view
st.markdown("---")

# Helper function to calculate document overlap
def calculate_overlap(vanilla_pids, model_pids, cutoffs=[100, 300, 500, 1000]):
    """Calculate document overlap at different depths"""
    overlap_stats = {}
    for cutoff in cutoffs:
        # Ensure we don't exceed the available results
        actual_cutoff = min(cutoff, len(vanilla_pids), len(model_pids))
        
        if actual_cutoff == 0:
            overlap_stats[cutoff] = {
                'common_docs': 0,
                'total_docs': 0,
                'overlap_percent': 0
            }
            continue
            
        vanilla_set = set(vanilla_pids[:actual_cutoff])
        model_set = set(model_pids[:actual_cutoff])
        
        common_docs = vanilla_set.intersection(model_set)
        
        # Calculate percentage of overlap (number of common docs / number of docs at this cutoff)
        overlap_percent = (len(common_docs) / actual_cutoff) * 100
        
        overlap_stats[cutoff] = {
            'common_docs': len(common_docs),
            'total_docs': actual_cutoff,
            'overlap_percent': round(overlap_percent, 2)
        }
    
    return overlap_stats

# Function to highlight new documents not in vanilla model
def highlight_new_docs(df, vanilla_docs, cutoff=None):
    """
    Apply styling to highlight documents that aren't present in the vanilla model results
    
    Args:
        df: DataFrame to style
        vanilla_docs: Set of document IDs from vanilla model
        cutoff: Optional cutoff to only consider top K results
    
    Returns:
        Styled DataFrame
    """
    # If cutoff is specified, limit the comparison to top K results
    if cutoff and cutoff < len(df):
        df_subset = df.iloc[:cutoff].copy()
    else:
        df_subset = df.copy()
    
    # Create a boolean mask for rows with document IDs not in vanilla_docs
    new_docs_mask = ~df_subset['Document ID'].isin(vanilla_docs)
    
    # Create a style function that highlights new documents
    def highlight_new(row):
        if row.name in df_subset[new_docs_mask].index:
            return ['background-color: #a6e3ff'] * len(row)
        return [''] * len(row)
    
    # Apply the styling
    return df.style.apply(highlight_new, axis=1)

# Optional: Add a button to compare all models
if st.button("Compare All Models") and query:
    # First, get results from the vanilla model
    vanilla_name = "ColBERT v2 Vanilla"
    vanilla_config = checkpoints[vanilla_name]
    
    with st.spinner(f"Getting results from {vanilla_name} for comparison..."):
        vanilla_searcher = get_searcher(
            vanilla_config["root"],
            vanilla_config["index"],
            nranks,
            nbits,
            ncells
        )
        
        if vanilla_searcher:
            vanilla_pids, vanilla_ranks, vanilla_scores = vanilla_searcher.search(query, k=k_results)
            vanilla_ranks = vanilla_ranks[:len(vanilla_pids)]
            
            # Create the vanilla results dataframe
            vanilla_results_data = {
                "Rank": vanilla_ranks,
                "Document ID": vanilla_pids,
                "Score": vanilla_scores,
            }
            vanilla_df = pd.DataFrame(vanilla_results_data)
            vanilla_df = vanilla_df.merge(all_collection, how='left')
            
            # Create sets of vanilla document IDs at different cutoffs for comparison
            vanilla_docs = {}
            cutoffs = [100, 300, 500, 1000]
            for cutoff in cutoffs:
                if cutoff <= len(vanilla_pids):
                    vanilla_docs[cutoff] = set(vanilla_pids[:cutoff])
                else:
                    vanilla_docs[cutoff] = set(vanilla_pids)
        else:
            st.error(f"Could not initialize searcher for {vanilla_name}")
            vanilla_pids = []
            vanilla_df = pd.DataFrame()
            vanilla_docs = {100: set(), 300: set(), 500: set(), 1000: set()}

    # Create tabs for each model
    tabs = st.tabs(list(checkpoints.keys()))
    
    # Dictionary to store overlap results
    model_overlaps = {}
    
    for i, (checkpoint_name, checkpoint_config) in enumerate(checkpoints.items()):
        with tabs[i]:
            with st.spinner(f"Searching with {checkpoint_name}..."):
                try:
                    searcher = get_searcher(
                        checkpoint_config["root"], 
                        checkpoint_config["index"],
                        nranks,
                        nbits,
                        ncells
                    )
                    
                    if searcher:
                        # Perform the search
                        start_time = time.time()
                        pids, ranks, scores = searcher.search(query, k=k_results)
                        ranks=ranks[:len(pids)]
                        end_time = time.time()
                        search_time = end_time - start_time
                        
                        st.write(f"Found {len(pids)} results in {search_time:.3f} seconds")
                        
                        if len(pids) > 0:
                            # Create a results DataFrame
                            results_data = {
                                "Rank": ranks,
                                "Document ID": pids,
                                "Score": scores,
                            }
                            
                            results_df = pd.DataFrame(results_data)
                            results_df = results_df.merge(all_collection, how='left')
                            
                            # Display results with highlighting if this is not the vanilla model
                            if checkpoint_name == vanilla_name:
                                st.dataframe(results_df, use_container_width=True)
                            else:
                                # Add a color legend
                                st.markdown(
                                    """
                                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <div style="background-color: #a6e3ff; width: 20px; height: 20px; margin-right: 8px;"></div>
                                        <span>New documents (not in vanilla model)</span>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                                
                                # Get top 1000 documents (or all if less) for highlighting
                                max_display = min(1000, len(results_df))
                                top_docs_df = results_df.iloc[:max_display]
                                
                                # Apply highlighting based on the full vanilla set
                                styled_df = highlight_new_docs(top_docs_df, set(vanilla_pids))
                                st.dataframe(styled_df, use_container_width=True)
                            
                            # Calculate overlap with vanilla model (if this is not the vanilla model)
                            if checkpoint_name != vanilla_name and vanilla_pids:
                                overlap_stats = calculate_overlap(vanilla_pids, pids)
                                model_overlaps[checkpoint_name] = overlap_stats
                        else:
                            st.info("No results found for your query.")
                    else:
                        st.error(f"Could not initialize searcher for {checkpoint_name}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)  # Display full traceback for debugging
    
    # Display a consolidated overlap summary in JSON format
    if model_overlaps:
        st.header("Document Overlap Analysis")
        
        # Create a formatted, user-friendly JSON structure
        formatted_overlaps = {}
        for model_name, overlap_data in model_overlaps.items():
            formatted_overlaps[model_name] = {}
            for cutoff, stats in overlap_data.items():
                formatted_overlaps[model_name][f"Top {cutoff}"] = {
                    "Common Documents": stats['common_docs'],
                    "Total Documents": stats['total_docs'],
                    "Overlap Percentage": f"{stats['overlap_percent']}%"
                }
        
        # Display as formatted JSON with syntax highlighting
        st.json(formatted_overlaps)