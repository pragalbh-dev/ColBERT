import streamlit as st
import pandas as pd
import numpy as np
import concurrent.futures
import time
import plotly.express as px
import plotly.graph_objects as go
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from colbert.ranking.score_fusion import (
    dbsf, relative_score_fusion, or_operation_fusion, logical_query_fusion,
    ranx_fusion
)

# Set wide layout for the entire app
st.set_page_config(layout="wide", page_title="ColBERT MultiQuery Search Demo")

# Add custom CSS for better styling
st.markdown("""
<style>
    /* Improve spacing and layout */
    .main > div {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Make dataframes use full width */
    div[data-testid="stDataFrame"] > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Style headers */
    h1, h2, h3 {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Better table styling */
    .stDataFrame table {
        width: 100% !important;
    }
    
    /* Improve button alignment */
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("ColBERT MultiQuery Search Demo")
st.write("Build complex logical queries across multiple ColBERT indexes")

# Define the checkpoint paths and their corresponding indexes
# You should modify these paths according to your actual checkpoints and indexes
checkpoints = {
    "ColBERT v2 Vanilla": {
        "root": "./experiments/colbert_aspect_training/colbertv2-vanilla",
        "index": "colbert_v2.tuned=false.nbits=2.aspects=all.form=NA"
    },
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

# Score fusion settings in the sidebar
st.sidebar.header("Score Fusion Settings")
fusion_method = st.sidebar.selectbox(
    "Fusion method for AND operations:",
    ["dbsf", "multi_dbsf", "rsf", "ranx", "direct"],
    help="Method to combine scores: Distribution-based Score Fusion (dbsf/multi_dbsf), Relative Score Fusion (rsf), ranx fusion (ranx), or direct filtering (direct)"
)

# Initialize default values
std_dev = 3.0
rsf_method = "minmax"
ranx_method = "bordafuse"
ranx_norm = None

# Show appropriate parameters based on selected fusion method
if fusion_method in ["dbsf", "multi_dbsf"]:
    std_dev = st.sidebar.slider(
        "Standard Deviation (std_dev)", 
        min_value=1.0, 
        max_value=5.0, 
        value=3.0, 
        step=0.1,
        help="Number of standard deviations to define distribution tails"
    )
    
    if fusion_method == "dbsf":
        st.sidebar.info("DBSF combines normalized scores using weighted sum (alpha parameter)")
    else:
        st.sidebar.info("multi_dbsf uses maximum normalized score for each document")
        
elif fusion_method == "rsf":
    rsf_method = st.sidebar.selectbox(
        "RSF Normalization Method",
        ["minmax", "rank", "softmax"],
        help="Method for normalizing scores in Relative Score Fusion"
    )
elif fusion_method == "ranx":
    ranx_method = st.sidebar.selectbox(
        "Ranx Fusion Method",
        ["bordafuse", "combsum", "combmnz", "rrf", "logistic"],
        help="Fusion algorithm to use in ranx"
    )
    
    ranx_norm_options = [None, "min-max", "sum", "max", "std", "rank"]
    ranx_norm_index = 0  # Default to None
    
    ranx_norm = st.sidebar.selectbox(
        "Ranx Normalization Strategy",
        ranx_norm_options,
        index=ranx_norm_index,
        help="Normalization strategy to apply before fusion in ranx"
    )

# Alpha is only needed for dbsf (not multi_dbsf) and rsf
if fusion_method in ["dbsf", "rsf"]:
    alpha = st.sidebar.slider(
        "Alpha (weight)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Weight for the first query's scores (1-alpha for the second)"
    )
else:
    alpha = 0.5  # Default value that won't be used

# Advanced settings collapsible section
with st.sidebar.expander("Advanced Search Settings"):
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
    all_collection = pd.read_csv('./data/all_collection.tsv', sep='\t', header=None, index_col=None)
    all_collection.columns = ['Document ID', 'Document']
    return all_collection

all_collection = get_collections()

# Initialize session state for storing results data for visualization
if 'result_scores' not in st.session_state:
    st.session_state.result_scores = None

# Function to perform search in parallel
def search_query(searcher, query, k=1000):
    """Execute a search for a single query and return results as a dataframe"""
    try:
        start_time = time.time()
        pids, ranks, scores = searcher.search(query, k=k)
        end_time = time.time()
        
        # Ensure ranks matches the length of pids
        ranks = ranks[:len(pids)]
        
        # Create a results DataFrame
        results_data = {
            "doc_id": pids,
            "rank": ranks,
            "score": scores,
        }
        
        results_df = pd.DataFrame(results_data)
        return results_df, end_time - start_time
    except Exception as e:
        st.error(f"Error searching for query '{query}': {str(e)}")
        return pd.DataFrame(columns=["doc_id", "rank", "score"]), 0

# Function to display search results
def display_results(combined_results, total_time):
    """
    Display search results in a formatted way
    
    Args:
        combined_results: DataFrame with search results
        total_time: Time taken to search
    """
    if combined_results is not None and not combined_results.empty:
        # Display the results in a full-width container
        st.header("Search Results")
        st.write(f"Found {len(combined_results)} results in {total_time:.3f} seconds")
        
        # Add document content from collection if available
        results_with_content = combined_results.copy()
        
        # Ensure document ID column is correctly named
        if "doc_id" in results_with_content.columns:
            results_with_content.rename(columns={"doc_id": "Document ID"}, inplace=True)
        
        # Handle score columns - could be 'score' or 'fused_score'
        score_column = None
        if "score" in results_with_content.columns:
            score_column = "score"
            results_with_content.rename(columns={"score": "Score"}, inplace=True)
            # Save scores for visualization
            st.session_state.result_scores = combined_results["score"].values
        elif "fused_score" in results_with_content.columns:
            score_column = "fused_score"
            results_with_content.rename(columns={"fused_score": "Score"}, inplace=True)
            # Save scores for visualization
            st.session_state.result_scores = combined_results["fused_score"].values
        else:
            st.session_state.result_scores = None
        
        # If there are scores but no ranks, generate ranks based on score order
        if "rank" not in results_with_content.columns and "Score" in results_with_content.columns:
            results_with_content["rank"] = results_with_content["Score"].rank(ascending=False).astype(int)
        
        if "rank" in results_with_content.columns:
            results_with_content.rename(columns={"rank": "Rank"}, inplace=True)
        
        # Merge with collection data if available
        if 'all_collection' in globals():
            results_with_content = results_with_content.merge(all_collection, on="Document ID", how="left")
        
        # Configure column widths for better display
        column_config = {}
        
        # Auto-sizing column configuration
        if "Rank" in results_with_content.columns:
            column_config["Rank"] = st.column_config.NumberColumn(width="small")
        
        if "Document ID" in results_with_content.columns:
            column_config["Document ID"] = st.column_config.TextColumn(width="medium")
            
        if "Score" in results_with_content.columns:
            column_config["Score"] = st.column_config.NumberColumn(
                format="%.4f",
                width="medium"
            )
            
        if "Document" in results_with_content.columns:
            column_config["Document"] = st.column_config.TextColumn(width="large")
        
        # Create a full-width container and display the dataframe at maximum width
        st.markdown('<style>div[data-testid="stDataFrame"] > div { width: 100% !important; }</style>', unsafe_allow_html=True)
        
        # Display the dataframe with custom settings
        st.dataframe(
            results_with_content,
            use_container_width=True,
            height=600,  # Set a taller height to show more results
            column_config=column_config,
            hide_index=True  # Hide index for cleaner display
        )
    else:
        st.info("No results found for your query or an error occurred.")

# Function to execute searches in parallel and combine results
def execute_logical_query(checkpoint_config, query_components, k=1000):
    """
    Execute a logical query with multiple components
    
    Args:
        checkpoint_config: Dictionary with checkpoint information
        query_components: List of dicts with query text and operations
        k: Number of results to retrieve
        
    Returns:
        pd.DataFrame: Combined results after applying logical operations
    """
    # Get the searcher
    searcher = get_searcher(
        checkpoint_config["root"], 
        checkpoint_config["index"],
        nranks,
        nbits,
        ncells
    )
    
    if not searcher:
        return None, 0
    
    # Extract queries and operations
    queries = [component["query"] for component in query_components if component["query"].strip()]
    operations = [component["operation"] for component in query_components[1:] if component["query"].strip()]
    
    if not queries:
        st.warning(f"No valid queries found in components: {query_components}")
        return None, 0
    
    st.info(f"Processing queries: {queries} with operations: {operations}")
    
    # Execute searches in parallel
    result_dfs = []
    search_times = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(search_query, searcher, query, k): query 
            for query in queries
        }
        
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                df, search_time = future.result()
                result_dfs.append(df)
                search_times.append(search_time)
            except Exception as e:
                st.error(f"Error processing query '{query}': {str(e)}")
                return None, 0
    
    total_search_time = sum(search_times)
    
    # If only one query, return its results
    if len(result_dfs) == 1:
        return result_dfs[0], total_search_time
    
    # Apply logical operations to combine results
    start_time = time.time()
    
    # Prepare parameters for logical query fusion
    fusion_params = {
        "fusion_method": fusion_method,
        "std_dev": std_dev if fusion_method in ["dbsf", "multi_dbsf"] else 3.0,
        "alpha": alpha,
        "rsf_method": rsf_method if fusion_method == "rsf" else "minmax",
        "ranx_method": ranx_method if fusion_method == "ranx" else "bordafuse",
        "ranx_norm": ranx_norm if fusion_method == "ranx" else None
    }
    
    # Apply logical query fusion
    try:
        combined_df = logical_query_fusion(result_dfs, operations, **fusion_params)
        
        fusion_time = time.time() - start_time
        
        return combined_df, total_search_time + fusion_time
    except ValueError as e:
        # Handle specific validation errors from logical_query_fusion
        st.error(f"Error during logical fusion: {str(e)}")
        return None, total_search_time
    except Exception as e:
        # Handle other unexpected errors
        st.error(f"Unexpected error during fusion: {str(e)}")
        return None, total_search_time

# Initialize session state for queries if not already done
if 'query_components' not in st.session_state:
    st.session_state.query_components = [{"query": "", "operation": ""}]

# Function to add query component
def add_query_component():
    st.session_state.query_components.append({"query": "", "operation": "AND"})

# Function to remove query component
def remove_query_component(index):
    if len(st.session_state.query_components) > 1:
        st.session_state.query_components.pop(index)

# Function to update query text
def update_query_text(index):
    # Get the value directly from session state using the input's key
    st.session_state.query_components[index]["query"] = st.session_state[f"query_{index}"]

# Function to update operation
def update_operation(index):
    # Get the value directly from session state using the select box's key
    st.session_state.query_components[index]["operation"] = st.session_state[f"operation_{index}"]

# Main query builder UI
st.header("Query Builder")
st.write("Build your logical query by adding query components and specifying operations between them.")

# Add warning about logical operation restrictions
if fusion_method in ['ranx', 'direct']:
    st.warning(
        "⚠️ **Important**: When using 'ranx' or 'direct' AND fusion methods, you cannot use OR operations "
        "to combine their results with other AND operations. This is because their score distributions are "
        "not compatible for proper OR fusion. Consider using 'dbsf', 'multi_dbsf', or 'rsf' instead."
    )

# Create query builder interface
for i, component in enumerate(st.session_state.query_components):
    # Adjust column ratios for better use of space
    if i == 0:
        cols = st.columns([4, 1])  # First query gets more space and no operation
        cols[0].text_input(
            f"Query {i+1}", 
            value=component["query"],
            key=f"query_{i}",
            on_change=update_query_text,
            args=(i,)
        )
    else:
        # For subsequent queries, show operation selector and query input with remove button
        cols = st.columns([1, 3, 1])
        with cols[0]:
            operation = st.selectbox(
                f"Operation",
                ["AND", "OR", "NOT"],
                index=["AND", "OR", "NOT"].index(component["operation"]) if component["operation"] in ["AND", "OR", "NOT"] else 0,
                key=f"operation_{i}",
                on_change=update_operation,
                args=(i,)
            )
        
        # Query input
        cols[1].text_input(
            f"Query {i+1}", 
            value=component["query"],
            key=f"query_{i}",
            on_change=update_query_text,
            args=(i,)
        )
        
        # Remove button (except for the first component)
        cols[2].button("❌", key=f"remove_{i}", on_click=remove_query_component, args=(i,))

# Add button row with better spacing
button_cols = st.columns([1, 1, 3])
button_cols[0].button("Add Query", on_click=add_query_component)
if button_cols[1].button("Execute Search", type="primary"):
    # Filter out empty query components
    query_components = [comp for comp in st.session_state.query_components if comp["query"].strip()]
    if not query_components:
        st.warning("Please enter at least one query.")
    else:
        # Display info about what's being executed
        st.info(f"Executing search with {len(query_components)} queries: " + 
               ", ".join([f'"{comp["query"]}"' for comp in query_components]))
        
        # Execute the logical query
        with st.spinner("Searching..."):
            results, search_time = execute_logical_query(
                checkpoints[selected_checkpoint],
                query_components,
                k=k_results
            )
            
            # Display results
            display_results(results, search_time)

# Add information about logical operations
st.markdown("---")
with st.expander("About Logical Operations"):
    st.markdown("""
    ### Logical Query Operations
    
    This demo supports three logical operations between queries:
    
    - **AND**: Returns documents that match both queries, combining scores using the selected fusion method
    - **OR**: Returns documents that match either query, combining them using Reciprocal Rank Fusion (RRF)
    - **NOT**: Excludes documents that match the second query from the results of the first query
    
    ### Score Fusion Methods
    
    For AND operations, you can choose from these score fusion methods:
    
    - **Distribution-based Score Fusion (DBSF)**: 
       - **dbsf**: Normalizes scores based on their statistical distribution and combines with weighted sum
       - **multi_dbsf**: Normalizes scores based on their statistical distribution but takes the maximum score for each document
    - **Relative Score Fusion (RSF)**: Normalizes scores based on their relative values within each result set
    - **Ranx Fusion**: Uses the ranx library to apply various fusion methods like Borda count, CombSUM, etc.
    - **Direct**: Simple filtering to keep only documents present in both result sets (no score fusion)
    
    ### OR Operations and Reciprocal Rank Fusion
    
    For OR operations, we use Reciprocal Rank Fusion (RRF) which:
    - Converts scores to ranks in each result set
    - Calculates a combined score using the formula `1/(k + rank)` for each document
    - Sums these scores across all sources where the document appears
    - This approach handles results from different scoring systems much better than simple max-score fusion
    
    ### Restrictions on Complex Logical Expressions
    
    When combining multiple operations, there are some restrictions:
    
    - **Valid AND-OR Combinations**: When using OR to combine results of two AND operations, the AND operations must use one of these methods:
      - dbsf
      - multi_dbsf
      - rsf
    
    - **Invalid Combinations**: If either AND operation uses 'ranx' or 'direct' methods, an error will be raised
    
    This restriction exists because the score distributions from these methods may not be compatible for proper OR operations.
    
    #### DBSF Variants
    
    - **dbsf**: The original implementation that uses a weighted sum of normalized scores (controlled by alpha)
    - **multi_dbsf**: An alternative implementation that takes the maximum normalized score for each document
    
    #### Ranx Fusion Options
    
    When using Ranx fusion for AND operations, you can select from various algorithms:
    
    - **bordafuse**: Fusion using the Borda count method
    - **combsum**: Simple sum of scores
    - **combmnz**: Sum of scores multiplied by number of non-zero entries
    - **rrf**: Reciprocal Rank Fusion
    - **logistic**: Logistic regression fusion
    
    You can also apply different normalization strategies before fusion:
    
    - **min-max**: Normalize scores to [0,1] range
    - **sum**: Divide scores by their sum
    - **max**: Divide scores by their maximum value
    - **std**: Standardize scores (z-score)
    - **rank**: Convert scores to ranks
    
    **Note**: When using ranx fusion with multiple 'AND' operations, all dataframes are processed together in a single fusion operation for better results, rather than sequentially combining pairs.
    
    Adjust parameters in the sidebar to control how scores are combined.
    """)

# Add score distribution visualization
st.markdown("---")
with st.expander("Score Distribution Analysis"):
    if st.session_state.result_scores is not None and len(st.session_state.result_scores) > 0:
        st.subheader("Score Distribution in Search Results")
        
        # Create histogram of scores
        fig = px.histogram(
            x=st.session_state.result_scores,
            nbins=50,
            labels={"x": "Score Value"},
            title="Distribution of Relevance Scores",
            color_discrete_sequence=["#3366cc"],
        )
        
        # Add a vertical line for the mean score
        mean_score = np.mean(st.session_state.result_scores)
        median_score = np.median(st.session_state.result_scores)
        
        fig.add_vline(x=mean_score, line_dash="dash", line_color="red", 
                      annotation_text=f"Mean: {mean_score:.4f}", 
                      annotation_position="top right")
        
        fig.add_vline(x=median_score, line_dash="dash", line_color="green", 
                      annotation_text=f"Median: {median_score:.4f}", 
                      annotation_position="top left")
        
        # Add a box plot below the histogram for more detailed distribution insights
        fig2 = px.box(
            y=st.session_state.result_scores,
            labels={"y": "Score Value"},
            title="Score Distribution Box Plot",
            color_discrete_sequence=["#ff7f0e"],
        )
        
        # Calculate some statistics about the score distribution
        min_score = np.min(st.session_state.result_scores)
        max_score = np.max(st.session_state.result_scores)
        std_dev = np.std(st.session_state.result_scores)
        
        # Display the plots
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Display distribution statistics as metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Minimum Score", f"{min_score:.4f}")
        col2.metric("Maximum Score", f"{max_score:.4f}")
        col3.metric("Mean Score", f"{mean_score:.4f}")
        col4.metric("Standard Deviation", f"{std_dev:.4f}")
        
        # Insights about the distribution
        st.markdown("### Distribution Insights")
        st.markdown("""
        Understanding score distribution helps analyze search quality:
        
        - **Wide distribution** suggests clear distinction between relevant and non-relevant results
        - **Narrow, high-value distribution** indicates most results are considered highly relevant
        - **Bimodal distribution** may indicate two distinct groups of results (e.g., different topics)
        - **Skewed distribution** might suggest bias in the search algorithm or data
        
        The box plot shows outliers and quartiles, helping identify unusual patterns in the scores.
        """)
    else:
        st.info("Execute a search to view score distribution analysis.")

# Additional information in the sidebar
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.markdown(
    """
    This demo allows you to build complex logical queries across ColBERT indexes.
    
    You can combine multiple queries using logical operations (AND, OR, NOT) and
    control how scores are fused using different methods.
    
    **Configure the checkpoint paths in the code to match your actual model paths.**
    """
) 