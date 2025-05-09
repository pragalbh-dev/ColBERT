import gradio as gr
import pandas as pd
import numpy as np
import torch
import time
import plotly.express as px
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from colbert.ranking.score_fusion_v3 import logical_query_fusion
import io
import csv
import tempfile
# Initialize the searcher once when the app starts
def initialize_searcher(root, index_name, nranks, nbits, ncells):
    """Initialize searcher once at startup"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Keep the Run context alive for the application's lifetime
        run_context = Run().context(RunConfig(nranks=nranks, experiment="colbert_aspect_training"))
        # Enter the context
        run_context.__enter__()
        
        config = ColBERTConfig(
            root=root,
            nbits=nbits,
            ncells=ncells,
            kmeans_niters=32,
            ndocs=100000,
            attend_to_mask_tokens=False
        )
        
        searcher = Searcher(index=index_name, config=config)
        return searcher, run_context
    except Exception as e:
        print(f"Failed to initialize searcher: {str(e)}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return None, None

# Function to search with a single query
def search_query(searcher, query, k):
    try:
        pids, ranks, scores = searcher.search(query, k=k)
        ranks = ranks[:len(pids)]
        scores = scores[:len(pids)]
        
        # Create results dataframe
        results_df = pd.DataFrame({
            "doc_id": pids,
            "rank": ranks,
            "score": scores,
        })
        
        return results_df
    except Exception as e:
        print(f"Error searching for query '{query}': {str(e)}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return pd.DataFrame(columns=["doc_id", "rank", "score"])

# Define logical query function with simplified interface
def logical_query_search(query1, operation1, query2, operation2, query3, model_selection, top_k, top_ke, fusion_method, alpha, std_dev):
    """Execute logical query with multiple components"""
    # Collect non-empty queries and operations
    all_queries = [query1, query2, query3]
    all_operations = [operation1, operation2]
    
    # Filter out empty queries
    query_components = []
    operations = []
    
    for i, query in enumerate(all_queries):
        if query and query.strip():
            query_components.append(query)
            if i > 0 and i-1 < len(all_operations):
                operations.append(all_operations[i-1])
    
    if not query_components:
        return None, "No valid queries provided", None
    
    # Get the correct searcher
    searcher = searchers.get(model_selection)
    if not searcher:
        return None, f"Searcher not initialized for {model_selection}", None
    
    try:
        # Execute searches for each query
        result_dfs = []
        start_time = time.time()
        
        for query in query_components:
            df = search_query(searcher, query, top_ke)
            if not df.empty:
                result_dfs.append(df)
        
        # If only one query, return its results
        if len(result_dfs) == 1:
            result_df = result_dfs[0]
            if len(result_df) > top_k:
                result_df = result_df.head(top_k)
                
            # Format for display
            result_df = result_df.rename(columns={"doc_id": "Document ID", "rank": "Rank", "score": "Score"})
            
            # Add document content if available
            if all_collection is not None:
                result_df = result_df.merge(all_collection, on="Document ID", how="left")
                
            end_time = time.time()
            message = f"Found {len(result_df)} results in {end_time - start_time:.3f} seconds"
            
            # Create score distribution plot
            scores_array = result_df["Score"].values if not result_df.empty else None
            plot_figure = create_score_distribution(scores_array) if scores_array is not None else None
            
            return result_df, message, plot_figure
        
        # Apply logical operations if we have multiple queries
        fusion_params = {
            "fusion_method": fusion_method,
            "std_dev": std_dev,
            "alpha": alpha,
            "rsf_method": "minmax",
            "ranx_method": "bordafuse",
            "ranx_norm": None
        }
        
        combined_df = logical_query_fusion(result_dfs, operations, **fusion_params)
        
        # Limit to top_k for display
        if len(combined_df) > top_k:
            combined_df = combined_df.head(top_k)
            
        # Format for display
        if "doc_id" in combined_df.columns:
            combined_df = combined_df.rename(columns={"doc_id": "Document ID"})
        if "rank" in combined_df.columns:
            combined_df = combined_df.rename(columns={"rank": "Rank"})
        if "score" in combined_df.columns:
            combined_df = combined_df.rename(columns={"score": "Score"})
        elif "fused_score" in combined_df.columns:
            combined_df = combined_df.rename(columns={"fused_score": "Score"})
        
        # Add document content
        if all_collection is not None:
            combined_df = combined_df.merge(all_collection, on="Document ID", how="left")
            
        end_time = time.time()
        message = f"Found {len(combined_df)} results in {end_time - start_time:.3f} seconds"
        
        # Create score distribution plot
        scores_array = None
        plot_figure = None
        
        if not combined_df.empty and "Score" in combined_df.columns:
            scores_array = combined_df["Score"].values
            # Create the plot using the function
            plot_figure = create_score_distribution(scores_array)
        
        return combined_df, message, plot_figure
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, f"Error during search: {str(e)}\n{error_detail}", None

# Create score distribution plot
def create_score_distribution(scores):
    if scores is None or len(scores) == 0:
        return None
    
    # Create histogram
    fig = px.histogram(
        x=scores,
        nbins=50,
        labels={"x": "Score Value"},
        title="Distribution of Relevance Scores"
    )
    
    # Add mean and median lines
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    
    fig.add_vline(x=mean_score, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_score:.4f}",
                  annotation_position="top right")
    
    fig.add_vline(x=median_score, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_score:.4f}",
                  annotation_position="top left")
    
    return fig

def dataframe_to_csv(df):
    """Convert dataframe to CSV and return as a downloadable file path"""
    if df is None or df.empty:
        return None
    
    try:
        # Use a temporary file to save the CSV
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(temp_file.name, index=False)
        
        # Return the file path for DownloadButton
        return temp_file.name
    except Exception as e:
        print(f"Error creating CSV: {str(e)}")
        return None

# Load the collection data
all_collection = None
try:
    all_collection = pd.read_csv('./data/all_collection.tsv', sep='\t', header=None, index_col=None)
    all_collection.columns = ['Document ID', 'Document']
except Exception as e:
    print(f"Warning: Could not load collection: {str(e)}")

# Define available checkpoints
checkpoints = {
    "ColBERT Aspect Token": {
        "root": "./experiments/colbert_aspect_training/none/run_1742874896/checkpoints/colbert-75",
        "index": "colbert_v2.tuned=true.nbits=2.aspects=all.form=nlq.mask=False.customer_mod=True.iter=75.exp=run_1742874896"
    },
    # "ColBERT v2 Vanilla": {
    #     "root": "./experiments/colbert_aspect_training/colbertv2-vanilla",
    #     "index": "colbert_v2.tuned=false.nbits=2.aspects=all.form=NA"
    # }
}

# Initialize searchers once at startup
searchers = {}
contexts = {}

print("Initializing searchers...")
for name, config in checkpoints.items():
    print(f"Initializing {name}...")
    searcher, context = initialize_searcher(
        config["root"], 
        config["index"],
        nranks=1,
        nbits=2,
        ncells=4
    )
    if searcher:
        searchers[name] = searcher
        contexts[name] = context
        print(f"Successfully initialized searcher for {name}")
    else:
        print(f"Failed to initialize searcher for {name}")

# Create Gradio UI with focus on multi-query and score distribution
with gr.Blocks(title="ColBERT MultiQuery Search Demo") as demo:
    # Store state for selected document
    selected_doc = gr.State(value=None)
    
    gr.Markdown("# ColBERT MultiQuery Search Demo")
    gr.Markdown("""
    Build complex logical queries across multiple ColBERT indexes.
    
    * Combine multiple queries with logical operations (AND, OR, NOT)
    * Choose from different score fusion methods
    * Analyze score distribution
    """)
    
    # Create tabs for search and document view
    with gr.Tabs() as tabs:
        with gr.Tab("Search") as search_tab:
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Group():
                        query1 = gr.Textbox(label="Query 1")
                        with gr.Row():
                            operation1 = gr.Dropdown(choices=["AND", "OR", "NOT"], value="AND", label="Operation")
                            query2 = gr.Textbox(label="Query 2")
                        with gr.Row():
                            operation2 = gr.Dropdown(choices=["AND", "OR", "NOT"], value="AND", label="Operation")
                            query3 = gr.Textbox(label="Query 3")
                    
                    with gr.Row():
                        search_button = gr.Button("Execute Search", variant="primary", scale=3)
                
                with gr.Column(scale=2):
                    with gr.Group():
                        model_select = gr.Dropdown(
                            choices=list(checkpoints.keys()),
                            value=list(checkpoints.keys())[0],
                            label="Select Model"
                        )
                        top_k = gr.Slider(minimum=10, maximum=1000, value=100, step=10, label="Results to display (top_k)")
                        top_ke = gr.Slider(minimum=1000, maximum=30000, value=10000, step=1000, label="Results per query (top_ke)")
                        
                    with gr.Group():
                        fusion_method = gr.Dropdown(
                            choices=["dbsf", "multi_dbsf", "rsf", "ranx", "direct"],
                            value="dbsf",
                            label="Fusion Method for AND operations"
                        )
                        alpha = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label="Alpha (weight for first query)")
                        std_dev = gr.Slider(minimum=1, maximum=5, value=3, step=0.1, label="Standard Deviation for DBSF")
            
            # Status message
            results_message = gr.Textbox(label="Status")
            
            # Results and visualizations
            with gr.Row():
                with gr.Column(scale=3):
                    # Make the dataframe selectable with row selection
                    results_df = gr.Dataframe(label="Search Results", interactive=True)
                with gr.Column(scale=2):
                    score_plot = gr.Plot(label="Score Distribution")
            
            # Add click instructions
            gr.Markdown("*Click on a row to view the document in the Document Viewer tab*")
            
            # Add information about operations
            with gr.Accordion("About Logical Operations", open=False):
                gr.Markdown("""
                ### Logical Query Operations
                
                This demo supports three logical operations between queries:
                
                - **AND**: Returns documents that match both queries, combining scores using the selected fusion method
                - **OR**: Returns documents that match either query, combining them using Reciprocal Rank Fusion
                - **NOT**: Excludes documents that match the second query from the results of the first query
                
                ### Score Fusion Methods
                
                For AND operations, you can choose from these methods:
                
                - **Distribution-based Score Fusion (DBSF)**: 
                   - **dbsf**: Normalizes scores based on their statistical distribution and combines with weighted sum
                   - **multi_dbsf**: Uses the maximum normalized score for each document
                - **Relative Score Fusion (RSF)**: Normalizes scores based on their relative values
                - **Ranx Fusion**: Uses various fusion methods like Borda count, CombSUM, etc.
                - **Direct**: Simple filtering to keep only documents present in both result sets
                """)
            
            # Button to trigger CSV download
            download_button = gr.Button("Download Results as CSV")

            # File output for downloading the CSV
            file_output = gr.File(label="Download Search Results")

            # Define the interaction for downloading the CSV
            download_button.click(
                fn=lambda df: dataframe_to_csv(df),
                inputs=[results_df],
                outputs=file_output
            )
        
        # Add the Document Viewer tab
        with gr.Tab("Document Viewer") as doc_tab:
            doc_id_display = gr.Textbox(label="Document ID", interactive=False)
            doc_score_display = gr.Number(label="Score", interactive=False, precision=4)
            doc_content = gr.Textbox(label="Document Content", interactive=False, lines=15)
    
    # Connect the search function
    search_button.click(
        logical_query_search,
        inputs=[
            query1, operation1, query2, operation2, query3,
            model_select, top_k, top_ke, fusion_method, alpha, std_dev
        ],
        outputs=[results_df, results_message, score_plot]
    )
    
    # Define a function to handle row selection
    def on_select_row(evt: gr.SelectData, results):
        if results is None or len(results) == 0:
            return None, None, "No document selected"
        
        # Get the selected row index
        row_index = evt.index[0] if isinstance(evt.index, list) else evt.index
        
        # Extract data from the selected row
        selected_row = results.iloc[row_index]
        
        doc_id = selected_row.get("Document ID", "N/A")
        score = selected_row.get("Score", 0.0)
        content = selected_row.get("Document", "No content available")
        
        # Return the values to update the document viewer
        return doc_id, score, content
    
    # Connect the row selection event to update the document viewer
    results_df.select(
        on_select_row,
        inputs=[results_df],
        outputs=[doc_id_display, doc_score_display, doc_content]
    )

# Cleanup function for when application exits
def cleanup():
    for name, context in contexts.items():
        try:
            if context:
                context.__exit__(None, None, None)
                print(f"Cleaned up context for {name}")
        except Exception as e:
            print(f"Error cleaning up context for {name}: {str(e)}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Register cleanup function
import atexit
atexit.register(cleanup)

# Start the Gradio app
if __name__ == "__main__":
    demo.queue().launch(debug=True, server_port=9000,share = True,server_name='172.16.98.80')






