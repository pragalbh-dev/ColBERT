import streamlit as st
import pandas as pd
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import colbert_score
import torch
import numpy as np

# Set wide layout for the entire app
st.set_page_config(layout="wide")

@st.cache_data
def get_test_docs():
    test_triples=pd.read_pickle('../synthetic_data/triples.test.pkl')
    test_triples_df=pd.DataFrame(test_triples,columns=['query','collection','negative'])
    test_triples_df['nlq']=test_triples_df['query'].apply(lambda x:' - '.join(x))
    test_positives_df=test_triples_df[['query','nlq','collection']].drop_duplicates()
    return test_positives_df

test_positives_df=get_test_docs()
test_positives_df_copy=test_positives_df.copy()
test_positives_df_copy['correct']=1

checkpoints= ['colbert-ir/colbertv2.0','./experiments/colbert_aspect_training/train_colbert/run_1742250943/checkpoints/colbert-50',
              './experiments/colbert_aspect_training/train_colbert/run_1742250943/checkpoints/colbert-200',
              ]


@st.cache_resource
def get_model_ckpt():
    config = ColBERTConfig(
                bsize=64,  # Small batch size for testing
                accumsteps=2,
                lr=5e-6,
                nway=2,  # Binary pairs for simplicity  
                query_maxlen=32,  
                doc_maxlen=512,   
                dim=128,
                similarity="cosine",
                use_ib_negatives=False,
                maxsteps=10000,  # Limit training steps
                warmup=250,
                val_check_interval=150,
                val_ema_alpha=0.95,
                attend_to_mask_tokens=True
            )
    
    ckpts = {checkpoint:Checkpoint(checkpoint, colbert_config=config) for checkpoint in checkpoints} 
    return ckpts

ckpts=get_model_ckpt()


def get_scores(query,docs):
    selected_model=st.session_state.get('selected_checkpoint','colbert-ir/colbertv2.0')
    
    ckpt=ckpts[selected_model]
    
    
    Q = ckpt.queryFromText([query])
    D = ckpt.docFromText(docs, bsize=1024)[0]
    D_mask = torch.ones(D.shape[:2], dtype=torch.long)
    scores = colbert_score(Q, D, D_mask).flatten().cpu().numpy().tolist()
    
    # Sort by descending score (highest first)
    indices = np.argsort(scores)[::-1]
    return scores, indices
    


def get_res(query):
    
    docs=test_positives_df.collection.to_list()

    res_df=pd.DataFrame(columns=['collection'])
    res_df['collection']= docs
    scores, indices = get_scores(query, docs)
    res_df['score']=scores
    
    # Create actual ranks based on sorted indices
    # Initialize with zeros
    ranks = np.zeros(len(scores), dtype=int)
    # Assign ranks (0-based) based on position in sorted list
    for rank, idx in enumerate(indices):
        ranks[idx] = rank
    res_df['rank'] = ranks
    
    res_df['nlq']=query
    res_df=res_df.merge(test_positives_df_copy[test_positives_df_copy['nlq']==query],on=['nlq','collection'],how='left')
    del res_df['query']
    
    res_df.correct=res_df.correct.fillna(0)
    
    # Sort by rank (ascending)
    res_df = res_df.sort_values('rank')
    
    return res_df

def get_comparison(query):
    
    query2=f'Consumer Industry - {query}'
    query1=f'Industry - {query}'
    
    industry_test_subset_df=test_positives_df[~test_positives_df.nlq.str.contains('Consumer Industry -')]
    consumer_test_subset_df=test_positives_df[test_positives_df.nlq.str.contains('Consumer Industry -')]
    
    pos_for_1=industry_test_subset_df[industry_test_subset_df['nlq']==query1].collection.to_list()
    pos_for_2=consumer_test_subset_df[consumer_test_subset_df['nlq']==query2].collection.to_list()
    queries=[query1,query2]
    docs=pos_for_1+pos_for_2
    query_score_dict={}
    query_ranking_dict={}
    for query in queries:
        scores,rankings = get_scores(query,docs)
        query_score_dict[query]=scores
        query_ranking_dict[query]=rankings
    
    # Create the comparison dataframe
    comparison_df = pd.DataFrame({'doc_text': docs})
    
    # Add scores for each query
    comparison_df[f'score_{query1}'] = query_score_dict[query1]
    comparison_df[f'score_{query2}'] = query_score_dict[query2]
    
    # Create rank columns (position in ranked list)
    for q in queries:
        # Initialize ranks with a high number
        ranks = np.full(len(docs), len(docs))
        # Fill in actual ranks (0 is highest)
        for rank, idx in enumerate(query_ranking_dict[q]):
            ranks[idx] = rank
        # Add to dataframe (add 1 so ranks start at 1 instead of 0)
        comparison_df[f'rank_{q}'] = ranks + 1
    
    # Add correctness markers
    comparison_df['correct_for'] = ''
    for i, doc in enumerate(docs):
        if doc in pos_for_1:
            comparison_df.loc[i, 'correct_for'] = query1
        if doc in pos_for_2:
            # Handle cases where a doc might be correct for both
            if comparison_df.loc[i, 'correct_for']:
                comparison_df.loc[i, 'correct_for'] += f' & {query2}'
            else:
                comparison_df.loc[i, 'correct_for'] = query2
    
    # Add binary correctness columns for each query (for color coding)
    comparison_df[f'is_correct_{query1}'] = comparison_df['doc_text'].apply(lambda x: 1 if x in pos_for_1 else 0)
    comparison_df[f'is_correct_{query2}'] = comparison_df['doc_text'].apply(lambda x: 1 if x in pos_for_2 else 0)
    
    # Sort by the first query's ranking for display
    comparison_df = comparison_df.sort_values(f'rank_{query1}')
    
    return comparison_df
    
# Initialize session state if needed
if 'selected_checkpoint' not in st.session_state:
    st.session_state.selected_checkpoint = checkpoints[0]

# Define list of base queries (without the aspect prefixes)
queries = list(set([q.split(' - ', 1)[1] for q in test_positives_df.nlq.tolist() 
                  if ' - ' in q]))

st.title("ColBERT Aspect-Based Search Demo")

# Sidebar for controls
with st.sidebar:
    st.header("Model & Query Selection")
    
    # Checkpoint selection
    selected_checkpoint = st.selectbox(
        "Select checkpoint",
        options=checkpoints,
        index=0,
        key="selected_checkpoint"
    )
    
    # Query selection
    selected_query = st.selectbox(
        "Select query",
        options=queries,
        index=0
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        fetch_results = st.button("Fetch Results", use_container_width=True)
    with col2:
        compare = st.button("Compare Aspects", use_container_width=True)

# Main area for results
if fetch_results:
    st.header(f"Results for: '{selected_query}'")
    st.subheader(f"Using model: {selected_checkpoint}")
    
    # Get results for industry aspect
    industry_query = f"Industry - {selected_query}"
    results_df = get_res(industry_query)
    
    # Display results with conditional formatting
    st.write("Industry Aspect Results:")
    st.dataframe(
        results_df.style.apply(
            lambda x: ['background-color: #8eff99' if v == 1 else 'background-color: #ff9999' for v in x], 
            axis=1, subset=['correct']
        ),
        use_container_width=True
    )
    
    # Show metrics - fixed to use bracket notation
    correct_pos = results_df[results_df.correct == 1]['rank'].values[0] if not results_df[results_df.correct == 1].empty else "Not found"
    st.metric("Correct document rank", correct_pos)

if compare:
    st.header(f"Aspect Comparison for: '{selected_query}'")
    st.subheader(f"Using model: {selected_checkpoint}")
    
    comparison_df = get_comparison(selected_query)
    
    # Show the base query column without the aspect prefix for cleaner display
    display_df = comparison_df.copy()
    
    # Get the query names for reference in column displays
    query1 = f"Industry - {selected_query}"
    query2 = f"Consumer Industry - {selected_query}"
    
    # Display the comparison results
    st.write("Document Rankings Comparison:")
    
    # First select columns we want to display
    columns_to_display = ['doc_text', f'rank_{query1}', f'rank_{query2}', 'correct_for']
    display_subset = display_df[columns_to_display].rename(columns={
        f'rank_{query1}': 'Industry Rank',
        f'rank_{query2}': 'Consumer Industry Rank',
        'doc_text': 'Document',
        'correct_for': 'Correct For Aspect'
    })
    
    # Then apply styling to the selected columns
    formatted_df = display_subset.style.apply(
        lambda x: ['background-color: #8eff99' if display_df.loc[x.name, f'is_correct_{query1}'] == 1 
                  else 'background-color: #ff9999' for _ in x], 
        axis=1, 
        subset=['Industry Rank']
    ).apply(
        lambda x: ['background-color: #8eff99' if display_df.loc[x.name, f'is_correct_{query2}'] == 1 
                  else 'background-color: #ff9999' for _ in x], 
        axis=1, 
        subset=['Consumer Industry Rank']
    )
    
    # Display the styled dataframe
    st.dataframe(formatted_df, use_container_width=True)
    
    # Calculate metrics - find the best (lowest) rank for each query's correct documents
    industry_correct = display_df[display_df[f'is_correct_{query1}'] == 1][f'rank_{query1}'].min() if not display_df[display_df[f'is_correct_{query1}'] == 1].empty else "Not found"
    consumer_correct = display_df[display_df[f'is_correct_{query2}'] == 1][f'rank_{query2}'].min() if not display_df[display_df[f'is_correct_{query2}'] == 1].empty else "Not found"
    
    # Display metrics with clearer labels
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Industry Aspect - Best Correct Doc Rank", industry_correct, 
                  help="Lower rank is better - shows the position of the highest-ranked correct document")
    with col2:
        st.metric("Consumer Industry Aspect - Best Correct Doc Rank", consumer_correct,
                  help="Lower rank is better - shows the position of the highest-ranked correct document")

if not fetch_results and not compare:
    st.info("Select a query and checkpoint, then click 'Fetch Results' or 'Compare Aspects'")
    