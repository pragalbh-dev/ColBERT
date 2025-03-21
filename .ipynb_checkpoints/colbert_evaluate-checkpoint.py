import json

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

def read_jsonl(file_path):
    """
    Read a JSONL file and return a list of parsed JSON objects.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        list: List of parsed JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


def evaluate(model_checkpoint,checkpoint_path,experiment_name,root_dir,all_collections_path='../data/all_collection.tsv',
             data_path="./experiments/colbert_aspect_training/run_1741616267/data",
             all_queries_path="../data/all_queries.tsv",all_labelled_positives_path='../data/all_positives.tsv',
             attend_to_mask_tokens=True,top_k=2000,depths=[10,50,100,500,1000]):
    
    print(f'Indexing documents with the the model : {model_checkpoint}')
    
    with Run().context(RunConfig(nranks=1, experiment=experiment_name):
    
        config = ColBERTConfig(
            nbits=2,
            root=root_dir,
            index_bsize=1024,ncells=8,kmeans_niters=48,attend_to_mask_tokens=attend_to_mask_tokens
        )
        
        indexer = Indexer(checkpoint=os.path.join(checkpoint_path,model_checkpoint), 
                          config=config)
         
        indexer.index(name=f"{experiment_name}.{model_checkpoint}.all.nbits=2",
                      collection=all_collections_path,
                      overwrite=True)
        
    print('INDEXING DONE : at ',f"{experiment_name}.{model_checkpoint}.all.nbits=2")
    print('SEARCHING in INDEX')
    
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):

        config = ColBERTConfig(
            root=root_dir,nbits=2,ncells=8,kmeans_niters=48,ndocs=4096,attend_to_mask_tokens=attend_to_mask_tokens)
        
        searcher = Searcher(index=f"{experiment_name}.{model_checkpoint}.all.nbits=2", config=config)
        # searcher = Searcher(index="run_1741616267.test.nbits=2", config=config)
        queries = Queries(all_queries_path)
        ranking = searcher.search_all(queries, k=top_k)
        
        output_path=ranking.save(f"rankings_{experiment_name}.{model_checkpoint}.all.nbits=2.attend_mask={attend_to_mask_tokens}.topk={topk}.ranking.tsv")

    print('calculating reranked results')

    rankings=pd.read_csv(output_path,
                     header=None,index_col=None,sep='\t')
    
    rankings.columns=['qid','doc_id','rank','score']
    
        
    ranking_collection=pd.read_csv(all_collections_path,
                              sep='\t',index_col=None,header=None)
    
    ranking_queries=pd.read_csv(all_queries_path,
                           sep='\t',index_col=None,header=None)
    
    ranking_collection.columns=['doc_id','collection']
    ranking_queries.columns=['qid','query']
    rankings=rankings.merge(ranking_collection)
    rankings=rankings.merge(ranking_queries)


    train_triples_path=os.path.join(data_path,'train/triples.train.colbert.jsonl')
    val_triples_path=os.path.join(data_path,'val/triples.train.colbert.oversampled.jsonl')
    test_triples_path=os.path.join(data_path,'test/triples.train.colbert.filtered.jsonl')
    
    
    train_queries_path=os.path.join(data_path,'train/queries.train.colbert.tsv')
    val_queries_path=os.path.join(data_path,'val/queries.train.colbert.tsv')
    test_queries_path=os.path.join(data_path,'train/test/queries.train.colbert.tsv')
    
    
    
    train_collections_path=os.path.join(data_path,'train/corpus.train.colbert.tsv')
    val_collections_path=os.path.join(data_path,'val/corpus.train.colbert.tsv')
    test_collections_path=os.path.join(data_path,'test/corpus.train.colbert.tsv')

    
    train_queries=pd.read_csv(train_queries_path,sep='\t',header=None,index_col=None)
    val_queries=pd.read_csv(val_queries_path,sep='\t',header=None,index_col=None)
    test_queries=pd.read_csv(test_queries_path,sep='\t',header=None,index_col=None)

    
    train_queries.columns=['qid','query']
    val_queries.columns=['qid','query']
    test_queries.columns=['qid','query']

    
    train_collections=pd.read_csv(train_collections_path,sep='\t',header=None,index_col=None)
    val_collections=pd.read_csv(val_collections_path,sep='\t',header=None,index_col=None)
    test_collections=pd.read_csv(test_collections_path,sep='\t',header=None,index_col=None)
    
    train_collections.columns=['doc_id','collection']
    val_collections.columns=['doc_id','collection']
    test_collections.columns=['doc_id','collection']    
    
    train_triples=read_jsonl(train_triples_path)
    val_triples=read_jsonl(val_triples_path)
    test_triples=read_jsonl(test_triples_path)


    val_triples_df=pd.DataFrame(val_triples, columns=['qid','pid','nid'])
    train_triples_df=pd.DataFrame(train_triples, columns=['qid','pid','nid'])
    test_triples_df=pd.DataFrame(test_triples, columns=['qid','pid','nid'])

    test_triples_df_pos=test_triples_df[['qid','pid']].drop_duplicates()
    train_triples_df_pos=train_triples_df[['qid','pid']].drop_duplicates()
    val_triples_df_pos=val_triples_df[['qid','pid']].drop_duplicates()
    
    test_triples_df_pos=test_triples_df_pos.merge(test_queries)

    train_triples_df_pos=train_triples_df_pos.merge(test_queries)
    val_triples_df_pos=val_triples_df_pos.merge(test_queries)

    test_triples_df_pos=test_triples_df_pos.merge(test_collections.rename(columns={'doc_id':'pid'}))
    train_triples_df_pos=train_triples_df_pos.merge(train_collections.rename(columns={'doc_id':'pid'}))
    val_triples_df_pos=val_triples_df_pos.merge(val_collections.rename(columns={'doc_id':'pid'}))

    val_triples_df_pos['set']='val'
    train_triples_df_pos['set']='train'
    test_triples_df_pos['set']='test'


    all_labelled_positives=pd.read_csv(all_labelled_positives_path,sep='\t',header=None,index_col=None)
    overall_set_positives=all_labelled_positives[['query','collection']].drop_duplicates()
    overall_set_positives['set_z']='unsampled_positive'

    rankings_set_div=rankings.merge(train_triples_df_pos[['collection','query','set']],how='left',on=['collection','query'])
    rankings_set_div=rankings_set_div.merge(test_triples_df_pos[['collection','query','set']],how='left',on=['collection','query'])
    rankings_set_div=rankings_set_div.merge(val_triples_df_pos[['collection','query','set']],how='left',on=['collection','query'])
    rankings_set_div=rankings_set_div.merge(overall_set_positives,on=['collection','query'],how='left')

    
    get_first_non_none = lambda str_list: next((s for s in str_list if not pd.isna(s)), None)

    rankings_set_div['positive']=rankings_set_div.apply(lambda x:get_first_non_none([x['set'], x['set_x'] , x['set_y'] , x['set_z'] ]),axis=1)


    cols=['query','collection','set']
    actual_number_positives_per_query=pd.concat([train_triples_df_pos[cols],test_triples_df_pos[cols],val_triples_df_pos[cols],
                                             overall_set_positives.rename(columns={'set_z':'set'})[cols]])
    
    actual_number_positives_per_query['positives']=actual_number_positives_per_query.groupby(['set','query'])\
    .collection.transform('count')
    actual_number_positives_per_query=actual_number_positives_per_query[['query','set','positives']].drop_duplicates()
    actual_number_positives_per_query=actual_number_positives_per_query.rename(columns={'positives':'true_positives'})
    actual_number_positives_per_query.loc[actual_number_positives_per_query['set']=='unsampled_positive','set']='total_with_unsampled'
    
    positives_count={}
    per_query_basis={}
    
    for d in depths: 
        ### we need to calculate positives per query at a given depth : why? 
            ## 1. need to know if there is any pattern in low precision queries, 
            ## 2. precision can only be calculated Per query since each query has different number of positives 
        
        per_query_basis[d]=rankings_set_div.groupby('query').apply(lambda x:\
                                                                   x[x['rank']<=d]['positive']\
                                                                   .value_counts().to_dict()).rename('positives_distribution')\
                                                                    .reset_index()
        positives_count[d]=rankings_set_div[rankings_set_div['rank']<=d].positive.value_counts().to_dict()

    
    out_dir=f"{'/'.join(output_path_.split('/')[:-1])}"
    rankings_set_div.to_pickle(f"{'/'.join(output_path_.split('/')[:-1])}/rankings_modified_{experiment_name}.{model_checkpoint}.all.nbits=2.attend_mask={attend_to_mask_tokens}.topk={topk}.ranking.pkl")

    with open(f'{out_dir}/per_query_basis_correct_responses.pkl','wb') as f:
        pickle.dump(per_query_basis,f)
    
        

    
        
    
        
        


    








    
    
    
        