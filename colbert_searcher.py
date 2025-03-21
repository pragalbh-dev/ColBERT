from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__=='__main__':
    with Run().context(RunConfig(nranks=4, experiment="colbert-ir/colbertv2.0")):

        config = ColBERTConfig(
            root="./experiments/colbert_aspect_training/colbertv2-vanilla",
            nbits=2,ncells=8,kmeans_niters=16,ndocs=4096,attend_to_mask_tokens=True,
        )
        
        searcher = Searcher(index="colbert_v2.nbits=2", config=config)
        # searcher = Searcher(index="run_1741616267.test.nbits=2", config=config)
        queries = Queries("../data/all_queries.tsv")
        ranking = searcher.search_all(queries, k=1000)
        ranking.save("rankings.colbert.vanilla.all.ranking.tsv")