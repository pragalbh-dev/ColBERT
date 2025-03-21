from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=4, experiment="colbert-ir/colbertv2.0")):

        config = ColBERTConfig(
            nbits=2,
            root="./experiments/colbert_aspect_training/colbertv2-vanilla",
            index_bsize=1024,ncells=8,kmeans_niters=32
        )

        indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", 
                          config=config)
        indexer.index(name="colbert_v2.nbits=2",
                      collection='../data/all_collection.tsv',
                      overwrite=True)