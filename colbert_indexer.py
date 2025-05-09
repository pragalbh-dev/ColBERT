from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="colbert_aspect_training")):

        config = ColBERTConfig(
            nbits=2,
            root="./experiments/colbert_aspect_training/colbertv2-tuned.all-aspects.nlq",
            index_bsize=512,ncells=8,kmeans_niters=32,doc_maxlen=512
        )

        indexer = Indexer(checkpoint="./experiments/colbert_aspect_training/none/run_1742739659/checkpoints/colbert-final", 
                          config=config)
        indexer.index(name="colbert_v2.tuned=true.nbits=2.aspects=all.form=nlq",
                      collection='./data/all_collection.tsv',
                      overwrite=True)