from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="run_1741616267_evaluation")):
    
        config = ColBERTConfig(
            nbits=2,
            root="./experiments/colbert_aspect_training/none/run_1741616267",
            index_bsize=1024
        )
    
        indexer = Indexer(checkpoint="./experiments/colbert_aspect_training/none/run_1741616267/checkpoints/colbert-best", 
                          config=config)
        indexer.index(name="run_1741616267.val.nbits=2",
                      collection="./experiments/colbert_aspect_training/run_1741616267/data/val/corpus.train.colbert.tsv",
                      overwrite=True)