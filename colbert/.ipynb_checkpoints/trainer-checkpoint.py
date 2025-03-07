from colbert.infra.run import Run
from colbert.infra.launcher import Launcher
from colbert.infra.config import ColBERTConfig, RunConfig

from colbert.training.training import train


class Trainer:
    def __init__(self, triples, queries, collection, config=None,tracker_config=None):
        self.config = ColBERTConfig.from_existing(config, Run().config)

        self.triples = triples
        self.queries = queries
        self.collection = collection
        self.tracker_config=tracker_config
        
    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def train(self, checkpoint='bert-base-uncased'):
        """
            Note that config.checkpoint is ignored. Only the supplied checkpoint here is used.
        """

        # Resources don't come from the config object. They come from the input parameters.
        # TODO: After the API stabilizes, make this "self.config.assign()" to emphasize this distinction.
        self.configure(triples=self.triples, queries=self.queries, collection=self.collection)
        self.configure(checkpoint=checkpoint)
        
        launcher = Launcher(train,tracker_config=tracker_config)

        self._best_checkpoint_path = launcher.launch(self.config, self.triples, self.queries, self.collection)


    def best_checkpoint_path(self):
        return self._best_checkpoint_path


# Create a custom trainer for single-GPU training that properly uses avoid_fork_if_possible
class SingleGPUTrainer(Trainer):
    def __init__(self, triples, queries, collection, config=None,tracker_config=None):
        super().__init__(triples, queries, collection, config, tracker_config=tracker_config)
        
    def train(self, checkpoint='bert-base-uncased'):
        """
        Override train method to use launch_without_fork for single-GPU training.
        This ensures that avoid_fork_if_possible=True is properly respected.
        """
        # Configure resources like the original train method
        self.configure(triples=self.triples, queries=self.queries, collection=self.collection)
        self.configure(checkpoint=checkpoint)
        
        # Create the launcher with the training function
        launcher = Launcher(train,tracker_config=self.tracker_config)
        
        # Check if we should avoid forking
        if hasattr(self.config, 'avoid_fork_if_possible') and self.config.avoid_fork_if_possible and self.config.nranks == 1:
            # Use launch_without_fork for single-GPU training
            self._best_checkpoint_path = launcher.launch_without_fork(self.config, self.triples, self.queries, self.collection)
        else:
            # Use standard launch for multi-GPU training
            self._best_checkpoint_path = launcher.launch(self.config, self.triples, self.queries, self.collection)
            