class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    def __getattr__(self, name):
        lower_name = name.lower()
        for attr in dir(self):
            if attr.lower() == lower_name:
                return getattr(self, attr)
        if lower_name.endswith('_dim'):
            stem = lower_name.rstrip('_dim')
            alt_name = f'n_{stem}s'
            for attr in dir(self):
                if attr.lower() == alt_name:
                    return getattr(self, attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

class TransitionConfig(Config):
    """Transition Model Configurations"""
    n_word_ids = None  # inferred
    n_tag_ids = None  # inferred
    n_deprel_ids = None  # inferred
    n_word_features = None  # inferred
    n_tag_features = None  # inferred
    n_deprel_features = None  # inferred
    n_classes = None  # inferred
    dropout = 0.5
    embed_size = None  # inferred
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.001

class GraphConfig(Config):
    """Graph Model Configurations"""
    n_arcs = 256
    n_labels = 64
    batch_size = 32
    dropout = 0.1
    n_epochs = 10
    lr = 2e-3
    hf_model_name = 'bert-base-cased'
    ud_corpus = ('English', 'EWT')
