from ..util.data import get_auto_embedding_dim
from .initializers import RandomNormal


class SequenceFeature(object):
    def __init__(self, name, vocab_size, embed_dim=None, pooling="mean", shared_with=None, padding_idx=None, initializer=RandomNormal(0, 0.0001)):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = get_auto_embedding_dim(vocab_size) if embed_dim is None else embed_dim
        self.pooling = pooling
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.initializer = initializer

    def __repr__(self):
        return f"<SequenceFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>"

    def get_embedding_layer(self):
        if not hasattr(self, "embed"):
            self.embed = self.initializer(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        return self.embed


class SparseFeature(object):
    def __init__(self, name, vocab_size, embed_dim=None, shared_with=None, padding_idx=None, initializer=RandomNormal(0, 0.0001)):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = get_auto_embedding_dim(vocab_size) if embed_dim is None else embed_dim
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.initializer = initializer

    def __repr__(self):
        return f"<SparseFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>"

    def get_embedding_layer(self):
        if not hasattr(self, "embed"):
            self.embed = self.initializer(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        return self.embed


class DenseFeature(object):
    def __init__(self, name, embed_dim=1):
        self.name = name
        self.embed_dim = embed_dim

    def __repr__(self):
        return f"<DenseFeature {self.name}>"