import torch


class RandomNormal(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, vocab_size, embed_dim, padding_idx=None):
        embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        torch.nn.init.normal_(embed.weight, self.mean, self.std)
        if padding_idx is not None:
            torch.nn.init.zeros_(embed.weight[padding_idx])
        return embed