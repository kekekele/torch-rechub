import torch.nn as nn


class ParameterFreeTokenMixer(nn.Module):
    def __init__(self, num_tokens, d_model, num_heads=None, dropout=0.0):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads) if num_heads is not None else int(num_tokens)
        if self.num_heads != self.num_tokens:
            raise ValueError("Parameter-free token mixing requires num_heads == num_tokens.")
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got d_model={self.d_model} num_heads={self.num_heads}")
        self.d_head = self.d_model // self.num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        split = x.reshape(batch_size, self.num_tokens, self.num_heads, self.d_head)
        shuffled = split.permute(0, 2, 1, 3)
        merged = shuffled.reshape(batch_size, self.num_heads, self.num_tokens * self.d_head)
        mixed = merged.reshape(batch_size, self.num_tokens, self.d_model)
        return self.dropout(mixed)