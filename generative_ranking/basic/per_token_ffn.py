import math

import torch
import torch.nn as nn


def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PerTokenFFN(nn.Module):
    def __init__(self, num_tokens, d_model, mult=4, dropout=0.0):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        hidden_dim = self.d_model * int(mult)
        self.dropout = nn.Dropout(dropout)
        self.W1 = nn.Parameter(torch.empty(self.num_tokens, self.d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_tokens, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_tokens, hidden_dim, self.d_model))
        self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W1, nonlinearity="linear")
        nn.init.kaiming_normal_(self.W2, nonlinearity="linear")

    def forward(self, x):
        hidden = torch.einsum("btd,tdh->bth", x, self.W1) + self.b1
        hidden = gelu(hidden)
        hidden = self.dropout(hidden)
        output = torch.einsum("bth,thd->btd", hidden, self.W2) + self.b2
        return self.dropout(output)