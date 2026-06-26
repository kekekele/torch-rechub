import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        var = torch.pow(x - avg, 2) + self.epsilon
        var = var.sum(dim=1, keepdim=True)
        ps = (x - avg) / torch.sqrt(var)
        ps = nn.Sigmoid()(ps)
        return ps * x + (1 - ps) * self.alpha * x


def activation_layer(act_name):
    if isinstance(act_name, str):
        name = act_name.lower()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "dice":
            return Dice()
        if name == "prelu":
            return nn.PReLU()
        if name == "softmax":
            return nn.Softmax(dim=1)
        if name == "leakyrelu":
            return nn.LeakyReLU()
    elif issubclass(act_name, nn.Module):
        return act_name()
    raise NotImplementedError