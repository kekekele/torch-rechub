import torch
import torch.nn as nn


class RegularizationLoss(nn.Module):
    def __init__(self, embedding_l1=0.0, embedding_l2=0.0, dense_l1=0.0, dense_l2=0.0):
        super().__init__()
        self.embedding_l1 = embedding_l1
        self.embedding_l2 = embedding_l2
        self.dense_l1 = dense_l1
        self.dense_l2 = dense_l2

    def forward(self, model):
        reg_loss = 0.0
        norm_params = set()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                for param in module.parameters():
                    norm_params.add(id(param))
        embedding_params = set()
        for module in model.modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                for param in module.parameters():
                    embedding_params.add(id(param))

        for param in model.parameters():
            if not param.requires_grad or id(param) in norm_params:
                continue
            if id(param) in embedding_params:
                if self.embedding_l1 > 0:
                    reg_loss += self.embedding_l1 * torch.sum(torch.abs(param))
                if self.embedding_l2 > 0:
                    reg_loss += self.embedding_l2 * torch.sum(param ** 2)
            else:
                if self.dense_l1 > 0:
                    reg_loss += self.dense_l1 * torch.sum(torch.abs(param))
                if self.dense_l2 > 0:
                    reg_loss += self.dense_l2 * torch.sum(param ** 2)
        return reg_loss