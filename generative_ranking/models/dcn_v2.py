import torch

from ..basic.layers import CrossNetMix, CrossNetV2, EmbeddingLayer, LR, MLP


class DCNv2(torch.nn.Module):
    def __init__(self, features, n_cross_layers, mlp_params, model_structure="parallel", use_low_rank_mixture=True, low_rank=32, num_experts=4, **kwargs):
        super().__init__()
        self.features = features
        self.dims = sum(fea.embed_dim for fea in features)
        self.embedding = EmbeddingLayer(features)
        if use_low_rank_mixture:
            self.crossnet = CrossNetMix(self.dims, n_cross_layers, low_rank=low_rank, num_experts=num_experts)
        else:
            self.crossnet = CrossNetV2(self.dims, n_cross_layers)
        self.model_structure = model_structure
        if self.model_structure not in ["crossnet_only", "stacked", "parallel"]:
            raise ValueError(f"Unsupported model_structure={self.model_structure}")
        if self.model_structure == "stacked":
            self.stacked_dnn = MLP(self.dims, output_layer=False, **mlp_params)
            final_dim = mlp_params["dims"][-1]
        elif self.model_structure == "parallel":
            self.parallel_dnn = MLP(self.dims, output_layer=False, **mlp_params)
            final_dim = mlp_params["dims"][-1] + self.dims
        else:
            final_dim = self.dims
        self.linear = LR(final_dim)

    def forward(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)
        cross_out = self.crossnet(embed_x)
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
        else:
            dnn_out = self.parallel_dnn(embed_x)
            final_out = torch.cat([cross_out, dnn_out], dim=1)
        y_pred = self.linear(final_out)
        return torch.sigmoid(y_pred.squeeze(1))