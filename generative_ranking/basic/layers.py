import torch
import torch.nn as nn

from .activation import activation_layer
from .features import DenseFeature, SequenceFeature, SparseFeature


class EmbeddingLayer(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.input_mask = InputMask()
        for fea in features:
            if fea.name in self.embed_dict:
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with is None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, SequenceFeature) and fea.shared_with is None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()

    def forward(self, x, features, squeeze_dim=False):
        sparse_emb, dense_values = [], []
        sparse_exists, dense_exists = False, False
        for fea in features:
            if isinstance(fea, SparseFeature):
                embed_name = fea.shared_with if fea.shared_with is not None else fea.name
                sparse_emb.append(self.embed_dict[embed_name](x[fea.name].long()).unsqueeze(1))
            elif isinstance(fea, SequenceFeature):
                if fea.pooling == "sum":
                    pooling_layer = SumPooling()
                elif fea.pooling == "mean":
                    pooling_layer = AveragePooling()
                elif fea.pooling == "concat":
                    pooling_layer = ConcatPooling()
                else:
                    raise ValueError(f"Unsupported sequence pooling: {fea.pooling}")
                fea_mask = self.input_mask(x, fea)
                embed_name = fea.shared_with if fea.shared_with is not None else fea.name
                sparse_emb.append(pooling_layer(self.embed_dict[embed_name](x[fea.name].long()), fea_mask).unsqueeze(1))
            else:
                dense = x[fea.name].float()
                dense_values.append(dense if dense.dim() > 1 else dense.unsqueeze(1))

        if dense_values:
            dense_exists = True
            dense_values = torch.cat(dense_values, dim=1)
        if sparse_emb:
            sparse_exists = True
            sparse_emb = torch.cat(sparse_emb, dim=1)

        if squeeze_dim:
            if dense_exists and not sparse_exists:
                return dense_values
            if sparse_exists and not dense_exists:
                return sparse_emb.flatten(start_dim=1)
            if sparse_exists and dense_exists:
                return torch.cat((sparse_emb.flatten(start_dim=1), dense_values), dim=1)
            raise ValueError("The input features can not be empty")
        if sparse_exists:
            return sparse_emb
        raise ValueError("Expected sparse or sequence features when squeeze_dim=False")


class InputMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, features):
        mask = []
        if not isinstance(features, list):
            features = [features]
        for fea in features:
            if isinstance(fea, (SparseFeature, SequenceFeature)):
                if fea.padding_idx is not None:
                    fea_mask = x[fea.name].long() != fea.padding_idx
                else:
                    fea_mask = x[fea.name].long() != -1
                mask.append(fea_mask.unsqueeze(1).float())
            else:
                raise ValueError("Only SparseFeature or SequenceFeature support mask generation")
        return torch.cat(mask, dim=1)


class LR(nn.Module):
    def __init__(self, input_dim, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return torch.sigmoid(self.fc(x)) if self.sigmoid else self.fc(x)


class ConcatPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x


class AveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1)
        sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
        non_padding_length = mask.sum(dim=-1)
        return sum_pooling_matrix / (non_padding_length.float() + 1e-16)


class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.sum(x, dim=1)
        return torch.bmm(mask, x).squeeze(1)


class MLP(nn.Module):
    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0, activation="relu"):
        super().__init__()
        dims = [] if dims is None else dims
        layers = []
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            x = x0 * self.w[i](x) + self.b[i] + x
        return x


class CrossNetMix(nn.Module):
    def __init__(self, input_dim, num_layers=2, low_rank=32, num_experts=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.u_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, input_dim, low_rank))) for _ in range(self.num_layers)])
        self.v_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, input_dim, low_rank))) for _ in range(self.num_layers)])
        self.c_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, low_rank, low_rank))) for _ in range(self.num_layers)])
        self.gating = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(self.num_experts)])
        self.bias = nn.ParameterList([nn.Parameter(nn.init.zeros_(torch.empty(input_dim, 1))) for _ in range(self.num_layers)])

    def forward(self, x):
        x_0 = x.unsqueeze(2)
        x_l = x_0
        for i in range(self.num_layers):
            output_of_experts = []
            gating_score_experts = []
            for expert_id in range(self.num_experts):
                gating_score_experts.append(self.gating[expert_id](x_l.squeeze(2)))
                v_x = torch.matmul(self.v_list[i][expert_id].t(), x_l)
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.c_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)
                uv_x = torch.matmul(self.u_list[i][expert_id], v_x)
                dot_ = x_0 * (uv_x + self.bias[i])
                output_of_experts.append(dot_.squeeze(2))
            output_of_experts = torch.stack(output_of_experts, 2)
            gating_score_experts = torch.stack(gating_score_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_experts.softmax(1))
            x_l = moe_out + x_l
        return x_l.squeeze()