import torch
import torch.nn as nn

from ..basic.features import DenseFeature, SparseFeature
from ..basic.layers import EmbeddingLayer
from ..basic.layer_norm import LayerNorm
from ..basic.per_token_ffn import PerTokenFFN, gelu
from ..basic.sparse_moe import PerTokenSparseMoE
from ..basic.token_mixing import ParameterFreeTokenMixer
from ..basic.tokenization import SemanticTokenizer


class RankMixerBlock(nn.Module):
    def __init__(self, num_tokens, d_model, num_heads, ffn_mult, token_dp=0.0, ffn_dp=0.0, ln_style="pre", use_moe=False, moe_experts=4, moe_l1_coef=0.0, moe_sparsity_ratio=1.0, moe_use_dtsi=True, moe_routing_type="relu_dtsi"):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln_style = str(ln_style).lower()
        self.use_moe = bool(use_moe)
        self.token_mixer = ParameterFreeTokenMixer(num_tokens, d_model, num_heads=num_heads, dropout=token_dp)
        if self.use_moe:
            self.per_token_ffn = PerTokenSparseMoE(num_tokens=num_tokens, d_model=d_model, mult=ffn_mult, num_experts=moe_experts, dropout=ffn_dp, l1_coef=moe_l1_coef, sparsity_ratio=moe_sparsity_ratio, use_dtsi=moe_use_dtsi, routing_type=moe_routing_type)
        else:
            self.per_token_ffn = PerTokenFFN(num_tokens=num_tokens, d_model=d_model, mult=ffn_mult, dropout=ffn_dp)
        self.moe_loss = None

    def forward(self, x):
        moe_loss = x.new_zeros(())
        if self.ln_style == "post":
            mixed = self.token_mixer(x)
            x = self.ln1(x + mixed)
            if self.use_moe:
                ffn_out, moe_loss = self.per_token_ffn(x)
            else:
                ffn_out = self.per_token_ffn(x)
            out = self.ln2(x + ffn_out)
        else:
            mixed = self.token_mixer(self.ln1(x))
            x = x + mixed
            if self.use_moe:
                ffn_out, moe_loss = self.per_token_ffn(self.ln2(x))
            else:
                ffn_out = self.per_token_ffn(self.ln2(x))
            out = x + ffn_out
        self.moe_loss = moe_loss
        return out


class RankMixerEncoder(nn.Module):
    def __init__(self, num_layers, num_tokens, d_model, num_heads, ffn_mult, token_dp=0.0, ffn_dp=0.0, ln_style="pre", use_moe=False, moe_experts=4, moe_l1_coef=0.0, moe_sparsity_ratio=1.0, moe_use_dtsi=True, moe_routing_type="relu_dtsi", use_final_ln=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            RankMixerBlock(num_tokens=num_tokens, d_model=d_model, num_heads=num_heads, ffn_mult=ffn_mult, token_dp=token_dp, ffn_dp=ffn_dp, ln_style=ln_style, use_moe=use_moe, moe_experts=moe_experts, moe_l1_coef=moe_l1_coef, moe_sparsity_ratio=moe_sparsity_ratio, moe_use_dtsi=moe_use_dtsi, moe_routing_type=moe_routing_type)
            for _ in range(int(num_layers))
        ])
        self.final_ln = LayerNorm(d_model) if use_final_ln else nn.Identity()
        self.moe_loss = None

    def forward(self, x):
        out = x
        moe_losses = []
        for block in self.blocks:
            out = block(out)
            if block.moe_loss is not None:
                moe_losses.append(block.moe_loss)
        self.moe_loss = torch.stack(moe_losses).sum() if moe_losses else x.new_zeros(())
        return self.final_ln(out)


class RankMixer(nn.Module):
    def __init__(self, features, sequence_features=None, d_model=128, num_layers=2, num_tokens=4, num_heads=None, ffn_mult=4, tokenizer_input_dim=None, semantic_groups=None, group_rules=None, token_projection="linear", seq_pool_modes=("mean",), include_seq_in_tokenization=True, add_cls_token=False, use_input_ln=True, input_dropout=0.0, token_mixing_dropout=0.0, ffn_dropout=0.0, head_dropout=0.0, ln_style="pre", use_final_ln=True, output_pooling="mean", use_moe=False, moe_experts=4, moe_l1_coef=0.0, moe_sparsity_ratio=1.0, moe_use_dtsi=True, moe_routing_type="relu_dtsi", return_moe_loss=None):
        super().__init__()
        self.features = features
        self.sequence_features = sequence_features or []
        self.sparse_features = [fea for fea in self.features if isinstance(fea, SparseFeature)]
        self.dense_features = [fea for fea in self.features if isinstance(fea, DenseFeature)]
        self.context_sequence_features = [fea for fea in self.features if isinstance(fea, SequenceFeature)]
        self.embedding_features = self.sparse_features + self.context_sequence_features + list(self.sequence_features)
        self.embedding = EmbeddingLayer(self.embedding_features) if self.embedding_features else None
        self.seq_pool_modes = [str(mode).lower() for mode in seq_pool_modes]
        self.include_seq_in_tokenization = bool(include_seq_in_tokenization)
        self.use_input_ln = bool(use_input_ln)
        self.add_cls_token = bool(add_cls_token)
        self.output_pooling = str(output_pooling).lower()
        self.use_moe = bool(use_moe)
        self.return_moe_loss = self.use_moe if return_moe_loss is None else bool(return_moe_loss)
        self.d_model = int(d_model)
        self.base_num_tokens = int(num_tokens)
        self.tokenizer_input_dim = int(tokenizer_input_dim) if tokenizer_input_dim is not None else self._infer_tokenizer_input_dim()

        self.feature_projectors = nn.ModuleDict()
        self.feature_input_dims = {}
        for fea in self.sparse_features:
            self._register_feature_projector(fea.name, fea.embed_dim)
        for fea in self.dense_features:
            self._register_feature_projector(fea.name, fea.embed_dim)
        for fea in self.context_sequence_features:
            self._register_feature_projector(fea.name, fea.embed_dim)
        for fea in self.sequence_features:
            for mode in self.seq_pool_modes:
                self._register_feature_projector(f"seq::{fea.name}::{mode}", fea.embed_dim)

        if not self.include_seq_in_tokenization:
            self.seq_append_projection = nn.Linear(self.tokenizer_input_dim, self.d_model)
            seq_token_count = len(self.sequence_features) * len(self.seq_pool_modes)
        else:
            self.seq_append_projection = None
            seq_token_count = 0

        self.tokenizer = SemanticTokenizer(feature_dims=self.feature_input_dims, target_tokens=self.base_num_tokens, d_model=self.d_model, semantic_groups=semantic_groups, group_rules=group_rules, token_projection=token_projection)
        token_count = self.base_num_tokens + seq_token_count + (1 if self.add_cls_token else 0)
        self.num_heads = int(num_heads) if num_heads is not None else token_count
        if self.num_heads != token_count:
            raise ValueError("RankMixer requires num_heads == token_count after token construction.")

        self.input_ln = LayerNorm(self.d_model) if self.use_input_ln else nn.Identity()
        self.input_dropout = nn.Dropout(input_dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02) if self.add_cls_token else None
        self.encoder = RankMixerEncoder(num_layers=num_layers, num_tokens=token_count, d_model=self.d_model, num_heads=self.num_heads, ffn_mult=ffn_mult, token_dp=token_mixing_dropout, ffn_dp=ffn_dropout, ln_style=ln_style, use_moe=self.use_moe, moe_experts=moe_experts, moe_l1_coef=moe_l1_coef, moe_sparsity_ratio=moe_sparsity_ratio, moe_use_dtsi=moe_use_dtsi, moe_routing_type=moe_routing_type, use_final_ln=use_final_ln)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head_dense1 = nn.Linear(self.d_model, self.d_model * 2)
        self.head_dense2 = nn.Linear(self.d_model * 2, self.d_model)
        self.ctr_logit = nn.Linear(self.d_model, 1)

    def _infer_tokenizer_input_dim(self):
        dims = [fea.embed_dim for fea in self.features]
        dims.extend(fea.embed_dim for fea in self.sequence_features or [])
        return int(dims[0]) if dims else self.d_model

    def _register_feature_projector(self, feature_name, input_dim):
        input_dim = int(input_dim)
        self.feature_input_dims[feature_name] = self.tokenizer_input_dim
        self.feature_projectors[feature_name] = nn.Identity() if input_dim == self.tokenizer_input_dim else nn.Linear(input_dim, self.tokenizer_input_dim)

    def _get_embedding_layer(self, feature):
        name = feature.shared_with if feature.shared_with is not None else feature.name
        return self.embedding.embed_dict[name]

    def _project_feature(self, feature_name, value):
        return self.feature_projectors[feature_name](value)

    def _sequence_mask(self, feature, x):
        values = x[feature.name].long()
        return values.ne(feature.padding_idx) if feature.padding_idx is not None else values.ne(-1)

    def _pool_sequence(self, seq_emb, mask, mode):
        valid = mask.float()
        if mode == "mean":
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return (seq_emb * valid.unsqueeze(-1)).sum(dim=1) / denom
        if mode == "max":
            masked = seq_emb.masked_fill(~mask.unsqueeze(-1), -1e9)
            output = masked.max(dim=1).values
            has_valid = mask.any(dim=1, keepdim=True)
            return torch.where(has_valid, output, torch.zeros_like(output))
        if mode == "target":
            counts = valid.sum(dim=1).long()
            last_idx = (counts - 1).clamp_min(0)
            batch_idx = torch.arange(seq_emb.size(0), device=seq_emb.device)
            gathered = seq_emb[batch_idx, last_idx]
            return torch.where(counts.unsqueeze(-1) > 0, gathered, torch.zeros_like(gathered))
        raise ValueError(f"Unsupported seq pool mode: {mode}")

    def _build_feature_map(self, x):
        feature_map = {}
        for fea in self.sparse_features:
            emb = self._get_embedding_layer(fea)(x[fea.name].long())
            feature_map[fea.name] = self._project_feature(fea.name, emb)
        for fea in self.dense_features:
            dense = x[fea.name].float()
            feature_map[fea.name] = self._project_feature(fea.name, dense if dense.dim() > 1 else dense.unsqueeze(1))
        for fea in self.context_sequence_features:
            if fea.pooling == "concat":
                raise ValueError("RankMixer item/context SequenceFeature does not support pooling='concat'; use mean/sum/max/target.")
            seq_emb = self._get_embedding_layer(fea)(x[fea.name].long())
            mask = self._sequence_mask(fea, x)
            pooled = self._pool_sequence(seq_emb, mask, fea.pooling)
            feature_map[fea.name] = self._project_feature(fea.name, pooled)
        seq_feature_map = {}
        for fea in self.sequence_features:
            if fea.pooling != "concat":
                raise ValueError("RankMixer sequence_features must use pooling='concat' to preserve token-level semantics.")
            seq_emb = self._get_embedding_layer(fea)(x[fea.name].long())
            mask = self._sequence_mask(fea, x)
            for mode in self.seq_pool_modes:
                pooled = self._pool_sequence(seq_emb, mask, mode)
                name = f"seq::{fea.name}::{mode}"
                seq_feature_map[name] = self._project_feature(name, pooled)
        if self.include_seq_in_tokenization:
            feature_map.update(seq_feature_map)
            return feature_map, None
        return feature_map, seq_feature_map

    def _append_sequence_tokens(self, tokens, seq_feature_map):
        if not seq_feature_map:
            return tokens
        seq_tokens = torch.stack([self.seq_append_projection(value) for value in seq_feature_map.values()], dim=1)
        return torch.cat([tokens, seq_tokens], dim=1)

    def forward(self, x):
        feature_map, seq_feature_map = self._build_feature_map(x)
        tokens = self.tokenizer(feature_map)
        if not self.include_seq_in_tokenization:
            tokens = self._append_sequence_tokens(tokens, seq_feature_map)
        if self.add_cls_token:
            cls_token = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = self.input_ln(tokens)
        tokens = self.input_dropout(tokens)
        encoded = self.encoder(tokens)
        if self.output_pooling in ("mean", "avg"):
            head_input = encoded.mean(dim=1)
        elif self.output_pooling == "cls":
            if not self.add_cls_token:
                raise ValueError("output_pooling='cls' requires add_cls_token=True.")
            head_input = encoded[:, 0, :]
        else:
            raise ValueError(f"Unknown output_pooling: {self.output_pooling}")
        head_input = self.head_dropout(head_input)
        head_hidden = gelu(self.head_dense1(head_input))
        head_hidden = self.head_dropout(head_hidden)
        head_hidden = gelu(self.head_dense2(head_hidden))
        y_pred = torch.sigmoid(self.ctr_logit(head_hidden).squeeze(1))
        if self.return_moe_loss:
            return y_pred, self.encoder.moe_loss
        return y_pred