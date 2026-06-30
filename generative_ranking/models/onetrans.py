import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..basic.features import DenseFeature, SequenceFeature, SparseFeature
from ..basic.layers import EmbeddingLayer

VALID_MASK_TYPES = {"paper_causal", "origin", "hard_mask", "bimask_soft", "bimask_hard"}


def linear_pyramid_schedule(total_tokens, ns_len, num_layers, align_to=32):
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if total_tokens < ns_len:
        raise ValueError("total_tokens must be greater than or equal to ns_len")
    if align_to <= 0:
        raise ValueError("align_to must be positive")

    if num_layers == 1:
        return [ns_len]

    # 金字塔 stack 中，query 长度从 total_tokens 逐层收缩到 ns_len。
    schedule = [total_tokens]
    for layer_idx in range(1, num_layers - 1):
        raw = total_tokens + (ns_len - total_tokens) * layer_idx / (num_layers - 1)
        target_len = int(round(raw))
        if align_to > 1 and total_tokens > align_to:
            target_len = int(round(target_len / align_to) * align_to)
        target_len = max(ns_len, min(schedule[-1], target_len))
        schedule.append(target_len)
    schedule.append(ns_len)
    return schedule


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rms).to(input_dtype) * self.weight


class FFNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.proj_1 = nn.Linear(input_dim, hidden_dim)
        self.proj_2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.proj_2(self.act(self.proj_1(x))))


class MixedCausalAttention(nn.Module):
    def __init__(self, ns_len, d_model, num_heads=4, if_mask=True, mask_type="paper_causal"):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if mask_type not in VALID_MASK_TYPES:
            raise ValueError(f"Unsupported mask_type: {mask_type}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.ns_len = ns_len
        self.if_mask = if_mask
        self.mask_type = mask_type
        self.dense = nn.Linear(d_model, d_model)
        # Mixed Q/K/V parameterization:
        # - seq token 使用共享投影（索引 self.ns_len）
        # - 每个 NS token 使用独立投影（索引 0..ns_len-1）
        self.kqv_list = nn.ModuleList(
            [nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)]) for _ in range(ns_len + 1)]
        )

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def create_attention_mask(self, query_len, key_len, device, dtype):
        if self.mask_type == "paper_causal":
            row_idx = torch.arange(query_len, device=device).unsqueeze(1)
            col_idx = torch.arange(key_len, device=device).unsqueeze(0)
            q_abs = row_idx + (key_len - query_len)
            allowed = col_idx <= q_abs
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            return mask.masked_fill(~allowed, torch.finfo(dtype).min)

        row_idx = torch.arange(query_len, device=device).unsqueeze(1)
        col_idx = torch.arange(key_len, device=device).unsqueeze(0)
        origin_allowed = (col_idx - row_idx) <= (self.ns_len - 1)
        if self.mask_type == "origin":
            return origin_allowed.to(dtype=dtype) + 1e-9
        if self.mask_type == "hard_mask":
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            return mask.masked_fill(~origin_allowed, torch.finfo(dtype).min)

        ns_query_rows = row_idx < self.ns_len
        strict_causal_allowed = col_idx <= row_idx
        if self.mask_type == "bimask_soft":
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            seq_allowed = (~ns_query_rows) & strict_causal_allowed
            return mask + seq_allowed.to(dtype=dtype)

        allowed = ns_query_rows | strict_causal_allowed
        mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def _cal_kqv(self, x, group_idx, proj_idx):
        return self.kqv_list[group_idx][proj_idx](x)

    def _project_one(self, x, proj_idx):
        seq_len = x.size(1) - self.ns_len
        outputs = []
        # 前面 seq_len 个 token 属于 S token，使用共享参数组 self.ns_len
        if seq_len > 0:
            outputs.append(self._cal_kqv(x[:, :seq_len, :], self.ns_len, proj_idx))
        # 后面 ns_len 个 token 属于 NS token，每个 token 使用独立的参数组
        for i in range(self.ns_len):
            start = seq_len + i
            outputs.append(self._cal_kqv(x[:, start : start + 1, :], i, proj_idx))
        return torch.cat(outputs, dim=1)

    def cal_mix_param_kqv(self, x):
        return self._project_one(x[0], 0), self._project_one(x[1], 1), self._project_one(x[2], 2)

    def forward(self, x):
        seq_len_k = x[0].size(1)
        seq_len_q = x[1].size(1)
        k, q, v = self.cal_mix_param_kqv(x)
        k = self.split_heads(k)
        q = self.split_heads(q)
        v = self.split_heads(v)

        attention_mask = None
        if self.if_mask:
            attention_mask = self.create_attention_mask(seq_len_q, seq_len_k, device=q.device, dtype=q.dtype)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(output.size(0), -1, self.d_model)
        return self.dense(output)


class OneTransBlock(nn.Module):
    def __init__(self, ns_len, d_model, num_heads=4, ffn_hidden=256, pyramid_stack_len=None, mask_type="paper_causal", use_checkpoint=False):
        super().__init__()
        self.ns_len = ns_len
        self.pyramid_stack_len = pyramid_stack_len
        self.use_checkpoint = use_checkpoint
        self.rms_0 = RMSNorm(d_model)
        self.rms_1 = RMSNorm(d_model)
        self.cma = MixedCausalAttention(ns_len=ns_len, d_model=d_model, num_heads=num_heads, mask_type=mask_type)
        self.ffn_list = nn.ModuleList([FFNLayer(d_model, ffn_hidden, d_model) for _ in range(ns_len + 1)])

    def cal_mix_param_ffn(self, x):
        outputs = []
        seq_len = x.size(1) - self.ns_len
        if seq_len > 0:
            outputs.append(self.ffn_list[self.ns_len](x[:, :seq_len, :]))
        for i in range(self.ns_len):
            start = seq_len + i
            outputs.append(self.ffn_list[i](x[:, start : start + 1, :]))
        return torch.cat(outputs, dim=1)

    def _forward_impl(self, x):
        x = self.rms_0(x)
        k_x, q_x, v_x = x, x, x
        # pyramid stack 只裁剪 query 侧长度，key/value 仍保留完整上下文。
        if self.pyramid_stack_len is not None and self.pyramid_stack_len >= self.ns_len:
            q_x = x[:, -self.pyramid_stack_len :, :]
        origin_x = q_x
        x = self.cma((k_x, q_x, v_x))
        x = origin_x + x
        origin_x = x
        x = self.rms_1(x)
        x = self.cal_mix_param_ffn(x)
        return origin_x + x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class MultiOneTransBlock(nn.Module):
    def __init__(self, ns_len, d_model, num_heads=4, ffn_hidden=256, n=4, pyramid_stack_len=None, mask_type="paper_causal", use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                OneTransBlock(
                    ns_len=ns_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_hidden=ffn_hidden,
                    pyramid_stack_len=pyramid_stack_len,
                    mask_type=mask_type,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(n)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class OneTrans(nn.Module):
    def __init__(
        self,
        features,
        sequence_features=None,
        max_seq_len=50,
        ns_len=4,
        d_model=128,
        num_heads=4,
        ffn_hidden=256,
        multi_num=4,
        num_pyramid_layers=6,
        pyramid_align=32,
        mask_type="paper_causal",
        use_sep_token=False,
        use_checkpoint=False,
        head_dropout=0.0,
    ):
        super().__init__()
        self.features = features
        self.sequence_features = sequence_features or []
        if not self.sequence_features:
            raise ValueError("OneTrans requires at least one sequence feature")

        self.sparse_features = [fea for fea in self.features if isinstance(fea, SparseFeature)]
        self.dense_features = [fea for fea in self.features if isinstance(fea, DenseFeature)]
        # 这里的 item_sequence_features 不是 OneTrans 的历史 S token 序列输入，
        # 而是“多值 item/静态属性”类型特征，需要先池化后再进入非序列 tokenizer。
        self.item_sequence_features = [fea for fea in self.features if isinstance(fea, SequenceFeature)]
        self.embedding_features = self.sparse_features + self.item_sequence_features + list(self.sequence_features)
        self.embedding = EmbeddingLayer(self.embedding_features) if self.embedding_features else None

        self.ns_len = int(ns_len)
        self.max_seq_len = int(max_seq_len)
        self.d_model = int(d_model)
        self.use_sep_token = bool(use_sep_token)

        self.non_seq_input_dim = sum(int(fea.embed_dim) for fea in self.sparse_features + self.dense_features + self.item_sequence_features)
        self.seq_input_dim = sum(int(fea.embed_dim) for fea in self.sequence_features)
        if self.non_seq_input_dim <= 0:
            raise ValueError("OneTrans requires non-sequential features for NS tokenization")
        if self.seq_input_dim <= 0:
            raise ValueError("OneTrans requires sequence feature embeddings for S tokenization")

        self.non_seq_tokenizer = nn.Linear(self.non_seq_input_dim, self.ns_len * self.d_model)
        self.seq_tokenizer = nn.Linear(self.seq_input_dim, self.d_model)
        if self.use_sep_token:
            self.sep_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        self.base_block = MultiOneTransBlock(
            ns_len=self.ns_len,
            d_model=self.d_model,
            num_heads=num_heads,
            ffn_hidden=ffn_hidden,
            n=multi_num,
            mask_type=mask_type,
            use_checkpoint=use_checkpoint,
        )
        total_tokens = self.ns_len + self.max_seq_len + (1 if self.use_sep_token else 0)
        schedule = linear_pyramid_schedule(total_tokens=total_tokens, ns_len=self.ns_len, num_layers=num_pyramid_layers, align_to=pyramid_align)
        self.stack_blocks = nn.ModuleList(
            [
                MultiOneTransBlock(
                    ns_len=self.ns_len,
                    d_model=self.d_model,
                    num_heads=num_heads,
                    ffn_hidden=ffn_hidden,
                    n=multi_num,
                    pyramid_stack_len=target_len,
                    mask_type=mask_type,
                    use_checkpoint=use_checkpoint,
                )
                for target_len in schedule
            ]
        )
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 1),
        )

    def _get_embedding_layer(self, feature):
        name = feature.shared_with if feature.shared_with is not None else feature.name
        return self.embedding.embed_dict[name]

    def _sequence_mask(self, feature, x):
        values = x[feature.name].long()
        return values.ne(feature.padding_idx) if feature.padding_idx is not None else values.ne(-1)

    def _pool_sequence(self, seq_emb, mask, mode):
        valid = mask.float()
        if mode == "mean":
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return (seq_emb * valid.unsqueeze(-1)).sum(dim=1) / denom
        if mode == "sum":
            return (seq_emb * valid.unsqueeze(-1)).sum(dim=1)
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
        raise ValueError(f"Unsupported sequence pooling mode: {mode}")

    def _build_non_seq_inputs(self, x):
        outputs = []
        for fea in self.sparse_features:
            outputs.append(self._get_embedding_layer(fea)(x[fea.name].long()))
        for fea in self.dense_features:
            dense = x[fea.name].float()
            outputs.append(dense if dense.dim() > 1 else dense.unsqueeze(1))
        for fea in self.item_sequence_features:
            # item_sequence_features 为多值特征，但不可使用 concat 池化；
            # 它们会先被归约为一个固定向量，再进入 NS token tokenizer。
            if fea.pooling == "concat":
                raise ValueError("OneTrans does not support concat pooling for non-sequential item features")
            seq_emb = self._get_embedding_layer(fea)(x[fea.name].long())
            mask = self._sequence_mask(fea, x)
            outputs.append(self._pool_sequence(seq_emb, mask, fea.pooling))
        return torch.cat(outputs, dim=1)

    def _build_seq_tokens(self, x):
        seq_parts = []
        for fea in self.sequence_features:
            # sequence_features 必须使用 concat 池化，以保持 S token 的时间步顺序。
            if fea.pooling != "concat":
                raise ValueError("OneTrans sequence features must use pooling='concat' to preserve token order")
            seq_parts.append(self._get_embedding_layer(fea)(x[fea.name].long()))
        seq_inputs = torch.cat(seq_parts, dim=-1)
        return self.seq_tokenizer(seq_inputs)

    def forward(self, x):
        batch_size = next(iter(x.values())).size(0)
        non_seq_inputs = self._build_non_seq_inputs(x)
        # non-seq 特征先聚合成一个向量，再切成固定数量的 NS token。
        ns_tokens = self.non_seq_tokenizer(non_seq_inputs).view(batch_size, self.ns_len, self.d_model)
        seq_tokens = self._build_seq_tokens(x)
        if self.use_sep_token:
            sep_token = self.sep_token.expand(batch_size, -1, -1)
            tokens = torch.cat([seq_tokens, sep_token, ns_tokens], dim=1)
        else:
            tokens = torch.cat([seq_tokens, ns_tokens], dim=1)

        # 先过 base block，再进入逐层缩短 query 的 pyramid stack。
        tokens = self.base_block(tokens)
        for block in self.stack_blocks:
            tokens = block(tokens)
        pooled = self.head_dropout(tokens.mean(dim=1))
        return torch.sigmoid(self.head(pooled).squeeze(1))