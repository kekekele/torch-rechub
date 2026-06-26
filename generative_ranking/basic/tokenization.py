import math
import re

import torch
import torch.nn as nn


def _sanitize_group_name(name):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return safe or "group"


def _looks_like_regex(pattern):
    if pattern.startswith("re:"):
        return True
    for token in ("^", "$", ".*", "[", "]", "(", ")", "|", "?"):
        if token in pattern:
            return True
    return False


def _normalize_groups(semantic_groups):
    if not semantic_groups:
        return []
    if isinstance(semantic_groups, dict):
        return [(str(k), list(v)) for k, v in semantic_groups.items()]
    groups = []
    for idx, item in enumerate(semantic_groups):
        if isinstance(item, dict):
            name = item.get("name", f"group_{idx}")
            feats = item.get("features") or item.get("patterns") or []
            groups.append((str(name), list(feats)))
        elif isinstance(item, (list, tuple)):
            groups.append((f"group_{idx}", list(item)))
    return groups


def _compile_group_rules(group_rules):
    if not group_rules:
        return []
    compiled = []
    for rule in group_rules:
        name = _sanitize_group_name(rule.get("name", "group"))
        patterns = [p for p in rule.get("patterns", []) if p]
        if patterns:
            compiled.append((name, [re.compile(p) for p in patterns]))
    return compiled


def _assign_semantic_groups(feature_names, group_rules):
    compiled = _compile_group_rules(group_rules)
    if not compiled:
        return list(range(len(feature_names)))
    grouped = []
    used = set()
    for _, patterns in compiled:
        indices = []
        for idx, feat in enumerate(feature_names):
            if idx in used:
                continue
            for pat in patterns:
                if pat.search(feat):
                    indices.append(idx)
                    used.add(idx)
                    break
        if indices:
            grouped.extend(indices)
    for idx in range(len(feature_names)):
        if idx not in used:
            grouped.append(idx)
    return grouped


class SemanticTokenizer(nn.Module):
    def __init__(self, feature_dims, target_tokens, d_model, semantic_groups=None, group_rules=None, token_projection="linear"):
        super().__init__()
        self.target_tokens = int(target_tokens)
        self.d_model = int(d_model)
        self.semantic_groups = semantic_groups
        self.group_rules = group_rules
        self.token_projection = str(token_projection).lower()
        if self.token_projection != "linear":
            raise ValueError("Only linear token projection is supported.")
        self.feature_dims = dict(feature_dims)
        self.group_projections = nn.ModuleDict()
        self.chunk_projection = None

    def _resolve_group_features(self, group_features, available_names):
        resolved = []
        for raw in group_features:
            if raw in available_names:
                resolved.append(raw)
                continue
            pattern = raw[3:] if raw.startswith("re:") else raw
            if _looks_like_regex(raw):
                regex = re.compile(pattern)
                for name in available_names:
                    if regex.search(name) and name not in resolved:
                        resolved.append(name)
        return resolved

    def _get_group_projection(self, group_name, input_dim, device):
        group_name = _sanitize_group_name(group_name)
        if group_name not in self.group_projections:
            self.group_projections[group_name] = nn.Linear(input_dim, self.d_model).to(device)
        return self.group_projections[group_name]

    def _get_chunk_projection(self, input_dim, device):
        if self.chunk_projection is None:
            self.chunk_projection = nn.Linear(input_dim, self.d_model).to(device)
        return self.chunk_projection

    def forward(self, feature_map):
        feature_names = list(feature_map.keys())
        if not feature_names:
            raise ValueError("SemanticTokenizer needs at least one feature.")
        groups = _normalize_groups(self.semantic_groups)
        batch_size = next(iter(feature_map.values())).size(0)
        device = next(iter(feature_map.values())).device
        if groups:
            tokens = []
            for group_name, group_features in groups:
                resolved = self._resolve_group_features(group_features, feature_names)
                if resolved:
                    tensors = [feature_map[name] for name in resolved]
                    concat = torch.cat(tensors, dim=-1)
                    input_dim = sum(feature_map[name].size(-1) for name in resolved)
                else:
                    concat = next(iter(feature_map.values())).new_zeros(batch_size, 1)
                    input_dim = 1
                projection = self._get_group_projection(group_name, input_dim, device)
                tokens.append(projection(concat))
            stacked = torch.stack(tokens, dim=1)
            if stacked.size(1) > self.target_tokens:
                stacked = stacked[:, : self.target_tokens, :]
            elif stacked.size(1) < self.target_tokens:
                pad = stacked.new_zeros(batch_size, self.target_tokens - stacked.size(1), self.d_model)
                stacked = torch.cat([stacked, pad], dim=1)
            return stacked
        ordered_indices = _assign_semantic_groups(feature_names, self.group_rules)
        ordered_names = [feature_names[i] for i in ordered_indices]
        ordered_embeddings = torch.stack([feature_map[name] for name in ordered_names], dim=1)
        feature_count = len(ordered_names)
        target_tokens = self.target_tokens if self.target_tokens > 0 else feature_count
        token_size = int(math.ceil(feature_count / float(target_tokens)))
        pad_needed = target_tokens * token_size - feature_count
        if pad_needed > 0:
            pad = ordered_embeddings.new_zeros(batch_size, pad_needed, ordered_embeddings.size(-1))
            ordered_embeddings = torch.cat([ordered_embeddings, pad], dim=1)
        flat = ordered_embeddings.reshape(batch_size, target_tokens, token_size * ordered_embeddings.size(-1))
        projection = self._get_chunk_projection(flat.size(-1), device)
        return projection(flat)