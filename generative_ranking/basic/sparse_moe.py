import torch
import torch.nn as nn

from .per_token_ffn import gelu


class PerTokenSparseMoE(nn.Module):
    def __init__(self, num_tokens, d_model, mult=4, num_experts=4, dropout=0.0, l1_coef=0.0, sparsity_ratio=1.0, use_dtsi=True, routing_type="relu_dtsi"):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.num_experts = int(num_experts)
        hidden_dim = self.d_model * int(mult)
        self.dropout = nn.Dropout(dropout)
        self.l1_coef = float(l1_coef)
        self.sparsity_ratio = float(sparsity_ratio) if sparsity_ratio else 1.0
        self.use_dtsi = bool(use_dtsi)
        self.routing_type = str(routing_type).lower()

        self.W1 = nn.Parameter(torch.empty(self.num_tokens, self.num_experts, self.d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_tokens, self.num_experts, hidden_dim, self.d_model))
        self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts, self.d_model))
        self.gate_w_train = nn.Parameter(torch.empty(self.num_tokens, self.d_model, self.num_experts))
        self.gate_b_train = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts))
        if self.use_dtsi:
            self.gate_w_infer = nn.Parameter(torch.empty(self.num_tokens, self.d_model, self.num_experts))
            self.gate_b_infer = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts))
        else:
            self.gate_w_infer = None
            self.gate_b_infer = None
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in (self.W1, self.W2, self.gate_w_train):
            nn.init.kaiming_normal_(parameter, nonlinearity="linear")
        if self.gate_w_infer is not None:
            nn.init.kaiming_normal_(self.gate_w_infer, nonlinearity="linear")

    def _router_logits(self, x, w, b):
        return torch.einsum("btd,tde->bte", x, w) + b

    def forward(self, x):
        hidden = torch.einsum("btd,tedh->bteh", x, self.W1) + self.b1
        hidden = gelu(hidden)
        hidden = self.dropout(hidden)
        expert_out = torch.einsum("bteh,tehd->bted", hidden, self.W2) + self.b2
        expert_out = self.dropout(expert_out)
        gate_train_logits = self._router_logits(x, self.gate_w_train, self.gate_b_train)
        if self.routing_type == "relu_dtsi":
            gate_train = torch.softmax(gate_train_logits, dim=-1)
        elif self.routing_type == "relu":
            gate_train = torch.relu(gate_train_logits)
        else:
            raise ValueError(f"Unsupported routing_type: {self.routing_type}")
        if self.use_dtsi:
            gate_infer_logits = self._router_logits(x, self.gate_w_infer, self.gate_b_infer)
            gate_infer = torch.relu(gate_infer_logits)
        else:
            gate_infer = gate_train
        gate = gate_train if self.training else gate_infer
        output = torch.sum(expert_out * gate.unsqueeze(-1), dim=2)
        if self.l1_coef > 0.0:
            scale = 1.0 / max(self.sparsity_ratio, 1e-6)
            l1_loss = self.l1_coef * scale * gate_infer.sum(dim=-1).mean()
        else:
            l1_loss = x.new_zeros(())
        return output, l1_loss