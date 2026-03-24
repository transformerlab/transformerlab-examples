"""
Litesearch — pretraining script for consumer GPUs.
Forked from Karpathy's autoresearch, optimized for 2GB–32GB+ VRAM.

Usage:
    python train.py                         # train for 5 minutes
    python train.py --export model.pth      # train then export model
    python train.py --export-dir exports/   # export to directory (auto-named)
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb


def detect_device_and_dtype():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Litesearch requires an NVIDIA GPU.")
    cap = torch.cuda.get_device_capability()
    use_bf16 = cap >= (7, 5)  # Turing+
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    print(f"GPU: {gpu_name} ({vram_mb:.0f} MB)")
    print(f"Compute capability: {cap[0]}.{cap[1]}")
    print(f"Dtype: {'bfloat16' if use_bf16 else 'float32 (Pascal fallback)'}")
    return device, use_bf16, dtype, cap, vram_mb


PEAK_FLOPS_TABLE = {
    ((6, 1), "fp32"): 11.3,  # GTX 1080 Ti
    ((7, 0), "fp32"): 14.0,  # Tesla V100
    ((7, 5), "bf16"): 23.0,  # RTX 2080 Ti
    ((7, 5), "fp32"): 11.5,
    ((8, 0), "bf16"): 312.0,  # A100
    ((8, 0), "fp32"): 19.5,
    ((8, 6), "bf16"): 130.0,  # RTX 3090
    ((8, 6), "fp32"): 35.6,
    ((8, 9), "bf16"): 165.0,  # RTX 4090
    ((8, 9), "fp32"): 82.6,
    ((9, 0), "bf16"): 989.5,  # H100
    ((9, 0), "fp32"): 67.0,
}


def get_peak_flops(cap, use_bf16):
    dtype_str = "bf16" if use_bf16 else "fp32"
    key = (cap, dtype_str)
    if key in PEAK_FLOPS_TABLE:
        return PEAK_FLOPS_TABLE[key] * 1e12
    if cap[0] >= 8:
        return 50.0e12
    return 15.0e12


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x, ve, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin):
        x = x + self.attn(norm(x), ve, cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self, dtype=torch.bfloat16):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(
                    block.attn.ve_gate.weight
                )  # sigmoid(0)*2 = 1.0 = neutral
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=dtype)
        for ve in self.value_embeds.values():
            ve.to(dtype=dtype)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = self.config.n_layer * 12 * h * q * t
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
        use_bf16=True,
    ):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
            + len(value_embeds_params)
            + len(resid_params)
            + len(x0_params)
        )
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(
                kind="adamw",
                params=lm_head_params,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=embedding_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=value_embeds_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=resid_params,
                lr=scalar_lr * 0.01,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=x0_params,
                lr=scalar_lr,
                betas=(0.96, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=weight_decay,
                )
            )
        optimizer = MuonAdamW(param_groups, use_bf16=use_bf16)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T].to(idx.device), self.sin[:, :T].to(idx.device)

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = gradient_checkpoint(block, x, ve, cos_sin, use_reentrant=False)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        return logits


polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(
    p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t
):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum_t,
    lr_t,
    wd_t,
    beta2_t,
    ns_steps,
    red_dim,
    use_bf16=True,
):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16() if use_bf16 else g.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(
        v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2
    )
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Muon for 2D matrix params, AdamW for everything else."""

    def __init__(self, param_groups, use_bf16=True):
        super().__init__(param_groups, defaults={})
        self.use_bf16 = use_bf16
        # 0-D CPU tensors to avoid torch.compile recompilation
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                num_params, *shape, dtype=dtype, device=device
            )
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(
                state_shape, dtype=dtype, device=device
            )
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
            self.use_bf16,
        )
        for param, updated in zip(params, stacked_params.unbind(0)):
            param.data.copy_(updated.data)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            try:
                if group["kind"] == "adamw":
                    self._step_adamw(group)
                elif group["kind"] == "muon":
                    self._step_muon(group)
            except Exception as e:
                print(f"Optimizer step error ({group['kind']}): {e}")
                raise


ASPECT_RATIO = 64
HEAD_DIM = 128

_CONFIG_CANDIDATES = [
    (24, None),
    (20, None),
    (16, None),
    (14, None),
    (12, None),
    (10, None),
    (8, None),
    (6, None),
    (5, None),
    (4, None),
    (3, None),
    (2, None),
]


def estimate_model_memory_mb(depth, n_embd, vocab_size, use_bf16):
    n_head = n_embd // HEAD_DIM
    n_kv_head = n_head
    attn_params = depth * 4 * n_embd * n_embd
    mlp_params = depth * 2 * n_embd * (4 * n_embd)
    embed_params = 2 * vocab_size * n_embd
    ve_params = (depth // 2) * vocab_size * n_kv_head * HEAD_DIM
    gate_params = (depth // 2) * 32 * n_kv_head
    scalar_params = 2 * depth
    total_params = (
        attn_params
        + mlp_params
        + embed_params
        + ve_params
        + gate_params
        + scalar_params
    )
    param_bytes = 2 if use_bf16 else 4
    model_mb = total_params * param_bytes / 1e6
    grad_mb = total_params * param_bytes / 1e6
    opt_mb = total_params * 5 / 1e6
    return model_mb + grad_mb + opt_mb, total_params


def compute_optimal_config(vram_mb, use_bf16, vocab_size):
    # Reserve 800MB for desktop/OS, use 50% of remainder
    # Actual peak VRAM ~1.8x estimate due to CUDA allocator + fragmentation
    budget = (vram_mb - 800) * 0.50
    budget = max(budget, 400)

    for depth, _ in _CONFIG_CANDIDATES:
        base_dim = depth * ASPECT_RATIO
        n_embd = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
        n_head = n_embd // HEAD_DIM

        model_total_mb, total_params = estimate_model_memory_mb(
            depth, n_embd, vocab_size, use_bf16
        )

        act_budget = budget - model_total_mb
        if act_budget < 64:
            continue

        for T in [2048, 1024, 768, 512, 384, 256]:
            for B in [64, 32, 16, 8, 4, 2, 1]:
                logits_mb = B * T * vocab_size * 4 / 1e6
                act_bytes = 2 if use_bf16 else 4
                layer_act_mb = B * T * n_embd * act_bytes / 1e6
                total_act_mb = logits_mb + layer_act_mb + 50

                if total_act_mb < act_budget:
                    tokens_per_step = B * T
                    total_batch = max(2**14, tokens_per_step)
                    total_batch = (
                        (total_batch + tokens_per_step - 1) // tokens_per_step
                    ) * tokens_per_step
                    est = model_total_mb + total_act_mb
                    pm = total_params / 1e6
                    print(
                        f"Auto-config: depth={depth}, n_embd={n_embd}, B={B}, T={T}, "
                        f"params={pm:.1f}M, est={est:.0f}MB / {vram_mb:.0f}MB"
                    )
                    return {
                        "depth": depth,
                        "n_embd": n_embd,
                        "n_head": n_head,
                        "n_kv_head": n_head,
                        "device_batch_size": B,
                        "max_seq_len": T,
                        "total_batch_size": total_batch,
                        "estimated_vram_mb": est,
                    }

    print("WARNING: Using absolute minimum config (2 layers, 128 dim, B=1, T=256)")
    return {
        "depth": 2,
        "n_embd": 128,
        "n_head": 1,
        "n_kv_head": 1,
        "device_batch_size": 1,
        "max_seq_len": 256,
        "total_batch_size": 2**14,
        "estimated_vram_mb": 300,
    }


TOTAL_BATCH_SIZE = 2**19
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0
DEPTH = 8
DEVICE_BATCH_SIZE = 128


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)


def run_training(config=None, lr_override=None, log_queue=None, stop_event=None):
    def log(msg, end="\n"):
        if log_queue is not None:
            log_queue.put(msg + end)
        else:
            print(msg, end=end, flush=True)

    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    device, use_bf16, dtype, cap, vram_total_mb = detect_device_and_dtype()
    peak_flops = get_peak_flops(cap, use_bf16)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if use_bf16
        else torch.autocast("cuda", enabled=False)
    )

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    log(f"Vocab size: {vocab_size:,}")

    # If user didn't pass a config, auto-compute one from GPU VRAM and vocab size.
    if config is None:
        try:
            auto_cfg = compute_optimal_config(vram_total_mb, use_bf16, vocab_size)
            log("Auto-config computed from device VRAM")
            log(
                f"  depth={auto_cfg['depth']}, n_embd={auto_cfg['n_embd']}, device_B={auto_cfg['device_batch_size']}, T={auto_cfg['max_seq_len']}"
            )
            config = auto_cfg
        except Exception as e:
            log(f"Auto-config failed ({e}), falling back to defaults")

    depth = config["depth"]
    n_embd = config["n_embd"]
    n_head = config.get("n_head", n_embd // HEAD_DIM)
    n_kv_head = config.get("n_kv_head", n_head)
    device_batch_size = config["device_batch_size"]
    max_seq_len = config["max_seq_len"]
    total_batch_size = config.get("total_batch_size", TOTAL_BATCH_SIZE)

    matrix_lr = lr_override if lr_override is not None else MATRIX_LR

    model_config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
    )
    log(f"Model config: {asdict(model_config)}")

    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights(dtype=dtype)

    param_counts = model.num_scaling_params()
    log("Parameter counts:")
    for key, value in param_counts.items():
        log(f"  {key:24s}: {value:,}")
    num_params = param_counts["total"]
    num_flops_per_token = model.estimate_flops()
    log(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = device_batch_size * max_seq_len
    if total_batch_size % tokens_per_fwdbwd != 0:
        total_batch_size = (
            (total_batch_size + tokens_per_fwdbwd - 1) // tokens_per_fwdbwd
        ) * tokens_per_fwdbwd
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=matrix_lr,
        weight_decay=WEIGHT_DECAY,
        use_bf16=use_bf16,
    )

    if cap[0] >= 7:
        try:
            model = torch.compile(model, dynamic=False)
            log("torch.compile enabled")
        except Exception as e:
            log(f"torch.compile failed ({e}), running eager mode")
    else:
        log("torch.compile skipped (GPU too old for Triton)")

    from prepare import make_dataloader as _make_dataloader

    train_loader = _make_dataloader(tokenizer, device_batch_size, max_seq_len, "train")
    x, y, epoch = next(train_loader)

    log(f"Time budget: {TIME_BUDGET}s")
    log(f"Gradient accumulation steps: {grad_accum_steps}")
    log(f"Total batch size: {total_batch_size:,} tokens")
    log(f"Peak FLOPs: {peak_flops / 1e12:.1f} TFLOPS")
    if not use_bf16:
        log("NOTE: Running in fp32 mode (Pascal GPU). Expect ~2x memory vs bfloat16.")
    log("")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0
    crashed = False

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                log("\nTraining stopped by user.")
                break

            torch.cuda.synchronize()
            t0 = time.time()
            for micro_step in range(grad_accum_steps):
                with autocast_ctx:
                    loss = model(x, y)
                train_loss = loss.detach()
                loss = loss / grad_accum_steps
                loss.backward()
                x, y, epoch = next(train_loader)

            progress = min(total_training_time / TIME_BUDGET, 1.0)
            lrm = get_lr_multiplier(progress)
            muon_momentum = get_muon_momentum(step)
            muon_weight_decay = get_weight_decay(progress)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group["kind"] == "muon":
                    group["momentum"] = muon_momentum
                    group["weight_decay"] = muon_weight_decay
            optimizer.step()
            model.zero_grad(set_to_none=True)

            train_loss_f = train_loss.item()

            if math.isnan(train_loss_f) or train_loss_f > 100:
                log("FAIL — loss exploding")
                crashed = True
                break

            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0

            if step > 10:
                total_training_time += dt

            ema_beta = 0.9
            smooth_train_loss = (
                ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            )
            debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
            pct_done = 100 * progress
            tok_per_sec = int(total_batch_size / dt)
            mfu = 100 * num_flops_per_token * total_batch_size / dt / peak_flops
            remaining = max(0, TIME_BUDGET - total_training_time)
            current_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            log(
                f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | vram: {current_vram_mb:.0f}MB | remaining: {remaining:.0f}s"
            )

            # GC causes ~500ms stalls, so disable after first step
            if step == 0:
                gc.collect()
                gc.freeze()
                gc.disable()
            elif (step + 1) % 5000 == 0:
                gc.collect()

            step += 1

            if step > 10 and total_training_time >= TIME_BUDGET:
                break

    except Exception as e:
        log(f"\nTraining crashed: {e}")
        crashed = True

    log("")

    total_tokens = step * total_batch_size

    val_bpb = 0.0
    if not crashed and step > 10:
        try:
            model.eval()
            with autocast_ctx:
                import prepare

                eval_steps = max(30, min(int(vram_total_mb / 100), 100))
                cap_tokens = eval_steps * device_batch_size * max_seq_len
                old_eval = prepare.EVAL_TOKENS
                prepare.EVAL_TOKENS = min(old_eval, cap_tokens)
                log(f"Evaluating ({eval_steps} steps)...")
                val_bpb = evaluate_bpb(model, tokenizer, device_batch_size)
                prepare.EVAL_TOKENS = old_eval
        except Exception as e:
            log(f"Evaluation failed: {e}")
            crashed = True

    t_end = time.time()
    steady_state_mfu = (
        100
        * num_flops_per_token
        * total_batch_size
        * max(0, step - 10)
        / max(total_training_time, 1e-6)
        / peak_flops
        if total_training_time > 0
        else 0
    )
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    log("---")
    log(f"val_bpb:          {val_bpb:.6f}")
    log(f"training_seconds: {total_training_time:.1f}")
    log(f"total_seconds:    {t_end - t_start:.1f}")
    log(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    log(f"mfu_percent:      {steady_state_mfu:.2f}")
    log(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    log(f"num_steps:        {step}")
    log(f"num_params_M:     {num_params / 1e6:.1f}")
    log(f"depth:            {depth}")
    log(f"dtype:            {'bf16' if use_bf16 else 'fp32'}")

    return {
        "val_bpb": val_bpb,
        "training_seconds": total_training_time,
        "total_seconds": t_end - t_start,
        "peak_vram_mb": peak_vram_mb,
        "mfu_percent": steady_state_mfu,
        "total_tokens_M": total_tokens / 1e6,
        "num_steps": step,
        "num_params_M": num_params / 1e6,
        "depth": depth,
        "n_embd": n_embd,
        "crashed": crashed,
        "model": model,
        "config": model_config,
        "use_bf16": use_bf16,
        "device_batch_size": device_batch_size,
        "total_batch_size": total_batch_size,
    }


def continue_training(prev_result, lr_override=None, log_queue=None, stop_event=None):
    def log(msg, end="\n"):
        if log_queue is not None:
            log_queue.put(msg + end)
        else:
            print(msg, end=end, flush=True)

    model = prev_result["model"]
    model_config = prev_result["config"]
    use_bf16 = prev_result["use_bf16"]
    device = next(model.parameters()).device
    cap = torch.cuda.get_device_capability()
    peak_flops = get_peak_flops(cap, use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if use_bf16
        else torch.autocast("cuda", enabled=False)
    )

    depth = model_config.n_layer
    n_embd = model_config.n_embd
    max_seq_len = model_config.sequence_len
    device_batch_size = prev_result.get("device_batch_size", DEVICE_BATCH_SIZE)
    total_batch_size = prev_result.get("total_batch_size", TOTAL_BATCH_SIZE)

    tokenizer = Tokenizer.from_directory()
    num_flops_per_token = model.estimate_flops()

    tokens_per_fwdbwd = device_batch_size * max_seq_len
    if total_batch_size % tokens_per_fwdbwd != 0:
        total_batch_size = (
            (total_batch_size + tokens_per_fwdbwd - 1) // tokens_per_fwdbwd
        ) * tokens_per_fwdbwd
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=lr_override if lr_override is not None else MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
        use_bf16=use_bf16,
    )

    from prepare import make_dataloader as _make_dataloader

    train_loader = _make_dataloader(tokenizer, device_batch_size, max_seq_len, "train")
    x, y, epoch = next(train_loader)

    log(f"Continuing training for {TIME_BUDGET}s...")
    log(
        f"Model: depth={depth}, d={n_embd}, params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )
    log("")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t_start = time.time()
    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0
    crashed = False
    num_params = sum(p.numel() for p in model.parameters())

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                log("\nTraining stopped by user.")
                break

            torch.cuda.synchronize()
            t0 = time.time()
            for micro_step in range(grad_accum_steps):
                with autocast_ctx:
                    loss = model(x, y)
                train_loss = loss.detach()
                loss = loss / grad_accum_steps
                loss.backward()
                x, y, epoch = next(train_loader)

            progress = min(total_training_time / TIME_BUDGET, 1.0)
            lrm = get_lr_multiplier(progress)
            muon_momentum = get_muon_momentum(step)
            muon_weight_decay = get_weight_decay(progress)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group["kind"] == "muon":
                    group["momentum"] = muon_momentum
                    group["weight_decay"] = muon_weight_decay
            optimizer.step()
            model.zero_grad(set_to_none=True)

            train_loss_f = train_loss.item()

            if math.isnan(train_loss_f) or train_loss_f > 100:
                log("FAIL — loss exploding")
                crashed = True
                break

            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0

            if step > 10:
                total_training_time += dt

            ema_beta = 0.9
            smooth_train_loss = (
                ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            )
            debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
            pct_done = 100 * progress
            tok_per_sec = int(total_batch_size / dt)
            mfu = 100 * num_flops_per_token * total_batch_size / dt / peak_flops
            remaining = max(0, TIME_BUDGET - total_training_time)
            current_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            log(
                f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | vram: {current_vram_mb:.0f}MB | remaining: {remaining:.0f}s"
            )

            if step == 0:
                gc.collect()
                gc.freeze()
                gc.disable()
            elif (step + 1) % 5000 == 0:
                gc.collect()

            step += 1

            if step > 10 and total_training_time >= TIME_BUDGET:
                break

    except Exception as e:
        log(f"\nTraining crashed: {e}")
        crashed = True

    log("")

    total_tokens = step * total_batch_size

    val_bpb = 0.0
    if not crashed and step > 10:
        try:
            model.eval()
            with autocast_ctx:
                import prepare

                eval_steps = max(
                    30,
                    min(
                        int(
                            torch.cuda.get_device_properties(0).total_mem
                            / 1024
                            / 1024
                            / 100
                        ),
                        100,
                    ),
                )
                cap_tokens = eval_steps * device_batch_size * max_seq_len
                old_eval = prepare.EVAL_TOKENS
                prepare.EVAL_TOKENS = min(old_eval, cap_tokens)
                log(f"Evaluating ({eval_steps} steps)...")
                val_bpb = evaluate_bpb(model, tokenizer, device_batch_size)
                prepare.EVAL_TOKENS = old_eval
        except Exception as e:
            log(f"Evaluation failed: {e}")
            crashed = True

    t_end = time.time()
    steady_state_mfu = (
        100
        * num_flops_per_token
        * total_batch_size
        * max(0, step - 10)
        / max(total_training_time, 1e-6)
        / peak_flops
        if total_training_time > 0
        else 0
    )
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    log("---")
    log(f"val_bpb:          {val_bpb:.6f}")
    log(f"training_seconds: {total_training_time:.1f}")
    log(f"total_seconds:    {t_end - t_start:.1f}")
    log(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    log(f"mfu_percent:      {steady_state_mfu:.2f}")
    log(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    log(f"num_steps:        {step}")
    log(f"num_params_M:     {num_params / 1e6:.1f}")
    log(f"depth:            {depth}")
    log(f"dtype:            {'bf16' if use_bf16 else 'fp32'}")

    return {
        "val_bpb": val_bpb,
        "training_seconds": total_training_time,
        "total_seconds": t_end - t_start,
        "peak_vram_mb": peak_vram_mb,
        "mfu_percent": steady_state_mfu,
        "total_tokens_M": total_tokens / 1e6,
        "num_steps": step,
        "num_params_M": num_params / 1e6,
        "depth": depth,
        "n_embd": n_embd,
        "crashed": crashed,
        "model": model,
        "config": model_config,
        "use_bf16": use_bf16,
        "device_batch_size": device_batch_size,
        "total_batch_size": total_batch_size,
    }


def export_model(result, output_path):
    if result["crashed"]:
        print("Cannot export crashed model")
        return False
    try:
        torch.save(
            {
                "state_dict": result["model"].state_dict(),
                "config": asdict(result["config"]),
                "use_bf16": result["use_bf16"],
            },
            output_path,
        )
        print(f"Saved to {output_path}")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def generate(model, tokenizer, prompt, max_tokens=128, temperature=0.7, top_p=0.9):
    gc.enable()
    gc.collect()

    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_tokens):
            if input_ids.size(1) > model.config.sequence_len:
                input_ids = input_ids[:, -model.config.sequence_len :]
            logits = model(input_ids[:, -model.config.sequence_len :])
            logits = logits[:, -1, :] / temperature
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            next_id = sorted_idx.gather(-1, next_idx)
            if next_id.item() == tokenizer.get_bos_token_id():
                break
            input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export", type=str, default=None, help="Export model to path after training"
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Export to directory with auto-named file",
    )
    args = parser.parse_args()

    result = run_training()

    if args.export:
        export_model(result, args.export)
    elif args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(args.export_dir, f"litesearch_{ts}.pth")
        export_model(result, path)

    if result["crashed"]:
        exit(1)
