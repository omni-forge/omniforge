#!/usr/bin/env python3
"""OmniForge decoder-only transformer implemented in PyTorch.

Source: ZIP1 (primary) with improvements from ZIP2/ZIP4 (GQA support).
Includes: RMSNorm, RoPE, Flash Attention, SwiGLU, weight tying, generate.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as project_config


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    n_layers: int
    n_heads: int
    hidden_dim: int
    ffn_dim: int
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    rope_base: float = 10000.0
    use_flash_attention: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    eod_token_id: int = 4
    n_kv_heads: Optional[int] = None

    def __post_init__(self) -> None:
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normed


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        cos = self.cos_cached[:, :, start_pos: start_pos + seq_len, :].to(device=q.device, dtype=q.dtype)
        sin = self.sin_cached[:, :, start_pos: start_pos + seq_len, :].to(device=q.device, dtype=q.dtype)
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.hidden_dim = cfg.hidden_dim
        self.head_dim = cfg.hidden_dim // cfg.n_heads
        self.use_flash_attention = cfg.use_flash_attention and hasattr(F, "scaled_dot_product_attention")

        # Separate Q, K, V projections for GQA support
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim, cfg.context_length, cfg.rope_base)
        causal_mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, cfg.context_length, cfg.context_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Repeat K/V heads if using GQA (n_kv_heads < n_heads)
        if self.n_kv_heads < self.n_heads:
            n_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_repeat, dim=1)
            v = v.repeat_interleave(n_repeat, dim=1)

        if self.use_flash_attention:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attn = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
            attn = self.attn_dropout(attn)
            y = attn @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.resid_dropout(self.out_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.hidden_dim, bias=False)
        self.w3 = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_dim, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_dim, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class OmniForge(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.hidden_dim, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        # Weight tying: embed and lm_head share weights
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @classmethod
    def from_config(cls) -> "OmniForge":
        cfg = ModelConfig(
            vocab_size=project_config.VOCAB_SIZE,
            context_length=project_config.CONTEXT_LENGTH,
            n_layers=project_config.N_LAYERS,
            n_heads=project_config.N_HEADS,
            hidden_dim=project_config.HIDDEN_DIM,
            ffn_dim=project_config.FFN_DIM,
            dropout=project_config.DROPOUT,
            rms_norm_eps=project_config.RMS_NORM_EPS,
            rope_base=project_config.ROPE_BASE,
            use_flash_attention=project_config.USE_FLASH_ATTENTION,
            pad_token_id=project_config.PAD_TOKEN_ID,
            bos_token_id=project_config.BOS_TOKEN_ID,
            eos_token_id=project_config.EOS_TOKEN_ID,
            eod_token_id=project_config.EOD_TOKEN_ID,
        )
        return cls(cfg)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[model] Total parameters: {total:,}")
        print(f"[model] Trainable parameters: {trainable:,}")
        return total, trainable

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.cfg.context_length:
            raise ValueError(f"Sequence length {seq_len} exceeds context length {self.cfg.context_length}")
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.cfg.context_length:]
            logits = self(idx_cond)[:, -1, :]
            if not do_sample:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                temperature = max(float(temperature), 1e-6)
                logits = logits / temperature
                if top_k and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    logits = logits.masked_fill(logits < values[:, [-1]], torch.finfo(logits.dtype).min)
                if top_p and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative = torch.cumsum(probs, dim=-1)
                    sorted_mask = cumulative > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(sorted_mask, torch.finfo(sorted_logits.dtype).min)
                    logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
                    logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids


def main() -> None:
    model = OmniForge.from_config()
    model.count_parameters()


if __name__ == "__main__":
    main()