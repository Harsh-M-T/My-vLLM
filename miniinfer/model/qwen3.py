import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from miniinfer.config import ModelConfig
from miniinfer.model.base import ModelBase


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q: [bsz, num_heads, seq_len, head_dim]
    # k: [bsz, num_kv_heads, seq_len, head_dim]
    # cos/sin: [bsz, seq_len, head_dim] or [seq_len, head_dim]
    # Need to expand cos/sin to match q and k shapes
    cos = cos.unsqueeze(1)  # [bsz, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # hidden_states: [bsz, num_kv_heads, seq_len, head_dim]
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = positions.max().item() + 1
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(x.dtype)
            self._sin_cached = emb.sin().to(x.dtype)
        cos = self._cos_cached[positions]
        sin = self._sin_cached[positions]
        return cos, sin


class Qwen3MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_heads_per_head = config.num_kv_heads_per_head
        self.scaling = self.head_dim**-0.5
        self.rotary_emb = rotary_emb

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_heads * config.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_heads * config.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(hidden_states, positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_cache, v_cache = kv_cache
        k_cache[:, :, :q_len] = k
        v_cache[:, :, :q_len] = v

        # Repeat k/v for GQA
        k = repeat_kv(k, self.num_kv_heads_per_head)
        v = repeat_kv(v, self.num_kv_heads_per_head)
        k_cache_full = repeat_kv(k_cache, self.num_kv_heads_per_head)
        v_cache_full = repeat_kv(v_cache, self.num_kv_heads_per_head)

        output = self._flash_attn_with_kvcache(q, k, v, k_cache_full, v_cache_full)

        output = output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(output)

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        try:
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seq_len,
                max_seqlen_k=max_seq_len,
                softmax_scale=self.scaling,
                causal=True,
            )
        except ImportError:
            return self._sdpa_varlen(q, k, v, cu_seqlens, max_seq_len)

    def _flash_attn_with_kvcache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        try:
            from flash_attn import flash_attn_with_kvcache
            return flash_attn_with_kvcache(q, k_cache, v_cache, k, v, causal=True)
        except ImportError:
            return self._sdpa_with_kvcache(q, k, v, k_cache, v_cache)

    def _sdpa_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        batch_size = cu_seqlens.shape[0] - 1
        output = torch.empty_like(q)
        for i in range(batch_size):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            q_i = q[:, start:end]
            k_i = k[:, start:end]
            v_i = v[:, start:end]
            attn = torch.matmul(q_i, k_i.transpose(-2, -1)) * self.scaling
            mask = torch.triu(torch.ones_like(attn), diagonal=1).bool()
            attn.masked_fill_(mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            output[:, start:end] = torch.matmul(attn, v_i)
        return output

    def _sdpa_with_kvcache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_heads, q_len, head_dim = q.shape
        _, _, kv_len, _ = k_cache.shape
        k_full = k_cache[:, :, :kv_len]
        v_full = v_cache[:, :, :kv_len]
        attn = torch.matmul(q, k_full.transpose(-2, -1)) * self.scaling
        mask = torch.triu(
            torch.ones(q_len, kv_len, device=q.device),
            diagonal=kv_len - q_len + 1,
        ).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v_full)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx, rotary_emb)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, kv_cache, cu_seqlens, max_seq_len)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nn.Module, ModelBase):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self._config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, i, self.rotary_emb) for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @property
    def config(self) -> ModelConfig:
        return self._config

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
        block_tables: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                positions,
                kv_caches[i],
                cu_seqlens,
                max_seq_len,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, model_path: str) -> None:
        weight_map = {}
        shards = [
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        for shard in shards:
            try:
                weights = load_file(f"{model_path}/{shard}")
                weight_map.update(weights)
            except FileNotFoundError:
                continue

        state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in weight_map:
                param.data.copy_(weight_map[name])
            else:
                hf_name = name.replace("layers.", "model.layers.")
                if hf_name in weight_map:
                    param.data.copy_(weight_map[hf_name])

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda", dtype: str = "float16"):
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config = ModelConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            rope_theta=hf_config.rope_theta,
            rms_norm_eps=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
        )
        model = cls(config)
        model.load_weights(model_path)
        model.to(device=device, dtype=getattr(torch, dtype))
        model.eval()
        return model