
import torch

from miniinfer.config import EngineConfig
from miniinfer.model.base import ModelBase
from miniinfer.sequence import Sequence


class ModelRunner:
    def __init__(self, config: EngineConfig, model: ModelBase):
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._init_kv_caches()

    def _init_kv_caches(self):
        model_config = self.model.config
        max_blocks = (
            self.config.max_model_len + self.config.block_size - 1
        ) // self.config.block_size
        max_tokens = max_blocks * self.config.block_size
        for _ in range(model_config.num_layers):
            k_cache = torch.zeros(
                (1, model_config.num_kv_heads, max_tokens, model_config.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            v_cache = torch.zeros_like(k_cache)
            self.kv_caches.append((k_cache, v_cache))

    def prepare_inputs(self, seqs: list[Sequence]) -> dict:
        max_len = max(s.total_len for s in seqs)
        batch_size = len(seqs)
        
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        positions = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        
        for i, seq in enumerate(seqs):
            seq_len = seq.total_len
            token_ids = seq.prompt_token_ids + seq.output_token_ids
            input_ids[i, :seq_len] = torch.tensor(token_ids, dtype=torch.long, device=self.device)
            positions[i, :seq_len] = torch.arange(
                seq.num_cached_tokens,
                seq.num_cached_tokens + seq_len,
                device=self.device,
            )

        return {
            "input_ids": input_ids,
            "positions": positions,
            "cu_seqlens": None,
            "max_seq_len": max_len,
        }

    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> torch.Tensor:
        all_logits = []
        for seq in seqs:
            inputs = self.prepare_inputs([seq])
            logits = self.model.forward(
                input_ids=inputs["input_ids"],
                positions=inputs["positions"],
                kv_caches=self.kv_caches,
                cu_seqlens=inputs["cu_seqlens"],
                max_seq_len=inputs["max_seq_len"],
            )
            all_logits.append(logits)
        return torch.cat(all_logits, dim=1)

    def capture_cuda_graphs(self, batch_sizes: list[int]) -> None:
        pass

    def run_cuda_graph(self, batch_size: int, **kwargs) -> torch.Tensor:
        return self.run(kwargs["seqs"], kwargs["is_prefill"])