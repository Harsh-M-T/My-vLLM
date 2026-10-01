from dataclasses import dataclass

import torch
import torch.nn.functional as F

from miniinfer.sequence import Sequence


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    stop_token_ids: list[int] | None = None
    seed: int | None = None


class Sampler:
    def __init__(self, params: SamplingParams):
        self.params = params
        if params.seed is not None:
            torch.manual_seed(params.seed)

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.params.temperature == 0.0:
            return torch.argmax(logits, dim=-1)

        logits = logits / self.params.temperature

        if self.params.top_k > 0:
            top_k = min(self.params.top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1:]
            logits.masked_fill_(indices_to_remove, float("-inf"))

        if self.params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.params.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits.masked_fill_(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def apply_sampling(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        # logits: [1, total_seq_len, vocab_size] or [batch, seq_len, vocab_size]
        if logits.dim() == 3 and logits.shape[0] == 1:
            # Flattened batch: [1, total_seq_len, vocab_size]
            logits = logits.squeeze(0)  # [total_seq_len, vocab_size]
            last_logits = []
            offset = 0
            for seq in seqs:
                seq_len = seq.total_len
                last_logits.append(logits[offset + seq_len - 1])
                offset += seq_len
            last_logits = torch.stack(last_logits)
        else:
            # Batched: [batch, seq_len, vocab_size]
            last_logits = []
            for i, seq in enumerate(seqs):
                seq_len = seq.total_len
                last_logits.append(logits[i, seq_len - 1])
            last_logits = torch.stack(last_logits)
        sampled = self.sample(last_logits)
        return sampled.tolist()