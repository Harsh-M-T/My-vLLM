from abc import ABC, abstractmethod

import torch

from miniinfer.config import ModelConfig


class ModelBase(ABC):
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
        block_tables: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def load_weights(self, model_path: str) -> None:
        pass

    @property
    @abstractmethod
    def config(self) -> ModelConfig:
        pass