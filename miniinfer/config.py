from dataclasses import dataclass


@dataclass
class EngineConfig:
    model_path: str
    block_size: int = 16
    max_num_seqs: int = 64
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    dtype: str = "float16"
    device: str = "cuda"
    enable_prefix_caching: bool = True
    enable_cuda_graph: bool = True
    cuda_graph_batch_sizes: list[int] | None = None
    max_num_batched_tokens: int = 4096

    def __post_init__(self):
        if self.cuda_graph_batch_sizes is None:
            self.cuda_graph_batch_sizes = [1, 2, 4, 8, 16, 32, 64]


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool

    @property
    def num_kv_heads_per_head(self) -> int:
        return self.num_heads // self.num_kv_heads