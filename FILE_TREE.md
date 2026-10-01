# mini-infer File Tree & Module Interfaces

```
mini-infer/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── example.py
├── bench.py
├── miniinfer/
│   ├── __init__.py
│   ├── config.py
│   ├── sequence.py
│   ├── block_manager.py
│   ├── scheduler.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── qwen3.py
│   │   └── base.py
│   ├── attention.py
│   ├── model_runner.py
│   ├── sampler.py
│   ├── engine.py
│   └── llm.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_block_manager.py
    ├── test_scheduler.py
    ├── test_sampler.py
    ├── test_e2e_greedy.py
    └── fixtures/
        └── tiny_config.json
```

---

## Module Interfaces

### `config.py`
```python
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
    cuda_graph_batch_sizes: list[int] = None  # e.g., [1, 2, 4, 8, 16, 32, 64]
    max_num_batched_tokens: int = 4096

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
```

### `sequence.py`
```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"

@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0  # for prefix caching

    @property
    def num_prompt_tokens(self) -> int: ...
    @property
    def num_output_tokens(self) -> int: ...
    @property
    def total_len(self) -> int: ...
    def append_token(self, token_id: int) -> None: ...
    def is_finished(self) -> bool: ...
    def get_prompt_len(self) -> int: ...
```

### `block_manager.py`
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    hash: Optional[int] = None  # hash of full block content for prefix caching
    prev_hash: Optional[int] = None  # chained hash for prefix sharing

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True): ...
    
    def allocate(self, seq: Sequence, num_tokens: int) -> list[int]: ...
    def free(self, seq: Sequence) -> None: ...
    def can_allocate(self, num_blocks: int) -> bool: ...
    def get_block_table(self, seq: Sequence) -> list[int]: ...
    def touch_blocks(self, seq: Sequence) -> None: ...  # inc ref count
    def release_blocks(self, seq: Sequence) -> None: ...  # dec ref count
    
    # Prefix caching
    def compute_block_hash(self, block: Block, token_ids: list[int]) -> int: ...
    def find_cached_blocks(self, token_ids: list[int]) -> list[tuple[int, int]]: ...  # [(block_id, num_tokens)]
```

### `scheduler.py`
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SchedulerOutput:
    prefill_seqs: list[Sequence]
    decode_seqs: list[Sequence]
    preempted_seqs: list[Sequence]
    num_batched_tokens: int

class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager): ...
    
    def add_seq(self, seq: Sequence) -> None: ...
    def abort_seq(self, seq_id: int) -> None: ...
    def schedule(self) -> SchedulerOutput: ...
    
    # Internal
    def _schedule_prefill(self) -> tuple[list[Sequence], int]: ...
    def _schedule_decode(self) -> tuple[list[Sequence], int]: ...
    def _preempt(self, seqs: list[Sequence]) -> list[Sequence]: ...
```

### `model/qwen3.py`
```python
import torch
import torch.nn as nn
from miniinfer.model.base import ModelBase
from miniinfer.config import ModelConfig

class Qwen3Model(ModelBase):
    def __init__(self, config: ModelConfig): ...
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...
    
    def load_weights(self, model_path: str) -> None: ...  # from HF safetensors
```

### `model/base.py`
```python
from abc import ABC, abstractmethod
import torch
from miniinfer.config import ModelConfig

class ModelBase(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor: ...
    
    @abstractmethod
    def load_weights(self, model_path: str) -> None: ...
    
    @property
    @abstractmethod
    def config(self) -> ModelConfig: ...
```

### `attention.py`
```python
import torch
from typing import Optional, Tuple

def attention_with_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_tables: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seq_len: Optional[int] = None,
) -> torch.Tensor: ...
    # Uses flash_attn_with_kvcache if available, else SDPA fallback

def attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor: ...
    # Uses flash_attn_varlen_func if available, else SDPA fallback
```

### `model_runner.py`
```python
import torch
from typing import Optional
from miniinfer.sequence import Sequence
from miniinfer.model.base import ModelBase
from miniinfer.config import EngineConfig

class ModelRunner:
    def __init__(self, config: EngineConfig, model: ModelBase): ...
    
    def prepare_inputs(self, seqs: list[Sequence]) -> dict: ...
        # Returns: input_ids, positions, cu_seqlens, max_seq_len, block_tables, slot_mapping
    
    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> torch.Tensor: ...  # logits [total_tokens, vocab_size]
    
    def capture_cuda_graphs(self, batch_sizes: list[int]) -> None: ...
    def run_cuda_graph(self, batch_size: int, input_ids: torch.Tensor, ...) -> torch.Tensor: ...
```

### `sampler.py`
```python
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    stop_token_ids: Optional[list[int]] = None
    seed: Optional[int] = None

class Sampler:
    def __init__(self, params: SamplingParams): ...
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor: ...
        # Returns sampled token_ids [batch_size]
    
    def apply_sampling(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]: ...
```

### `engine.py`
```python
from typing import Optional
from miniinfer.sequence import Sequence
from miniinfer.scheduler import Scheduler
from miniinfer.model_runner import ModelRunner
from miniinfer.sampler import Sampler, SamplingParams
from miniinfer.config import EngineConfig
from transformers import AutoTokenizer

class Engine:
    def __init__(self, config: EngineConfig): ...
    
    def step(self) -> list[Sequence]: ...
        # One scheduler + runner + sampler iteration
    
    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[dict]: ...
        # Returns list of {"text": str, "token_ids": list[int], "finish_reason": str}
    
    def _create_sequences(self, prompts: list[str], sampling_params: SamplingParams) -> list[Sequence]: ...
    def _check_stop(self, seq: Sequence, new_token: int) -> bool: ...
```

### `llm.py`
```python
from miniinfer.engine import Engine
from miniinfer.config import EngineConfig
from miniinfer.sampler import SamplingParams

class LLM:
    def __init__(
        self,
        model: str,
        max_num_seqs: int = 64,
        max_model_len: int = 2048,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.9,
        dtype: str = "float16",
        enable_prefix_caching: bool = True,
        enable_cuda_graph: bool = True,
    ): ...
    
    def generate(
        self,
        prompts: list[str] | str,
        sampling_params: SamplingParams,
    ) -> list[dict]: ...
```