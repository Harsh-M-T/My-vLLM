from miniinfer.config import EngineConfig
from miniinfer.engine import Engine
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
    ):
        config = EngineConfig(
            model_path=model,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            enable_cuda_graph=enable_cuda_graph,
        )
        self.engine = Engine(config)

    def generate(
        self,
        prompts: list[str] | str,
        sampling_params: SamplingParams,
    ) -> list[dict]:
        if isinstance(prompts, str):
            prompts = [prompts]
        return self.engine.generate(prompts, sampling_params)