from transformers import AutoTokenizer

from miniinfer.config import EngineConfig, ModelConfig
from miniinfer.engine import Engine
from miniinfer.model import Qwen3Model
from miniinfer.sampler import SamplingParams


def test_e2e_with_tiny_model():
    config = ModelConfig(
        vocab_size=50257,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    
    model = Qwen3Model(config)
    model.eval()
    
    engine_config = EngineConfig(
        model_path="dummy",
        max_num_seqs=4,
        max_model_len=128,
        device="cpu",
        dtype="float32",
    )
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    class TestEngine(Engine):
        def __init__(self, model, tokenizer, config):
            self.config = config
            self.tokenizer = tokenizer
            self.model = model
            from miniinfer.model_runner import ModelRunner
            from miniinfer.scheduler import Scheduler
            self.runner = ModelRunner(config, model)
            self.scheduler = Scheduler(config, None)
            self.seq_counter = 0
    
    engine = TestEngine(model, tokenizer, engine_config)
    params = SamplingParams(temperature=0.0, max_tokens=10, seed=42)
    
    prompts = ["Hello world"]
    outputs = engine.generate(prompts, params)
    
    print(f"Prompt: {prompts[0]}")
    print(f"Output: {outputs[0]['text']}")
    print(f"Tokens: {outputs[0]['token_ids']}")
    print(f"Finish reason: {outputs[0]['finish_reason']}")
    print("E2E test passed!")


if __name__ == "__main__":
    test_e2e_with_tiny_model()