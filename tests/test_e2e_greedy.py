import pytest
import torch

from miniinfer.model import Qwen3Model


@pytest.fixture
def tiny_model(tiny_config):
    model = Qwen3Model(tiny_config)
    model.eval()
    return model


def test_model_forward_shape(tiny_model, tiny_config):
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    kv_caches = []
    for _ in range(tiny_config.num_layers):
        k = torch.zeros(batch_size, tiny_config.num_kv_heads, seq_len, tiny_config.head_dim)
        v = torch.zeros_like(k)
        kv_caches.append((k, v))

    logits = tiny_model.forward(input_ids, positions, kv_caches)
    assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)


def test_model_deterministic(tiny_model, tiny_config):
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    positions = torch.tensor([[0, 1, 2, 3, 4]])
    kv_caches = []
    for _ in range(tiny_config.num_layers):
        k = torch.zeros(1, tiny_config.num_kv_heads, 5, tiny_config.head_dim)
        v = torch.zeros_like(k)
        kv_caches.append((k, v))

    logits1 = tiny_model.forward(input_ids, positions, kv_caches)
    logits2 = tiny_model.forward(input_ids, positions, kv_caches)
    assert torch.allclose(logits1, logits2)


@pytest.mark.gpu
@pytest.mark.model
def test_e2e_greedy_vs_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from miniinfer.llm import LLM
    from miniinfer.sampler import SamplingParams

    model_path = "Qwen/Qwen3-0.6B"
    prompts = [
        "Hello, how are you?",
        "The capital of France is",
        "Write a short poem about AI.",
        "1 + 1 =",
        "Translate to French: Hello world",
    ]

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)

    llm = LLM(model_path, max_num_seqs=4, max_model_len=256, dtype="float16")
    params = SamplingParams(temperature=0.0, max_tokens=32)

    hf_outputs = []
    for prompt in prompts:
        inputs = hf_tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        out = hf_model.generate(**inputs, max_new_tokens=32, do_sample=False)
        hf_outputs.append(
            hf_tokenizer.decode(
                out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
        )

    mini_outputs = llm.generate(prompts, params)
    mini_texts = [o["text"] for o in mini_outputs]

    for hf_text, mini_text in zip(hf_outputs, mini_texts):
        assert hf_text.strip() == mini_text.strip(), f"Mismatch:\nHF: {hf_text}\nMini: {mini_text}"