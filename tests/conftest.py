import pytest
import torch

from miniinfer.config import ModelConfig


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=1000,
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


@pytest.fixture
def device():
    return torch.device("cpu")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "model: mark test as requiring model weights")
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="GPU not available"))