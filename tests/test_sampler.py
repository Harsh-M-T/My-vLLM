import torch

from miniinfer.sampler import Sampler, SamplingParams
from miniinfer.sequence import Sequence


def test_sampler_greedy():
    params = SamplingParams(temperature=0.0, seed=42)
    sampler = Sampler(params)

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    token = sampler.sample(logits)
    assert token.item() == 3


def test_sampler_temperature():
    params = SamplingParams(temperature=1.0, seed=42)
    sampler = Sampler(params)

    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    tokens = [sampler.sample(logits).item() for _ in range(100)]
    assert len(set(tokens)) > 1


def test_sampler_top_k():
    params = SamplingParams(temperature=1.0, top_k=2, seed=42)
    sampler = Sampler(params)

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    tokens = [sampler.sample(logits).item() for _ in range(50)]
    assert all(t in [2, 3] for t in tokens)


def test_sampler_top_p():
    params = SamplingParams(temperature=1.0, top_p=0.5, seed=42)
    sampler = Sampler(params)

    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
    tokens = [sampler.sample(logits).item() for _ in range(50)]
    assert all(t == 0 for t in tokens)


def test_sampler_apply_sampling():
    params = SamplingParams(temperature=0.0, seed=42)
    sampler = Sampler(params)

    seq1 = Sequence(seq_id=0, prompt_token_ids=[1, 2])
    seq1.output_token_ids = [5]
    seq2 = Sequence(seq_id=1, prompt_token_ids=[3, 4])
    seq2.output_token_ids = [6]
    # Format: [1, total_seq_len, vocab_size]
    logits = torch.tensor([[
        [0.0, 0.0, 0.0, 0.0],  # seq1 token 1
        [0.0, 0.0, 0.0, 0.0],  # seq1 token 2
        [0.0, 0.0, 0.0, 10.0], # seq1 token 5 (last)
        [0.0, 0.0, 0.0, 0.0],  # seq2 token 3
        [0.0, 0.0, 0.0, 0.0],  # seq2 token 4
        [0.0, 10.0, 0.0, 0.0], # seq2 token 6 (last)
    ]])

    tokens = sampler.apply_sampling(logits, [seq1, seq2])
    assert tokens == [3, 1]