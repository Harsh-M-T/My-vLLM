import pytest

from miniinfer.config import EngineConfig
from miniinfer.scheduler import Scheduler
from miniinfer.sequence import Sequence, SequenceStatus


@pytest.fixture
def config():
    return EngineConfig(
        model_path="dummy",
        max_num_seqs=4,
        max_model_len=128,
        max_num_batched_tokens=64,
    )


def test_scheduler_admission(config):
    scheduler = Scheduler(config, None)
    seq1 = Sequence(seq_id=0, prompt_token_ids=[1]*10)
    seq2 = Sequence(seq_id=1, prompt_token_ids=[1]*20)

    scheduler.add_seq(seq1)
    scheduler.add_seq(seq2)

    output = scheduler.schedule()
    assert len(output.prefill_seqs) == 2
    assert seq1.status.name == "RUNNING"
    assert seq2.status.name == "RUNNING"


def test_scheduler_budget(config):
    scheduler = Scheduler(config, None)
    seq1 = Sequence(seq_id=0, prompt_token_ids=[1]*40)
    seq2 = Sequence(seq_id=1, prompt_token_ids=[1]*40)

    scheduler.add_seq(seq1)
    scheduler.add_seq(seq2)

    output = scheduler.schedule()
    assert len(output.prefill_seqs) == 1


def test_scheduler_decode_phase(config):
    scheduler = Scheduler(config, None)
    seq = Sequence(seq_id=0, prompt_token_ids=[1]*10)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)

    output = scheduler.schedule()
    assert len(output.decode_seqs) == 1
    assert seq in output.decode_seqs


def test_scheduler_abort(config):
    scheduler = Scheduler(config, None)
    seq = Sequence(seq_id=0, prompt_token_ids=[1]*10)
    scheduler.add_seq(seq)

    scheduler.abort_seq(0)
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0