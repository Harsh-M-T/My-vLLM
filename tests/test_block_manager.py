from miniinfer.sequence import Sequence, SequenceStatus


def test_sequence_properties():
    seq = Sequence(seq_id=0, prompt_token_ids=[1, 2, 3, 4])
    assert seq.num_prompt_tokens == 4
    assert seq.num_output_tokens == 0
    assert seq.total_len == 4
    assert seq.get_prompt_len() == 4
    assert seq.get_last_token() == 4

    seq.append_token(5)
    seq.append_token(6)
    assert seq.num_output_tokens == 2
    assert seq.total_len == 6
    assert seq.get_last_token() == 6


def test_sequence_status():
    seq = Sequence(seq_id=0, prompt_token_ids=[1, 2, 3])
    assert seq.status == SequenceStatus.WAITING
    seq.status = SequenceStatus.RUNNING
    assert seq.status == SequenceStatus.RUNNING
    seq.status = SequenceStatus.FINISHED
    assert seq.is_finished()


def test_block_manager_basic():
    from miniinfer.block_manager import BlockManager

    manager = BlockManager(num_blocks=10, block_size=16)
    assert manager.can_allocate(1)

    blocks = manager.allocate(Sequence(seq_id=0, prompt_token_ids=[1]*16), 16)
    assert len(blocks) == 1
    assert blocks[0] == 0

    seq = Sequence(seq_id=1, prompt_token_ids=[1]*32)
    blocks = manager.allocate(seq, 32)
    assert len(blocks) == 2

    manager.free(seq)
    assert manager.can_allocate(2)


def test_block_manager_ref_counting():
    from miniinfer.block_manager import BlockManager

    manager = BlockManager(num_blocks=10, block_size=16)
    seq1 = Sequence(seq_id=0, prompt_token_ids=[1]*16)
    seq2 = Sequence(seq_id=1, prompt_token_ids=[1]*16)

    blocks1 = manager.allocate(seq1, 16)
    manager.touch_blocks(seq1)
    blocks2 = manager.allocate(seq2, 16)
    manager.touch_blocks(seq2)

    assert manager.blocks[blocks1[0]].ref_count == 1
    assert manager.blocks[blocks2[0]].ref_count == 1

    manager.release_blocks(seq1)
    assert manager.blocks[blocks1[0]].ref_count == 0


def test_block_manager_prefix_caching():
    from miniinfer.block_manager import BlockManager

    manager = BlockManager(num_blocks=10, block_size=16, enable_prefix_caching=True)
    tokens = list(range(32))

    cached = manager.find_cached_blocks(tokens)
    assert cached == []

    seq = Sequence(seq_id=0, prompt_token_ids=tokens)
    _ = manager.allocate(seq, 32)
    manager.touch_blocks(seq)
    manager.release_blocks(seq)

    _ = Sequence(seq_id=1, prompt_token_ids=tokens)
    cached = manager.find_cached_blocks(tokens)
    assert len(cached) == 2