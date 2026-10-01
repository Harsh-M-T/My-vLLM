from dataclasses import dataclass

from miniinfer.config import EngineConfig
from miniinfer.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    prefill_seqs: list[Sequence]
    decode_seqs: list[Sequence]
    preempted_seqs: list[Sequence]
    num_batched_tokens: int


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: object | None):
        self.config = config
        self.block_manager = block_manager
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add_seq(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def abort_seq(self, seq_id: int) -> None:
        self.waiting = [s for s in self.waiting if s.seq_id != seq_id]
        self.running = [s for s in self.running if s.seq_id != seq_id]

    def schedule(self) -> SchedulerOutput:
        prefill_seqs = []
        decode_seqs = []
        preempted = []
        num_batched_tokens = 0

        for seq in self.waiting[:]:
            if num_batched_tokens + seq.total_len <= self.config.max_num_batched_tokens:
                seq.status = SequenceStatus.RUNNING
                prefill_seqs.append(seq)
                num_batched_tokens += seq.total_len
                self.waiting.remove(seq)
                self.running.append(seq)
            else:
                break

        for seq in self.running[:]:
            if not seq.is_finished():
                decode_seqs.append(seq)
                num_batched_tokens += 1

        return SchedulerOutput(prefill_seqs, decode_seqs, preempted, num_batched_tokens)