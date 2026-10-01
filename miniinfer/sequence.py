from dataclasses import dataclass, field
from enum import Enum


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
    num_computed_tokens: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def get_prompt_len(self) -> int:
        return self.num_prompt_tokens

    def get_last_token(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]