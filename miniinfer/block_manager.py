from dataclasses import dataclass

from miniinfer.sequence import Sequence


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    hash: int | None = None
    prev_hash: int | None = None


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_list = list(range(num_blocks))
        self.block_hashes = {}

    def allocate(self, seq: Sequence, num_tokens: int) -> list[int]:
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if not self.can_allocate(num_blocks):
            raise RuntimeError("Not enough free blocks")

        allocated = []
        for _ in range(num_blocks):
            block_id = self.free_list.pop(0)
            self.blocks[block_id].ref_count = 0
            allocated.append(block_id)
        seq.block_table = allocated
        return allocated

    def free(self, seq: Sequence) -> None:
        for block_id in seq.block_table:
            self.blocks[block_id].ref_count = 0
            self.free_list.append(block_id)
        seq.block_table = []

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self.free_list) >= num_blocks

    def get_block_table(self, seq: Sequence) -> list[int]:
        return seq.block_table

    def touch_blocks(self, seq: Sequence) -> None:
        for block_id in seq.block_table:
            self.blocks[block_id].ref_count += 1
        if self.enable_prefix_caching:
            self._compute_and_store_hashes(seq)

    def _compute_and_store_hashes(self, seq: Sequence) -> None:
        token_ids = seq.prompt_token_ids + seq.output_token_ids
        prev_hash = None
        for i, block_id in enumerate(seq.block_table):
            start = i * self.block_size
            end = start + self.block_size
            chunk = token_ids[start:end]
            if len(chunk) < self.block_size:
                break
            h = hash(tuple(chunk))
            if prev_hash is not None:
                h = hash((prev_hash, h))
            self.blocks[block_id].hash = h
            self.blocks[block_id].prev_hash = prev_hash
            self.block_hashes[h] = block_id
            prev_hash = h

    def release_blocks(self, seq: Sequence) -> None:
        for block_id in seq.block_table:
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self.free_list.append(block_id)

    def compute_block_hash(self, block: Block, token_ids: list[int]) -> int:
        h = hash(tuple(token_ids))
        if block.prev_hash is not None:
            h = hash((block.prev_hash, h))
        return h

    def find_cached_blocks(self, token_ids: list[int]) -> list[tuple[int, int]]:
        if not self.enable_prefix_caching:
            return []

        results = []
        prev_hash = None
        for i in range(0, len(token_ids), self.block_size):
            chunk = token_ids[i:i + self.block_size]
            if len(chunk) < self.block_size:
                break
            h = hash(tuple(chunk))
            if prev_hash is not None:
                h = hash((prev_hash, h))
            if h in self.block_hashes:
                block_id = self.block_hashes[h]
                if self.blocks[block_id].ref_count == 0:
                    results.append((block_id, self.block_size))
                    prev_hash = h
                else:
                    break
            else:
                break
        return results