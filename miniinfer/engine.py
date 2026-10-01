from transformers import AutoTokenizer

from miniinfer.config import EngineConfig
from miniinfer.model import Qwen3Model
from miniinfer.model_runner import ModelRunner
from miniinfer.sampler import Sampler, SamplingParams
from miniinfer.scheduler import Scheduler
from miniinfer.sequence import Sequence, SequenceStatus


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = Qwen3Model.from_pretrained(
            config.model_path,
            device=config.device,
            dtype=config.dtype,
        )
        self.runner = ModelRunner(config, self.model)
        self.scheduler = Scheduler(config, None)
        self.seq_counter = 0

    def step(self) -> list[Sequence]:
        sched_output = self.scheduler.schedule()
        all_seqs = sched_output.prefill_seqs + sched_output.decode_seqs
        if not all_seqs:
            return []

        is_prefill = len(sched_output.prefill_seqs) > 0
        logits = self.runner.run(all_seqs, is_prefill)
        sampled_tokens = self.sampler.apply_sampling(logits, all_seqs)

        finished = []
        for seq, token_id in zip(all_seqs, sampled_tokens):
            seq.append_token(token_id)
            if self._check_stop(seq, token_id):
                seq.status = SequenceStatus.FINISHED
                finished.append(seq)

        return finished

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[dict]:
        self.sampler = Sampler(sampling_params)
        seqs = self._create_sequences(prompts, sampling_params)
        for seq in seqs:
            self.scheduler.add_seq(seq)

        outputs = {seq.seq_id: {"text": "", "token_ids": [], "finish_reason": ""} for seq in seqs}

        while any(not seq.is_finished() for seq in seqs):
            finished = self.step()
            for seq in finished:
                outputs[seq.seq_id]["text"] = self.tokenizer.decode(seq.output_token_ids)
                outputs[seq.seq_id]["token_ids"] = seq.output_token_ids
                is_stop = seq.get_last_token() in (sampling_params.stop_token_ids or [])
                outputs[seq.seq_id]["finish_reason"] = "stop" if is_stop else "length"

        return [outputs[seq.seq_id] for seq in seqs]

    def _create_sequences(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[Sequence]:
        seqs = []
        for prompt in prompts:
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            seq = Sequence(
                seq_id=self.seq_counter,
                prompt_token_ids=token_ids,
                status=SequenceStatus.WAITING,
            )
            self.seq_counter += 1
            seqs.append(seq)
        return seqs

    def _check_stop(self, seq: Sequence, new_token: int) -> bool:
        if seq.num_output_tokens >= self.sampler.params.max_tokens:
            return True
        if self.sampler.params.stop_token_ids and new_token in self.sampler.params.stop_token_ids:
            return True
        return False