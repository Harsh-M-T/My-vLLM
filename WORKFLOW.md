# mini-infer Workflow Documentation

This document describes the complete request-to-response workflow of mini-infer, a minimal LLM inference engine implementing vLLM's core ideas.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Component Architecture](#component-architecture)
3. [Request Lifecycle](#request-lifecycle)
4. [Detailed Component Walkthrough](#detailed-component-walkthrough)
5. [Key Algorithms](#key-algorithms)
6. [Data Flow Diagram](#data-flow-diagram)
7. [Configuration & Tuning](#configuration--tuning)

---

## High-Level Overview

mini-infer processes batches of prompts through a continuous batching loop with paged KV cache management. The engine separates **prefill** (prompt processing) and **decode** (token generation) phases, allowing multiple requests at different stages to share GPU compute efficiently.

**Core loop:**
```
while requests remain:
    1. Scheduler selects sequences for this step (prefill + decode)
    2. ModelRunner builds inputs and runs forward pass
    3. Sampler generates next tokens
    4. Engine updates sequences, checks stop conditions
    5. Finished sequences are returned to user
```

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                            LLM (User API)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                           Engine                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Scheduler   │──│ModelRunner  │──│  Sampler    │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────┐               │
│  │           BlockManager (KV Cache)            │               │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Qwen3Model     │
                    │  (PyTorch)      │
                    └─────────────────┘
```

| Component | Responsibility |
|-----------|----------------|
| **LLM** | User-facing API, tokenizer handling |
| **Engine** | Main loop, coordinates scheduler/runner/sampler |
| **Scheduler** | Admission, prefill/decode split, token budget, preemption |
| **ModelRunner** | Input tensor construction, KV cache allocation, forward pass |
| **Sampler** | Temperature, top-k, top-p, stop token detection |
| **BlockManager** | Paged KV blocks, ref counting, prefix caching via chained hashing |
| **Qwen3Model** | Model architecture (RMSNorm, RoPE, GQA, SwiGLU) + weight loading |

---

## Request Lifecycle

### 1. Initialization (`LLM.__init__`)
```python
llm = LLM("Qwen/Qwen3-0.6B", max_num_seqs=64, max_model_len=2048, device="cuda")
```
- Loads tokenizer from Hugging Face
- Creates `EngineConfig` with user parameters
- Instantiates `Engine`:
  - Loads `Qwen3Model` from safetensors weights
  - Creates `ModelRunner` (allocates KV cache tensors)
  - Creates `Scheduler` with `BlockManager`
  - Creates `Sampler` (per-request, via `SamplingParams`)

### 2. Request Submission (`LLM.generate`)
```python
outputs = llm.generate(["Hello", "How are you?"], SamplingParams(temperature=0.7))
```
- Tokenizes each prompt string → `prompt_token_ids`
- Creates `Sequence` objects with `status=WAITING`
- Registers sequences with `Scheduler.add_seq()`
- Enters generation loop until all sequences `FINISHED`

### 3. Generation Loop (`Engine.generate` → `Engine.step`)

#### Step A: Scheduling (`Scheduler.schedule`)
```python
output = scheduler.schedule()
# Returns SchedulerOutput:
#   prefill_seqs:  sequences in prefill phase (new prompts)
#   decode_seqs:   sequences in decode phase (generating tokens)
#   preempted_seqs: sequences evicted due to memory pressure
#   num_batched_tokens: total tokens in this batch
```

**Scheduling policy:**
1. **Prefill admission**: Pull from `waiting` queue while `num_batched_tokens + seq.total_len <= max_num_batched_tokens`
2. **Decode continuation**: All non-finished `running` sequences get 1 token budget each
3. **Preemption**: If no free blocks, evict lowest-priority running sequences (recompute later)

#### Step B: Input Preparation (`ModelRunner.prepare_inputs`)
For each scheduled sequence:
- Concatenates `prompt_token_ids + output_token_ids`
- Builds padded batch tensor `[batch, max_seq_len]`
- Computes `positions` (absolute positions for RoPE)
- Returns dict with `input_ids`, `positions`, `cu_seqlens`, `max_seq_len`

#### Step C: Forward Pass (`ModelRunner.run` → `Qwen3Model.forward`)
```
input_ids [B, L] ──▶ embed_tokens ──▶ hidden_states [B, L, H]
                           │
                           ▼
              ┌────────────────────────┐
              │  For each decoder layer │
              ├────────────────────────┤
              │  RMSNorm               │
              │  Attention (GQA + RoPE)│──▶ writes to KV cache
              │  RMSNorm               │
              │  SwiGLU MLP            │
              │  Residual add          │
              └────────────────────────┘
                           │
                           ▼
                    RMSNorm (final)
                           │
                           ▼
                    lm_head ──▶ logits [B, L, V]
```

**KV Cache Write (per layer):**
- `k_cache[batch, :, :q_len] = k` (after RoPE)
- `v_cache[batch, :, :q_len] = v`
- For decode: only last position written; prefill: full prompt written

#### Step D: Sampling (`Sampler.apply_sampling`)
```python
# Extract last token logits for each sequence
last_logits = logits[batch_idx, seq_len - 1, :]  # [B, V]

# Apply sampling pipeline:
1. Temperature scaling: logits /= temperature
2. Top-k: keep top k logits, mask rest to -inf
3. Top-p (nucleus): cumulative prob threshold, mask tail
4. Softmax → probabilities
5. Multinomial sample (or argmax if temp=0)
```

#### Step E: Sequence Update & Stop Check
```python
for seq, token_id in zip(all_seqs, sampled_tokens):
    seq.append_token(token_id)
    if seq.num_output_tokens >= max_tokens:        → FINISHED (length)
    if token_id in stop_token_ids:                 → FINISHED (stop)
    if seq.total_len >= max_model_len:             → FINISHED (length)
```

Finished sequences are removed from scheduler, added to output dict.

### 4. Response Assembly
```python
outputs = [
    {"text": tokenizer.decode(seq.output_token_ids),
     "token_ids": seq.output_token_ids,
     "finish_reason": "stop" | "length"}
    for seq in original_sequences
]
```

---

## Detailed Component Walkthrough

### BlockManager (Paged KV Cache)

**Block structure:**
```
Block 0: [token_0 ... token_15]  hash=H(prev_hash, tokens)
Block 1: [token_16 ... token_31] hash=H(prev_hash, tokens)
...
```

**Allocation (`allocate`):**
1. Compute `num_blocks = ceil(num_tokens / block_size)`
2. Pop `num_blocks` from `free_list`
3. Set `ref_count = 0` (incremented by `touch_blocks`)
4. Assign to `seq.block_table`

**Prefix Caching (`touch_blocks` → `_compute_and_store_hashes`):**
```python
prev_hash = None
for i, block_id in enumerate(seq.block_table):
    chunk = token_ids[i*block_size : (i+1)*block_size]
    h = hash(tuple(chunk))
    if prev_hash: h = hash((prev_hash, h))
    blocks[block_id].hash = h
    blocks[block_id].prev_hash = prev_hash
    block_hashes[h] = block_id
    prev_hash = h
```

**Prefix Lookup (`find_cached_blocks`):**
```python
for each full block in prompt:
    h = hash(chunk) chained with prev_hash
    if h in block_hashes and blocks[block_id].ref_count == 0:
        reuse block_id
    else:
        break  # prefix chain broken
```

**Reference Counting:**
- `touch_blocks`: `ref_count += 1` (sequence actively using)
- `release_blocks`: `ref_count -= 1`; if 0 → return to `free_list`, clear hash

### Scheduler (Continuous Batching)

**State machine per sequence:**
```
WAITING ──(admitted)──▶ RUNNING ──(prefill done)──▶ RUNNING (decode) ──(finished)──▶ FINISHED
     ▲                                                                             │
     └──────────────(preempted)────────────────────────────────────────────────────┘
```

**Token Budget:**
- `max_num_batched_tokens` limits total tokens per step
- Prefill sequences consume `seq.total_len` tokens
- Decode sequences consume 1 token each
- Ensures OOM-safe batching

**Preemption Strategy:**
- When blocks exhausted, evict from `running` (decode phase first)
- Evicted sequences return to `waiting` with `num_computed_tokens` preserved
- On re-admission, BlockManager finds cached prefix blocks

### ModelRunner (Batching & KV Cache)

**KV Cache Layout:**
```python
# Per layer: [max_batch, num_kv_heads, max_tokens, head_dim]
k_cache = torch.zeros(max_num_seqs, num_kv_heads, max_model_len, head_dim)
v_cache = torch.zeros_like(k_cache)
```

**Per-sequence execution (current implementation):**
- Processes each sequence individually (batch_size=1)
- Concatenates logits along sequence dimension for sampler
- *Future: true batched prefill with `cu_seqlens` and flash-attn varlen*

### Qwen3Model (Architecture)

**Decoder Layer:**
```
x ──▶ RMSNorm ──▶ Attention (GQA + RoPE) ──▶ + ──▶ RMSNorm ──▶ SwiGLU ──▶ +
```

**Grouped Query Attention:**
- `num_heads = 32`, `num_kv_heads = 8` → `num_kv_heads_per_head = 4`
- Q: `[B, 32, L, 128]`, K/V: `[B, 8, L, 128]`
- Repeat K/V 4× to match Q heads before attention

**Rotary Embeddings:**
- Precompute `cos/sin` up to `max_position_embeddings`
- `apply_rotary_pos_emb` with `cos.unsqueeze(1)` broadcast over heads

---

## Key Algorithms

### Chained Block Hashing (Prefix Caching)
```
Block 0 hash = H(tokens_0:15)
Block 1 hash = H(Block 0 hash, tokens_16:31)
Block 2 hash = H(Block 1 hash, tokens_32:47)
...
```
- Enables O(1) prefix sharing: identical prefix → identical hash chain
- Only full blocks participate (partial blocks break chain)
- Hash map: `block_hash → block_id` (only for `ref_count == 0` blocks)

### Continuous Batching with Phase Separation
```
Time ──────────────────────────────────────────────────────▶

Request A: [PREFILL: 100 tok] [DECODE: 1] [DECODE: 1] [DECODE: 1] ...
Request B:          [PREFILL: 50 tok] [DECODE: 1] [DECODE: 1] ...
Request C:                     [PREFILL: 200 tok] [DECODE: 1] ...

GPU Utilization: ████████████████████████████████████████████
```
- Prefill: compute-intensive, memory-bound (large matmuls)
- Decode: memory-bound (small matmuls, KV cache reads)
- Interleaving keeps GPU saturated

### Flash-Attention Integration (Optional)
- **Varlen prefill**: `flash_attn_varlen_func` with `cu_seqlens`
- **Decode with KV cache**: `flash_attn_with_kvcache`
- Fallback: pure PyTorch SDPA implementation

---

## Data Flow Diagram

```
USER PROMPTS
     │
     ▼
┌──────────────────────────────────────┐
│ LLM.generate()                       │
│ - Tokenize prompts                   │
│ - Create Sequence objects            │
│ - engine.generate(sequences, params) │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Engine.generate()                    │
│ sampler = Sampler(params)            │
│ scheduler.add_seq() for each         │
│ while any not finished:              │
│   step()                             │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Engine.step()                        │
│ sched_output = scheduler.schedule()  │
│ logits = runner.run(seqs, is_prefill)│
│ tokens = sampler.apply_sampling()    │
│ update sequences, check stops        │
└──────────────┬───────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌─────────────┐   ┌─────────────┐
│ Scheduler   │   │ ModelRunner │
│ schedule()  │   │ prepare_inputs()
│ - prefill   │   │   - input_ids
│ - decode    │   │   - positions
│ - budget    │   │   - cu_seqlens
│ - preempt   │   └──────┬──────┘
└─────────────┘          │
                         ▼
              ┌──────────────────────┐
              │ Qwen3Model.forward() │
              │ - embed_tokens       │
              │ - N × DecoderLayer   │
              │   - RMSNorm          │
              │   - Attention        │
              │     - QKV proj       │
              │     - RoPE           │
              │     - GQA repeat     │
              │     - KV cache write │
              │     - Flash/SDPA     │
              │   - SwiGLU           │
              │ - final RMSNorm      │
              │ - lm_head            │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Sampler.apply_sampling() │
              │ - extract last logits│
              │ - temp / top-k / top-p │
              │ - softmax + multinomial│
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Engine: update seqs  │
              │ - append token       │
              │ - check stops        │
              │ - mark FINISHED      │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Return outputs       │
              │ tokenizer.decode()   │
              └──────────────────────┘
```

---

## Configuration & Tuning

### EngineConfig Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `block_size` | 16 | KV cache block granularity (tokens/block) |
| `max_num_seqs` | 64 | Max concurrent sequences |
| `max_model_len` | 2048 | Max context length |
| `max_num_batched_tokens` | 4096 | Token budget per scheduler step |
| `gpu_memory_utilization` | 0.9 | Fraction of GPU mem for KV cache |
| `enable_prefix_caching` | True | Share blocks for common prefixes |
| `enable_cuda_graph` | True | Capture decode graphs for fixed batch sizes |
| `cuda_graph_batch_sizes` | [1,2,4,8,16,32,64] | Batch sizes to capture |

### Memory Estimation

```
KV cache per layer = 2 * max_num_seqs * num_kv_heads * max_model_len * head_dim * dtype_bytes
Total KV = num_layers * KV_per_layer

Example (Qwen3-0.6B, max_num_seqs=64, max_model_len=2048, fp16):
  24 layers * 2 * 64 * 8 * 2048 * 128 * 2 bytes ≈ 1.2 GB
```

### Performance Tips

1. **Increase `max_num_batched_tokens`** for higher throughput (if GPU memory allows)
2. **Enable prefix caching** for workloads with shared system prompts
3. **Use CUDA graphs** for decode (auto-enabled on GPU)
4. **Tune `block_size`**: smaller = less internal fragmentation, larger = less metadata overhead
5. **Flash-attention**: install `flash-attn` for 2-3× prefill speedup

---

## Extension Points

| Feature | Location | Effort |
|---------|----------|--------|
| Multi-GPU (tensor parallel) | `ModelRunner`, `Qwen3Model` | High |
| Streaming output | `Engine.generate` | Medium |
| Speculative decoding | `Engine.step`, new `DraftModel` | High |
| Quantization (AWQ/GPTQ) | `Qwen3Model.load_weights` | Medium |
| Other model families | New `model/*.py` + `ModelBase` | Medium |
| OpenAI-compatible server | New `server.py` | Medium |

---

## Debugging & Profiling

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Profile CUDA kernels
import torch.profiler
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    outputs = llm.generate(prompts, params)
prof.export_chrome_trace("trace.json")

# Check KV cache utilization
print(f"Free blocks: {len(block_manager.free_list)} / {block_manager.num_blocks}")
print(f"Block hashes cached: {len(block_manager.block_hashes)}")
```

---

## Summary

mini-infer implements a complete offline LLM inference pipeline:

1. **Paged KV cache** enables memory-efficient, flexible batching
2. **Continuous batching** with prefill/decode separation maximizes GPU utilization
3. **Prefix caching** via chained block hashing accelerates shared-prompt workloads
4. **Pure PyTorch model** with optional Flash-Attention kernels
5. **Clean module boundaries** make each component testable and replaceable

The codebase is designed to be readable (~1500 lines core) while demonstrating production-grade techniques from vLLM.