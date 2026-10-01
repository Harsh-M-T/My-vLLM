import argparse
import json
import platform
import random
import sys
import time

import torch

from miniinfer import LLM, SamplingParams


def get_system_info():
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def generate_workload(num_requests: int, min_len: int, max_len: int, seed: int = 42):
    random.seed(seed)
    prompts = []
    for _ in range(num_requests):
        input_len = random.randint(min_len, max_len)
        _ = random.randint(min_len, max_len)  # output_len not used for token generation
        token_ids = [random.randint(100, 10000) for _ in range(input_len)]
        prompts.append(token_ids)
    return prompts


def run_mini_infer(
    model_path: str,
    prompts: list[list[int]],
    sampling_params: SamplingParams,
    device: str,
):
    llm = LLM(model_path, max_num_seqs=64, max_model_len=2048, device=device, dtype="float16")
    
    # Convert token IDs to strings for the API
    prompt_strings = []
    for p in prompts:
        prompt_strings.append(" ".join(str(t) for t in p))
    
    start = time.perf_counter()
    outputs = llm.generate(prompt_strings, sampling_params)
    end = time.perf_counter()
    
    total_tokens = sum(len(o["token_ids"]) for o in outputs)
    return total_tokens, end - start


def run_hf_generate(
    model_path: str,
    prompts: list[list[int]],
    sampling_params: SamplingParams,
    device: str,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    input_texts = [" ".join(str(t) for t in p) for p in prompts]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(model.device)
    
    gen_kwargs = {
        "max_new_tokens": sampling_params.max_tokens,
        "do_sample": sampling_params.temperature > 0,
        "temperature": sampling_params.temperature if sampling_params.temperature > 0 else None,
        "top_p": sampling_params.top_p if sampling_params.top_p < 1.0 else None,
        "top_k": sampling_params.top_k if sampling_params.top_k > 0 else None,
        "pad_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    
    start = time.perf_counter()
    outputs = model.generate(**inputs, **gen_kwargs)
    end = time.perf_counter()
    
    total_tokens = sum(len(o) - len(inputs.input_ids[0]) for o in outputs)
    return total_tokens, end - start


def run_vllm(
    model_path: str,
    prompts: list[list[int]],
    sampling_params: SamplingParams,
    device: str,
):
    try:
        from vllm import LLM as VLLM
        from vllm import SamplingParams as VLLMSamplingParams
    except ImportError:
        print("vLLM not installed, skipping...")
        return None, None
    
    vllm_sampling = VLLMSamplingParams(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        max_tokens=sampling_params.max_tokens,
    )
    
    llm = VLLM(model_path, max_num_seqs=64, max_model_len=2048, dtype="float16")
    prompt_strings = [" ".join(str(t) for t in p) for p in prompts]
    
    start = time.perf_counter()
    outputs = llm.generate(prompt_strings, vllm_sampling)
    end = time.perf_counter()
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return total_tokens, end - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--min-len", type=int, default=100)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true")
    args = parser.parse_args()
    
    print("System Info:")
    sys_info = get_system_info()
    for k, v in sys_info.items():
        print(f"  {k}: {v}")
    print()
    
    print(
        f"Generating workload: {args.num_requests} requests, "
        f"lengths {args.min_len}-{args.max_len}"
    )
    prompts = generate_workload(args.num_requests, args.min_len, args.max_len, args.seed)
    print(f"Prompt lengths: {[len(p) for p in prompts[:5]]}...")
    print()
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        top_p=1.0,
        top_k=-1,
        seed=args.seed,
    )
    
    results = {}
    
    print("Running mini-infer...")
    tokens, elapsed = run_mini_infer(args.model, prompts, sampling_params, args.device)
    results["mini-infer"] = {"tokens": tokens, "time": elapsed, "tok_s": tokens / elapsed}
    print(f"  Tokens: {tokens}, Time: {elapsed:.2f}s, Tokens/s: {tokens/elapsed:.0f}")
    print()
    
    if not args.skip_hf:
        print("Running HF generate...")
        tokens, elapsed = run_hf_generate(args.model, prompts, sampling_params, args.device)
        results["hf-generate"] = {"tokens": tokens, "time": elapsed, "tok_s": tokens / elapsed}
        print(f"  Tokens: {tokens}, Time: {elapsed:.2f}s, Tokens/s: {tokens/elapsed:.0f}")
        print()
    
    if not args.skip_vllm:
        print("Running vLLM...")
        tokens, elapsed = run_vllm(args.model, prompts, sampling_params, args.device)
        if tokens is not None:
            results["vllm"] = {"tokens": tokens, "time": elapsed, "tok_s": tokens / elapsed}
            print(f"  Tokens: {tokens}, Time: {elapsed:.2f}s, Tokens/s: {tokens/elapsed:.0f}")
        print()
    
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Engine':<20} {'Tokens':>10} {'Time (s)':>10} {'Tokens/s':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['tokens']:>10} {r['time']:>10.2f} {r['tok_s']:>10.0f}")
    
    output = {
        "system": sys_info,
        "workload": {
            "num_requests": args.num_requests,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "seed": args.seed,
        },
        "results": results,
    }
    
    with open("bench_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to bench_results.json")


if __name__ == "__main__":
    main()