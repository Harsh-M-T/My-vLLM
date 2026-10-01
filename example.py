from miniinfer import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-0.6B", max_num_seqs=64, max_model_len=2048, device="cuda")
params = SamplingParams(temperature=0.6, max_tokens=256, top_p=0.95)
outputs = llm.generate(["Hello, how are you?"], params)
print(outputs[0]["text"])