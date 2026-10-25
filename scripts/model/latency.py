# how long one edit takes for each of the 4 systems, at batch size 1 like a user of the demo
# same gpu and the same 30 test rows (10 per edit type) for all of them, greedy, one warm-up run first
# run after both fine-tunes: CUDA_VISIBLE_DEVICES=0 python latency.py, writes $B/outputs/results/latency.json

import os, sys, json, time, random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompts import messages

B = os.environ["B"]
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
N = 10   # rows per edit type

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # the same 30 rows for every system
    rng = random.Random(0)
    rows = []
    for t in ["t1", "t2", "t3"]:
        all_rows = [json.loads(l) for l in open(f"{B}/data/final/test/{t}.jsonl")]
        rows += [dict(r, type=t) for r in rng.sample(all_rows, N)]

    # one base model with both adapters, each request picks one
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=0.85, seed=0,
              enable_lora=True, max_lora_rank=16)
    loras = {"patch": LoRARequest("patch", 1, f"{B}/checkpoints/patch/final"),
             "rewrite": LoRARequest("rewrite", 2, f"{B}/checkpoints/rewrite/final")}

    res = {}
    for system in ["zs_patch", "sft_patch", "zs_rewrite", "sft_rewrite"]:
        mode = system.split("_")[1]
        lora = loras[mode] if system.startswith("sft") else None   # zero-shot = no adapter
        params = SamplingParams(temperature=0.0, max_tokens=768 if mode == "patch" else 4096)

        # warm-up run (loads the adapter), not timed
        llm.chat([messages(mode, rows[0]["instruction"], rows[0]["original"])], SamplingParams(max_tokens=8),
                 lora_request=lora, use_tqdm=False)

        # one row at a time
        lat, tok = [], []
        for r in rows:
            s = time.time()
            o = llm.chat([messages(mode, r["instruction"], r["original"])], params, lora_request=lora, use_tqdm=False)
            lat.append(time.time() - s)
            tok.append(len(o[0].outputs[0].token_ids))

        by_type = {}
        for t in ["t1", "t2", "t3"]:
            by_type[t] = sum(l for l, r in zip(lat, rows) if r["type"] == t) / N
        res[system] = dict(rows=len(lat), latency_mean_s=sum(lat) / len(lat), tokens_mean=sum(tok) / len(tok),
                           latency_by_type=by_type)
        print(system, res[system], flush=True)

    os.makedirs(f"{B}/outputs/results", exist_ok=True)
    json.dump(res, open(f"{B}/outputs/results/latency.json", "w"), indent=1)
