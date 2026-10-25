import os, sys, json, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompts import messages

B = os.environ["B"]
SYSTEM = os.environ["SYSTEM"]
DATASET = os.environ["DATASET"]
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODE = "rewrite" if "rewrite" in SYSTEM else "patch"
if SYSTEM.startswith("sft"):
    ADAPTER = f"{B}/checkpoints/{MODE}/final"
else:
    ADAPTER = None
out_path = f"{B}/outputs/preds/{SYSTEM}_{DATASET}.jsonl"

# rows with id, instruction, program
def load_rows():
    if DATASET == "test":
        rows = []
        for t in ["t1", "t2", "t3"]:
            for line in open(f"{B}/data/final/test/{t}.jsonl"):
                r = json.loads(line)
                rows.append(dict(id=f"{t}_{r['uuid']}", instruction=r["instruction"], program=r["original"]))
        return rows
    import pandas as pd
    df = pd.read_parquet(f"{B}/data/benchcad/edit-bench/data/edit_bench-00000-of-00001.parquet",
                         columns=["record_id", "instruction", "orig_code"])
    rows = []
    for r in df.itertuples():
        program = r.orig_code if r.orig_code.endswith("\n") else r.orig_code + "\n"
        rows.append(dict(id=str(r.record_id), instruction=r.instruction, program=program))
    return rows


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = load_rows()
    convs = [messages(MODE, r["instruction"], r["program"]) for r in rows]

    # base model once; vllm adds the lora adapter on top when a request asks for it
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=0.85, seed=0,
              enable_lora=ADAPTER is not None, max_lora_rank=16)
    # temperature 0 = greedy. a patch is short, a whole program can be long
    params = SamplingParams(temperature=0.0, max_tokens=768 if MODE == "patch" else 4096)
    lora = LoRARequest("adapter", 1, ADAPTER) if ADAPTER else None

    t0 = time.time()
    outs = llm.chat(convs, params, lora_request=lora, use_tqdm=True)
    total = time.time() - t0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_tok = 0
    with open(out_path, "w") as f:
        for r, o in zip(rows, outs):
            g = o.outputs[0]
            n_tok += len(g.token_ids)
            f.write(json.dumps(dict(id=r["id"], output=g.text, n_tokens=len(g.token_ids), finish=g.finish_reason)) + "\n")
    meta = dict(system=SYSTEM, dataset=DATASET, rows=len(rows), batch_seconds=total, output_tokens=n_tok)
    json.dump(meta, open(out_path.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(meta)
