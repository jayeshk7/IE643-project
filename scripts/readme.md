# scripts

Run the laptop scripts from this folder with the project venv (`uv sync` in the repo root makes it). Their paths
start with `../`, which is the repo root, so they only work from here:

```
cd scripts
../.venv/bin/python <script>
```

None of them take arguments. You pick the split by editing the lines at the top (`SPLIT`, or the commented-out train / val / test lines). Only `filter_data.py` creates its output folder, so make the others first, e.g. `mkdir -p ../data/train ../final_data_split/train ../final_triplet/train`.

## 1. Data

Run these once per split, in this order.

- `data_filtration/get_zero_to_cad100k.py` downloads one split of Zero-To-CAD-100k (uuid + CadQuery code) into
  `data/<split>/<uuid>.py`. Uncomment the `load_dataset` line and the `open` line for the split you want.
- `data_filtration/filter_data.py` keeps the programs that build exactly one solid (no assemblies), with a 30 s and
  1.5 GB limit per program. Output: `filtered_data/filtered_<split>/`.
- `data_filtration/sample_10k.py` copies `N` random programs (seed 0) into `final_data_split/<split>/`. Set `source`,
  `destination` and `N` (10000 for train, 1000 for val and test).

## 2. Edits (original, edited, patch)

- `dataset_creation/create_t1.py` - T1: scale one number, either a top-level parameter or a number inside a call.
- `dataset_creation/create_t2.py` - T2: add one line at the end that rotates or mirrors the whole part.
- `dataset_creation/create_t3.py` - T3: set one length to k times another length (also saves the `relation`).

Each one reads `final_data_split/<SPLIT>/`, makes at most one edit per program and overwrites `final_triplet/<SPLIT>/t1.jsonl` (`t2`, `t3`).

```
nohup nice -n 10 ../.venv/bin/python dataset_creation/create_t1.py > ../run_log_t1.txt 2>&1 &
```

## 3. GPU server

The rest ran on a GPU server. These scripts find everything through the env variable `B`:

- `$B/data/{split}/t*.jsonl` - edits without instructions (copy `../final_triplet/{split}/` here)
- `$B/data/final/{split}/t*.jsonl` - edits with instructions (the same files as `data/` in the HF dataset)
- `$B/data/benchcad/edit-bench/data/edit_bench-00000-of-00001.parquet` - BenchCAD edit-bench
- `$B/outputs/` and `$B/checkpoints/{patch,rewrite}/final/` - outputs and adapters

Scripts were run in this order:

1. `instructions/gen_instructions.py` - Phi-4 (vLLM) writes 3 instructions per edit, keeps the first that passes the checks, else we use a template. Split over GPUs with `SHARD` / `NSHARDS`. Output: `$B/outputs/instructions/shard*.jsonl`.
2. `instructions/add_instructions.py` - adds the instructions to the edits, giving `$B/data/final/{split}/t*.jsonl`.
3. `model/build_sft.py` - the fine-tuning files, one with the patch as the answer and one with the whole program.
4. `model/train_lora.py` - the LoRA fine-tune (`MODE=patch` or `rewrite`, run with torchrun). `RESUME=<checkpoint dir>` continues a stopped run. Saves the adapter in `$B/checkpoints/{MODE}/final/` and the losses in `log_history.json`.
5. `model/infer.py` - greedy outputs for a whole test set (`SYSTEM` = zs_patch, zs_rewrite, sft_patch or sft_rewrite,
   `DATASET` = test or benchcad). Output is saved in `$B/outputs/preds/`.
6. `model/latency.py` - seconds per edit at batch size 1 for all four methods, on the same GPU and the same 30 rows.
7. `eval/evaluate.py` - applies each output (with `eval/apply_patch.py`), runs the program in its own process with a
   120 s limit and scores it. `SYSTEM=copy` is the no-edit baseline. Output: `$B/outputs/scores/`.
8. `eval/summarize.py` - the results tables and failure counts in `$B/outputs/results/`.
9. `eval/plots.py` - the dataset, loss and results figures.
10. `eval/pick_examples.py`, then `eval/render.py` - picks example rows by a fixed rule (seed 0) and draws the original, the target and each system's edit.

`interface/app.py` is the Gradio demo, and `interface/setup_demo.sh` sets it up on a new machine from Hugging Face.
