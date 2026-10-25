# Editing CAD programs from plain-English instructions

IE 643 course project by Jayesh Kandpal (25M0819), team Tired Perceptron.

The idea: you have a CadQuery program and you want one change, like "increase the plate length from 80 to 120". Instead of asking an LLM to rewrite the whole program, we fine-tuned Qwen2.5-Coder-7B-Instruct with LoRA so that it writes only a program diff. Once the diff is applied, the edited program is run, and the solid it builds is compared with the target solid.

The dataset and the two trained LoRA adapters are on Hugging Face:

- dataset: https://huggingface.co/datasets/jajamken/DLTP_project
- adapters (`patch_lora/` and `rewrite_lora/`): https://huggingface.co/jajamken/DLTP_project

## Results

There are two test sets: our own test split (2,800 edits on 1,000 programs) and BenchCAD edit-bench (748 edits), which the model never sees during training. The main number is normalized IoU:

    clip((IoU(edit, target) - IoU(original, target)) / (1 - IoU(original, target)), 0, 1)

so 0 means no better than leaving the program as it is, and 1 means the target solid exactly. Exact match means the edited program has the same Python AST as the target. Latency is the mean time per edit on the same 30 test rows, on one RTX A6000 with vLLM and batch size 1.

| system | test: normalized IoU | test: exact % | avg output tokens | avg sec / edit | BenchCAD: normalized IoU |
|---|---|---|---|---|---|
| no edit (copy the original) | 0 | 0 | 0 | - | 0 |
| zero-shot, patch | 0.214 | 11.2 | 175 | 3.69 | 0.105 |
| zero-shot, whole-program rewrite | 0.877 | 52.8 | 481 | 10.77 | **0.450** |
| fine-tuned rewrite (baseline) | **0.988** | **99.6** | 470 | 11.63 | 0.266 |
| fine-tuned patch (ours) | 0.947 | 95.1 | **84** | **2.15** | 0.119 |

On our test split the patch model gets close to the rewrite model while using about 5x fewer tokens, and it is about 5x faster. On BenchCAD both fine-tuned models do badly and the zero-shot rewrite is best. BenchCAD programs are written differently and include edit types we never trained on.

## How the data was made

1. Downloaded Zero-To-CAD-100k (100,516 CadQuery programs) and kept the programs that build exactly one solid.
2. Sampled 10,000 train / 1,000 val / 1,000 test programs, keeping Zero-To-CAD's own splits.
3. Made three kinds of edits with scripts: T1 changes one number, T2 rotates or mirrors the whole part, T3 sets one length to k times another length.
4. Phi-4 writes the instruction for each edit. It is told exactly what changed and writes 3 candidates, and we keep the first one that passes a few simple checks (a template sentence if none does).

## Code

Everything is in `scripts/`, and `scripts/readme.md` has a bit more about each script.

- `data_filtration/` - download, the one-solid filter, sampling the splits
- `dataset_creation/` - the T1, T2 and T3 edit generators
- `instructions/` - Phi-4 instructions, and merging them into the data
- `model/` - prompt format, fine-tuning files, LoRA training, inference, latency
- `eval/` - applying patches, scoring, results tables, plots and 3D renders
- `interface/` - the Gradio demo and a script that sets it up

## Demo

The interface needs one GPU (the bf16 weights alone are about 15 GB), both adapters in
`$B/checkpoints/{patch,rewrite}/final/` and the test split in `$B/data/final/test/`:

```bash
export B=/some/big/dir TMPDIR=/some/big/dir/tmp && mkdir -p $TMPDIR   # app.py needs both
CUDA_VISIBLE_DEVICES=0 python scripts/interface/app.py
# on your laptop: ssh -N -L 7860:localhost:7860 <user>@<server>, then open http://localhost:7860
```

On a fresh machine, `B=/some/big/dir bash scripts/interface/setup_demo.sh` makes its own venv and downloads the base model, the adapters and the test split from Hugging Face (about 22 GB). The run command is at the top of the script. On an A6000 the setup took a few minutes.

## Credits

- Zero-To-CAD-100k (ADSKAILab, Apache-2.0) for the source programs
- BenchCAD edit-bench (CC-BY-4.0) as the second test set
- Qwen2.5-Coder-7B-Instruct (Apache-2.0) as the base model
- Phi-4 (Microsoft, MIT) for writing the instructions
- CadQuery, OpenCascade (OCP), vLLM, Hugging Face transformers and PEFT

## AI Usage Declaration

- Several programs were written with the help of AI. However, we decided what each script does and we reviewed everything.
