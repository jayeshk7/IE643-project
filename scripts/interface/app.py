# the demo: paste or upload a cadquery program, type an instruction, and see the model's patch, the edited
# program and both parts in 3d, with a few checks and the iou
# CUDA_VISIBLE_DEVICES=0 python app.py on the gpu machine, then on the laptop
#   ssh -N -L 7860:localhost:7860 <user>@<gpu-server> and open http://localhost:7860

import os, sys, json, time, random, tempfile
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path += [os.path.join(HERE, "..", "model"), os.path.join(HERE, "..", "eval")]
import torch, gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from prompts import messages, strip_fence
from apply_patch import apply_patch, PatchError

B = os.environ["B"]
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
# our two trained lora adapters (from hugging face, see setup_demo.sh)
ADAPTERS = {"patch": f"{B}/checkpoints/patch/final", "rewrite": f"{B}/checkpoints/rewrite/final"}
# each model choice in the ui: which adapter and which prompt. None means no adapter, i.e. the zero-shot base model
CHOICES = {"Fine-tuned patch model (ours)": ("patch", "patch"),
           "Fine-tuned full-rewrite model (baseline)": ("rewrite", "rewrite"),
           "Zero-shot base model, patch prompt": (None, "patch")}
TIMEOUT = 60   # seconds for running the programs + the geometry checks

# the base model, loaded once (bf16, ~15 GB on the gpu)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
# both adapters on the same base model, by name. each lora layer then has the frozen W plus
# one pair of A, B matrices for "patch" and one for "rewrite"
model = PeftModel.from_pretrained(model, ADAPTERS["patch"], adapter_name="patch")
model.load_adapter(ADAPTERS["rewrite"], adapter_name="rewrite")
model.eval()
with torch.no_grad():   # warm-up, so the first real edit is not slowed by cuda init
    model.generate(**tok("import cadquery as cq", return_tensors="pt").to(model.device), max_new_tokens=8, do_sample=False)

# 3 examples per edit type from our test split, so the target is known
examples = {}
rng = random.Random(0)
for t, label in [("t1", "T1 value change"), ("t2", "T2 rotate / mirror"), ("t3", "T3 relative value")]:
    rows = [json.loads(l) for l in open(f"{B}/data/final/test/{t}.jsonl")]
    for i, r in enumerate(rng.sample(rows, 3)):
        examples[f"{label} #{i + 1} ({r['uuid'][:8]})"] = r


# the model's reply for one edit: same prompt as training, greedy decoding
def generate(choice, instruction, program):
    adapter, mode = CHOICES[choice]
    text = tok.apply_chat_template(messages(mode, instruction, program), add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").to(model.device)
    max_new = 768 if mode == "patch" else 4096   # a patch is short, a whole program can be long
    start = time.time()
    with torch.no_grad():
        if adapter is None:
            # zero-shot: lora switched off, only the base weights W are used
            with model.disable_adapter():
                out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
        else:
            # fine-tuned: every lora layer computes W x + (alpha / r) * B A x with this adapter's A and B
            model.set_adapter(adapter)
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
    reply = out[0, ids["input_ids"].shape[1]:]   # only the new tokens, not the prompt
    return tok.decode(reply, skip_special_tokens=True), mode, len(reply), time.time() - start


# ---------- geometry in a child process (cadquery can hang) ----------

def _geometry(original, edited, target, conn):
    import cadquery as cq
    from evaluate import run_program, occt_iou, volume
    res, d = {}, tempfile.mkdtemp(dir=os.environ["TMPDIR"])
    stage = "original program"   # which step failed, for the error message
    try:
        a = run_program(original)
        res["vol_orig"] = volume(a)
        res["stl_orig"] = f"{d}/original.stl"
        cq.exporters.export(cq.Compound.makeCompound(a), res["stl_orig"])
        stage = "edited program"
        b = run_program(edited)
        res["n_solids"], res["vol_edit"] = len(b), volume(b)
        res["stl_edit"] = f"{d}/edited.stl"
        cq.exporters.export(cq.Compound.makeCompound(b), res["stl_edit"])
        res["iou_vs_original"] = occt_iou(a, b)
        if target:
            stage = "ground-truth program"
            t = run_program(target)
            res["iou_vs_target"], res["iou_orig_target"] = occt_iou(b, t), occt_iou(a, t)
    except Exception as e:
        res["error"] = f"{stage}: {type(e).__name__}: {e}"[:300]
    conn.send(res)

def geometry(original, edited, target):
    parent, child = mp.Pipe()
    p = mp.get_context("fork").Process(target=_geometry, args=(original, edited, target, child))
    p.start(); p.join(TIMEOUT)
    if p.is_alive():
        p.kill(); p.join()
        return {"timeout": True}
    return parent.recv() if parent.poll() else {"error": "geometry process (it crashed)"}


# normalized iou like in the report: clip((iou(edited, target) - iou(original, target)) / (1 - iou(original, target)), 0, 1)
def normalized(g):
    i, b = g.get("iou_vs_target"), g.get("iou_orig_target")
    if i is None or b is None or b >= 1 - 1e-9:
        return None
    return min(max((i - b) / (1 - b), 0.0), 1.0)

def run(choice, instruction, program, example):
    if not program.endswith("\n"):
        program += "\n"
    # the ground truth is known only if the example's program and instruction are both unchanged
    ex = examples.get(example)
    target = ex["edited"] if ex and ex["original"] == program and ex["instruction"].strip() == instruction.strip() else None
    raw, mode, n_tokens, secs = generate(choice, instruction, program)
    lines = [f"model: {choice}", f"output tokens: {n_tokens}, generation time: {secs:.1f} s"]
    if mode == "patch":
        try:
            edited = apply_patch(program, strip_fence(raw))
            lines.append("patch applied: yes")
        except PatchError as e:
            lines.append(f"patch applied: NO ({e})")
            if target is not None:
                lines.append("normalized IoU: 0 (a failed edit counts as 0)")
            return raw, "", None, None, "\n".join(lines)
    else:
        edited = strip_fence(raw)
    g = geometry(program, edited, target)
    if g.get("timeout"):
        lines.append(f"geometry check: timed out after {TIMEOUT} s (this part is slow to build and compare), not scored")
    elif "error" in g:
        lines.append(f"FAILED in the {g['error']}")
        if target is not None:
            lines.append("normalized IoU: 0 (a failed edit counts as 0)")
    else:
        lines.append(f"solids in the edited part: {g['n_solids']}")
        lines.append(f"volume: {g['vol_orig']:.1f} -> {g['vol_edit']:.1f}")
        if g.get("iou_vs_original") is not None:
            lines.append(f"IoU(edited, original): {g['iou_vs_original']:.4f}")
        if target is not None and g.get("iou_vs_target") is not None:
            lines.append(f"IoU(edited, ground-truth target): {g['iou_vs_target']:.4f}")
        if target is not None and normalized(g) is not None:
            lines.append(f"normalized IoU: {normalized(g):.3f} (1 = target reached, 0 = no better than the original)")
    if target is not None:
        from evaluate import same_ast   # same exact-match check as evaluate.py
        lines.append(f"exact match (same AST as the ground truth): {'yes' if same_ast(edited, target) else 'no'}")
    return raw, edited, g.get("stl_orig"), g.get("stl_edit"), "\n".join(lines)

def load_example(name):
    r = examples[name]
    return r["original"], r["instruction"]

def load_file(f):
    return open(f).read() if f else ""


with gr.Blocks(title="CAD edit") as demo:
    gr.Markdown("## Natural-language editing of CadQuery programs\nIE 643 course project, team Tired Perceptron. "
                "Give a CadQuery program (the part must be in a variable `result`) and an edit instruction.")
    with gr.Row():
        with gr.Column():
            example = gr.Dropdown(list(examples), label="Example from our test split (optional)")
            upload = gr.File(label="or upload a .py file", file_types=[".py"])
            program = gr.Code(label="Original CadQuery program", language="python", lines=22)
            instruction = gr.Textbox(label="Edit instruction", placeholder="Increase the plate length from 80 to 120.")
            choice = gr.Radio(list(CHOICES), value=list(CHOICES)[0], label="Model")
            go = gr.Button("Edit", variant="primary")
        with gr.Column():
            raw = gr.Code(label="Model output (patch or program)", language="python", lines=10)
            status = gr.Textbox(label="Checks", lines=7)
            with gr.Row():
                view_a = gr.Model3D(label="Original part")
                view_b = gr.Model3D(label="Edited part")
            edited = gr.Code(label="Edited program", language="python", lines=12)
    example.change(load_example, example, [program, instruction])
    upload.change(load_file, upload, program)
    go.click(run, [choice, instruction, program, example], [raw, edited, view_a, view_b, status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)))
