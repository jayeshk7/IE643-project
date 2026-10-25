# draws the original part, the target and each system's edit side by side
# IDS="t1_<uuid>,t3_<uuid>" SYSTEMS="zs_patch,sft_patch" DATASET=test python render.py
# saves $B/outputs/figures/qual_{DATASET}_{id}.png (OUT=<dir> for another folder, FS=19 for bigger slide labels)

import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path += [HERE, os.path.join(HERE, "..", "model")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import cadquery as cq
from apply_patch import apply_patch
from prompts import strip_fence
from evaluate import run_program

B = os.environ["B"]
IDS = os.environ["IDS"].split(",")
SYSTEMS = os.environ["SYSTEMS"].split(",")
DATASET = os.environ.get("DATASET", "test")
out_dir = os.environ.get("OUT", f"{B}/outputs/figures")
FS = float(os.environ.get("FS", 9))   # label size
NAMES = {"zs_patch": "zero-shot patch", "zs_rewrite": "zero-shot rewrite", "sft_rewrite": "fine-tuned rewrite",
         "sft_patch": "fine-tuned patch (ours)"}
LIGHT = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])

# id -> original program, target program and instruction
def refs():
    out = {}
    if DATASET == "test":
        for t in ["t1", "t2", "t3"]:
            for line in open(f"{B}/data/final/test/{t}.jsonl"):
                r = json.loads(line)
                out[f"{t}_{r['uuid']}"] = dict(original=r["original"], target=r["edited"], instruction=r["instruction"])
    else:
        import pandas as pd
        df = pd.read_parquet(f"{B}/data/benchcad/edit-bench/data/edit_bench-00000-of-00001.parquet",
                             columns=["record_id", "instruction", "orig_code", "gt_code"])
        for r in df.itertuples():
            out[str(r.record_id)] = dict(original=r.orig_code, target=r.gt_code, instruction=r.instruction)
    return out

# id -> what the system wrote
def preds(system):
    out = {}
    for line in open(f"{B}/outputs/preds/{system}_{DATASET}.jsonl"):
        r = json.loads(line)
        out[r["id"]] = r["output"]
    return out

# id -> the system's scores (empty if it was not scored)
def scores(system):
    out = {}
    path = f"{B}/outputs/scores/{system}_{DATASET}.jsonl"
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out[r["id"]] = r
    return out

# matplotlib 3d has no lighting, so each face is shaded by its angle to one light direction
def draw(ax, code, title, box, color):
    ax.set_title(title, fontsize=FS)
    ax.set_axis_off()
    if code is None:
        ax.text2D(0.15, 0.5, "patch did not apply", transform=ax.transAxes, color="#d03030", fontsize=FS)
        code = ""
    try:
        solids = run_program(code, DATASET == "benchcad")
        verts, faces = cq.Compound.makeCompound(solids).tessellate(box[1] * 0.004)
        tris = np.array([(p.x, p.y, p.z) for p in verts])[np.array(faces)]   # the 3 corners of every triangle
        normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
        shade = 0.35 + 0.65 * np.abs(normal @ LIGHT)
        colors = np.array(matplotlib.colors.to_rgb(color)) * shade[:, None]
        ax.add_collection3d(Poly3DCollection(tris, facecolors=colors, edgecolor="none"))
    except Exception as e:
        if code:
            ax.text2D(0.15, 0.5, f"program failed:\n{type(e).__name__}", transform=ax.transAxes, color="#d03030", fontsize=FS)
    center, half = box
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)

# a cube (center, half side) around both the original and the target, so both are drawn at the same scale
def box(codes):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for code in codes:
        try:
            bb = cq.Compound.makeCompound(run_program(code, DATASET == "benchcad")).BoundingBox()
        except Exception:
            continue
        lo = np.minimum(lo, [bb.xmin, bb.ymin, bb.zmin])
        hi = np.maximum(hi, [bb.xmax, bb.ymax, bb.zmax])
    if not np.isfinite(lo).all():   # neither program runs
        return np.zeros(3), 1.0
    return (lo + hi) / 2, (hi - lo).max() / 2 * 1.05


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    ref = refs()
    outputs = {s: preds(s) for s in SYSTEMS}
    score = {s: scores(s) for s in SYSTEMS}
    for i in IDS:
        r = ref[i]
        panels = [("original", r["original"], "#9a9890"), ("ground truth", r["target"], "#1baf7a")]
        for s in SYSTEMS:
            out = outputs[s].get(i, "")
            # the edited program: the reply itself for rewrite, the patched original for patch (None if that fails)
            try:
                if "rewrite" in s:
                    code = strip_fence(out)
                else:
                    code = apply_patch(r["original"], strip_fence(out))
            except Exception:
                code = None
            title = NAMES.get(s, s)
            iou = score[s].get(i, {}).get("occt_iou")
            if iou is not None:
                title += f"\nIoU with target {iou:.3f}"
            panels.append((title, code, "#2a78d6"))

        b = box([r["original"], r["target"]])
        fig = plt.figure(figsize=(3.2 * len(panels), 3.6))
        for k, (title, code, color) in enumerate(panels):
            draw(fig.add_subplot(1, len(panels), k + 1, projection="3d"), code, title, b, color)
        fig.suptitle(f"Instruction: {r['instruction'][:160]}", fontsize=FS + 1, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(f"{out_dir}/qual_{DATASET}_{i[:20]}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("saved", i)
