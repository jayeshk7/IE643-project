# the figures for the report and slides: dataset stats, loss curves, results
# python plots.py, saves into $B/outputs/figures/ (loss and results figures are skipped if their files are missing)

import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = os.environ["B"]
out = f"{B}/outputs/figures"
os.makedirs(out, exist_ok=True)

# one color per edit type and per system, gray is the no-edit baseline
TYPE_COLOR = {"t1": "#2a78d6", "t2": "#eb6834", "t3": "#1baf7a"}
SYSTEMS = [("copy", "no-edit copy", "#9a9890"), ("zs_patch", "zero-shot patch", "#2a78d6"),
           ("zs_rewrite", "zero-shot rewrite", "#eb6834"), ("sft_rewrite", "fine-tuned rewrite", "#1baf7a"),
           ("sft_patch", "fine-tuned patch (ours)", "#eda100")]
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
NAMES = {"t1": "T1 value change", "t2": "T2 rotate / mirror", "t3": "T3 relative value"}

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{out}/{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------- dataset ----------
rows = []
for split in ["train", "val", "test"]:
    for t in ["t1", "t2", "t3"]:
        for line in open(f"{B}/data/final/{split}/{t}.jsonl"):
            r = json.loads(line)
            rows.append(dict(split=split, type=t, iou=r["iou"], n_instr=len(r["instruction"].split())))
d = pd.DataFrame(rows)
train = d[d.split == "train"]

# rows per split and edit type
fig, ax = plt.subplots(figsize=(6, 3.2))
splits = ["train", "val", "test"]
x = np.arange(3)
for k, t in enumerate(["t1", "t2", "t3"]):
    counts = [len(d[(d.split == s) & (d.type == t)]) for s in splits]
    bars = ax.bar(x + (k - 1) * 0.26, counts, 0.24, color=TYPE_COLOR[t], label=NAMES[t])
    ax.bar_label(bars, labels=[f"{n:,}" for n in counts], fontsize=7, color=INK2, padding=2)
ax.set_xticks(x, ["train (10,000 programs)", "val (996)", "test (1,000)"])
ax.set_yscale("log")
ax.set_ylabel("triplets (log scale)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Triplets per split and edit type", color=INK, fontsize=11, loc="left")
save(fig, "data_rows")

# how much each edit changes the shape: iou(original, edited)
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
for ax, t in zip(axes, ["t1", "t2", "t3"]):
    ax.hist(train[train.type == t].iou, bins=40, range=(0, 1), color=TYPE_COLOR[t], edgecolor="#fcfcfb", linewidth=0.5)
    ax.set_title(NAMES[t], color=INK, fontsize=10, loc="left")
    ax.set_xlabel("IoU(original, edited)")
axes[0].set_ylabel("train triplets")
save(fig, "data_iou")

# instruction length
fig, ax = plt.subplots(figsize=(5, 2.8))
for t in ["t1", "t2", "t3"]:
    ax.hist(train[train.type == t].n_instr, bins=range(0, 40), histtype="step", linewidth=2, color=TYPE_COLOR[t],
            label=NAMES[t])
ax.set_xlabel("words per instruction")
ax.set_ylabel("train triplets")
ax.legend(frameon=False, fontsize=8)
save(fig, "data_instruction_length")


# ---------- training curves ----------
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True)
have = False
for ax, mode, color in zip(axes, ["patch", "rewrite"], ["#eda100", "#1baf7a"]):
    path = f"{B}/checkpoints/{mode}/log_history.json"
    if not os.path.exists(path):
        continue
    have = True
    log = json.load(open(path))
    # the trainer rounds the train loss to 4 decimals, so it is drawn as dots and the zeros are dropped (log axis)
    train_loss = [(e["step"], e["loss"]) for e in log if "loss" in e and e["loss"] > 0]
    val_loss = [(e["step"], e["eval_loss"]) for e in log if "eval_loss" in e]
    ax.plot([s for s, _ in train_loss], [l for _, l in train_loss], color=color, linestyle="none", marker="o",
            markersize=3, label="train (mean of 10 steps)")
    ax.plot([s for s, _ in val_loss], [l for _, l in val_loss], color=INK2, linewidth=2, marker="o", markersize=5,
            label="val (300 rows)")
    if mode == "rewrite":   # resumed on 4 gpus at step 600, still 32 rows per step
        ax.axvline(600, color=INK2, linewidth=1, linestyle="--")
        ax.text(610, 0.02, "resumed on 4 GPUs", fontsize=7, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("step (batch 32)")
    ax.set_ylabel("loss per reply token")
    ax.set_title(f"{mode} fine-tune", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
if have:
    save(fig, "train_loss")
else:
    plt.close(fig)


# ---------- results ----------
path = f"{B}/outputs/results/summary.csv"
if os.path.exists(path):
    res = pd.read_csv(path)
    present = [s for s in SYSTEMS if s[0] in set(res.system)]
    # one bar chart each: (test set, column of summary.csv, y label, groups on the x axis)
    charts = [("test", "norm_occt", "normalized IoU improvement (OCCT)", ["all", "t1", "t2", "t3"]),
              ("test", "exact", "exact match % (same AST)", ["all", "t1", "t2", "t3"]),
              ("benchcad", "valid", "edited program gives a valid solid %", ["all", "T1", "T2", "T3", "T4", "T5"])]
    for dataset, metric, label, types in charts:
        sub = res[res.dataset == dataset]
        if sub.empty:
            continue
        # the no-edit copy is always 0 on our test split, so it is not drawn there
        shown = [s for s in present if not (dataset == "test" and s[0] == "copy")]
        fig, ax = plt.subplots(figsize=(1.3 * len(types) + 2.5, 3.2))
        w = 0.8 / len(shown)   # bar width: the systems stand side by side in each group
        for k, (s, name, color) in enumerate(shown):
            v = [sub[(sub.system == s) & (sub.type == t)][metric].mean() for t in types]
            offset = (k - (len(shown) - 1) / 2) * w
            ax.bar(np.arange(len(types)) + offset, v, w * 0.92, color=color, label=name)
        ax.set_xticks(np.arange(len(types)), [t.upper() if t != "all" else "all" for t in types])
        ax.set_ylabel(label)
        if dataset == "test":
            title = "our test split (2,800 edits)"
        else:
            title = "BenchCAD edit-bench (748 edits)"
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8, ncol=len(shown), loc="upper center", bbox_to_anchor=(0.5, -0.12))
        save(fig, f"res_{dataset}_{metric}")

    # output length: patch vs rewrite
    sub = res[(res.type == "all") & (res.dataset == "test")]
    if len(sub):
        fig, ax = plt.subplots(figsize=(5, 2.8))
        ps = [s for s in present if s[0] != "copy"]
        v = [sub[sub.system == s].out_tokens.mean() for s, _, _ in ps]
        bars = ax.barh([name for _, name, _ in ps], v, color=[color for _, _, color in ps])
        ax.bar_label(bars, labels=[f"{x:.0f}" for x in v], fontsize=8, color=INK2, padding=3)
        ax.set_xlabel("output tokens per edit (our test split)")
        save(fig, "res_tokens")

# ---------- accuracy vs time per edit ----------
path = f"{B}/outputs/results/latency.json"
if os.path.exists(path) and os.path.exists(f"{B}/outputs/results/summary.csv"):
    lat = json.load(open(path))
    sub = res[(res.type == "all") & (res.dataset == "test")]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for s, name, color in SYSTEMS:
        if s not in lat:
            continue
        x = lat[s]["latency_mean_s"]
        y = sub[sub.system == s].norm_occt.mean()
        ax.plot(x, y, "o", color=color, markersize=9)
        dy = -4 if y > 0.5 else 4   # label just below the dot when it is high up, else just above
        ax.annotate(f"{name}\n{x:.2f} s, {y:.3f}", (x, y), textcoords="offset points", xytext=(8, dy), fontsize=8,
                    color=INK)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("seconds per edit (1 A6000, batch 1, 30 test rows)")
    ax.set_ylabel("normalized IoU improvement (test)")
    ax.set_title("accuracy vs time per edit", color=INK, fontsize=11, loc="left")
    save(fig, "res_tradeoff")
