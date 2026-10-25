# picks the example rows for the 3d figures with a fixed rule instead of by hand (seed 0)
# per edit type: one row where our patch model is right and the zero-shot patch is wrong, and one where ours is wrong
# right = same ast as the target on our test split, normalized iou >= 0.95 on benchcad
# on benchcad only rows whose original program runs for us (the copy row is valid)
# DATASET=test python pick_examples.py, then pass the printed IDS to render.py

import os, json, random

B = os.environ["B"]
DATASET = os.environ.get("DATASET", "test")

def scores(system):
    return {r["id"]: r for r in map(json.loads, open(f"{B}/outputs/scores/{system}_{DATASET}.jsonl"))}

def correct(r):
    if DATASET == "test":
        return bool(r.get("exact"))
    # benchcad: the copy row's iou is iou(original, target)
    v = (r.get("occt_iou") or 0.0) if r.get("valid") else 0.0
    b = copy[r["id"]].get("occt_iou")
    return b is not None and b < 1 and (v - b) / (1 - b) >= 0.95

ours, zs, copy = scores("sft_patch"), scores("zs_patch"), scores("copy")
rng = random.Random(0)
ids = []
for t in sorted({r["type"] for r in ours.values()}):
    rows = [i for i, r in ours.items() if r["type"] == t and (DATASET == "test" or copy[i].get("valid"))]
    win = [i for i in rows if correct(ours[i]) and not correct(zs[i])]
    fail = [i for i in rows if not correct(ours[i])]
    for kind, pool in [("win", win), ("fail", fail)]:
        if pool:
            i = rng.choice(sorted(pool))
            ids.append(i)
            print(t, kind, f"(of {len(pool)})", i, "| ours:", ours[i].get("error") or f"iou {ours[i].get('occt_iou')}",
                  "| zero-shot:", zs[i].get("error") or f"iou {zs[i].get('occt_iou')}")
print("IDS=" + ",".join(ids))
