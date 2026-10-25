import os
import sys, ast, json, time
import multiprocessing as mp

for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"   # one thread for numpy etc., the same setting as our runs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path += [HERE, os.path.join(HERE, "..", "model")]   # apply_patch, prompts
from apply_patch import apply_patch, PatchError
from prompts import strip_fence


B = os.environ["B"]
SYSTEM = os.environ.get("SYSTEM", "copy")    # defaults are there because the interface imports this file
DATASET = os.environ.get("DATASET", "test")
TIMEOUT = 120      # seconds per row
if SYSTEM == "copy":
    MODE = "copy"
elif "rewrite" in SYSTEM:
    MODE = "rewrite"
else:
    MODE = "patch"
out_path = f"{B}/outputs/scores/{SYSTEM}_{DATASET}.jsonl"

import cadquery as cq


# ---------- the test rows: id -> original program + target ----------

def load_refs():
    refs = {}
    if DATASET == "test":
        # our test split: the target is the edited program
        for t in ["t1", "t2", "t3"]:
            for line in open(f"{B}/data/final/test/{t}.jsonl"):
                r = json.loads(line)
                refs[f"{t}_{r['uuid']}"] = dict(type=t, original=r["original"], target_code=r["edited"], target_step=None,
                                                site_kind=r.get("site_kind"))
    else:
        # benchcad: the target solid comes as a step file (we write each one to disk once)
        import pandas as pd
        df = pd.read_parquet(f"{B}/data/benchcad/edit-bench/data/edit_bench-00000-of-00001.parquet")
        os.makedirs(f"{B}/data/benchcad/steps", exist_ok=True)
        for r in df.itertuples():
            step = f"{B}/data/benchcad/steps/{r.record_id}_gt.step"
            if not os.path.exists(step):
                open(step, "wb").write(r.gt_step)
            refs[str(r.record_id)] = dict(type=r.category, original=r.orig_code, target_code=r.gt_code,
                                          target_step=step, edit_type=r.edit_type)
    return refs


# ---------- geometry (runs in the child) ----------

# run a program and return the solids of its part
# the part = `result` (zero-to-cad style), else the last object passed to show_object.
# shown_first (benchcad): the last shown object wins, since a benchcad program displays its part with
# show_object(result); an edit to `result` after that call does not change what the program shows
def run_program(code, shown_first=False):
    shown = []
    ns = {"show_object": lambda obj, *a, **k: shown.append(obj)}   # cadquery has no show_object, we record it
    exec(code, ns)
    if shown_first and shown:
        obj = shown[-1]
    else:
        obj = ns.get("result", shown[-1] if shown else None)
    if obj is None:
        raise ValueError("no result")
    if isinstance(obj, cq.Workplane):
        return obj.solids().vals()
    if isinstance(obj, cq.Shape):
        return obj.Solids()
    raise ValueError("result is not a shape")

def volume(solids):
    return sum(s.Volume() for s in solids)

# exact iou = volume(a and b) / volume(a or b), over all solid pairs (benchcad targets can have several)
# None if the two boolean orders disagree (occt failed silently)
def occt_iou(a, b):
    va, vb = volume(a), volume(b)
    i1 = sum(x.intersect(y).Volume() for x in a for y in b)
    i2 = sum(y.intersect(x).Volume() for x in a for y in b)
    if abs(i1 - i2) > 1e-4 * min(va, vb) or not 0 <= i1 <= min(va, vb) * (1 + 1e-4):
        return None
    return i1 / (va + vb - i1)

# all geometry scores of one edited program, sent back to the parent
def score(ref, edited, conn):
    # occt on one cpu core and one thread, without parallel booleans: the settings of our runs
    os.sched_setaffinity(0, {os.cpu_count() - 1})
    from OCP.OSD import OSD_ThreadPool
    from OCP.BOPAlgo import BOPAlgo_Options
    OSD_ThreadPool.DefaultPool_s(1)
    BOPAlgo_Options.SetParallelMode_s(False)
    res = {}
    try:
        pred = run_program(edited, shown_first=DATASET == "benchcad")
        res["n_solids"] = len(pred)
        res["valid"] = bool(pred) and all(s.isValid() for s in pred) and volume(pred) > 0
        if res["valid"]:
            if ref["target_step"]:
                target = cq.importers.importStep(ref["target_step"]).solids().vals()
            else:
                target = run_program(ref["target_code"])
            res["occt_iou"] = occt_iou(pred, target)
    except Exception as e:
        res["error"] = type(e).__name__
    conn.send(res)   # a small dict, so it never blocks on a full pipe

# run score() in a child process with a time limit
def score_in_child(ref, edited):
    parent, child = mp.Pipe()
    p = mp.get_context("fork").Process(target=score, args=(ref, edited, child))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():
        p.kill()
        p.join()
        return {"error": "timeout"}
    if not parent.poll():   # died without sending anything (occt segfault)
        return {"error": "crash"}
    try:
        return parent.recv()
    except EOFError:
        return {"error": "crash"}

# exact match: same python syntax tree (so spaces and comments don't matter)
def same_ast(a, b):
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return False


if __name__ == "__main__":
    refs = load_refs()
    if MODE == "copy":
        preds = [dict(id=k, output=None, n_tokens=0) for k in refs]
    else:
        preds = [json.loads(l) for l in open(f"{B}/outputs/preds/{SYSTEM}_{DATASET}.jsonl")]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    with open(out_path, "w") as out:
        for i, p in enumerate(preds):
            ref = refs[p["id"]]
            row = dict(id=p["id"], type=ref["type"], system=SYSTEM, dataset=DATASET, n_tokens=p.get("n_tokens"),
                       site_kind=ref.get("site_kind"), edit_type=ref.get("edit_type"))

            # model output -> edited program
            if MODE == "copy":
                edited = ref["original"]
                row["applied"] = True
            elif MODE == "rewrite":
                edited = strip_fence(p["output"])
                row["applied"] = True
            else:
                try:
                    edited = apply_patch(ref["original"], strip_fence(p["output"]))
                    row["applied"] = True
                except PatchError as e:
                    edited = None
                    row["applied"] = False
                    row["error"] = "patch_" + str(e).replace(" ", "_")
            row["exact"] = edited is not None and same_ast(edited, ref["target_code"])

            # edited program -> geometry scores
            if edited is not None:
                row.update(score_in_child(ref, edited))
            out.write(json.dumps(row) + "\n")

            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(preds)} rows, {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {len(preds)} rows to {out_path} in {time.time() - t0:.0f}s")
