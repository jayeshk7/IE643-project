import os, ast, json, random, difflib
import multiprocessing as mp
from collections import Counter
from itertools import zip_longest
import cadquery

SPLIT = "train"
# SPLIT = "val"
# SPLIT = "test"

source = f"../final_data_split/{SPLIT}/"
final_triplet_path = f"../final_triplet/{SPLIT}/t1.jsonl"

MAX_TRIES = 6                          # candidate edits per script, we keep the first one that works
TIMEOUT = 120                          # seconds per script (original + all its candidates)
SCALES = [0.5, 0.75, 1.25, 1.5, 2.0]   # new value = old value * one of these


# ---------- where can we edit (python ast) ----------

def find_edit_sites(src):
    tree = ast.parse(src)

    # how much each name is read & assigned in the whole script
    loads, stores = Counter(), Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loads[node.id] += 1
            else:
                stores[node.id] += 1

    sites = []
    for node in tree.body:   # top-level lines only
        # only `name = number`
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name, lit = node.targets[0].id, node.value
        # negative numbers are UnaryOp, not Constant, so they are skipped here too
        if not (isinstance(lit, ast.Constant) and type(lit.value) in (int, float)):
            continue
        text = ast.get_source_segment(src, lit)
        # skip zero, unused names, names assigned twice, and odd literals like 1e-3
        if lit.value <= 0 or loads[name] == 0 or stores[name] != 1 or not text.replace(".", "", 1).isdigit():
            continue
        # where the number is: line + start / end column (utf-8 byte offsets)
        sites.append(dict(kind="param", name=name, value=lit.value, text=text, lineno=lit.lineno,
                          col=lit.col_offset, end_col=lit.end_col_offset))
    return sites

def find_call_sites(src):
    sites = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        for lit in node.args + [k.value for k in node.keywords]:
            if not (isinstance(lit, ast.Constant) and type(lit.value) in (int, float)):
                continue
            text = ast.get_source_segment(src, lit)
            if lit.value <= 0 or not text.replace(".", "", 1).isdigit():
                continue
            sites.append(dict(kind="call", value=lit.value, text=text, lineno=lit.lineno,
                              col=lit.col_offset, end_col=lit.end_col_offset))
    return sites


# ---------- make the edit ----------

# pick a new value for a site: returns (new value, new text) or None
def propose(site, rng):
    old = site["value"]

    # ints stay ints
    if isinstance(old, int):
        cands = sorted({round(old * k) for k in SCALES} - {old, 0})
        if not cands:
            return None
        new = rng.choice(cands)
        return new, str(new)

    # floats keep the original's number of decimals (at least 1)
    if "." in site["text"]:
        decimals = max(len(site["text"].split(".")[1]), 1)
    else:
        decimals = 1
    cands = sorted({f"{old * k:.{decimals}f}" for k in SCALES} - {f"{old:.{decimals}f}"})
    cands = [c for c in cands if float(c) > 0]
    if not cands:
        return None
    text = rng.choice(cands)
    return float(text), text

def apply_edit(src, site, new_text):
    lines = src.split("\n")
    line = lines[site["lineno"] - 1].encode()
    line = line[:site["col"]] + new_text.encode() + line[site["end_col"]:]
    lines[site["lineno"] - 1] = line.decode()
    return "\n".join(lines)

# lines with their "\n", the way difflib wants them (s always ends with "\n")
def split_lines(s):
    return [l + "\n" for l in s.split("\n")[:-1]]

def make_patch(original, edited):
    diff = difflib.unified_diff(split_lines(original), split_lines(edited), "a/part.py", "b/part.py", n=3)
    return "".join(diff)


# ---------- run the scripts + check the geometry (in a child process) ----------

def build(program):
    code = {}
    exec(program, code)
    return code["result"].solids().vals()

# is this edit good
def check_edit(a, va, edited):
    try:
        solids = build(edited)
        if len(solids) != 1:   # must still be exactly one solid
            return None
        b = solids[0]
        vb = b.Volume()
        if not b.isValid() or vb <= 0:
            return None
        # iou = volume(a and b) / volume(a or b)
        inter = a.intersect(b).Volume()
        inter2 = b.intersect(a).Volume()

        # occt sometimes gets this wrong silently: both orders must agree and fit inside both solids
        if abs(inter - inter2) > 1e-4 * min(va, vb) or not 0 <= inter <= min(va, vb) * (1 + 1e-4):
            return None
        iou = inter / (va + vb - inter)
    except Exception:
        return None
    # the edit must change the geometry
    # (occt booleans run in parallel, so even the same script moves volume / iou by ~1e-5)
    dvol = abs(vb - va) / va
    if dvol <= 1e-4 and 1 - iou <= 1e-4:
        return None
    return iou, va, vb

# runs in the child: build the original once, then try the edits in order and stop at the first good one
# sends "bad_original", or one entry per edit tried (None = rejected)
def check_script(original, edits, conn):
    try:
        solids = build(original)
        good_original = len(solids) == 1 and solids[0].isValid() and solids[0].Volume() > 0
    except Exception:
        good_original = False
    if not good_original:
        conn.send("bad_original")
        return
    a = solids[0]
    va = a.Volume()

    results = []
    for edited in edits:
        res = check_edit(a, va, edited)
        results.append(res)
        if res is not None:   # one edit per script is enough
            break
    conn.send(results)

def run_script(original, edits):
    parent, child = mp.Pipe()
    p = mp.Process(target=check_script, args=(original, edits, child))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():   # timeout
        p.kill()
        p.join()
        return []
    if not parent.poll():   # crashed without sending anything
        return []
    results = parent.recv()
    if results == "bad_original":
        return None
    return results


# ---------- one script -> at most one triplet ----------

def process(path):
    uuid = os.path.basename(path)[:-3]

    with open(path) as f:
        original = f.read()
    if not original.endswith("\n"):   # 20% of files, the patch needs it
        original += "\n"

    rng = random.Random(uuid)   # same edits every run

    # both kinds of sites, shuffled, random kind first, then alternate the two kinds
    # so about half of the kept edits are named params and half are numbers inside calls
    kinds = [find_edit_sites(original), find_call_sites(original)]
    for k in kinds:
        rng.shuffle(k)
    rng.shuffle(kinds)
    sites = [s for pair in zip_longest(*kinds) for s in pair if s is not None]

    # up to MAX_TRIES candidate edits
    cands = []
    for site in sites:
        prop = propose(site, rng)
        if prop:
            new_value, new_text = prop
            cands.append((site, new_text))
        if len(cands) == MAX_TRIES:
            break
    if not cands:
        return []

    edits = [apply_edit(original, site, new_text) for site, new_text in cands]
    results = run_script(original, edits)
    if results is None:
        return []

    rows = []
    for (site, _), edited, res in zip(cands, edits, results):
        if res is None:
            continue
        iou, va, vb = res
        # site_kind: param or call, to compare the model on both later (benchcad is all call)
        rows.append(dict(uuid=uuid, edit_type="T1_literal_replace", site_kind=site["kind"],
                         iou=iou, vol_orig=va, vol_edit=vb,
                         original=original, edited=edited, patch=make_patch(original, edited)))
    return rows


if __name__ == "__main__":

    # add a slice like [:20] here to go through fewer scripts
    files = sorted(os.listdir(source))

    with open(final_triplet_path, "w") as file:
        for f in files:
            for row in process(source + f):
                file.write(json.dumps(row) + "\n")

    print(f"created triples using {len(files)} files from {source}")
