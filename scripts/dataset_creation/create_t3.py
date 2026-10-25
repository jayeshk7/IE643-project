import os, json, random
from create_t1 import find_edit_sites, apply_edit, run_script, make_patch   # same sites + checks as t1

SPLIT = "train"
# SPLIT = "val"
# SPLIT = "test"

source = f"../final_data_split/{SPLIT}/"
final_triplet_path = f"../final_triplet/{SPLIT}/t3.jsonl"

MAX_TRIES = 6   # candidate edits per script, we keep the first one that works
RATIOS = [0.25, 0.5, 1, 1.5, 2, 3]
NOT_LENGTH = {"angle", "deg", "degree", "degrees", "count", "num", "number", "n"}   # words that mean not a length


# a name is a length unless one of its separated words says angle or count (hole_count, angle_deg, n_teeth)
def is_length(name):
    words = name.lower().split("_")
    return not any(word in NOT_LENGTH for word in words)

# value as text: ints stay ints, floats get up to 4 decimals (2.5000 -> 2.5, 3.0000 -> 3.0)
def fmt(value, is_int):
    if is_int:
        return str(int(value))
    text = f"{value:.4f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text

# all (target site, new value text, relation) candidates, shuffled, first MAX_TRIES
def candidates(sites, rng):
    sites = [s for s in sites if is_length(s["name"])]
    cands = []
    for target in sites:
        is_int = isinstance(target["value"], int)
        for ref in sites:
            # the other parameter, of the same type (int with int, float with float)
            if ref is target or type(ref["value"]) is not type(target["value"]):
                continue
            for k in RATIOS:
                new = k * ref["value"]
                if is_int and new != int(new):
                    continue
                # same edit size as t1: the new value is 0.5x to 2x of the old value
                if not 0.5 <= new / target["value"] <= 2:
                    continue
                text = fmt(new, is_int)
                if float(text) == target["value"]:   # no change
                    continue
                cands.append((target, text, f"{target['name']} = {k} * {ref['name']}"))
    rng.shuffle(cands)
    return cands[:MAX_TRIES]

def process(path):
    uuid = os.path.basename(path)[:-3]

    with open(path) as f:
        original = f.read()
    if not original.endswith("\n"):   # the patch needs it
        original += "\n"

    rng = random.Random(uuid)   # same edits every run

    cands = candidates(find_edit_sites(original), rng)
    if not cands:
        return []

    edits = [apply_edit(original, target, text) for target, text, _ in cands]
    results = run_script(original, edits)
    if results is None:
        return []

    rows = []
    for (_, _, relation), edited, res in zip(cands, edits, results):
        if res is None:
            continue
        iou, va, vb = res
        # relation goes to the instruction llm, the patch alone looks like a t1 edit
        rows.append(dict(uuid=uuid, edit_type="T3_relative_compute", relation=relation,
                         iou=iou, vol_orig=va, vol_edit=vb,
                         original=original, edited=edited, patch=make_patch(original, edited)))
    return rows


if __name__ == "__main__":

    # add a slice like [:20] here to go through fewer scripts
    files = sorted(os.listdir(source))

    n = 0
    with open(final_triplet_path, "w") as file:
        for f in files:
            for row in process(source + f):
                file.write(json.dumps(row) + "\n")
                n += 1

    print(f"created {n} t3 triples using {len(files)} files from {source}")
