import os, json, random
from create_t1 import run_script, make_patch

SPLIT = "train"
# SPLIT = "val"
# SPLIT = "test"

source = f"../final_data_split/{SPLIT}/"
final_triplet_path = f"../final_triplet/{SPLIT}/t2.jsonl"

MAX_TRIES = 6   # candidate edits per script, we keep the first one that works
ANGLES = [30, 45, 60, 90, 120, 180]
AXES = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
PLANES = ["XY", "YZ", "XZ"]


# all 18 rotations (x / y / z axis through the origin, 6 angles) + 3 mirrors, shuffled, first MAX_TRIES
def candidate_lines(rng):
    lines = []
    for axis in AXES:
        for angle in ANGLES:
            lines.append(f"result = result.rotate((0, 0, 0), {axis}, {angle})")
    for plane in PLANES:
        lines.append(f'result = result.mirror("{plane}")')
    rng.shuffle(lines)
    return lines[:MAX_TRIES]

def process(path):
    uuid = os.path.basename(path)[:-3]

    with open(path) as f:
        original = f.read()
    if not original.endswith("\n"):   # the patch needs it
        original += "\n"

    rng = random.Random(uuid)   # same edits every run

    # add the line at the very end, so it applies to the final result
    edits = [original + line + "\n" for line in candidate_lines(rng)]
    results = run_script(original, edits)
    if results is None:
        return []

    rows = []
    for edited, res in zip(edits, results):
        if res is None:
            continue
        iou, va, vb = res
        # rotate / mirror can't change the volume, so a change means occt got one of the shapes wrong
        if abs(vb - va) / va > 1e-3:
            continue
        rows.append(dict(uuid=uuid, edit_type="T2_chain_transform",
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

    print(f"created {n} t2 triples using {len(files)} files from {source}")
