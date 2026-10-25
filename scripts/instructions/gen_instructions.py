import os, re, json

B = os.environ["B"]
SHARD = int(os.environ.get("SHARD", 0))
NSHARDS = int(os.environ.get("NSHARDS", 1))
MODEL = "microsoft/phi-4"
N_CANDIDATES = 3
SPLITS = ["train", "val", "test"]
TYPES = ["t1", "t2", "t3"]
out_path = f"{B}/outputs/instructions/shard{SHARD}.jsonl"

AXIS = {(1, 0, 0): "X", (0, 1, 0): "Y", (0, 0, 1): "Z"}

# words that show the ratio was written in the instruction, per ratio
RATIO_WORDS = {0.25: ["quarter", "0.25", "1/4", "one-fourth", "one fourth"],
               0.5: ["half", "0.5", "1/2"],
               1.0: ["equal", "same", "match"],
               1.5: ["1.5", "one and a half", "one-and-a-half", "3/2"],
               2.0: ["twice", "double", "2 times", "two times", "2x", "2×"],
               3.0: ["three times", "3 times", "triple", "thrice", "3x", "3×"]}

# the ratio in words, for the template sentence
RATIO_TEXT = {0.25: "one quarter of the", 0.5: "half the", 1.0: "the", 1.5: "1.5 times the", 2.0: "twice the", 3.0: "three times the"}

# system prompt: style rules + made-up examples (no benchcad sentences)
SYSTEM = """You write edit instructions for CadQuery (Python CAD) programs, the way an engineer asks a colleague for one change to a part.
Rules:
1. Exactly one imperative sentence, at most 35 words.
2. Name the feature in plain English by what it is on the part (for example "the wall thickness", "the chamfer on the edges", "the box height", "the number of holes in the ring"). Never write variable names, underscores, code, function calls or line numbers.
3. Value change: start with a verb such as Increase, Decrease, Change or Make, and give the old and the new value as numbers ("from A to B").
4. Rotation or mirror: say it applies to the entire part, give the angle in degrees and the axis (X, Y or Z), or the mirror plane (XY, YZ or XZ).
5. Relation: start with "Set the ... equal to", express the factor in words (for example "twice", "half of", "one quarter of", "1.5 times", "three times") and name both features. Do not state the resulting number. You may add that the other feature is unchanged.
6. Output only the instruction sentence, nothing else.

Examples (made up, for style only):
- Increase the bracket thickness from 4 to 6.
- Decrease the slot length from 30.0 to 22.5.
- Rotate the entire part by 45 degrees about the Z axis.
- Mirror the entire part across the XZ plane.
- Set the rib height equal to twice the base thickness; the base thickness is unchanged."""


# ---------- facts about the edit, read from the patch ----------

# the removed lines (-) and the added lines (+) of the patch
def changed_lines(patch):
    minus, plus = [], []
    for l in patch.splitlines():
        if l.startswith("-") and not l.startswith("---"):
            minus.append(l[1:])
        if l.startswith("+") and not l.startswith("+++"):
            plus.append(l[1:])
    return minus, plus

# the patch without its header lines (---, +++, @@)
def hunk(patch):
    return "\n".join(l for l in patch.splitlines() if not l.startswith(("---", "+++", "@@")))

# a plain number like 40 or 2.5 (not part of a name like x2)
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")

# the one number that differs between the old and the new line: (old, new, position in the old line)
def changed_number(old_line, new_line):
    old_nums = list(NUM.finditer(old_line))
    new_nums = list(NUM.finditer(new_line))
    for x, y in zip(old_nums, new_nums):
        if x.group() != y.group():
            return x.group(), y.group(), x.start()
    return None

# name of the innermost call around position pos, e.g. "chamfer" for the 0.5 in .chamfer(0.5)
def enclosing_call(line, pos):
    depth = 0
    i = pos
    while i > 0:   # walk left and count brackets
        i -= 1
        if line[i] == ")":
            depth += 1
        elif line[i] == "(":
            if depth == 0:
                m = re.search(r"(\w+)\s*$", line[:i])   # the word just before the "("
                return m.group(1) if m else None
            depth -= 1
    return None

# wall_thickness -> "wall thickness"
def words(name):
    return name.replace("_", " ").strip()

# everything the prompt and the checks need to know about one edit
def facts(row, t):
    minus, plus = changed_lines(row["patch"])
    f = dict(type=t, hunk=hunk(row["patch"]))

    # t2: rotate (axis + angle) or mirror (plane), read from the added line
    if t == "t2":
        line = plus[0]
        m = re.search(r"rotate\(\(0, 0, 0\), \((\d), (\d), (\d)\), (\d+)\)", line)
        if m:
            axis = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            f.update(op="rotate", axis=AXIS[axis], angle=m.group(4))
        else:
            f.update(op="mirror", plane=re.search(r'mirror\("(\w+)"\)', line).group(1))
        return f

    # t1 / t3: one number changed on one line
    old, new, pos = changed_number(minus[0], plus[0])
    f.update(old=old, new=new, line=minus[0].strip())

    # what the number belongs to: a named parameter, or a call like .chamfer(0.5)
    if t == "t1" and row["site_kind"] == "call":
        func = enclosing_call(minus[0], pos)
        kw = re.match(r"\s*(\w+)\s*=\s*$", minus[0][:pos])   # keyword argument alone on its line, e.g. "width = 10,"
        if func is None and kw:
            f.update(kind="param", name=kw.group(1))
        else:
            f.update(kind="call", func=func)
    else:
        f.update(kind="param", name=minus[0].split("=")[0].strip())

    # t3: the ratio k and the other parameter, from the relation "target = k * ref"
    if t == "t3":
        target, rest = row["relation"].split(" = ")
        k, ref = rest.split(" * ")
        m = re.search(rf"^{re.escape(ref)}\s*=\s*(.+)$", row["original"], re.M)   # the line that defines ref
        f.update(k=float(k), ref=ref, ref_line=m.group(0).strip() if m else ref)
    return f

# the user message for phi-4: what the edit is, plus the diff
def user_message(f):
    if f["type"] == "t2":
        if f["op"] == "rotate":
            what = f"Rotate the whole part by {f['angle']} degrees about the {f['axis']} axis through the origin."
        else:
            what = f"Mirror the whole part across the {f['plane']} plane."
        return f"Edit: {what}\nThe line added at the end of the program:\n{f['hunk']}"
    if f["type"] == "t3":
        return (f"Edit: set the parameter on `{f['line']}` to {f['k']:g} times the parameter on `{f['ref_line']}`"
                f" (it becomes {f['new']}).\nDiff:\n{f['hunk']}")
    return f"Edit: change the value {f['old']} to {f['new']} on the line `{f['line']}`.\nDiff:\n{f['hunk']}"


# ---------- checks on a candidate instruction ----------

# is the number `value` written in the text?
def has_number(text, value):
    return any(float(x) == float(value) for x in NUM.findall(text))

# is the parameter named in the text? at least one word (longer than 2 letters) of its name
def name_mentioned(text, name):
    ws = [w for w in name.lower().split("_") if len(w) > 2]
    return not ws or any(w in text.lower() for w in ws)

def valid(text, f):
    if not 15 <= len(text) <= 300:
        return False
    if "\n" in text or "_" in text or "`" in text:
        return False
    if re.search(r"\bline\s*\d", text, re.I) or re.search(r"\w\(", text):
        return False
    low = text.lower()

    if f["type"] == "t2":
        if f["op"] == "rotate":
            # says rotate, the angle, and the axis letter
            axis_said = re.search(rf"\b{f['axis']}\b", text, re.I) is not None
            return "rotat" in low and has_number(text, f["angle"]) and axis_said
        # says mirror (or reflect) and the plane
        return ("mirror" in low or "reflect" in low) and f["plane"].lower() in low

    if f["type"] == "t3":
        # the ratio in words, and both parameters
        ratio_said = any(w in low for w in RATIO_WORDS[f["k"]])
        return ratio_said and name_mentioned(text, f["name"]) and name_mentioned(text, f["ref"])

    # t1: the new value, and the parameter (a number inside a call has no name to check)
    if not has_number(text, f["new"]):
        return False
    return f["kind"] == "call" or name_mentioned(text, f["name"])

# fallback sentence when no candidate passes
def template(f):
    if f["type"] == "t2":
        if f["op"] == "rotate":
            return f"Rotate the entire part by {f['angle']} degrees about the {f['axis']} axis."
        return f"Mirror the entire part across the {f['plane']} plane."
    if f["type"] == "t3":
        r = words(f["ref"])
        return f"Set the {words(f['name'])} equal to {RATIO_TEXT[f['k']]} {r}; the {r} is unchanged."
    verb = "Increase" if float(f["new"]) > float(f["old"]) else "Decrease"
    if f["kind"] == "call":
        what = f"the {words(f['func'])} value" if f["func"] else "the value"
        return f"{verb} {what} from {f['old']} to {f['new']}."
    return f"{verb} the {words(f['name'])} from {f['old']} to {f['new']}."

# first line of the reply, without quotes around it
def clean(text):
    text = text.strip().strip('"').strip()
    return text.split("\n")[0].strip()


if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    # all rows of all splits; this gpu takes every NSHARDS-th row
    jobs = []   # (split, type, row)
    i = 0
    for split in SPLITS:
        for t in TYPES:
            for line in open(f"{B}/data/{split}/{t}.jsonl"):
                if i % NSHARDS == SHARD:
                    jobs.append((split, t, json.loads(line)))
                i += 1

    fs = [facts(row, t) for _, t, row in jobs]
    convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(f)}] for f in fs]

    llm = LLM(MODEL, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, enable_prefix_caching=True, seed=SHARD)
    params = SamplingParams(n=N_CANDIDATES, temperature=0.8, top_p=0.95, max_tokens=80, seed=SHARD)
    outs = llm.chat(convs, params, use_tqdm=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as file:
        for (split, t, row), f, out in zip(jobs, fs, outs):
            cands = [clean(o.text) for o in out.outputs]
            good = [c for c in cands if valid(c, f)]
            if good:
                instr, source = good[0], "phi-4"
            else:
                instr, source = template(f), "template"
            file.write(json.dumps(dict(split=split, type=t, uuid=row["uuid"], instruction=instr, source=source,
                                       candidates=cands)) + "\n")
    print(f"wrote {len(jobs)} rows to {out_path}")
