# adds the phi-4 instructions to the triplets, as the fields instruction and instruction_source
# run it after gen_instructions.py has finished on all shards: python add_instructions.py
# writes $B/data/final/{split}/t*.jsonl

import os, glob, json
from collections import Counter

B = os.environ["B"]
SPLITS = ["train", "val", "test"]
TYPES = ["t1", "t2", "t3"]

# all instructions from the shard files, by (split, type, uuid)
instr = {}
for path in glob.glob(f"{B}/outputs/instructions/shard*.jsonl"):
    for line in open(path):
        r = json.loads(line)
        instr[r["split"], r["type"], r["uuid"]] = (r["instruction"], r["source"])

stats = Counter()
for split in SPLITS:
    os.makedirs(f"{B}/data/final/{split}", exist_ok=True)
    for t in TYPES:
        with open(f"{B}/data/final/{split}/{t}.jsonl", "w") as out:
            for line in open(f"{B}/data/{split}/{t}.jsonl"):
                row = json.loads(line)
                text, source = instr[split, t, row["uuid"]]   # every row must have one
                row["instruction"] = text
                row["instruction_source"] = source
                out.write(json.dumps(row) + "\n")
                stats[split, t, source] += 1

# count per split, type and source
for k in sorted(stats):
    print(*k, stats[k])
