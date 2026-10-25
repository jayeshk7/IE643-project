import os, json
from prompts import messages

B = os.environ["B"]
SPLITS = ["train", "val", "test"]
TYPES = ["t1", "t2", "t3"]

os.makedirs(f"{B}/data/sft", exist_ok=True)
for mode in ["patch", "rewrite"]:
    for split in SPLITS:
        n = 0
        with open(f"{B}/data/sft/{mode}_{split}.jsonl", "w") as out:
            for t in TYPES:
                for line in open(f"{B}/data/final/{split}/{t}.jsonl"):
                    r = json.loads(line)
                    if mode == "patch":
                        target = r["patch"]
                    else:
                        target = r["edited"]
                    msgs = messages(mode, r["instruction"], r["original"], target)
                    out.write(json.dumps(dict(uuid=r["uuid"], type=t, messages=msgs)) + "\n")
                    n += 1
        print(mode, split, n)

# 1. A dataset row is a set of separate fields: uuid, original program, edited, patch, instruction, iou, and so on.
# 2. Qwen is a chat model, so it trains on conversations. build_sft.py turns each row into three messages. example:
# [system]    You edit CadQuery programs ... reply with only a unified diff ...
# [user]      Instruction: Increase the box width from 10 to 20.
#             Program:
#             import cadquery as cq
#             result = cq.Workplane('XY').box(20,10,5).faces('>Z').shell(-1)
#             --- a/part.py
#             +++ b/part.py
#             @@ -1,2 +1,2 @@
#             -result = cq.Workplane('XY').box(20,10,5)...
#             +result = cq.Workplane('XY').box(20,20,5)...
