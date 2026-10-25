import os, glob, json, math
import numpy as np, pandas as pd

B = os.environ["B"]
out_dir = f"{B}/outputs/results"
nan = float("nan")

# all score rows of all systems and both test sets
rows = []
for path in glob.glob(f"{B}/outputs/scores/*.jsonl"):
    for line in open(path):
        rows.append(json.loads(line))

# iou(original, target) of each row = the iou of the no-edit copy baseline on that row
base = {}
for r in rows:
    if r["system"] == "copy":
        iou = r.get("occt_iou")
        base[r["dataset"], r["id"]] = nan if iou is None else iou

def normalized(iou, b):
    if b is None or math.isnan(b) or b >= 1 - 1e-9:   # original already equals the target: no headroom
        return nan
    return float(np.clip((iou - b) / (1 - b), 0, 1))

# per-row scores used in the tables
for r in rows:
    r["valid"] = r.get("valid") is True   # missing (patch did not apply) = not valid

    # iou0: the iou with failures counted as 0; nan = valid solid but unscorable
    if not r["valid"]:
        r["iou0"] = 0.0
    elif r.get("occt_iou") is None:
        r["iou0"] = nan
    else:
        r["iou0"] = r["occt_iou"]
    if math.isnan(r["iou0"]):
        r["norm_occt"] = nan
    else:
        r["norm_occt"] = normalized(r["iou0"], base.get((r["dataset"], r["id"]), nan))

df = pd.DataFrame(rows)

# one line of the table, for a group of rows
def table(g):
    return pd.Series(dict(
        n=len(g),
        applied=100 * g.applied.mean(),                  # % patches that apply
        valid=100 * g.valid.mean(),                      # % edited programs that build a valid solid
        one_solid=100 * (g.n_solids == 1).mean(),
        exact=100 * g.exact.mean(),                      # % same ast as the target
        unscorable=int((g.valid & g.occt_iou.isna()).sum()),
        occt_iou=g.iou0.mean(),
        norm_occt=g.norm_occt.mean(),                    # the main metric
        out_tokens=g.n_tokens.mean(),
    ))

# table lines for each dataset, system and value of one column (that value goes in the "type" column)
def lines(d, column, prefix=""):
    t = d.groupby(["dataset", "system", column]).apply(table).reset_index()
    t["type"] = prefix + t[column]
    return t

# all rows, per edit type, t1 split by site kind (param / call), benchcad by its own 15 edit types
df["all"] = "all"
t1 = df[(df.dataset == "test") & (df.type == "t1")]
bc = df[df.dataset == "benchcad"]
parts = [lines(df, "all"), lines(df, "type")]
if len(t1):
    parts.append(lines(t1, "site_kind", "t1_"))
if len(bc):
    parts.append(lines(bc, "edit_type", "bc_"))
res = pd.concat(parts)[["dataset", "type", "system"] + list(table(df.iloc[:1]).index)]
res = res.sort_values(["dataset", "type", "system"])

os.makedirs(out_dir, exist_ok=True)
res.to_csv(f"{out_dir}/summary.csv", index=False)
with open(f"{out_dir}/summary.md", "w") as f:
    f.write(res.round(3).to_markdown(index=False))
print(res[res.type == "all"].round(3).to_string(index=False))

# failure counts per test set x system: first word of each row's error, ok = no error
err = df.error.fillna("ok").astype(str).str.split(":").str[0].str[:40]
errs = pd.crosstab(err, [df.dataset, df.system])
errs.to_csv(f"{out_dir}/errors.csv")
with open(f"{out_dir}/errors.md", "w") as f:
    f.write(errs.to_markdown())
