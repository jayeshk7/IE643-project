from datasets import load_dataset

# ds = load_dataset("ADSKAILab/Zero-To-CAD-100k", split="train", streaming=True, columns=['uuid', 'cadquery_file'])
# ds = load_dataset("ADSKAILab/Zero-To-CAD-100k", split="validation", streaming=True, columns=['uuid', 'cadquery_file'])
ds = load_dataset("ADSKAILab/Zero-To-CAD-100k", split="test", streaming=True, columns=['uuid', 'cadquery_file'])

ds_iterator = iter(ds)

while True:
    try:
        sample = next(ds_iterator)
        code = sample["cadquery_file"].decode('utf-8')

        # with open(f"../data/train/{sample["uuid"]}.py", mode="w", encoding="utf-8") as file:
        # with open(f"../data/val/{sample["uuid"]}.py", mode="w", encoding="utf-8") as file:
        with open(f"../data/test/{sample["uuid"]}.py", mode="w", encoding="utf-8") as file:
            file.write(code)
    except StopIteration:
        break
