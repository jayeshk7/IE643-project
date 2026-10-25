import os, random, shutil

source = "../filtered_data/filtered_train/"
# source = "../filtered_data/filtered_val/"
# source = "../filtered_data/filtered_test/"

N = 10000  # 10000 for train, 1000 for val / test

files = sorted(os.listdir(source))
sample = random.Random(0).sample(files, N)

# print(sample[0])

destination = "../final_data_split/train"
# destination = "../final_data_split/test"
# destination = "../final_data_split/val"

for file in sample:
    src_path = os.path.join(source, file)
    dst_path = os.path.join(destination, file)
    shutil.copy2(src_path, dst_path)

print(f"Successfully copied {len(sample)} files to {destination}")
