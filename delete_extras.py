import os
import glob
print('running')
all_files = glob.glob("./outputs/**/ckpt_2000_final.pt", recursive=True) + glob.glob("./outputs/**/ckpt_2000.pt", recursive=True)

for path in all_files:
    print(path)

input('confirm deletion')

for path in all_files:
    os.remove(path)