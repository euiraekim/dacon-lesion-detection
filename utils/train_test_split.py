from glob import glob
import shutil
import os
import random
import json
from tqdm import tqdm

def create_folder(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

data_path = '../data'
save_path = '../data/data_splited'

files = glob(f'{data_path}/train_all/*.json')

random.seed(42)
random.shuffle(files)

train_path = os.path.join(save_path, "train")
create_folder(train_path)

val_path = os.path.join(save_path, "valid")
create_folder(val_path)

num_all = len(files)
num_valid = round(0.1 * num_all)

idx = 0
for file in tqdm(files):
    if idx < num_all - num_valid:
        shutil.copyfile(file, "%s/train/%s" % (save_path, os.path.basename(file)))
    else:
        shutil.copyfile(file, "%s/valid/%s" % (save_path, os.path.basename(file)))
    idx += 1