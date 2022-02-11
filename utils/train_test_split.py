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

files = glob(f'{data_path}/train_all/*.json')

random.seed(42)
random.shuffle(files)

train_path = os.path.join(data_path, "train")
create_folder(train_path)

val_path = os.path.join(data_path, "valid")
create_folder(val_path)

num_all = len(files)
num_valid = round(0.1 * num_all)

idx = 0
for file in files:
    if idx < num_all - num_valid:
        shutil.copyfile(file, "%s/train/%s" % (data_path, os.path.basename(file)))
    else:
        shutil.copyfile(file, "%s/valid/%s" % (data_path, os.path.basename(file)))
    idx += 1