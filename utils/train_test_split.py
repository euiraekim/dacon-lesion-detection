from glob import glob
import shutil
import os
import random
import json

random.seed(42)

data_path = '../data'
save_path = '../data/data_splited'

files = glob('%s/train/*.json' % data_path)

random.shuffle(files)

if os.path.exists(save_path):
    shutil.rmtree(save_path)

train_path = os.path.join(save_path, "train")
if not os.path.exists(train_path):
    os.makedirs(train_path)
val_path = os.path.join(save_path, "valid")
if not os.path.exists(val_path):
    os.makedirs(val_path)

num_all = len(files)
num_valid = round(0.1 * num_all)

idx = 0
for file in files:
    if idx < num_all - num_valid:
        shutil.copyfile(file, "%s/train/%s" % (save_path, os.path.basename(file)))
    else:
        shutil.copyfile(file, "%s/valid/%s" % (save_path, os.path.basename(file)))
    idx += 1