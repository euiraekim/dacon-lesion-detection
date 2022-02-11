import numpy as np
import json
import base64
import os
import random
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from glob import glob

from collections import defaultdict

base_path = "../data"

def convert_to_coco(
    json_paths,
    save_path,
    image_path=None
):
    """
        only for train dataset
    """
    res = defaultdict(list)
    
    categories = {
        '01_ulcer': 1,
        '02_mass': 2,
        '04_lymph': 3,
        '05_bleeding': 4
    }

    if image_path:
        if not os.path.exists(image_path):
            os.makedirs(image_path)
    
    n_id = 0
    for json_path in tqdm(json_paths):
        with open(json_path, 'r') as f:
            tmp = json.load(f)

        if image_path:
            image = BytesIO(base64.b64decode(tmp['imageData']))
            image = Image.open(image).convert('RGB')
            image.save(os.path.join(image_path, tmp['file_name'].split(".")[0]+".jpg"))
        
        image_id = int(tmp['file_name'].split('_')[-1][:6])
        res['images'].append({
            'id': image_id,
            'width': tmp['imageWidth'],
            'height': tmp['imageHeight'],
            'file_name': tmp['file_name'].split(".")[0]+".jpg", # 변경
        })
        
        for shape in tmp['shapes']:
            box = np.array(shape['points']) # 추가
            x1, y1, x2, y2 = \
                    min(box[:, 0]), min(box[:, 1]), max(box[:, 0]), max(box[:, 1])
            
            w, h = x2 - x1, y2 - y1
            
            res['annotations'].append({
                'id': n_id,
                'image_id': image_id,
                'category_id': categories[shape['label']],
                'area': w * h,
                'bbox': [x1, y1, w, h],
                'iscrowd': 0,
            })
            n_id += 1
    
    for name, id in categories.items():
        res['categories'].append({
            'id': id,
            'name': name,
        })
        
    with open(save_path, 'w') as f:
        json.dump(res, f)

random.seed(42)

train_file = glob(os.path.join(base_path, 'train/*.json'))
valid_file = glob(os.path.join(base_path, 'valid/*.json'))

convert_to_coco(train_file, os.path.join(base_path, 'train_annotations.json'), os.path.join(base_path, "train_images"))
convert_to_coco(valid_file, os.path.join(base_path, 'valid_annotations.json'), os.path.join(base_path, "valid_images"))


# 테스트 데이터
test_path = '../data/test'

test_files = sorted(glob(os.path.join(test_path, '*')))

test_json_list = []
for file in tqdm(test_files):
    with open(file, "r") as json_file:
        test_json_list.append(json.load(json_file))
        
image_path = os.path.join(base_path, "test_images")
if not os.path.exists(image_path):
    os.makedirs(image_path)

for sample in tqdm(test_json_list):
    
    image_id = sample['file_name'].split(".")[0]
    image = BytesIO(base64.b64decode(sample['imageData']))
    image = Image.open(image).convert('RGB')
    
    image.save(os.path.join(base_path, "test_images", image_id+".jpg"))