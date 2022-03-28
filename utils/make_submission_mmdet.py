import os
from glob import glob
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from mmdet.apis import init_detector, inference_detector

parser = argparse.ArgumentParser()
parser.add_argument('-data', '--data', default='../data/data_splited/test_images',dest='data')
parser.add_argument('-config', '--config', default='../ckpts/faster_rcnn_swin-l_ms/final.py',dest='config')
parser.add_argument('-weight', '--weight', default='../ckpts/faster_rcnn_swin-l_ms/best.pth',dest='weight')
parser.add_argument('-save', '--save', default='../result_csv/faster_rcnn_swin-l_ms.csv',dest='save')
options = parser.parse_args()

image_path = options.data
config_path = options.config
checkpoint_path = options.weight
save_path = options.save

model = init_detector(config_path, checkpoint_path, device='cuda:0')

def make_prediction(image_path):
    predictions = {
    'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
    'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[] }

    image_list = glob(os.path.join(image_path, '*'))

    for file_path in tqdm(image_list):
        image_file_path = os.path.join(image_path, os.path.basename(file_path).split('.')[0] + '.jpg')
        result = inference_detector(model, image_file_path)

        # 1개의 이미지의 result마다 class 개수인 4번 만큼 루프를 돈다.
        for j, v in enumerate(result):
            for det in v.tolist():
                predictions['file_name'].append(os.path.basename(file_path).split('.')[0] + '.json')
                predictions['class_id'].append(j+1)
                predictions['confidence'].append(det[4])
                predictions['point1_x'].append(det[0])
                predictions['point1_y'].append(det[1])
                predictions['point2_x'].append(det[2])
                predictions['point2_y'].append(det[1])
                predictions['point3_x'].append(det[2])
                predictions['point3_y'].append(det[3])
                predictions['point4_x'].append(det[0])
                predictions['point4_y'].append(det[3])
        # if i == 50:
        #     break
    return predictions

result = make_prediction(image_path)
result_df = pd.DataFrame(result)

# confidence 기준으로 내림차순으로 정렬한 후 위 30000행을 csv로 저장
result_df = result_df.sort_values(by=['confidence'], axis=0, ascending=False)
result_df[:30000].to_csv(save_path, index=False)