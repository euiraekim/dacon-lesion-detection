import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
from glob import glob

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.copy()
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y

parser = argparse.ArgumentParser()
parser.add_argument('-data', '--data', default='../yolo/yolov5/runs/detect/exp',dest='data')
parser.add_argument('-save', '--save', default='../result_csv/submission_yolov5.csv',dest='save')
options = parser.parse_args()

result_path = options.data
result_label = glob(os.path.join(result_path, 'labels/*.txt'))

results = {
    'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
    'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[]
}

for i in tqdm(result_label):
    with open(i,'r') as f:
        file_name = os.path.basename(i).replace('.txt','.json')
        img_name = file_name.replace('.json','.jpg')
        ow,oh,_ = cv2.imread(os.path.join(result_path, img_name)).shape
        
        for line in f.readlines():
            label,xc,yc,w,h,score = line[:-1].split(' ')
            xc,yc,w,h,score = list(map(float,[xc,yc,w,h,score]))
            if score > 0.22:
                xc,w = np.array([xc,w]) * ow
                yc,h = np.array([yc,h]) * oh

                x_min,y_min,x_max,y_max = np.array(xywh2xyxy([xc,yc,w,h])).astype(int)

                results['file_name'].append(file_name)
                results['class_id'].append(label)
                results['confidence'].append(score)
                results['point1_x'].append(x_min)
                results['point1_y'].append(y_min)
                results['point2_x'].append(x_max)
                results['point2_y'].append(y_min)
                results['point3_x'].append(x_max)
                results['point3_y'].append(y_max)
                results['point4_x'].append(x_min)
                results['point4_y'].append(y_max)
            
df = pd.DataFrame(results)
df['class_id'] = df['class_id'].apply(lambda x:int(x)+1)

df = df.sort_values(by=['confidence'], axis=0, ascending=False)
df[:30000].to_csv(options.save, index=False)