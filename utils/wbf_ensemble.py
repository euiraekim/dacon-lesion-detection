import pandas as pd
import os
from tqdm import tqdm
import argparse

from ensemble_boxes import weighted_boxes_fusion

parser = argparse.ArgumentParser()
parser.add_argument('-img-path', '--img-path', default='../data/data_splited/data_coco/test_images/',dest='image_path')
parser.add_argument('-result-path', '--result-path', default='../result_csv/',dest='result_path')
options = parser.parse_args()

# test image 경로 
image_path = options.image_path

# 각 모델의 result csv파일이 있는 경로
result_path = options.result_path
csv_path_list = os.listdir(result_path)

img_h = 576

iou_thr = 0.4
skip_box_thr = 0.0001

# 모든 모델의 result csv 파일에 가져와서 list에 저장
csv_files = [pd.read_csv(os.path.join(result_path, i)) for i in csv_path_list]

# 각 csv 파일을 wbf하기 좋게 형태를 바꿔줌
for csv_file in csv_files:
    csv_file['x1'] = csv_file['point1_x'] / img_h
    csv_file['y1'] = csv_file['point1_y'] / img_h
    csv_file['x2'] = csv_file['point3_x'] / img_h
    csv_file['y2'] = csv_file['point3_y'] / img_h
    csv_file.drop(['point1_x', 'point1_y', 'point2_x', 'point2_y', 'point3_x', 'point3_y', 'point4_x', 'point4_y'], axis=1, inplace=True)
    csv_file.set_index(keys=['file_name'], drop=True, inplace=True)

# print(csv_files)

image_list = os.listdir(image_path)
file_name_list = [os.path.basename(i).split('.')[0] + '.json' for i in image_list]

ensembled_result = {
    'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
    'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[]
}

# 모든 test image 파일에 대하여 루프를 돌며 wbf를 실행한다.
for file_name in tqdm(file_name_list):
    label_list = []
    score_list = []
    box_list = []
    
    for csv_file in csv_files:
        if file_name in csv_file.index:
            label_list.append(csv_file.loc[[file_name]]['class_id'].values.tolist())
            score_list.append(csv_file.loc[[file_name]]['confidence'].values.tolist())
            box_list.append(csv_file.loc[[file_name]][['x1', 'y1', 'x2', 'y2']].values.tolist())
            
    # 한 이미지 파일에 대해 여러 모델 결과 WBF 앙상블하기
    boxes, scores, labels = weighted_boxes_fusion(
                            box_list, score_list, label_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # 0~1로 정규화되어있던 boxes에 height를 곱해주고 int형으로 변환
    boxes = boxes * img_h

    ensembled_result['file_name'].extend([file_name for i in range(len(boxes))])
    ensembled_result['class_id'].extend(labels)
    ensembled_result['confidence'].extend(scores)
    ensembled_result['point1_x'].extend(boxes[:, 0])
    ensembled_result['point1_y'].extend(boxes[:, 1])
    ensembled_result['point2_x'].extend(boxes[:, 2])
    ensembled_result['point2_y'].extend(boxes[:, 1])
    ensembled_result['point3_x'].extend(boxes[:, 2])
    ensembled_result['point3_y'].extend(boxes[:, 3])
    ensembled_result['point4_x'].extend(boxes[:, 0])
    ensembled_result['point4_y'].extend(boxes[:, 3])

df = pd.DataFrame(ensembled_result)
df['class_id'] = df['class_id'].apply(lambda x:int(x))

df = df.sort_values(by=['confidence'], axis=0, ascending=False)
print(len(df))
df[:30000].to_csv(os.path.join(result_path, 'ensemble.csv'), index=False)