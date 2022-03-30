# 데이콘 병변 검출 대회 (Object Detection)

대회 링크 : https://dacon.io/competitions/official/235855/overview/description


## 요약

사용 모델 (3개) -> WBF ensemble
* Faster RCNN / backbone: Swin Transformer - L
* RetinaNet / backbone: Swin Transformer - T
* YOLOv5 + TTA


## 데이터 전처리

```
mkdir data
```

다운 받은 대회 데이터를 data 폴더에 넣고, train 폴더의 이름을 train_all로 바꿔준다.

학습 데이터(90%)와 검증 데이터(10%) 분리
```
bash scripts/train_test_split.sh
```

coco format으로 변환
```
bash scripts/convert_to_coco.sh
```

yolo format으로 변환
```
bash scripts/convert_to_yolo.sh
```


## mmdetection 학습 (Faster RCNN / RetinaNet)
mmdetection/custom_configs에 각 모델의 config 파일 있음

학습
```
bash train_mmdet.sh
```


## mmdetection 제출 파일 만들기
```
bash make_submission_mmdet.sh
```


## YOLOv5 학습
```
bash train_yolov5.sh
```


## YOLOv5 제출 파일 만들기

Inference
```
bash inference_yolov5.sh
```

제출 파일 생성
```
bash make_submission_mmdet.sh
```


## WBF 앙상블

```
bash wbf_ensemble.sh
```