cd yolo/yolov5
pip install -r requirements.txt

python train.py --img 576 --batch 8 --epochs 150 --cfg models/yolov5x.yaml --data ../../data/data_yolo/endoscopy.yaml --weights yolov5x.pt --project ../../ckpts/yolov5x-endoscopy --save-period 1 --name endoscopy --device 0 --multi-scale
cd ../..