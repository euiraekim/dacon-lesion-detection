cd yolo/yolov5
pip install -r requirements.txt
python detect.py --source ../../data/data_splited/data_yolo/images/test --save-txt --save-conf --weight ../../ckpts/yolov5x-endoscopy/endoscopy/weights/best.pt --imgsz 576 --device 0 --augment
cd ../..