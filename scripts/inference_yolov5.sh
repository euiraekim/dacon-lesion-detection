cd yolo/yolov5
pip install -r requirements.txt
python detect.py --source ../../data/test_images --save-txt --save-conf --weight ../../ckpts/yolov5x-endoscopy/endoscopy/weights/best.pt --imgsz 576 --device 0 --augment
cd ../..