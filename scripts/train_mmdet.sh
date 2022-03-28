pip install openmim
mim install mmdet

cd mmdetection
chmod +x tools/dist_train.sh
bash tools/dist_train.sh custom_configs/faster_rcnn_swin-l_ms/final.py 1 --work-dir ../ckpts/faster_rcnn_swin-l_ms
bash tools/dist_train.sh custom_configs/retinanet_swin-t_ms/final.py 1 --work-dir ../ckpts/retinanet_swin-t_ms
cd ..