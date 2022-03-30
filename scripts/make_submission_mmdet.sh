cd utils
python make_submission_mmdet.py  --data ../data/data_splited/test_images --config ../ckpts/faster_rcnn_swin-l_ms/final.py --weight ../ckpts/faster_rcnn_swin-l_ms/best.pth --save ../result_csv/faster_rcnn_swin-l_ms.csv
python make_submission_mmdet.py  --data ../data/data_splited/test_images --config ../ckpts/retinanet_swin-t_ms/final.py --weight ../ckpts/retinanet_swin-t_ms/best.pth --save ../result_csv/retinanet_swin-t_ms.csv
cd ..