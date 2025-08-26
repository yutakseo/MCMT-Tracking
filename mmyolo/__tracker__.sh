#wget -P checkpoint https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth # noqa: E501, E261.
"""
python /workspace/mmyolo/demo/video_demo.py \
    /workspace/mmyolo/demo/demo.mp4 \
    /workspace/mmyolo/configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
    /workspace/mmyolo/checkpoint/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
    --out /workspace/results/demo_result.mp4"""


python /workspace/mmyolo/demo/video_demo.py \
    /workspace/mmyolo/demo/demo.mp4 \
    /workspace/weights/yolox_x.py \
    /workspace/weights/epoch_250.pth \
    --out /workspace/results/demo_result.mp4