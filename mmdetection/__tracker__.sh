
INPUT_VIDEO=/workspace/mmdetection/demo/demo.mp4
CONFIG=/workspace/mmdetection/configs/bytetrack/yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval.py
TRACK_CHECK=/workspace/weights/epoch_250.pth
THR=0.1
OUTPUT_PATH=/workspace/results/demo_result.mp4



python /workspace/mmdetection/demo/mot_demo.py \
    $INPUT_VIDEO \
    $CONFIG \
    --checkpoint $TRACK_CHECK \
    --score-thr $THR \
    --out $OUTPUT_PATH \
    --fps 10
