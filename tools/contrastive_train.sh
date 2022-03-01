echo "SEGCONTRAST PRE-TRAINING"

python3 contrastive_train.py \
    --dataset-name SemanticKITTI \
    --data-dir ./Datasets/SemanticKITTI \
    --batch-size 8 \
    --epochs 200 \
    --lr 0.12 \
    --num-points 20000 \
    --use-cuda \
    --use-intensity \
    --segment-contrast \
    --checkpoint segcontrast