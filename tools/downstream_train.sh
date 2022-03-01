# Scratch
# train and move checkpoints annd move logs
echo "SCRATCH BLOCK"
python3 downstream_train.py --use-cuda --use-intensity --checkpoint scratch_1scan --contrastive --percentage-labels 0.0001 --batch-size 2 --sparse-model MinkUNet --epochs 600 & wait;\
python3 downstream_train.py --use-cuda --use-intensity --checkpoint scratch_0p001 --contrastive --percentage-labels 0.001 --batch-size 2 --sparse-model MinkUNet --epochs 300 & wait;\
python3 downstream_train.py --use-cuda --use-intensity --checkpoint scratch_0p01 --contrastive --percentage-labels 0.01 --batch-size 2 --sparse-model MinkUNet --epochs 120 & wait;\
python3 downstream_train.py --use-cuda --use-intensity --checkpoint scratch_0p1 --contrastive --percentage-labels 0.1 --batch-size 2 --sparse-model MinkUNet --epochs 40 & wait;\
python3 downstream_train.py --use-cuda --use-intensity --checkpoint scratch_0p5 --contrastive --percentage-labels 0.5 --batch-size 2 --sparse-model MinkUNet --epochs 20 & wait;\
python3 downstream_train.py --use-cuda --use-intensity --checkpoint scratch_1p0 --contrastive --percentage-labels 1.0 --batch-size 2 --sparse-model MinkUNet --epochs 15 & wait;\

# Depth Contrast
# train and move checkpoints annd move logs
#echo "DEPTH CONTRAST BLOCK"
#python3 downstream_train.py --use-cuda --use-intensity --checkpoint depth_contrast_1scan --contrastive --percentage-labels 0.0001 --load-checkpoint --batch-size 2 --sparse-model MinkUNetSMLP --epochs 600 --depth & wait;\
#python3 downstream_train.py --use-cuda --use-intensity --checkpoint depth_contrast_0p001 --contrastive --percentage-labels 0.001 --load-checkpoint --batch-size 2 --sparse-model MinkUNetSMLP --epochs 300 --depth & wait;\
# python3 downstream_train.py --use-cuda --use-intensity --checkpoint depth_contrast_0p01 --contrastive --percentage-labels 0.01 --load-checkpoint --batch-size 2 --sparse-model MinkUNetSMLP --epochs 120 --depth & wait;\
#python3 downstream_train.py --use-cuda --use-intensity --checkpoint depth_contrast_0p1 --contrastive --percentage-labels 0.1 --load-checkpoint --batch-size 2 --sparse-model MinkUNetSMLP --epochs 40 --depth & wait;\
#python3 downstream_train.py --use-cuda --use-intensity --checkpoint depth_contrast_0p5 --contrastive --percentage-labels 0.5 --load-checkpoint --batch-size 2 --sparse-model MinkUNetSMLP --epochs 20 --depth & wait;\
#python3 downstream_train.py --use-cuda --use-intensity --checkpoint depth_contrast_1p0 --contrastive --percentage-labels 1.0 --load-checkpoint --batch-size 2 --sparse-model MinkUNetSMLP --epochs 15 --depth & wait;\

# Point Contrast
# train and move checkpoints annd move logs
#echo "SEMANTICPOSS FINE-TUNING"
#python3 downstream_train.py --use-cuda --use-intensity --checkpoint point_contrast_1scan --contrastive --percentage-labels 0.0001 --load-checkpoint --batch-size 2 --sparse-model MinkUNet --epochs 600 --point & wait;\
#python3 downstream_train.py --use-cuda --use-intensity --checkpoint point_contrast_0p001 --contrastive --percentage-labels 0.001 --load-checkpoint --batch-size 2 --sparse-model MinkUNet --epochs 300 --point & wait;\
#python3 downstream_train.py --use-cuda --use-intensity --checkpoint point_contrast_0p01 --contrastive --percentage-labels 0.01 --load-checkpoint --batch-size 2 --sparse-model MinkUNet --epochs 120 --point & wait;\
#python3 downstream_train.py --use-cuda --use-intensity --checkpoint point_contrast_0p1 --contrastive --percentage-labels 0.1 --load-checkpoint --batch-size 2 --sparse-model MinkUNet --epochs 40 --point & wait;\
#python3 downstream_train.py --use-cuda --use-intensity --checkpoint point_contrast_0p5 --contrastive --percentage-labels 0.5 --load-checkpoint --batch-size 2 --sparse-model MinkUNet --epochs 20 --point & wait;\
#python3 downstream_train.py --use-cuda --use-intensity --checkpoint point_contrast_1p0 --contrastive --percentage-labels 1.0 --load-checkpoint --batch-size 2 --sparse-model MinkUNet --epochs 15 --point & wait;\

# Segment Contrast
# train and move checkpoints annd move logs
echo "SEGMENT CONTRAST BLOCK"
python3 downstream_train.py --use-cuda --use-intensity --checkpoint segment_contrast_0p001 --contrastive --percentage-labels 0.001 --load-checkpoint --batch-size 2 --sparse-model MinkUNet --epochs 300 & wait;\
python3 downstream_train.py --use-cuda --use-intensity --checkpoint segment_contrast_0p01 --contrastive --percentage-labels 0.01 --load-checkpoint --batch-size 2 --sparse-model MinkUNet --epochs 120 & wait;\
python3 downstream_train.py --use-cuda --use-intensity --checkpoint segment_contrast_0p1 --contrastive --percentage-labels 0.1 --load-checkpoint --batch-size 2 --sparse-model MinkUNet --epochs 40 & wait;\
python3 downstream_train.py --use-cuda --use-intensity --checkpoint segment_contrast_0p5 --contrastive --percentage-labels 0.5 --load-checkpoint --batch-size 2 --sparse-model MinkUNet --epochs 20 & wait;\
python3 downstream_train.py --use-cuda --use-intensity --checkpoint segment_contrast_1p0 --contrastive --percentage-labels 1.0 --load-checkpoint --batch-size 2 --sparse-model MinkUNet --epochs 15 & wait;\
