# SegContrast

**[Paper](http://www.ipb.uni-bonn.de/pdfs/nunes2022ral-icra.pdf)** **|** **[Video](https://www.youtube.com/watch?v=kotRb_ySnIw)**

![](pics/overview.png)

Installing pre-requisites:

`sudo apt install build-essential python3-dev libopenblas-dev`

`pip3 install -r requirements.txt`

`pip3 install torch ninja`

Installing MinkowskiEngine with CUDA support:

`pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps`

**Note:** We have released a new representation learning method based on temporal associations ([TARL](https://github.com/PRBonn/TARL)) that achieves better performance than SegContrast.

# SegContrast with Docker

Inside the `docker/` directory there is a `Dockerfile` to build an image to run SegContrast. You can build the image from scratch or download the image from docker hub by:

```
docker pull nuneslu/segcontrast:minkunet
```

Then start the container with:

```
docker run --gpus all -it --rm -v /PATH/TO/SEGCONTRAST:/home/segcontrast segcontrast /bin/zsh
```

# Data Preparation

Download [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) inside the directory ```./Datasets/SemanticKITTI/datasets```. The directory structure should be:

```
./
└── Datasets/
    └── SemanticKITTI
        └── dataset
          └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	├── 000000.bin
            |   |	├── 000001.bin
            |   |	└── ...
            │   └── labels/ 
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            ├── 08/ # for validation
            ├── 11/ # 11-21 for testing
            └── 21/
                └── ...
```

# Pretrained Weights
- SegContrast pretraining [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/segcontrast_pretrain.zip)
- Fine-tuned semantic segmentation
    - 0.1% labels [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/semantic_segmentation_weights/semseg_finetune_0p001.zip)
    - 1% labels [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/semantic_segmentation_weights/semseg_finetune_0p01.zip)
    - 10% labels [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/semantic_segmentation_weights/semseg_finetune_0p1.zip)
    - 50% labels [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/semantic_segmentation_weights/semseg_finetune_0p5.zip)
    - 100% labels [weights](https://www.ipb.uni-bonn.de/html/projects/segcontrast/semantic_segmentation_weights/semseg_finetune_1p0.zip)

# Reproducing the results

Run the following to start the pre-training:

```
python3 contrastive_train.py --use-cuda --use-intensity --segment-contrast --checkpoint segcontrast
```

The default parameters, e.g., learning rate, batch size and epochs are already the same as the paper.

After pre-training you can run the downstream fine-tuning with:

```
python3 downstream_train.py --use-cuda --use-intensity --checkpoint \
        segment_contrast --contrastive --load-checkpoint --batch-size 2 \
        --sparse-model MinkUNet --epochs 15
```

We provide in `tools` the `contrastive_train.sh` and `downstream_train.sh` scripts to reproduce the results pre-training and fine-tuning with the different label percentages shown on the paper:

For pre-training:

```
./tools/contrastive_train.sh
```

Then for fine-tuning:

```
./tools/downstream_train.sh
```

Finally, to compute the IoU metrics use:

```
./tools/eval_train.sh
```

# Citation

If you use this repo, please cite as :

```
@article{nunes2022ral,
    author = {L. Nunes and R. Marcuzzi and X. Chen and J. Behley and C. Stachniss},
    title = {{SegContrast: 3D Point Cloud Feature Representation Learning through Self-supervised Segment Discrimination}},
    journal = {{IEEE Robotics and Automation Letters (RA-L)}},
    year = 2022,
    doi = {10.1109/LRA.2022.3142440},
    issn = {2377-3766},
    volume = {7},
    number = {2},
    pages = {2116-2123},
    url = {http://www.ipb.uni-bonn.de/pdfs/nunes2022ral-icra.pdf},
}
```
