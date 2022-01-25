# SegContrast: 3D Point Cloud Feature Representation Learning through Self-supervised Segment Discrimination

**Abstract -** Semantic scene interpretation is essential for autonomous systems to operate in complex scenarios. While deep learning-based methods excel at this task,they rely on vast amounts of labeled data that is tedious to generate and might not cover all relevant classes sufficiently. Self-supervised representation learning has the prospect of reducing the amount of required labeled data by learning descriptive representations from unlabeled data. In this paper, we address the problem of representation learning for 3D point cloud data in the context of autonomous driving. We propose a new contrastive learning approach that aims at learning the structural context of the scene. Our approach extracts class-agnostic segments over the point cloud and applies the contrastive loss over these segments to discriminate between similar and dissimilar structures. We apply our method on data recorded with a 3D LiDAR. We show that our method achieves competitive performance and can learn a more descriptive feature representation than other state-of-the-art self-supervised contrastive point cloud methods.

Source code for our work soon to be published at RA-L:

```
@article{nunes2022ral,
  author = {Lucas Nunes and Rodrigo Marcuzzi and Xieyuanli Chen and Jens Behley and Cyrill Stachniss},
  title = {{SegContrast: 3D Point Cloud Feature Representation Learning through Self-supervised Segment Discrimination}},
  journal = {IEEE Robotics and Automation Letters (RA-L)},
  year = {2022},
  volume = {7},
  number = {2},
  pages = {2116-2123},
  doi={10.1109/LRA.2022.3142440}
}
```

More information on the article and code will be published soon.
