import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from data_utils.data_map import *
from pcd_utils.pcd_preprocess import *
from pcd_utils.pcd_transforms import *
import MinkowskiEngine as ME
import torch
import json

warnings.filterwarnings('ignore')

class SemanticPOSSDataLoader(Dataset):
    def __init__(self, root,  split='train', pre_training=True, resolution=0.05, percentage=None, intensity_channel=False):
        self.root = root
        self.augmented_dir = 'augmented_views'
        self.n_clusters = 50

        if not os.path.isdir(os.path.join(self.root, self.augmented_dir)):
            os.makedirs(os.path.join(self.root, self.augmented_dir))
        self.resolution = resolution
        self.intensity_channel = intensity_channel

        self.seq_ids = {}
        self.seq_ids['train'] = [ '00', '01', '02', '03' ]
        self.seq_ids['validation'] = [ '04', '05' ]
        self.pre_training = pre_training
        self.split = split

        assert (split == 'train' or split == 'validation')
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list(split)

        if split == 'train':
            self.train_set_percent(percentage)

        print('The size of %s data is %d'%(split,len(self.points_datapath)))

    def train_set_percent(self, percentage):
        if percentage is None or percentage == 1.0:
            return

        percentage = str(percentage)
        # the stratified point clouds are pre-defined on this percentiles_split.json file
        with open('tools/percentiles_poss_split.json', 'r') as p:
            splits = json.load(p)

            assert (percentage in splits)

            self.points_datapath = []
            self.labels_datapath = []

            for seq in splits[percentage]:
                self.points_datapath += splits[percentage][seq]['points']
                self.labels_datapath += splits[percentage][seq]['labels']

        return

    def datapath_list(self, split):
        self.points_datapath = []
        self.labels_datapath = []

        for seq in self.seq_ids[split]:
            point_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'velodyne')
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            self.points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

            try:
                label_seq_path = os.path.join(self.root, 'dataset', 'sequences', seq, 'labels')
                point_seq_label = os.listdir(label_seq_path)
                point_seq_label.sort()
                self.labels_datapath += [ os.path.join(label_seq_path, label_file) for label_file in point_seq_label ]
            except:
                pass

    def transforms(self, points):
        if self.pre_training:
            points = np.expand_dims(points, axis=0)
            points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
            points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
            points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
            points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])
            points[:,:,:3] = jitter_point_cloud(points[:,:,:3])
            #points[:,:,:3] = random_point_dropout(points[:,:,:3])
            points = random_drop_n_cuboids(points)

            return np.squeeze(points, axis=0)
        elif self.split == 'train':
            theta = torch.FloatTensor(1,1).uniform_(0, 2*np.pi).item()
            scale_factor = torch.FloatTensor(1,1).uniform_(0.95, 1.05).item()
            rot_mat = np.array([[np.cos(theta),
                                    -np.sin(theta), 0],
                                [np.sin(theta),
                                    np.cos(theta), 0], [0, 0, 1]])

            points[:, :3] = np.dot(points[:, :3], rot_mat) * scale_factor
            return points
        else:
            return points


    def __len__(self):
        return len(self.points_datapath)

    def _get_augmented_item(self, index):
        # we need to preprocess the data to get the cuboids and guarantee overlapping points
        # so if we never have done this we do and save this preprocessing
        cluster_path = os.path.join(self.root, self.augmented_dir, f'{index}.npy')
        if os.path.isfile(cluster_path):
            # if the preprocessing is done and saved already we simply load it
            points_set = np.load(cluster_path)
            # Px5 -> [x, y, z, i, c] where i is the intesity and c the Cluster associated to the point

        else:
            # if not we load the full point cloud and do the preprocessing saving it at the end
            points_set = np.fromfile(self.points_datapath[index], dtype=np.float32)
            points_set = points_set.reshape((-1, 4))

            # remove ground and get clusters from point cloud
            points_set = clusterize_pcd(points_set, self.n_clusters)
            #visualize_pcd_clusters(points_set)

            # Px5 -> [x, y, z, i, c] where i is the intesity and c the Cluster associated to the point
            np.save(cluster_path, points_set)


        points_i = random_cuboid_point_cloud(points_set.copy())
        points_i = self.transforms(points_i)
        points_j = random_cuboid_point_cloud(points_set.copy())
        points_j = self.transforms(points_j)

        if not self.intensity_channel:
            points_i = points_i[:, :3]
            points_j = points_j[:, :3]

        # now the point set returns [x,y,z,i,c] always
        return points_i, points_j

    def _get_item(self, index):
        points_set = np.fromfile(self.points_datapath[index], dtype=np.float32)
        points_set = points_set.reshape((-1, 4))

        labels = np.fromfile(self.labels_datapath[index], dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF

        #remap labels to learning values
        #labels = np.vectorize(poss_map.get)(labels)
        labels = np.asarray([ poss_map.get(lbl, 0) for lbl in labels ])
        labels = np.expand_dims(labels, axis=-1)
        unlabeled = labels[:,0] == 0

        # remove unlabeled points
        labels = np.delete(labels, unlabeled, axis=0)
        points_set = np.delete(points_set, unlabeled, axis=0)
        points_set[:, :3] = self.transforms(points_set[:, :3])

        if not self.intensity_channel:
            points_set = points_set[:, :3]

        # now the point set return [x,y,z,i] always
        return points_set, labels.astype(np.int32)

    def __getitem__(self, index):
        return self._get_augmented_item(index) if self.pre_training else self._get_item(index)
