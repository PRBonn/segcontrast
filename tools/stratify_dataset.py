import numpy as np
import os
import json

train_seqs = [ '00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
#train_seqs = [ '01' ]
percentiles = [ 0.001 ]

full_content = {
  1: False,
  2: False,
  3: False,
  4: False,
  5: False,
  6: False,
  7: False,
  8: False,
  9: False,
  10: False,
  11: False,
  12: False,
  13: False,
  14: False,
  15: False,
  16: False,
  17: False,
  18: False,
  19: False,
}

learning_map = {
  0 : 0,     # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car"
  253: 7,    # "moving-bicyclist"
  254: 6,    # "moving-person"
  255: 8,    # "moving-motorcyclist"
  256: 5,    # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
  257: 5,    # "moving-bus" mapped to "moving-other-vehicle" -----------mapped
  258: 4,    # "moving-truck"
  259: 5,    # "moving-other-vehicle"
}

root = '../Datasets/SemanticKITTI'
points_path = 'data_odometry_velodyne'
labels_path = 'data_odometry_labels'
present_classes = []

points_datapath = {}

def evaluate_percentile(label_files, seq_content):
    for class_ in present_classes:
        seq_content[class_] = False

    percentile_content = full_content.copy()
    for file_ in label_files:
        labels = np.fromfile(file_, dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF

        #remap labels to learning values
        labels = np.vectorize(learning_map.get)(labels)

        for class_ in np.unique(labels):
            percentile_content[class_] = True

    
    #print(list(percentile_content.values()), list(seq_content.values()))
    return np.all(np.equal(list(percentile_content.values()), list(seq_content.values())))


if __name__ == '__main__':
    for seq in train_seqs:
        print('Listing Sequence ', seq)
        seq_content = full_content.copy()
        points_datapath[seq] = {'points': [], 'labels': [], 'seq_content': None}
        point_seq_path = os.path.join(root, points_path, 'dataset', 'sequences', seq, 'velodyne')
        point_seq_bin = os.listdir(point_seq_path)
        point_seq_bin.sort()
        points_datapath[seq]['points'] += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]

        label_seq_path = os.path.join(root, labels_path, 'dataset', 'sequences', seq, 'labels')
        point_seq_label = os.listdir(label_seq_path)
        point_seq_label.sort()

        for label_file in point_seq_label:
            points_datapath[seq]['labels'] += [ os.path.join(label_seq_path, label_file) ]
            labels = np.fromfile(points_datapath[seq]['labels'][-1], dtype=np.uint32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF

            #remap labels to learning values
            labels = np.vectorize(learning_map.get)(labels)

            for class_ in np.unique(labels):
                seq_content[class_] = True
            
        points_datapath[seq]['seq_content'] = seq_content

    percentiles_paths = {}

    for percentile in percentiles:
        percentiles_paths[percentile] = {}
        print('PERCENTILE :', percentile)
        for seq in train_seqs:
            print('SEQUENCE: ', seq)
            percentiles_paths[percentile][seq] = {'points': [], 'labels': []}
            seq_percent = max(1,int(len(points_datapath[seq]['labels']) * percentile))
            print(len(points_datapath[seq]['labels']), ' -> ', seq_percent)

            labels_n_index = np.array(list(enumerate(points_datapath[seq]['labels'])))[:,0]
            percentile_ind = np.random.choice(labels_n_index, seq_percent, replace=False).astype(int)
            percentile_seq = [ points_datapath[seq]['labels'][i] for i in percentile_ind ]
            tries = 0
            while not evaluate_percentile(percentile_seq, points_datapath[seq]['seq_content']) and tries < 1000:
                labels_n_index = np.array(list(enumerate(points_datapath[seq]['labels'])))[:,0]
                percentile_ind = np.random.choice(labels_n_index, seq_percent, replace=False).astype(int)
                percentile_seq = [ points_datapath[seq]['labels'][i] for i in percentile_ind ]
                tries += 1

            for class_ in points_datapath[seq]['seq_content']:
                if class_ not in present_classes:
                    present_classes.append(class_)

            for i in percentile_ind:
                percentiles_paths[percentile][seq]['points'] += [ points_datapath[seq]['points'][i] ]
                percentiles_paths[percentile][seq]['labels'] += [ points_datapath[seq]['labels'][i] ]

    with open('percentiles_redo.json', 'w+') as f:
        json.dump(percentiles_paths, f)

    


