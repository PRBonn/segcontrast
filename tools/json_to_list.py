import json
import numpy as np

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

def class_contents(labels_files, lbl_count):
    for file_ in labels_files['labels']:
        labels = np.fromfile('../' + file_, dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF

        #remap labels to learning values
        labels = np.vectorize(learning_map.get)(labels)
        classes, counts = np.unique(labels, return_counts=True)

        for class_, count in zip(classes, counts):
            lbl_count[class_] += count

    return lbl_count


splits = None
with open('percentiles_split.json', 'r') as f:
    splits = json.load(f)

for percentile in splits:
    print(f'PERCENT: {percentile}')
    lbl_count = [ 0 for _ in range(20) ]
    for seq in splits[percentile]:
        lbl_count = class_contents(splits[percentile][seq], lbl_count)


    lbl_count = np.array(lbl_count)
    class_dist = lbl_count / np.sum(lbl_count)
    for class_ in range(20):
        print(f'{class_}: {round(class_dist[class_],5)}')
    print(f'\t- CLASS DIST: {class_dist}')
