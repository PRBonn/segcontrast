import numpy as np
import torch
import torchvision
from PIL import Image
from math import ceil, floor
import argparse
from numpy import inf
import MinkowskiEngine as ME
import os
from utils import *
from data_utils.collations import numpy_to_sparse_tensor
import open3d as o3d
from tqdm import tqdm
from data_utils.data_map import color_map, labels, labels_poss
from data_utils.ioueval import iouEval

def sparse_tensor_to_pcd(coords, feats, sparse_resolution, shift=False):
    pcd = o3d.geometry.PointCloud()

    points = args.sparse_resolution * coords.numpy()

    colors = [ color_map[int(label)] for label in feats.numpy() ]
    colors = np.asarray(colors) / 255.
    colors = colors[:, ::-1]

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if shift:
        shift_size = (pcd.get_max_bound()[1] - pcd.get_min_bound()[1]) * 1.5
        points[:, 1] = points[:, 1] + shift_size
        pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def model_pipeline(model, data, args):
    eval = iouEval(n_classes=len(content.keys()), ignore=0)

    for iter_n, (x_coord, x_feats, x_label) in enumerate(tqdm(data)):
        x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)

        if 'UNet' in args.sparse_model:
            y = y[:, 0]
        else:
            y = torch.from_numpy(np.asarray(y))
            y = y[:, 0]

        h = model['model'](x)
        z = model['classifier'](h)

        y = y.cuda() if args.use_cuda else y

        # accumulate accuracy
        pred = z.max(dim=1)[1]
        eval.addBatch(pred.long().cpu().numpy(), y.long().cpu().numpy())

        #curr_acc = eval.getacc()
        #miou, _ = eval.getIoU()
        #print(f'\t- Acc: {curr_acc}\tmIoU: {miou}', end='\r')


        if args.visualize_pcd:
            pcd_gt = sparse_tensor_to_pcd(x.C[:, 1:].cpu(), y.cpu(), args.sparse_resolution, shift=True)
            pcd_pred = sparse_tensor_to_pcd(x.C[:, 1:].cpu(), pred.cpu(), args.sparse_resolution)

            o3d.visualization.draw_geometries([pcd_gt, pcd_pred])

    acc = eval.getacc()
    mean_iou, class_iou = eval.getIoU()
    # return the epoch mean loss
    return acc, mean_iou, class_iou


def run_inference(model, args):
    data_val = data_loaders[args.dataset_name](root=args.data_dir, split='validation',
                                                intensity_channel=args.use_intensity, pre_training=False, resolution=args.sparse_resolution)

    # create the data loader for train and validation data
    val_loader = torch.utils.data.DataLoader(
        data_val,
        batch_size=args.batch_size,
        collate_fn=SparseCollation(args.sparse_resolution, inf),
        shuffle=True,
    )

    # retrieve validation loss
    model_acc, model_miou, model_class_iou = model_pipeline(model, val_loader, args)
    print(f'\nModel Acc.: {model_acc}\tModel mIoU: {model_miou}\n\n- Per Class mIoU:')
    for class_ in range(model_class_iou.shape[0]):
        print(f'\t{labels[class_]}: {model_class_iou[class_].item()}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')

    parser.add_argument('--dataset-name', type=str, default='ModelNet40',
                        help='Name of dataset (default: ModelNet40')
    parser.add_argument('--data-dir', type=str, default='./Datasets/ModelNet/modelnet40_normal_resampled',
                        help='Path to dataset (default: ./Datasets/ModelNet/modelnet40_normal_resampled')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='using cuda (default: True')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device id (default: 0')
    parser.add_argument('--feature-size', type=int, default=128,
                        help='Feature output size (default: 128')
    parser.add_argument('--sparse-resolution', type=float, default=0.01,
                        help='Sparse tensor resolution (default: 0.01')
    parser.add_argument('--sparse-model', type=str, default='SparseResNet14',
                        help='Sparse model to be used (default: SparseResNet14')
    parser.add_argument('--use-normals', action='store_true', default=False,
                        help='use points normals (default: False')
    parser.add_argument('--log-dir', type=str, default='checkpoint/downstream_task',
                        help='logging directory (default: checkpoint/downstream_task)')
    parser.add_argument('--best', type=str, default='bestloss',
                        help='best loss or accuracy over training (default: bestloss)')
    parser.add_argument('--checkpoint', type=str, default='sem_seg',
                        help='model checkpoint (default: sem_seg)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input inference batch-size')
    parser.add_argument('--visualize-pcd', action='store_true', default=False,
                        help='visualize inference point cloud (default: False')
    parser.add_argument('--use-intensity', action='store_true', default=False,
                        help='use intensity channel (default: False')

    args = parser.parse_args()

    if args.use_cuda:
        dtype = torch.cuda.FloatTensor
        device = torch.device("cuda")
        torch.cuda.set_device(0)
        print('GPU')
    else:
        dtype = torch.FloatTensor
        device = torch.device("cpu")

    set_deterministic()

    # define backbone architecture
    resnet = get_model(args, dtype)
    resnet.eval()

    classifier = get_classifier_head(args, dtype)
    classifier.eval()

    model_filename = f'{args.best}_model_{args.checkpoint}.pt'
    classifier_filename = f'{args.best}_model_head_{args.checkpoint}.pt'
    print(model_filename, classifier_filename)
    # load pretained weights
    if os.path.isfile(f'{args.log_dir}/{model_filename}') and os.path.isfile(f'{args.log_dir}/{classifier_filename}'):
       checkpoint = torch.load(f'{args.log_dir}/{model_filename}')
       resnet.load_state_dict(checkpoint['model'])
       epoch = checkpoint['epoch']

       checkpoint = torch.load(f'{args.log_dir}/{classifier_filename}')
       classifier.load_state_dict(checkpoint['model'])

       print(f'Loading model: {args.checkpoint}, from epoch: {epoch}')
    else:
       print('Trained model not found!')
       import sys
       sys.exit()
    
    model = {'model': resnet, 'classifier': classifier}
    run_inference(model, args)
