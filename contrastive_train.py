from trainer.semantic_kitti_contrastive_trainer import SemanticKITTIContrastiveTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from utils import *
from losses.contrastive import ContrastiveLoss
import argparse
from numpy import inf
import MinkowskiEngine as ME

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')

    parser.add_argument('--dataset-name', type=str, default='SemanticKITTI',
                        help='Name of dataset (default: SemanticKITTI')
    parser.add_argument('--data-dir', type=str, default='./Datasets/SemanticKITTI',
                        help='Path to dataset (default: ./Datasets/SemanticKITTI')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=2.4e-1,
                        help='learning rate (default: 2.4e-1')
    parser.add_argument("--decay-lr", default=1e-4, action="store", type=float,
                        help='Learning rate decay (default: 1e-4')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='Tau temperature smoothing (default 0.1)')
    parser.add_argument('--log-dir', type=str, default='checkpoint/contrastive',
                        help='logging directory (default: checkpoint/contrastive)')
    parser.add_argument('--checkpoint', type=str, default='contrastive_checkpoint',
                        help='model checkpoint (default: contrastive_checkpoint)')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='using cuda (default: False')
    parser.add_argument('--feature-size', type=int, default=128,
                        help='Feature output size (default: 128')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device id (default: 0')
    parser.add_argument('--num-points', type=int, default=20000,
                        help='Number of points sampled from point clouds (default: 20000')
    parser.add_argument('--sparse-resolution', type=float, default=0.05,
                        help='Sparse tensor resolution (default: 0.05')
    parser.add_argument('--sparse-model', type=str, default='MinkUNet',
                        help='Sparse model to be used (default: MinkUNet')
    parser.add_argument('--use-intensity', action='store_true', default=False,
                        help='use points intensity (default: False')
    parser.add_argument('--load-checkpoint', action='store_true', default=False,
                        help='load checkpoint (default: True')
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Number steps to accumulate gradient')
    parser.add_argument('--segment-contrast', action='store_true', default=False,
                        help='Use segments patches for contrastive learning (default: False')

    args = parser.parse_args()

    if args.use_cuda:
        dtype = torch.cuda.FloatTensor
        device = torch.device("cuda")
        print('GPU')
    else:
        dtype = torch.FloatTensor
        device = torch.device("cpu")

    data_train, data_test = get_dataset(args)
    train_loader, _ = get_data_loader(data_train, data_test, args)
    criterion = nn.CrossEntropyLoss().cuda()

    model = get_moco_model(args, dtype)

    if torch.cuda.device_count() > 1:
        model_sem_kitti = SemanticKITTIContrastiveTrainer(model, criterion, train_loader, args)
        trainer = Trainer(gpus=-1, accelerator='ddp', max_epochs=args.epochs, accumulate_grad_batches=args.accum_steps)
        trainer.fit(model_sem_kitti)

    else:
        model_sem_kitti = SemanticKITTIContrastiveTrainer(model, criterion, train_loader, args)
        trainer = Trainer(gpus=[0], max_epochs=args.epochs, accumulate_grad_batches=args.accum_steps)
        trainer.fit(model_sem_kitti)
