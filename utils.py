import MinkowskiEngine as ME
import numpy as np
from data_utils.collations import SparseAugmentedCollation, SparseCollation
from data_utils.datasets.SemanticKITTIDataLoader import SemanticKITTIDataLoader
from data_utils.datasets.SemanticPOSSDataLoader import SemanticPOSSDataLoader
from models.minkunet import *
from models.moco import *
from models.blocks import ProjectionHead, SegmentationClassifierHead
from data_utils.data_map import content, content_indoor

sparse_models = {
    'MinkUNet': MinkUNet,
}

data_loaders = {
    'SemanticKITTI': SemanticKITTIDataLoader,
    'SemanticPOSS': SemanticPOSSDataLoader,
}

data_class = {
    'SemanticKITTI': 20,
    'SemanticPOSS': 14,
}

def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

def list_parameters(models):
    optim_params = []
    for model in models:
        optim_params += list(models[model].parameters())

    return optim_params

def get_model(args, dtype, pre_training=False):
    return sparse_models[args.sparse_model](
        in_channels=4 if args.use_intensity else 3,
        out_channels=latent_features[args.sparse_model],
    )#.type(dtype)

def get_projection_head(args, dtype):
    return ProjectionHead(in_channels=latent_features[args.sparse_model], out_channels=args.feature_size)#.type(dtype)

def get_moco_model(args, dtype):
    return MoCo(sparse_models[args.sparse_model], ProjectionHead, dtype, args)

def get_classifier_head(args, dtype):
    if 'UNet' in args.sparse_model:
        return SegmentationClassifierHead(
                in_channels=latent_features[args.sparse_model], out_channels=data_class[args.dataset_name]
            )#.type(dtype)
    else:
        return ClassifierHead(
                in_channels=latent_features[args.sparse_model], out_channels=data_class[args.dataset_name]
            )#.type(dtype)

def get_optimizer(optim_params, args):
    if 'UNet' in args.sparse_model:
        optimizer = torch.optim.SGD(optim_params, lr=args.lr, momentum=0.9, weight_decay=args.decay_lr)
    else:
        optimizer = torch.optim.Adam(optim_params, lr=args.lr, weight_decay=args.decay_lr)

    return optimizer

def get_class_weights(dataset):
    weights = list(content.values()) if dataset == 'SemanticKITTI' else list(content_indoor.values())

    weights = torch.from_numpy(np.asarray(weights)).float()
    if torch.cuda.is_available():
        weights = weights.cuda()

    return weights

def write_summary(writer, summary_id, report, epoch):
    writer.add_scalar(summary_id, report, epoch)

def get_dataset(args, pre_training=True):
    percent_labels = 1.0 if pre_training else args.percentage_labels
    segment_contrast = False if not pre_training else args.segment_contrast
    data_train = data_loaders[args.dataset_name](root=args.data_dir, split='train', percentage=percent_labels, 
                                                    intensity_channel=args.use_intensity, pre_training=pre_training, resolution=args.sparse_resolution)
    data_test = data_loaders[args.dataset_name](root=args.data_dir, split='validation', percentage=percent_labels, 
                                                    intensity_channel=args.use_intensity, pre_training=pre_training, resolution=args.sparse_resolution)

    return data_train, data_test

def get_data_loader(data_train, data_test, args, pre_training=True):
    collate_fn = None

    if pre_training:
        collate_fn = SparseAugmentedCollation(args.sparse_resolution, args.num_points, args.segment_contrast)
    else:
        collate_fn = SparseCollation(args.sparse_resolution, args.num_points)

    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0
    )

    return train_loader, test_loader
