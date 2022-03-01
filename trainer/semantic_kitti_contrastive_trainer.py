import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import iou
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
from data_utils.data_map import labels, content
from data_utils.ioueval import iouEval
from data_utils.collations import *
from numpy import inf, pi, cos, mean
from functools import partial

class SemanticKITTIContrastiveTrainer(pl.LightningModule):
    def __init__(self, model, criterion, train_loader, params, pre_training=True):
        super().__init__()
        self.moco_model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.params = params
        self.segment_contrast = self.params.segment_contrast
        self.writer = SummaryWriter(f'runs/{params.checkpoint}')
        self.iter_log = 100
        self.loss_eval = []
        self.train_step = 0

        if self.params.load_checkpoint:
            self.load_checkpoint()

    ############################################################################################################################################
    # FORWARD                                                                                                                                  #
    ############################################################################################################################################

    def forward(self, xi, xj, s=None):
        return self.moco_model(xi, xj, s)

    ############################################################################################################################################

    ############################################################################################################################################
    # TRAINING                                                                                                                                 #
    ############################################################################################################################################

    def pre_training_segment_step(self, batch, batch_nb):
        (xi_coord, xi_feats, si), (xj_coord, xj_feats, sj) = batch

        xi, xj = collate_points_to_sparse_tensor(xi_coord, xi_feats, xj_coord, xj_feats)

        out_seg, tgt_seg = self.forward(xi, xj, [si, sj])
        loss = self.criterion(out_seg, tgt_seg)

        self.contrastive_iter_callback(loss.item())

        return {'loss': loss}

    def pre_training_step(self, batch, batch_nb):
        (xi_coord, xi_feats, _), (xj_coord, xj_feats, _) = batch

        xi, xj = collate_points_to_sparse_tensor(xi_coord, xi_feats, xj_coord, xj_feats)
        output, target = self.forward(xi, xj)
        loss = self.criterion(output, target)

        self.contrastive_iter_callback(loss.item())

        return {'loss': loss}

    def training_step(self, batch, batch_nb):
        self.train_step += 1
        torch.cuda.empty_cache()
        return self.pre_training_segment_step(batch, batch_nb) if self.segment_contrast else self.pre_training_step(batch, batch_nb)

    def training_epoch_end(self, outputs):
        avg_loss = torch.FloatTensor([ x['loss'] for x in outputs ]).mean()
        epoch_dict = {'avg_loss': avg_loss}

        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            self.checkpoint_callback()

    ############################################################################################################################################

    ############################################################################################################################################
    # CALLBACKS                                                                                                                                #
    ############################################################################################################################################

    def checkpoint_callback(self):
        if self.current_epoch % 10 == 0:
            self.save_checkpoint(f'epoch{self.current_epoch}')

        if self.current_epoch == self.params.epochs - 1:
            self.save_checkpoint(f'lastepoch{self.current_epoch}')

    def contrastive_iter_callback(self, batch_loss, batch_pcd_loss=None, batch_segment_loss=None):
        # after each iteration we log the losses on tensorboard
        self.loss_eval.append(batch_loss)

        if self.train_step % self.iter_log == 0:
            self.write_summary(
                'training/learning_rate',
                self.scheduler.get_lr()[0],
                self.train_step,
            )

            # loss
            self.write_summary(
                'training/loss',
                mean(self.loss_eval),
                self.train_step,
            )

            self.loss_eval = []

    ############################################################################################################################################

    ############################################################################################################################################
    # SUMMARY WRITERS                                                                                                                          #
    ############################################################################################################################################

    def write_summary(self, summary_id, report, iter):
        self.writer.add_scalar(summary_id, report, iter)

    def contrastive_mesh_writer(self):
        val_iterator = iter(self.train_loader)

        # get just the first iteration(BxNxM) validation set point clouds
        x, y = next(val_iterator)
        z = self.forward(x)
        for i in range(self.params.batch_size):
            points = x.C.cpu().numpy()
            labels = z.max(dim=1)[1].cpu().numpy()

            batch_ind = points[:, 0] == i
            points = expand_dims(points[batch_ind][:, 1:], 0) * self.params.sparse_resolution
            colors = array([ color_map[lbl][::-1] for lbl in labels[batch_ind] ])
            colors = expand_dims(colors, 0)

            point_size_config = {
                'material': {
                    'cls': 'PointsMaterial',
                    'size': 0.3
                }
            }
        
            self.writer.add_mesh(
                f'validation_vis_{i}/point_cloud',
                torch.from_numpy(points),
                torch.from_numpy(colors),
                config_dict=point_size_config,
                global_step=self.current_epoch,
            )

        del val_iterator

    ############################################################################################################################################

    ############################################################################################################################################
    # CHECKPOINT HANDLERS                                                                                                                      #
    ############################################################################################################################################

    def load_checkpoint(self):
        self.configure_optimizers()

        # load model, best loss and optimizer
        file_name = f'{self.params.log_dir}/bestloss_model_{self.params.checkpoint}.pt'
        checkpoint = torch.load(file_name)
        self.moco_model.model_q.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # load model head
        file_name = f'{self.params.log_dir}/bestloss_model_head_{self.params.checkpoint}.pt'
        checkpoint = torch.load(file_name)
        self.moco_model.head_q.load_state_dict(checkpoint['model'])

    def save_checkpoint(self, checkpoint_id):
        # save the best loss checkpoint
        print(f'Writing model checkpoint for {checkpoint_id}')
        state = {
            'model': self.moco_model.model_q.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.log_dir}/{checkpoint_id}_model_{self.params.checkpoint}.pt'

        torch.save(state, file_name)

        state = {
            'model': self.moco_model.head_q.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
        }
        file_name = f'{self.params.log_dir}/{checkpoint_id}_model_head_{self.params.checkpoint}.pt'

        torch.save(state, file_name)
        torch.save(self.state_dict(), f'checkpoint/contrastive/{checkpoint_id}_full_model_{self.params.checkpoint}.pt')

    ############################################################################################################################################

    ############################################################################################################################################
    # OPTIMIZER CONFIG                                                                                                                         #
    ############################################################################################################################################

    def configure_optimizers(self):
        # define optimizers
        optimizer = torch.optim.SGD(self.moco_model.parameters(), lr=self.params.lr, momentum=0.9, weight_decay=self.params.decay_lr, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.params.epochs, eta_min=self.params.lr / 1000)

        self.optimizer = optimizer
        self.scheduler = scheduler

        return [optimizer], [scheduler]

    ############################################################################################################################################

    #@pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    #@pl.data_loader
    def val_dataloader(self):
        pass

    #@pl.data_loader
    def test_dataloader(self):
        pass
