import os
import cv2
import random
import numpy as np
import smplx
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from src.human_hand_prior.utils.io_utils import load_config
from src.human_hand_prior.utils.train_utils import make_deterministic, geodesic_loss_R
from src.human_hand_prior.utils.rot_utils import aa2matrot
from src.human_hand_prior.utils.data_utils import copy2cpu
from src.human_hand_prior.models.handposer_model import HandPoser
from src.human_hand_prior.data.load import build_data_loader
from pytorch_lightning.core import LightningModule

class LightingHandPoser(LightningModule):
    def __init__(self, config):
        super(LightingHandPoser, self).__init__()
        self.hp_ps = load_config(**config)
        make_deterministic(self.hp_ps.general.random_seed)

        self.expr_name = self.hp_ps.general.expr_name
        self.dataset_names = self.hp_ps.general.dataset_names
        self.dataset_dir = self.hp_ps.general.dataset_dir
        self.log_dir = self.hp_ps.general.log_dir
        self.snapshot_dir = self.hp_ps.general.snapshot_dir


        self.hp_model = HandPoser(self.hp_ps)
        mano_dir = self.hp_ps.general.mano_dir
        self.mano_layer = smplx.create(mano_dir, 'mano', use_pca=False, is_rhand=True).eval() # only right hand

    def forward(self, pose):
        result = self.hp_model(pose)
        return result

    ### About data
    def train_dataloader(self):
        return build_data_loader(self.dataset_names, self.dataset_dir, 'train',
                                 **self.hp_ps.train_params)

    def val_dataloader(self):
        return build_data_loader(self.dataset_names, self.dataset_dir, 'val',
                                 **self.hp_ps.val_params)

    ### About optimizer
    def configure_optimizers(self):
        gen_params = [a[1] for a in self.hp_model.named_parameters() if a[1].requires_grad]
        gen_optimizer_class = getattr(optim, self.hp_ps.train_params.optimizer_type)
        gen_optimizer = gen_optimizer_class(gen_params, **self.hp_ps.train_params.optimizer_args)

        lr_sched_class = getattr(lr_scheduler, self.hp_ps.train_params.lr_scheduler_type)
        gen_lr_scheduler = lr_sched_class(gen_optimizer, **self.hp_ps.train_params.lr_scheduler_args)

        schedulers = [
            {
                'scheduler': gen_lr_scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            },
        ]
        return [gen_optimizer], schedulers

    ### About training

    def training_step(self, batch):
        d_rec = self(batch['hand_pose'].view(-1, 45))
        loss =self._compute_loss(batch, d_rec)
        train_loss = loss['weighted_loss']['loss_total']
        tensorboard_logs = {'train_loss': train_loss}
        progress_bar = {k: copy2cpu(v) for k, v in loss['weighted_loss'].items()}
        self.log("train_loss", train_loss)
        return {'loss' : train_loss }#, 'progress_bar':progress_bar, 'tblog': tensorboard_logs}


    ### About validation
    def validation_step(self, batch, batch_idx):
        d_rec = self(batch['hand_pose'].view(-1, 45))
        loss = self._compute_loss(batch, d_rec)
        val_loss = loss['unweighted_loss']['loss_total']

        progress_bar = {k: copy2cpu(v) for k, v in loss['unweighted_loss'].items()}
        self.log("val_loss_step", val_loss)
        return {'val_loss_step': copy2cpu(val_loss)}#, 'progress_bar':progress_bar, 'tblog':progress_bar}

    def validation_epoch_end(self, outputs):
        metrics = {'val_loss': np.nanmean(np.concatenate([v['val_loss_step'] for v in outputs]))}
        metrics = {k: torch.as_tensor(v) for k, v in metrics.items()}
        self.log("val_loss", metrics['val_loss'])
        return {'val_loss': metrics['val_loss'], 'log': metrics}


    ### About loss
    def _compute_loss(self, d_ori, d_rec):
        device = d_rec['Z_hand_mean'].device
        l1_loss = nn.L1Loss(reduction='mean')
        geodesic_loss = geodesic_loss_R(reduction='mean')
        batch_size, latent_dim =d_rec['Z_hand_mean'].shape


        w_KL = self.hp_ps.train_params.loss_weight_KL
        w_vert = self.hp_ps.train_params.loss_weight_vert
        w_matrot = self.hp_ps.train_params.loss_weight_matrot
        w_jtr = self.hp_ps.train_params.loss_weight_jtr

        q_z = d_rec['q_z']
        root_pose = torch.zeros([batch_size, 3], dtype=torch.float32).to(device)
        root_pose.requires_grad = False

        with torch.no_grad():
            mano_ori = self.mano_layer(global_orient=root_pose, hand_pose=d_ori['hand_pose'], betas=d_ori['hand_shape'].detach())#, trans=trans)
        mano_rec = self.mano_layer(global_orient=root_pose, hand_pose=d_rec['hand_pose'], betas=d_ori['hand_shape'].detach())#, trans=trans)

        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((batch_size, latent_dim), device=device, requires_grad=False),
            scale=torch.ones((batch_size, latent_dim), device=device, requires_grad=False))
        weighted_loss_dict = {
            'loss_KL' : w_KL * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1])),
            'loss_vert' : w_vert * l1_loss(mano_ori.vertices, mano_rec.vertices)
        }

        if (self.current_epoch < self.hp_ps.train_params.keep_extra_loss_terms_until_epoch):
            weighted_loss_dict['loss_matrot'] = w_matrot*geodesic_loss(d_rec['hand_pose_matrot'].view(-1,3,3), aa2matrot(d_ori['hand_pose'].view(-1,3)))
            weighted_loss_dict['loss_jtr'] = w_jtr*l1_loss(mano_ori.joints, mano_rec.joints)
        weighted_loss_dict['loss_total'] = torch.stack(list(weighted_loss_dict.values())).sum()

        with torch.no_grad():
            unweighted_loss_dict = {'loss_vert': torch.sqrt(torch.pow(mano_rec.vertices -mano_ori.vertices, 2).sum(-1)).mean()}
            unweighted_loss_dict['loss_total'] = torch.cat(
                list({k: v.view(-1) for k, v in unweighted_loss_dict.items()}.values()), dim=-1).sum().view(1)

        return {'weighted_loss' : weighted_loss_dict,
                'unweighted_loss' : unweighted_loss_dict}





