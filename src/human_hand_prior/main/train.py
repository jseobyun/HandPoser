import os
import glob
from dotmap import DotMap
import pytorch_lightning as pl
from src.human_hand_prior.utils.io_utils import load_config
from src.human_hand_prior.manager.lightning_wrapper import LightingHandPoser
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

def train_handposer_once(hp_ps):
    resume_from_checkpoint = hp_ps.train_params.continue_train
    model = LightingHandPoser(hp_ps)

    tblogger = TensorBoardLogger(model.log_dir, name='tensorboard')
    lr_monitor = LearningRateMonitor()

    checkpoint_callback = ModelCheckpoint(
        dirpath=model.snapshot_dir,
        filename="%s_{epoch:03d}_{val_loss:.5f}"%model.expr_name,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min')

    early_stop_callback = EarlyStopping(**model.hp_ps.train_params.early_stopping)

    last_checkpoint = None
    if resume_from_checkpoint:
        available_ckpts = sorted(glob.glob(os.path.join(model.snapshot_dir, '*.ckpt')), key=os.path.getmtime)
        if len(available_ckpts) > 0 :
            last_checkpoint = available_ckpts[-1]


    trainer = pl.Trainer(gpus=hp_ps.train_params.num_gpu,
                         weights_summary='top',
                         plugins=[DDPPlugin(find_unused_parameters=True)],
                         callbacks=[lr_monitor, checkpoint_callback, early_stop_callback], #
                         max_epochs=hp_ps.train_params.num_epoch,
                         logger=tblogger)
    trainer.fit(model) # ckpt_path = last_chekcpoint


if __name__ == '__main__':
    default_ps_fname = glob.glob(os.path.join(os.path.dirname(__file__), '*.yaml'))[0]
    hp_ps = load_config(default_ps_fname)
    train_handposer_once(hp_ps)


