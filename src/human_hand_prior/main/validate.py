import os
import glob
from dotmap import DotMap
import pytorch_lightning as pl
from src.human_hand_prior.utils.io_utils import load_config
from src.human_hand_prior.manager.lightning_wrapper import LightingHandPoser
from pytorch_lightning.plugins import DDPPlugin

def test_handposer_once(hp_ps):
    model = LightingHandPoser(hp_ps)
    last_checkpoint = None

    available_ckpts = sorted(glob.glob(os.path.join(model.snapshot_dir, '*.ckpt')), key=os.path.getmtime)
    if len(available_ckpts) > 0 :
        last_checkpoint = available_ckpts[-1]
        print("The latest checkpoint : ", last_checkpoint)


    trainer = pl.Trainer(gpus=hp_ps.train_params.num_gpu,
                         weights_summary='top',
                         plugins=[DDPPlugin(find_unused_parameters=True)])
    trainer.validate(model, ckpt_path=last_checkpoint)



if __name__ == '__main__':
    default_ps_fname = glob.glob(os.path.join(os.path.dirname(__file__), '*.yaml'))[0]
    hp_ps = load_config(default_ps_fname)
    test_handposer_once(hp_ps)


