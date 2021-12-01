import os
import cv2
import glob
import torch
from src.human_hand_prior.utils.io_utils import load_config
from src.human_hand_prior.manager.lightning_wrapper import LightingHandPoser
from src.human_hand_prior.utils.vis_utils import vis_mano


def demo_handposer_once(hp_ps):
    model = LightingHandPoser(hp_ps)
    last_checkpoint = None

    available_ckpts = sorted(glob.glob(os.path.join(model.snapshot_dir, '*.ckpt')), key=os.path.getmtime)
    if len(available_ckpts) > 0:
        last_checkpoint = available_ckpts[-1]
        print("The latest checkpoint : ", last_checkpoint)

    model = model.load_from_checkpoint(config=hp_ps, checkpoint_path=last_checkpoint)
    model.eval()

    for i in range(300):
        random_z = torch.randn(1, 32).to(torch.float32)
        mean_val = torch.mean(torch.abs(random_z)).numpy()
        with torch.no_grad():
            pose_rec = model.hp_model.decode(random_z)
        img = vis_mano(pose_rec['hand_pose'][0].reshape(15, 3))
        img_name = str(i)+'_dist'+format(mean_val, '.3f') + '.png'
        cv2.imwrite(os.path.join(model.vis_dir, img_name), img)



if __name__=='__main__':
    default_ps_fname = glob.glob(os.path.join(os.path.dirname(__file__), '*.yaml'))[0]
    hp_ps = load_config(default_ps_fname)
    hp_ps.val_params.vis = False

    demo_handposer_once(hp_ps)
