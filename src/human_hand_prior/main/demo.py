import os
import cv2
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
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

    cmap = plt.get_cmap('YlGnBu')
    colors = [cmap(i) for i in np.linspace(0, 1, 200)]
    colors = [(c[0], c[1], c[2]) for c in colors]

    num_demo = 5000
    random_zs = torch.randn(num_demo, 32).to(torch.float32)

    random_zs_np = random_zs.numpy()
    z_emb = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(random_zs_np)
    dist = np.sqrt(np.sum(z_emb**2, axis=1))
    max_dist = np.max(dist)
    dist = dist/max_dist * 200
    idx = dist.astype(int)
    idx[idx>=200] = 199
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    for i in range(num_demo):
        x, y = z_emb[i]
        if i % 200 != 0:
            continue
        random_z = random_zs[i:i + 1, :]
        with torch.no_grad():
            pose_rec = model.hp_model.decode(random_z)
        img = vis_mano(pose_rec['hand_pose'][0].reshape(15, 3))
        img_name = str(i)+f'_x{x}'+f'_y{y}' + '.png'
        cv2.imwrite(os.path.join(model.vis_dir, img_name), img)

    for i, (x, y) in enumerate(z_emb):
        ax.scatter(x, y, s=1, c=colors[idx[i]], marker='o')
    plt.savefig(os.path.join(model.vis_dir, 'scatter.png'))



if __name__=='__main__':
    default_ps_fname = glob.glob(os.path.join(os.path.dirname(__file__), '*.yaml'))[0]
    hp_ps = load_config(default_ps_fname)
    hp_ps.val_params.vis = False

    demo_handposer_once(hp_ps)
