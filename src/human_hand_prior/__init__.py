import os
from .utils.io_utils import load_model, load_config
from .manager.lightning_wrapper import LightingHandPoser

def create():
    hp_ps = load_config(os.path.join(os.path.dirname(__file__), 'main/config.yaml'))
    hp_ps.general.snapshot_dir = os.path.join(os.path.dirname(__file__), 'output/snapshot')
    hp = load_model(hp_ps, LightingHandPoser)
    return hp

