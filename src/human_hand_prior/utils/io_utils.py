import os
import json
import yaml
import glob
from dotmap import DotMap


def load_config(default_ps_fname=None, **kwargs):
    if isinstance(default_ps_fname, str):
        assert os.path.exists(default_ps_fname), FileNotFoundError(default_ps_fname)
        assert default_ps_fname.lower().endswith('.yaml'), NotImplementedError('Only .yaml files are accepted.')
        default_ps = yaml.safe_load(open(default_ps_fname, 'r'))
    else:
        default_ps = {}

    default_ps.update(kwargs)

    return DotMap(default_ps, _dynamic=False)

def load_json(json_path):
    if not os.path.exists(json_path):
        print(f"No such file or dir : {json_path}")
        return None
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def load_model(hp_ps, model_code):
    expr_name = str(hp_ps.general.expr_name)
    snapshot_dir = hp_ps.general.snapshot_dir
    snapshot_list = os.listdir(snapshot_dir)
    expr_snapshots = [os.path.join(snapshot_dir, snapshot) for snapshot in snapshot_list if snapshot.startswith(expr_name) and snapshot.endswith('.ckpt')]
    if len(expr_snapshots) == 0:
        raise FileNotFoundError(f"Could not found experiment {expr_name} model in snapshot directory.")
    available_ckpts = sorted(expr_snapshots, key=os.path.getmtime)
    last_ckpt = available_ckpts[-1]
    pl_model = model_code(hp_ps)
    pl_model.load_from_checkpoint(config=hp_ps, checkpoint_path=last_ckpt)
    model = pl_model.hp_model
    model.eval()
    print(f"{last_ckpt} model is loaded.")
    return model