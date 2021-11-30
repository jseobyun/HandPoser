import os
import json
import yaml
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