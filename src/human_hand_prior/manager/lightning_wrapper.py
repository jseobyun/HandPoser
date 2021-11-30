
from src.human_hand_prior.utils.io_utils import load_config
from src.human_hand_prior.utils.train_utils import make_deterministic
from src.human_hand_prior.models.handposer_model import HandPoser
from pytorch_lightning.core import LightningModule
class Trainer(LightningModule):
    def __init__(self, config):
        super(Trainer, self).__init__()
        hp_ps = load_config(**config)
        make_deterministic(hp_ps.general.random_seed)

        self.expr_name = hp_ps.general.expr_name
        self.dataset_names = hp_ps.general.dataset_names
        self.dataset_dir = hp_ps.general.datset_dir

        self.hp_model = HandPoser(hp_ps)

    def forward(self, pose):
        return self.hp_model(pose)




