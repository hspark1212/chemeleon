import torch
import pytorch_lightning as pl

from chemeleon.modules.chemeleon import Chemeleon


class ChemeleonReinforce(pl.LightningModule):
    def __init__(self, _config):
        super().__init__()
        self.save_hyperparameters(_config)
        self.module = Chemeleon.load_from_checkpoint(_config["model_path"])

    @torch.no_grad()
    def sample_fn(self, n_atoms: torch.Tensor, text_input: list[str]):
        trajectory_container = self.module.sample(
            n_atoms=n_atoms,
            n_samples=1,
            text_input=text_input,
            return_trajectory=True,
            return_atoms=False,
        )
        return trajectory_container
