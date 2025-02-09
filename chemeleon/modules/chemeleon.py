from typing import Union, Any, Dict, List, Optional
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


from chemeleon.constants import (
    PATH_CLIP_GENERAL_TEXT,
    PATH_CHEMELEON_GENERAL_TEXT,
    PATH_CLIP_COMPOSITION,
    PATH_CHEMELEON_COMPOSITION,
    CHECKPOINT_URLS,
)
from chemeleon.utils.download import download_file
from chemeleon.modules.base_module import BaseModule
from chemeleon.modules.cspnet import CSPNet, SinusoidalTimeEmbeddings
from chemeleon.utils.diff_utils import (
    BetaScheduler,
    SigmaScheduler,
    d_log_p_wrapped_normal,
    D3PM,
)
from chemeleon.modules.schema import TrajectoryStep, TrajectoryContainer
from chemeleon.text_encoder.text_encoder import TextEncoder
from chemeleon.text_encoder.crystal_clip import CrystalClip


class Chemeleon(BaseModule):
    def __init__(self, _config: Dict[str, Any], **kwargs):
        super().__init__(_config)
        self.save_hyperparameters(_config)
        # time embedding
        self.time_embed = SinusoidalTimeEmbeddings(_config["time_dim"])
        # text embedding
        self.text_guide = _config["text_guide"]
        if self.text_guide:
            if "path_ckpt_clip" in kwargs:
                pretrained_clip_model = CrystalClip.load_from_checkpoint(
                    kwargs["path_ckpt_clip"]
                )
            else:
                pretrained_clip_model = None
            self.cond_drop_prob = _config["cond_drop_prob"]
            self.text_encoder = TextEncoder(
                text_encoder_name=_config["text_encoder"],
                text_embed_dim=_config["text_embed_dim"],
                max_text_len=_config["max_text_len"],
                text_dim=_config["text_dim"],
                trainable_text_encoder=_config["trainable_text_encoder"],
                pretrained_clip_model=pretrained_clip_model,
            )
        # set scheduler
        self.num_timesteps = _config["timesteps"]
        self.beta_scheduler = BetaScheduler(
            timesteps=self.num_timesteps, scheduler_mode=_config["beta_schedule"]
        )
        self.sigma_scheduler = SigmaScheduler(timesteps=self.num_timesteps)
        # d3pm for atom types
        self.max_atoms = _config["max_atoms"]
        self.d3pm = D3PM(
            beta_scheduler=self.beta_scheduler,
            num_timesteps=_config["timesteps"],
            max_atoms=_config["max_atoms"],
            d3pm_hybrid_coeff=_config["d3pm_hybrid_coeff"],
        )
        # lattice noise
        self.mask_lattice_matrix = (
            torch.tensor([[1, 0, 1], [1, 1, 1], [0, 0, 1]]).bool().to(self.device)
        )  # all lattice matrix are re-defined from a function "from_paramters" in pymatgen
        # it will zero out three elements of the lattice matrix
        # decoder
        self.decoder = CSPNet(
            hidden_dim=_config["hidden_dim"],
            time_dim=_config["time_dim"],
            text_dim=_config["text_dim"] if self.text_guide else 0,
            num_layers=_config["num_layers"],
            max_atoms=_config["max_atoms"],
            act_fn=_config["act_fn"],
            dis_emb=_config["dis_emb"],
            num_freqs=_config["num_freqs"],
            edge_style=_config["edge_style"],
            cutoff=_config["cutoff"],
            max_neighbors=_config["max_neighbors"],
            ln=_config["ln"],
            ip=_config["ip"],
            smooth=_config["smooth"],
            pred_atom_types=_config["pred_atom_types"],
        )
        # set cost for loss
        self.cost_atom_types = _config["cost_atom_types"]
        self.cost_lattice = _config["cost_lattice"]
        self.cost_coords = _config["cost_coords"]

    @classmethod
    def load_general_text_model(cls, *args, **kwargs):
        path_ckpt_chemeleon = PATH_CHEMELEON_GENERAL_TEXT
        path_ckpt_clip = PATH_CLIP_GENERAL_TEXT

        # Check and download checkpoints if not exists
        if not os.path.exists(path_ckpt_chemeleon):
            url = CHECKPOINT_URLS["chemeleon_general_text"]
            print(f"Downloading Chemeleon checkpoint from {url}...")
            download_file(url, path_ckpt_chemeleon)

        if not os.path.exists(path_ckpt_clip):
            url = CHECKPOINT_URLS["clip_general_text"]
            print(f"Downloading CLIP checkpoint from {url}...")
            download_file(url, path_ckpt_clip)

        return cls.load_from_checkpoint(
            path_ckpt_chemeleon, path_ckpt_clip=path_ckpt_clip, **kwargs
        )

    @classmethod
    def load_composition_model(cls, *args, **kwargs):
        path_ckpt_chemeleon = PATH_CHEMELEON_COMPOSITION
        path_ckpt_clip = PATH_CLIP_COMPOSITION

        # Check and download checkpoints if not exists
        if not os.path.exists(path_ckpt_chemeleon):
            url = CHECKPOINT_URLS["chemeleon_composition"]
            print(f"Downloading Chemeleon checkpoint from {url}...")
            download_file(url, path_ckpt_chemeleon)

        if not os.path.exists(path_ckpt_clip):
            url = CHECKPOINT_URLS["clip_composition"]
            print(f"Downloading CLIP checkpoint from {url}...")
            download_file(url, path_ckpt_clip)

        return cls.load_from_checkpoint(
            path_ckpt_chemeleon, path_ckpt_clip=path_ckpt_clip, **kwargs
        )

    def forward(self, batch: Batch) -> Dict[str, Any]:
        """Calculate P_loss

        Args:
            batch (Batch): Batch
        Returns:
            Dict[str, Any]: results
        """
        batch_size = batch.num_graphs
        num_nodes = batch.num_nodes
        batched_t = self.beta_scheduler.uniform_sample_t(batch_size, self.device)  # [B]

        time_emb = self.time_embed(batched_t)  # [B, time_dim]

        alpha_cumprod = self.beta_scheduler.alphas_cumprod[batched_t]  # [B]

        c0 = torch.sqrt(alpha_cumprod)  # [B]
        c1 = torch.sqrt(1.0 - alpha_cumprod)  # [B]

        sigmas = self.sigma_scheduler.sigmas[batched_t]  # [B]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[batched_t]  # [B]

        # q_sample (x_t | x_0)
        # 1) d3pm for atom types
        batch_idx = batch.batch
        t_per_node = batched_t[batch_idx]  # [B_n]
        a_0 = batch.atom_types.long()  # [B_n]
        noise_atom_types = torch.rand(num_nodes, self.max_atoms).to(
            self.device
        )  # [B_n, max_atoms]
        x_t_atom_types = self.d3pm.q_sample(
            a_0, t_per_node, noise_atom_types
        )  # [B_n, max_atoms]

        # 2) variance-preserving for lattice
        l_0 = batch.lattices
        mask_lattice_matrix = self.mask_lattice_matrix.to(self.device)
        noise_lattice = torch.randn_like(l_0) * mask_lattice_matrix  # [B, 3, 3]
        x_t_lattice = c0[:, None, None] * l_0 + c1[:, None, None] * noise_lattice

        # 3) variance-exploding for fractional coordinates
        frac_coords = batch.frac_coords
        noise_coords = torch.randn_like(frac_coords)  # [B_n, 3]
        sigmas_per_atom = sigmas[batch_idx][:, None]  # [B_n, 1]
        sigmas_norm_per_atom = sigmas_norm[batch_idx][:, None]  # [B_n, 1]

        target_coords = d_log_p_wrapped_normal(
            sigmas_per_atom * noise_coords, sigmas_per_atom
        ) / torch.sqrt(
            sigmas_norm_per_atom
        )  # [B_n, 1]
        x_t_coords = (frac_coords + sigmas_per_atom * noise_coords) % 1.0

        # text embedding
        text_embeds = None
        if self.text_guide:
            text_embeds = self.text_encoder.get_text_embeds(
                batch.text, self.cond_drop_prob, device=self.device
            )  # [B, D]

        # predict noise lattice and fractional coordinates
        decoder_output = self.decoder(
            t=time_emb,
            frac_coords=x_t_coords,
            lattices=x_t_lattice,
            atom_types=x_t_atom_types,
            num_atoms=batch.natoms,
            node2graph=batch_idx,
            text_embeds=text_embeds,
        )
        pred_x_start_atom_types = decoder_output.atom_types_out  # [B_n, max_atoms]
        pred_noise_lattice = decoder_output.lattice_out  # [B, 3, 3]
        pred_noise_coords = decoder_output.coords_out  # [B_n, 3]

        # loss for atom_types (1) VB (2) CE loss
        true_q_posterior_logits = self.d3pm.q_posterior_logits(
            a_0, x_t_atom_types, t_per_node
        )
        pred_q_posterior_logits = self.d3pm.q_posterior_logits(
            pred_x_start_atom_types, x_t_atom_types, t_per_node, is_x_0_one_hot=True
        )
        vb_loss = self.d3pm.categorical_kl_logits(
            true_q_posterior_logits, pred_q_posterior_logits
        )
        ce_loss = F.cross_entropy(pred_x_start_atom_types.flatten(0, -2), a_0.flatten())
        loss_atom_types = vb_loss + ce_loss * self.d3pm.hybrid_coeff
        # loss for lattice
        loss_lattice = F.mse_loss(
            pred_noise_lattice.masked_select(mask_lattice_matrix),
            noise_lattice.masked_select(mask_lattice_matrix),
        )
        # loss for coords
        loss_coords = F.mse_loss(pred_noise_coords, target_coords)

        loss = (
            self.cost_atom_types * loss_atom_types
            + self.cost_lattice * loss_lattice
            + self.cost_coords * loss_coords
        )
        return {
            "loss": loss,
            "vb_loss_atom_types": vb_loss,
            "ce_loss_atom_types": ce_loss,
            "true_noise_lattice": noise_lattice.masked_select(mask_lattice_matrix),
            "pred_noise_lattice": pred_noise_lattice.masked_select(mask_lattice_matrix),
            "true_noise_coords": target_coords,
            "pred_noise_coords": pred_noise_coords,
        }

    def model_predictions(
        self,
        time_emb: torch.Tensor,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        batch_natoms: torch.Tensor,
        batch_idx: torch.Tensor,
        cond_scale: float,
        text_embeds: Optional[torch.Tensor] = None,
        null_text_embeds: Optional[torch.Tensor] = None,
    ):
        if self.text_guide:
            # conditional sampling
            conditional_decoder_output = self.decoder(
                t=time_emb,
                atom_types=atom_types,
                frac_coords=frac_coords,
                lattices=lattices,
                num_atoms=batch_natoms,
                node2graph=batch_idx,
                text_embeds=text_embeds,
            )
            pred_a_cond = conditional_decoder_output.atom_types_out
            pred_l_cond = conditional_decoder_output.lattice_out
            pred_x_cond = conditional_decoder_output.coords_out

            # unconditional sampling
            unconditional_decoder_output = self.decoder(
                t=time_emb,
                atom_types=atom_types,
                frac_coords=frac_coords,
                lattices=lattices,
                num_atoms=batch_natoms,
                node2graph=batch_idx,
                text_embeds=null_text_embeds,
            )
            pred_a_null = unconditional_decoder_output.atom_types_out
            pred_l_null = unconditional_decoder_output.lattice_out
            pred_x_null = unconditional_decoder_output.coords_out

            # conditional free guidance
            pred_a = (1 - cond_scale) * pred_a_null + cond_scale * pred_a_cond
            pred_l = (1 - cond_scale) * pred_l_null + cond_scale * pred_l_cond
            pred_x = (1 - cond_scale) * pred_x_null + cond_scale * pred_x_cond
        else:
            decoder_output = self.decoder(
                t=time_emb,
                atom_types=atom_types,
                frac_coords=frac_coords,
                lattices=lattices,
                num_atoms=batch_natoms,
                node2graph=batch_idx,
            )
            pred_a = decoder_output.atom_types_out
            pred_l = decoder_output.lattice_out
            pred_x = decoder_output.coords_out
        return pred_a, pred_l, pred_x

    @torch.no_grad()
    def _sample_generator(
        self,
        natoms: Union[int, List[int]],
        texts: Optional[Union[str, List[str]]] = None,
        cond_scale: float = 2.0,
        step_lr: float = 1e-5,
        return_atoms: bool = True,
    ):
        """Sample from the model

        Args:
            natoms (Union[int, List[int]]): number of atoms for each sample
                e.g., [10, 20, 30] or 10
            texts (Optional[Union[str, List[str]]], optional): text for conditional sampling.
                Defaults to None.
            cond_scale (float, optional): scale for conditional sampling.
                if cond_scale > 0.0, use conditional sampling
                Defaults to 2.0.
            step_lr (float, optional): step size for Langevin dynamics.
                Defaults to 1e-5.
            return_atoms (bool, optional): return atoms or not.
                Defaults to True.
        """
        # construct a batch
        if isinstance(natoms, int):
            natoms = [natoms]
        if texts is not None and isinstance(texts, str):
            texts = [texts]
            assert len(natoms) == len(
                texts
            ), "natoms and texts must have the same number of elements."

        data_list = [Data(x=torch.zeros(natom, 1), natoms=natom) for natom in natoms]
        batch = Batch.from_data_list(data_list)
        batch.to(self.device)

        # set properties
        batch_size = batch.num_graphs
        num_nodes = batch.num_nodes
        batch_idx = batch.batch
        batch_natoms = batch.natoms
        mask_lattice_matrix = self.mask_lattice_matrix.to(self.device)

        # sample from pure noise
        a_T = torch.full((num_nodes,), 0, dtype=torch.long).to(self.device)
        l_T = torch.randn(batch_size, 3, 3).to(self.device) * mask_lattice_matrix
        x_T = torch.randn(num_nodes, 3).to(self.device)

        time_start = self.beta_scheduler.timesteps

        trajectory_container = TrajectoryContainer(total_steps=time_start)

        # Initialize the first step of the trajectory at t=T
        trajectory_container[time_start] = TrajectoryStep(
            num_atoms=batch_natoms,
            atom_types=a_T,
            frac_coords=x_T % 1.0,
            lattices=l_T,
            batch_idx=batch_idx,
        )

        if self.text_guide:
            text_embeds = self.text_encoder.get_text_embeds(
                texts,
                cond_drop_prob=0.0,
                device=self.device,
            )
            null_text_embeds = self.text_encoder.get_text_embeds(
                texts,
                cond_drop_prob=1.0,
                device=self.device,
            )
        else:
            text_embeds = None
            null_text_embeds = None

        for t in tqdm(range(time_start, 0, -1)):
            batched_t = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
            time_emb = self.time_embed(batched_t)

            a_t = trajectory_container[t].atom_types
            x_t = trajectory_container[t].frac_coords
            l_t = trajectory_container[t].lattices

            ### Predictor ###
            pred_a, pred_l, pred_x = self.model_predictions(
                time_emb=time_emb,
                atom_types=a_t,
                frac_coords=x_t,
                lattices=l_t,
                batch_natoms=batch_natoms,
                batch_idx=batch_idx,
                cond_scale=cond_scale,
                text_embeds=text_embeds,
                null_text_embeds=null_text_embeds,
            )
            # update the state for atom types
            rand_a = (
                torch.rand((num_nodes, self.max_atoms)).to(self.device)
                if t > 1
                else torch.zeros((num_nodes, self.max_atoms)).to(self.device)
            )
            pred_x_start_logits = pred_a  # [B_n, max_atoms]
            a_t_minus_1, pred_q_posterior_logits = self.d3pm.p_logits(
                pred_x_start_logits=pred_x_start_logits,
                x_t_atom_types=a_t,
                t_per_node=batched_t[batch_idx],
                noise=rand_a,
                return_pred_q_posterior_logits=True,
            )
            # update the state for lattice
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_l = rand_l * mask_lattice_matrix
            # posterior mean and std
            mean_l = c0 * (l_t - c1 * pred_l)
            std_l = sigmas
            l_t_minus_1 = mean_l + std_l * rand_l
            l_t_minus_1 = l_t_minus_1 * mask_lattice_matrix
            # clip the lattice matrix when t = T (for the first step of prediction)
            # otherwise, the lattice matrix could be diverged when the first prediction is too large
            if t == time_start:
                l_t_minus_1 = l_t_minus_1.clip(-6, 6)
            # update the state for coords (0 -> 0.5 step)
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]
            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x**2 - adjacent_sigma_x**2

            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            # posterior mean and std
            mean_x = x_t - step_size * pred_x
            std_x = torch.sqrt(
                (adjacent_sigma_x**2 * (sigma_x**2 - adjacent_sigma_x**2))
                / (sigma_x**2)
            )
            x_t_minus_05 = mean_x + std_x * rand_x

            ### Corrector ###
            _, _, pred_x = self.model_predictions(
                time_emb=time_emb,
                atom_types=a_t_minus_1,
                frac_coords=x_t_minus_05,
                lattices=l_t_minus_1,
                batch_natoms=batch_natoms,
                batch_idx=batch_idx,
                cond_scale=cond_scale,
                text_embeds=text_embeds,
                null_text_embeds=null_text_embeds,
            )

            # update the state for coords (0.5 -> 1 step)
            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            trajectory_step = TrajectoryStep(
                num_atoms=batch_natoms,
                atom_types=a_t_minus_1,
                frac_coords=x_t_minus_1 % 1.0,
                lattices=l_t_minus_1,
                batch_idx=batch_idx,
                timestep=batched_t,
                q_posterior_logits_a=pred_q_posterior_logits,
                mean_l=mean_l,
                std_l=std_l,
                mean_x=mean_x,
                std_x=std_x,
            )
            trajectory_container[t - 1] = trajectory_step
            if return_atoms:
                yield trajectory_container.get_atoms(t=t - 1)
            else:
                yield trajectory_step

    def sample(
        self,
        text_input: str,
        n_atoms: int,
        n_samples: int,
        cond_scale: float = 2.0,
        step_lr: float = 1e-5,
        return_atoms: bool = True,
        return_trajectory: bool = False,
        stream: bool = False,
    ):
        natoms = [n_atoms] * n_samples
        texts = [text_input] * n_samples
        if stream:
            return self._sample_generator(
                natoms, texts, cond_scale, step_lr, return_atoms
            )
        else:
            trajectory = list(
                self._sample_generator(natoms, texts, cond_scale, step_lr, return_atoms)
            )
            if return_trajectory:
                return trajectory
            else:
                return trajectory[-1]
