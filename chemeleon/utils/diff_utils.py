# https://github.com/jiaor17/DiffCSP/diffcsp/pl_modules/diff_utils.py
import math
import numpy as np
import torch
import torch.nn as nn

DEFAULT_DTYPE = torch.get_default_dtype()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-((x + T * i) ** 2) / 2 / sigma**2)
    return p_


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + T * i) / sigma**2 * torch.exp(-((x + T * i) ** 2) / 2 / sigma**2)
    return p_ / p_wrapped_normal(x, sigma, N, T)


def sigma_norm(sigma, T=1.0, sn=10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T=T)
    return (normal_**2).mean(dim=0)


class BetaScheduler(nn.Module):
    def __init__(self, timesteps, scheduler_mode, beta_start=0.0001, beta_end=0.02):
        super(BetaScheduler, self).__init__()
        self.timesteps = timesteps
        if scheduler_mode == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "quadratic":
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Invalid scheduler mode: {scheduler_mode}")

        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        # posterior mean
        posterior_mean_coeff1 = torch.ones_like(betas)
        posterior_mean_coeff1[1:] = (
            betas[1:] * torch.sqrt(alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        )  # coeff for x0 = beta_t * sqrt(alpha_cumprod(t-1)) / (1 - alpha_cumprod(t))
        posterior_mean_coeff2 = torch.zeros_like(betas)
        posterior_mean_coeff2[1:] = (
            (1.0 - alphas_cumprod[:-1])
            * torch.sqrt(alphas[1:])
            / (1.0 - alphas_cumprod[1:])
        )  # coeff for xt = (1 - alpha_cumprod(t-1)) * sqrt(alpha(t)) / (1 - alpha_cumprod(t))
        # posterior variance
        sigmas = torch.zeros_like(betas)
        sigmas[1:] = (
            betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        )  # sigma_t = beta_t * (1 - alpha_cumprod(t-1)) / (1 - alpha_cumprod(t))
        sigmas = torch.sqrt(sigmas)

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(DEFAULT_DTYPE))

        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("posterior_mean_coeff1", posterior_mean_coeff1)
        register_buffer("posterior_mean_coeff2", posterior_mean_coeff2)
        register_buffer("sigmas", sigmas)

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps + 1), batch_size)
        return torch.from_numpy(ts).to(device)


class SigmaScheduler(nn.Module):
    def __init__(self, timesteps, sigma_begin=0.01, sigma_end=1.0):
        super(SigmaScheduler, self).__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = torch.FloatTensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps))
        )

        sigmas_norm_ = sigma_norm(sigmas)

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(DEFAULT_DTYPE))

        register_buffer("sigmas", torch.cat([torch.zeros([1]), sigmas], dim=0))
        register_buffer(
            "sigmas_norm", torch.cat([torch.ones([1]), sigmas_norm_], dim=0)
        )

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps + 1), batch_size)
        return torch.from_numpy(ts).to(device)


def prob_mask_like(shape: tuple, prob: float, device: torch.device) -> torch.Tensor:
    """
    For classifier free guidance. Creates a boolean mask for given input shape and probability of `True`.

    :param shape: Shape of mask.
    :param prob: Probability of True. In interval [0., 1.].
    :param device: Device to put the mask on. Should be the same as that of the tensor which it will be used on.
    :return: The mask.
    """
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# D3PM for atom types
class D3PM(nn.Module):
    def __init__(
        self,
        beta_scheduler: nn.Module,
        num_timesteps: int,
        max_atoms: int,
        d3pm_hybrid_coeff: float,
    ):
        super().__init__()
        self.beta_scheduler = beta_scheduler
        self.num_timesteps = num_timesteps
        self.max_atoms = max_atoms
        self.hybrid_coeff = d3pm_hybrid_coeff
        self.eps = 1.0e-6

        # transition matrix for absorbing
        q_one_step_mats = torch.stack(
            [
                self.get_absorbing_transition_mat(t)
                for t in range(0, self.num_timesteps + 1)
            ],
            dim=0,
        )
        self.register_buffer("q_one_step_mats", q_one_step_mats)

        # construct transition matrices for q(x_t | x_0)
        q_mat_t = self.q_one_step_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps + 1):
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = q_mat_t @ self.q_one_step_mats[t]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (
            self.num_timesteps + 1,
            self.max_atoms,
            self.max_atoms,
        )

        self.q_one_step_transposed = self.q_one_step_mats.transpose(1, 2)

    def get_absorbing_transition_mat(self, t: int):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Args:
            t (int): timestep.
            max_atoms (int): maximum number of atoms (103 + 1 for dummy atom).
                Defaults to 104.

        Returns:
            Q_t: transition matrix. shape = (max_atoms, max_atoms)
        """
        # get beta at timestep t
        beta_t = self.beta_scheduler.betas[t]

        diag = torch.full((self.max_atoms,), 1 - beta_t)
        mat = torch.diag(diag, 0)
        # add beta_t at first row
        mat[:, 0] += beta_t
        return mat

    def at(
        self,
        a: torch.Tensor,
        t_per_node: torch.Tensor,
        x: torch.Tensor,
    ):
        """Extract coefficients at specified timesteps t - 1 and conditioning data x.

        Args:
            a (torch.Tensor): matrix of coefficients. [num_timesteps, max_atoms, max_atoms]
            t_per_node (torch.Tensor): timesteps.[B_n]
            x (torch.Tensor): atom_types. [B_n]

        Returns:
            a[t, x] (torch.Tensor): coefficients at timesteps t and data x. [B_n, max_atoms]
        """
        a = a.to(x.device)
        bs = t_per_node.shape[0]
        t_per_node = t_per_node.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t_per_node - 1, x, :]

    def q_sample(
        self,
        x_0: torch.Tensor,
        t_per_node: torch.Tensor,
        noise: torch.Tensor,
    ):
        """Sample from q(x_t | x_0) (i.e. add noise to the data).
        q(x_t | x_0) = Categorical(x_t ; p = x_0 Q_{1...t})

        Args:
            x_0 (torch.Tensor): Image data at t=0. [B, C, H, W]
            t_per_node (torch.Tensor): Timesteps. [B_n]
            noise (torch.Tensor): Noise. [B_n, max_atoms]

        Returns:
            torch.Tensor: [B_n, max_atoms]
        """
        logits = torch.log(self.at(self.q_mats, t_per_node, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t_per_node: torch.Tensor,
        is_x_0_one_hot: bool = False,
    ):
        """Compute logits for q(x_{t-1} | x_t, x_0)."""
        if is_x_0_one_hot:
            x_0_logits = x_0.clone()
        else:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.max_atoms) + self.eps
            )

        assert x_0_logits.shape == x_t.shape + (self.max_atoms,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        fact1 = self.at(self.q_one_step_transposed, t_per_node, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)
        qmats2 = self.q_mats[t_per_node - 2]
        fact2 = torch.einsum("b...c, bcd -> b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t_per_node.reshape((t_per_node.shape[0], *[1] * (x_t.dim())))
        return torch.where(t_broadcast == 1, x_0_logits, out)

    def categorical_kl_logits(self, logits1, logits2, eps=1.0e-6):
        """KL divergence between categorical distributions.

        Distributions parameterized by logits.

        Args:
            logits1: logits of the first distribution. Last dim is class dim.
            logits2: logits of the second distribution. Last dim is class dim.
            eps: float small number to avoid numerical issues.

        Returns:
            KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
        """
        out = torch.softmax(logits1 + eps, dim=-1) * (
            torch.log_softmax(logits1 + eps, dim=-1)
            - torch.log_softmax(logits2 + eps, dim=-1)
        )
        return out.sum(dim=-1).mean()

    def p_logits(
        self,
        pred_x_start_logits: torch.Tensor,
        x_t_atom_types: torch.Tensor,
        t_per_node: torch.Tensor,
        noise: torch.Tensor,
        return_pred_q_posterior_logits: bool = False,
    ):
        pred_q_posterior_logits = self.q_posterior_logits(
            pred_x_start_logits, x_t_atom_types, t_per_node, is_x_0_one_hot=True
        )

        noise = torch.clamp(noise, min=self.eps, max=1.0)
        # if t == 1, use x_0_logits
        nonzero_mask = (
            (t_per_node != 1)
            .to(x_t_atom_types.dtype)
            .view(-1, *([1] * (x_t_atom_types.ndim)))
        )
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * nonzero_mask, dim=-1
        )
        if return_pred_q_posterior_logits:
            return sample, pred_q_posterior_logits
        else:
            return sample
