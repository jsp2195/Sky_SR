"""DDPM scheduler and sampler.

Provides:
    - linear / cosine beta schedules
    - q-sample (forward diffusion)
    - ancestral DDPM sampler conditioned on a side input
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(1e-8, 0.999)


@dataclass
class DiffusionTensors:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine"):
        self.timesteps = timesteps
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown schedule {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.t = DiffusionTensors(
            betas=betas.float(),
            alphas=alphas.float(),
            alphas_cumprod=alphas_cumprod.float(),
            alphas_cumprod_prev=alphas_cumprod_prev.float(),
            sqrt_alphas_cumprod=alphas_cumprod.sqrt().float(),
            sqrt_one_minus_alphas_cumprod=(1.0 - alphas_cumprod).sqrt().float(),
            posterior_variance=posterior_variance.float(),
        )

    def to(self, device: torch.device) -> "GaussianDiffusion":
        self.t = DiffusionTensors(**{k: v.to(device) for k, v in self.t.__dict__.items()})
        return self

    def _gather(self, a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        out = a.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sa = self._gather(self.t.sqrt_alphas_cumprod, t, x0.shape)
        sb = self._gather(self.t.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sa * x0 + sb * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sa = self._gather(self.t.sqrt_alphas_cumprod, t, x_t.shape)
        sb = self._gather(self.t.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sb * eps) / sa.clamp_min(1e-8)

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        cond: torch.Tensor,
        shape: tuple[int, int, int, int],
        sampling_steps: Optional[int] = None,
        clip_x0: bool = True,
        x0_clip_min: float = -1.0,
        x0_clip_max: float = 1.0,
        final_clip: bool = True,
        mean_only: bool = False,
        progress: bool = True,
    ) -> torch.Tensor:
        """Ancestral DDPM sampling.

        If ``sampling_steps`` < ``timesteps``, sub-samples a strided index list.
        ``clip_x0`` clamps the per-step x0 estimate to ``[x0_clip_min, x0_clip_max]``
        (default [-1, 1] for clean-image targets). ``final_clip`` clamps the
        returned tensor to ``[x0_clip_min, x0_clip_max]``. Both should be
        disabled or widened when the diffusion target is not a [-1, 1]-valued
        image (e.g. a scaled residual).

        ``mean_only=True`` returns the posterior mean (no per-step noise) for
        a deterministic posterior-mean sample; cheaper and far less variance
        than full ancestral sampling.
        """
        device = cond.device
        T = self.timesteps
        steps = sampling_steps or T
        if steps > T:
            steps = T
        # Strided list of timesteps from T-1 down to 0.
        ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn(shape, device=device)

        iterator = ts
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(ts, desc="ddpm-sample", leave=False)
            except Exception:
                pass

        for i, t_val in enumerate(iterator):
            t_batch = torch.full((shape[0],), int(t_val.item()), device=device, dtype=torch.long)
            eps = model(x, t_batch, cond)
            x0 = self.predict_x0_from_eps(x, t_batch, eps)
            if clip_x0:
                x0 = x0.clamp(x0_clip_min, x0_clip_max)

            # next index in the strided trajectory
            if i < len(ts) - 1:
                t_prev = int(ts[i + 1].item())
            else:
                t_prev = -1

            ac_t = self._gather(self.t.alphas_cumprod, t_batch, x.shape)
            if t_prev >= 0:
                ac_prev = self.t.alphas_cumprod[t_prev].view(1, 1, 1, 1)
            else:
                ac_prev = torch.tensor(1.0, device=device).view(1, 1, 1, 1)
            beta_tilde = ((1.0 - ac_prev) / (1.0 - ac_t)) * (1.0 - ac_t / ac_prev)
            mean = ac_prev.sqrt() * x0 + (1.0 - ac_prev - beta_tilde).clamp_min(0).sqrt() * eps

            if t_prev >= 0 and not mean_only:
                noise = torch.randn_like(x)
                x = mean + beta_tilde.clamp_min(1e-20).sqrt() * noise
            else:
                x = mean
        if final_clip:
            x = x.clamp(x0_clip_min, x0_clip_max)
        return x

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model,
        cond: torch.Tensor,
        shape: tuple[int, int, int, int],
        sampling_steps: Optional[int] = None,
        eta: float = 0.0,
        clip_x0: bool = True,
        x0_clip_min: float = -1.0,
        x0_clip_max: float = 1.0,
        final_clip: bool = True,
        progress: bool = True,
    ) -> torch.Tensor:
        """Deterministic DDIM sampler (eta=0 by default).

        Uses the same epsilon-prediction model. Each step computes x0 from
        the current x_t and eps, then projects to x_{t_prev} via the DDIM
        update. With eta=0 the trajectory is deterministic.
        """
        device = cond.device
        T = self.timesteps
        steps = sampling_steps or T
        if steps > T:
            steps = T
        ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn(shape, device=device)

        iterator = ts
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(ts, desc="ddim-sample", leave=False)
            except Exception:
                pass

        for i, t_val in enumerate(iterator):
            t_batch = torch.full((shape[0],), int(t_val.item()), device=device, dtype=torch.long)
            eps = model(x, t_batch, cond)
            x0 = self.predict_x0_from_eps(x, t_batch, eps)
            if clip_x0:
                x0 = x0.clamp(x0_clip_min, x0_clip_max)

            if i < len(ts) - 1:
                t_prev = int(ts[i + 1].item())
            else:
                t_prev = -1

            ac_t = self._gather(self.t.alphas_cumprod, t_batch, x.shape)
            if t_prev >= 0:
                ac_prev = self.t.alphas_cumprod[t_prev].view(1, 1, 1, 1)
            else:
                ac_prev = torch.tensor(1.0, device=device).view(1, 1, 1, 1)

            # DDIM update: x_{t-1} = sqrt(ac_prev) * x0 + sqrt(1 - ac_prev - sigma^2) * eps + sigma * noise
            sigma2 = (eta ** 2) * ((1.0 - ac_prev) / (1.0 - ac_t)) * (1.0 - ac_t / ac_prev)
            sigma = sigma2.clamp_min(0).sqrt()
            dir_term = (1.0 - ac_prev - sigma2).clamp_min(0).sqrt() * eps
            x = ac_prev.sqrt() * x0 + dir_term
            if t_prev >= 0 and eta > 0:
                x = x + sigma * torch.randn_like(x)
        if final_clip:
            x = x.clamp(x0_clip_min, x0_clip_max)
        return x
