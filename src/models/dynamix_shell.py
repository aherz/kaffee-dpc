"""Stochastic shell for DynaMix core."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from models.dynamix import DynaMix


class DynaMixShell(nn.Module):
	"""DynaMix stochastic shell."""

	def __init__(
		self,
		model: DynaMix,
		*,
		log_proc_std: torch.Tensor | float,
		log_obs_std: torch.Tensor | float,
		trainable: bool = True,
		covariance_transport: str = "jacobian",
	) -> None:
		super().__init__()
		self.model = model
		self.latent_dim = int(model.M)
		self.observed_dim = int(model.N)
		transport = str(covariance_transport).strip().lower()
		if transport not in {"jacobian", "identity"}:
			raise ValueError("covariance_transport must be 'jacobian' or 'identity'")
		self.covariance_transport = transport
		self._objective_context: torch.Tensor | None = None
		self._objective_precomputed_cnn: torch.Tensor | None = None
		dtype = next(model.parameters()).dtype
		device = next(model.parameters()).device
		proc = torch.as_tensor(log_proc_std, dtype=dtype, device=device).reshape(-1)
		obs = torch.as_tensor(log_obs_std, dtype=dtype, device=device).reshape(-1)
		if proc.numel() == 1:
			proc = proc.expand(int(model.M)).clone()
		if obs.numel() == 1:
			obs = obs.expand(int(model.N)).clone()
		if proc.numel() != int(model.M):
			raise ValueError(f"log_proc_std must have length {model.M}, got {proc.numel()}")
		if obs.numel() != int(model.N):
			raise ValueError(f"log_obs_std must have length {model.N}, got {obs.numel()}")
		if bool(trainable):
			self.log_proc_std = nn.Parameter(proc)
			self.log_obs_std = nn.Parameter(obs)
		else:
			self.register_buffer("log_proc_std", proc)
			self.register_buffer("log_obs_std", obs)

	@property
	def Q(self) -> torch.Tensor:
		return torch.diag(self.log_proc_std.exp().pow(2))

	@property
	def R(self) -> torch.Tensor:
		return torch.diag(self.log_obs_std.exp().pow(2))

	@property
	def Gamma(self) -> torch.Tensor:
		return self.Q

	@property
	def Lambda(self) -> torch.Tensor:
		return self.R

	@property
	def sigma_proc(self) -> torch.Tensor:
		return self.log_proc_std.exp().mean()

	@property
	def sigma_obs(self) -> torch.Tensor:
		return self.log_obs_std.exp().mean()

	def set_objective_context(self, context: torch.Tensor) -> None:
		self._objective_context = context
		self._objective_precomputed_cnn = self.model.precompute_cnn(context)

	def initialize_from_observation(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""DynaMix/AL-RNN latent initialization: z = x @ B, then z[:, :N] = x; where B is the encoder/embedding learned in Dynamix (core) training."""
		if x0.ndim != 2:
			raise ValueError(f"x0 must have shape (batch, N), got {x0.shape}")
		z = x0 @ self.model.B
		z[:, : int(self.model.N)] = x0
		sigma0 = self.Gamma.unsqueeze(0).expand(x0.shape[0], int(self.model.M), int(self.model.M)).clone()
		return z, sigma0

	def latent_column_from_observation(self, x0: torch.Tensor) -> torch.Tensor:
		"""Encode one observation into a latent (M, 1) array for scoring."""
		if x0.ndim == 1:
			x0_b = x0.unsqueeze(0)
		elif x0.ndim == 2 and int(x0.shape[0]) == 1:
			x0_b = x0
		else:
			raise ValueError(f"x0 must have shape (N,) or (1, N), got {tuple(x0.shape)}")
		if int(x0_b.shape[-1]) != int(self.model.N):
			raise ValueError(f"x0 has dim {int(x0_b.shape[-1])}, expected N={int(self.model.N)}")
		z_row, _ = self.initialize_from_observation(x0_b)
		return z_row.T.contiguous()

	def transition_and_jacobian(self, z_mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		if self._objective_context is None:
			raise RuntimeError("Call set_objective_context(context) before filtering objectives.")
		if z_mean.ndim != 2:
			raise ValueError(f"z_mean must have shape (batch, M), got {z_mean.shape}")
		batch = int(z_mean.shape[0])
		context = self._objective_context.expand(-1, batch, -1)
		precomputed = self.model.precompute_cnn(context)
		z_cols = z_mean.T.contiguous()
		z_pred = self.model.step(z_cols, context, precomputed_cnn=precomputed).T
		jacobians = []
		for idx in range(batch):
			z_col = z_mean[idx].unsqueeze(1)
			ctx_i = self._objective_context
			cnn_i = self._objective_precomputed_cnn
			if cnn_i is None:
				cnn_i = self.model.precompute_cnn(ctx_i)
			jacobians.append(compute_mixture_jacobian(self.model, z_col, ctx_i, cnn_i))
		return z_pred, torch.stack(jacobians, dim=0)

	def ensemble_forward(
		self,
		z_init: torch.Tensor,
		context: torch.Tensor,
		*,
		horizon: int,
		n_ensemble: int,
	) -> torch.Tensor:
		"""Run ensemble forecasts."""
		m_dim = int(z_init.shape[0])
		z = z_init.expand(m_dim, int(n_ensemble)).clone()
		ctx = context.expand(-1, int(n_ensemble), -1)
		cnn = self.model.precompute_cnn(ctx)
		sigma_q = self.log_proc_std.exp().unsqueeze(1)
		traj = torch.empty(int(horizon), m_dim, int(n_ensemble), device=z.device, dtype=z.dtype)
		for t in range(int(horizon)):
			z = self.model.step(z, ctx, precomputed_cnn=cnn)
			z = z + sigma_q * torch.randn_like(z)
			traj[t] = z
		return traj

	def windowed_nll(
		self,
		z_init: torch.Tensor,
		context: torch.Tensor,
		obs_window: torch.Tensor,
		*,
		n_ensemble: int = 256,
		burn_in_steps: int = 0,
	) -> torch.Tensor:
		"""Gaussian ensemble NLL per scored time step and observed dimension."""
		length, n_dim = obs_window.shape
		m_dim = int(z_init.shape[0])
		n_transitions = int(length) - 1
		if n_transitions < 1:
			raise ValueError("obs_window must contain an initial observation and at least one future observation")
		burn = max(0, min(int(burn_in_steps), n_transitions))
		z = z_init.expand(m_dim, int(n_ensemble)).clone()
		ctx = context.expand(-1, int(n_ensemble), -1)
		cnn = self.model.precompute_cnn(ctx)
		sigma_q = self.log_proc_std.exp().unsqueeze(1)
		sigma_r2 = self.log_obs_std.exp().pow(2)
		total = torch.zeros((), device=z.device, dtype=z.dtype)
		scored = 0
		for t in range(n_transitions):
			z = self.model.step(z, ctx, precomputed_cnn=cnn)
			z = z + sigma_q * torch.randn_like(z)
			if t < burn:
				continue
			y_pred = z[:n_dim]
			mu = y_pred.mean(dim=1)
			var = y_pred.var(dim=1) + sigma_r2
			var = var.clamp_min(1.0e-8)
			y_true = obs_window[t + 1]
			total = total + (0.5 * (torch.log(var) + (y_true - mu).pow(2) / var + math.log(2.0 * math.pi))).sum()
			scored += 1
		return total / float(max(scored, 1) * int(n_dim))

	def batched_windowed_nll(
		self,
		z_inits: list[torch.Tensor],
		contexts: list[torch.Tensor],
		obs_windows: list[torch.Tensor],
		*,
		n_ensemble: int = 256,
		burn_in_steps: int = 0,
	) -> torch.Tensor:
		if not z_inits:
			raise ValueError("z_inits must be non-empty")
		same_length = len({int(window.shape[0]) for window in obs_windows}) == 1
		if int(burn_in_steps) == 0 and same_length:
			obs = torch.stack(list(obs_windows), dim=0)
			return self._lead_averaged_nll(contexts[0], obs, n_ensemble=int(n_ensemble))
		total = torch.zeros((), device=z_inits[0].device, dtype=z_inits[0].dtype)
		for z0, context, obs in zip(z_inits, contexts, obs_windows, strict=True):
			total = total + self.windowed_nll(
				z0,
				context,
				obs,
				n_ensemble=int(n_ensemble),
				burn_in_steps=int(burn_in_steps),
			)
		return total / float(len(z_inits))

	def _lead_averaged_nll(
		self,
		context: torch.Tensor,
		obs: torch.Tensor,
		*,
		n_ensemble: int,
		chunk: int = 20,
	) -> torch.Tensor:
		"""Mean diagonal ensemble-Gaussian NLL over horizons 1,...,H.

		"obs" has shape (windows, H+1, N). The state at horizon 0 is the encoding
		of "obs[:, 0]", and horizon "h" is scored against "obs[:, h]".
		"""
		if obs.ndim != 3 or int(obs.shape[1]) < 2:
			raise ValueError(f"obs must have shape (windows, H+1, N), got {tuple(obs.shape)}")
		n_win, length, n_dim = (int(obs.shape[0]), int(obs.shape[1]), int(obs.shape[2]))
		horizon = length - 1
		m_dim = int(self.latent_dim)
		total = torch.zeros((), device=obs.device, dtype=obs.dtype)
		counted = 0
		sigma_r2 = self.log_obs_std.exp().pow(2)
		for start in range(0, n_win, int(chunk)):
			batch = obs[start : start + int(chunk)]
			b = int(batch.shape[0])
			encoded = []
			for i in range(b):
				encoded.append(self.latent_column_from_observation(batch[i, 0])[:, 0])
			z_row = torch.stack(encoded, dim=0)
			z = (
				z_row.transpose(0, 1)
				.unsqueeze(-1)
				.expand(m_dim, b, int(n_ensemble))
				.reshape(m_dim, b * int(n_ensemble))
				.contiguous()
			)
			ctx = context.expand(-1, b * int(n_ensemble), -1)
			cnn = self.model.precompute_cnn(ctx)
			sigma_q = self.log_proc_std.exp().unsqueeze(1)
			for step in range(horizon):
				z = self.model.step(z, ctx, precomputed_cnn=cnn)
				z = z + sigma_q * torch.randn_like(z)
				y = z[:n_dim].reshape(n_dim, b, int(n_ensemble))
				mu = y.mean(dim=-1)
				var = y.var(dim=-1, unbiased=True) + sigma_r2.unsqueeze(1)
				var = var.clamp_min(1.0e-8)
				y_true = batch[:, step + 1, :].transpose(0, 1)
				nll = 0.5 * (torch.log(var) + (y_true - mu).pow(2) / var + math.log(2.0 * math.pi))
				total = total + nll.sum()
				counted += b * n_dim
		return total / float(max(counted, 1))


def compute_mixture_jacobian(
	model: DynaMix,
	z: torch.Tensor,
	context: torch.Tensor,
	precomputed_cnn: torch.Tensor,
) -> torch.Tensor:
	"""Frozen-gate mixture Jacobian ("sum_k w_k J_k") for DynaMix AL-RNN experts."""
	with torch.no_grad():
		w_exp = model.gating_network(context, z, precomputed_cnn=precomputed_cnn)

	m_dim = int(model.M)
	n_experts = int(model.Experts)
	p_relu = int(model.P)
	if p_relu <= 0:
		mask = torch.ones(m_dim, 1, dtype=z.dtype, device=z.device)
	else:
		relu_active = (z[-p_relu:, :] > 0).to(z.dtype)
		mask = torch.cat([torch.ones(m_dim - p_relu, 1, dtype=z.dtype, device=z.device), relu_active], dim=0)

	a_all = torch.stack([expert.A for expert in model.experts])
	w_all = torch.stack([expert.W for expert in model.experts])
	j_all = torch.diag_embed(a_all) + w_all * mask.squeeze(1).unsqueeze(0).unsqueeze(1)
	return (w_exp[:, 0].view(n_experts, 1, 1) * j_all).sum(0)


def load_dynamix_shell_checkpoint(
	path: str | Path,
	*,
	model: DynaMix,
	trainable_shell: bool = False,
) -> DynaMixShell:
	"""Load a DynaMix shell checkpoint."""
	state = load_file(str(path))
	if "log_proc_std" not in state or "log_obs_std" not in state:
		raise KeyError(f"Checkpoint {path} is missing log_proc_std/log_obs_std")
	model_state: dict[str, torch.Tensor] = {}
	for key, value in state.items():
		if not key.startswith("model."):
			continue
		local_key = key.removeprefix("model.")
		if local_key == "gating_network.C1":
			local_key = "gating_network.mlp_layer1.weight"
		elif local_key == "gating_network.C2":
			local_key = "gating_network.mlp_layer2.weight"
		model_state[local_key] = value
	missing, unexpected = model.load_state_dict(model_state, strict=False)
	allowed_missing = {
		"gating_network.mlp_layer1.bias",
		"gating_network.mlp_layer2.bias",
	}
	if set(missing) - allowed_missing:
		raise RuntimeError(f"Unexpected missing model keys in {path}: {missing}")
	if unexpected:
		raise RuntimeError(f"Unexpected model keys in {path}: {unexpected}")
	return DynaMixShell(
		model,
		log_proc_std=state["log_proc_std"],
		log_obs_std=state["log_obs_std"],
		trainable=bool(trainable_shell),
	)


__all__ = ["DynaMixShell", "compute_mixture_jacobian", "load_dynamix_shell_checkpoint"]