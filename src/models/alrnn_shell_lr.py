"""AL-RNN stochastic shell with rank-r square-root covariances for RRKF.

See Schmidt et al., The Rank-Reduced Kalman Filter, NeurIPS 2023.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .alrnn import relu_transition_jacobian


def _distributed_low_rank_diagonal_factor(
	variance: torch.Tensor,
	r: int,
	*,
	generator: torch.Generator | None = None,
) -> torch.Tensor:
	"""Build an M x r factor with E[diag(SS^T)] ≈ variance on a random subspace."""

	if variance.ndim != 1:
		raise ValueError("variance must have shape (M,)")
	n = int(variance.shape[0])
	rank = min(int(r), n)
	gaussian = torch.randn(
		n,
		rank,
		dtype=variance.dtype,
		device=variance.device,
		generator=generator,
	)
	q, _ = torch.linalg.qr(gaussian, mode="reduced")
	target_std = torch.sqrt(variance.clamp_min(0.0)).unsqueeze(1)
	return q * target_std * ((n / rank) ** 0.5)


class ALRNNShellLR(torch.nn.Module):
	def __init__(
		self,
		*,
		latent_dim: int,
		gated_dim: int,
		observed_dim: int,
		A_init: torch.Tensor,
		W_init: torch.Tensor,
		h_init: torch.Tensor,
		embedding_init: torch.Tensor,
		proc_std_init: torch.Tensor,
		obs_std_init: torch.Tensor,
		filter_rank: int,
		train_core: bool = True,
		covariance_transport: str = "jacobian",
		proc_noise_sqrt_init: torch.Tensor | None = None,
	):
		super().__init__()
		self.latent_dim = int(latent_dim)
		self.gated_dim = int(gated_dim)
		self.observed_dim = int(observed_dim)
		self.filter_rank = int(filter_rank)
		if self.filter_rank < 1:
			raise ValueError("filter_rank must be >= 1")
		if self.filter_rank > self.latent_dim:
			raise ValueError("filter_rank cannot exceed latent_dim")

		transport = str(covariance_transport).strip().lower()
		if transport not in {"jacobian", "identity"}:
			raise ValueError("covariance_transport must be 'jacobian' or 'identity'")
		self.covariance_transport = transport

		self.A = torch.nn.Parameter(A_init.detach().clone(), requires_grad=bool(train_core))
		self.W = torch.nn.Parameter(W_init.detach().clone(), requires_grad=bool(train_core))
		self.h = torch.nn.Parameter(h_init.detach().clone(), requires_grad=bool(train_core))
		self.register_buffer("embedding", embedding_init.detach().clone())
		self.log_obs_std = torch.nn.Parameter(torch.log(obs_std_init.detach().clone().clamp(min=1e-6)))

		proc_std = proc_std_init.detach().clone().to(dtype=A_init.dtype).reshape(-1).clamp(min=1e-6)
		if int(proc_std.numel()) != self.latent_dim:
			raise ValueError(f"proc_std_init must have length {self.latent_dim}, got {proc_std.numel()}")
		if proc_noise_sqrt_init is None:
			factor0 = _distributed_low_rank_diagonal_factor(proc_std * proc_std, self.filter_rank)
		else:
			factor0 = proc_noise_sqrt_init.detach().clone()
			if tuple(factor0.shape) != (self.latent_dim, self.filter_rank):
				raise ValueError(
					f"proc_noise_sqrt_init must have shape {(self.latent_dim, self.filter_rank)}, "
					f"got {tuple(factor0.shape)}"
				)
		self.proc_noise_sqrt = torch.nn.Parameter(factor0)
		self._sampling_proc_std_exact: torch.Tensor | None = None
		self._sampling_obs_std_exact: torch.Tensor | None = None

	@classmethod
	def from_core_checkpoint(
		cls,
		checkpoint: dict[str, Any],
		*,
		proc_std_init: torch.Tensor,
		obs_std_init: torch.Tensor,
		filter_rank: int,
		train_core: bool,
		covariance_transport: str,
	) -> "ALRNNShellLR":
		state_dict = checkpoint["state_dict"]
		return cls(
			latent_dim=int(checkpoint["M"]),
			gated_dim=int(checkpoint["P"]),
			observed_dim=int(checkpoint["N"]),
			A_init=state_dict["A"],
			W_init=state_dict["W"],
			h_init=state_dict["h"],
			embedding_init=state_dict["B"],
			proc_std_init=proc_std_init,
			obs_std_init=obs_std_init,
			filter_rank=int(filter_rank),
			train_core=train_core,
			covariance_transport=covariance_transport,
		)

	@property
	def obs_std(self) -> torch.Tensor:
		return torch.exp(self.log_obs_std)

	@property
	def proc_std(self) -> torch.Tensor:
		"""Marginal process-noise stds sqrt(diag(SS^T))."""

		return torch.sqrt((self.proc_noise_sqrt * self.proc_noise_sqrt).sum(dim=-1).clamp_min(1e-12))

	def set_sampling_std_exact(
		self,
		*,
		proc_std: torch.Tensor | None = None,
		obs_std: torch.Tensor | None = None,
	) -> None:
		self._sampling_proc_std_exact = None
		self._sampling_obs_std_exact = None if obs_std is None else obs_std.detach().clone()

	def _sampling_proc_std(self) -> torch.Tensor:
		return self.proc_std.to(device=self.A.device, dtype=self.A.dtype)

	def _sampling_obs_std(self) -> torch.Tensor:
		if self._sampling_obs_std_exact is not None:
			return self._sampling_obs_std_exact.to(device=self.A.device, dtype=self.A.dtype)
		return self.obs_std.to(device=self.A.device, dtype=self.A.dtype)

	def _sample_process_noise(
		self,
		batch_shape: tuple[int, ...] | torch.Size,
		*,
		generator: torch.Generator | None = None,
	) -> torch.Tensor:
		"""Draw S @ ε with ε ~ N(0, I_r)."""

		eps = torch.randn(
			*batch_shape,
			self.filter_rank,
			dtype=self.A.dtype,
			device=self.A.device,
			generator=generator,
		)
		return eps @ self.proc_noise_sqrt.transpose(0, 1)

	@property
	def Gamma(self) -> torch.Tensor:
		return self.proc_noise_sqrt @ self.proc_noise_sqrt.transpose(0, 1)

	@property
	def Lambda(self) -> torch.Tensor:
		return torch.diag(self.obs_std ** 2)

	@property
	def Gamma_sqrt(self) -> torch.Tensor:
		return self.proc_noise_sqrt

	@property
	def Lambda_sqrt(self) -> torch.Tensor:
		return torch.diag(self.obs_std)

	def initialize_from_observation(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		z = (x0 @ self.embedding).detach()
		z[:, : self.observed_dim] = x0
		Sigma_sqrt0 = self.Gamma_sqrt.unsqueeze(0).expand(x0.shape[0], self.latent_dim, self.filter_rank).clone()
		return z, Sigma_sqrt0

	def gated_state(self, z: torch.Tensor) -> torch.Tensor:
		gated = z.clone()
		if self.gated_dim > 0:
			gated[:, -self.gated_dim :] = torch.relu(gated[:, -self.gated_dim :])
		return gated

	def transition_mean(self, z: torch.Tensor) -> torch.Tensor:
		"""Return the AL-RNN transition."""
		return self.A.unsqueeze(0) * z + self.gated_state(z) @ self.W.t() + self.h.unsqueeze(0)

	def relu_derivative(self, z: torch.Tensor) -> torch.Tensor:
		"""Elementwise ReLU derivative: 1 on linear coordinates, 1_{z>0} on gated coordinates."""
		dphi = torch.ones_like(z)
		if int(self.gated_dim) > 0:
			dphi[:, -self.gated_dim :] = (z[:, -self.gated_dim :] > 0.0).to(dtype=z.dtype)
		return dphi

	def transition_jacobian(self, z: torch.Tensor) -> torch.Tensor:
		return relu_transition_jacobian(self.A, self.W, z, gated_dim=self.gated_dim)

	def transition_and_jacobian(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return AL-RNN transition and Jacobian."""
		z_pred = self.transition_mean(z)
		if int(self.gated_dim) <= 0:
			jacobian = torch.diag(self.A).unsqueeze(0).expand(z.shape[0], self.latent_dim, self.latent_dim)
			return z_pred, jacobian

		dphi = self.relu_derivative(z)
		diagonal = torch.diag_embed(self.A.unsqueeze(0).expand(z.shape[0], -1))
		jacobian = diagonal + self.W.unsqueeze(0) * dphi.unsqueeze(-2)
		return z_pred, jacobian

	def transition_and_jacobian_factor(
		self,
		z: torch.Tensor,
		factor: torch.Tensor,
		*,
		transport_dtype: torch.dtype | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return (J(z) @ factor, f(z)) without materializing J.

		Uses "J = diag(A) + W @ diag(dphi)" with dphi = 1_{z>0} on gated coordinates, 1 on linear coordinates (piecewise ReLU derivative).
		"""
		if z.ndim != 2:
			raise ValueError(f"z must have shape (batch, M), got {tuple(z.shape)}")
		if factor.ndim != 3:
			raise ValueError(f"factor must have shape (batch, M, r), got {tuple(factor.shape)}")
		if factor.shape[:2] != z.shape:
			raise ValueError(
				"factor leading dimensions must match z: "
				f"z={tuple(z.shape)}, factor={tuple(factor.shape)}"
			)

		z_pred = self.transition_mean(z)
		product_dtype = factor.dtype if transport_dtype is None else transport_dtype
		factor_product = factor.to(dtype=product_dtype)
		A_product = self.A.to(dtype=product_dtype)
		if int(self.gated_dim) <= 0:
			return A_product.view(1, -1, 1) * factor_product, z_pred

		dphi = self.relu_derivative(z)
		dphi_product = dphi.to(dtype=product_dtype)
		W_product = self.W.to(dtype=product_dtype)
		weighted_factor = dphi_product.unsqueeze(-1) * factor_product
		transported = A_product.view(1, -1, 1) * factor_product + torch.matmul(
			W_product,
			weighted_factor,
		)
		return transported, z_pred

	def _apply_process_noise(
		self,
		z_state: torch.Tensor,
		*,
		generator: torch.Generator,
	) -> torch.Tensor:
		batch_shape: tuple[int, ...] = () if z_state.ndim == 1 else (int(z_state.shape[0]),)
		return z_state + self._sample_process_noise(batch_shape, generator=generator)

	@torch.no_grad()
	def rollout_observation_sequence(
		self,
		x0: np.ndarray | torch.Tensor,
		*,
		steps: int,
		seed: int,
		sample_process_noise: bool,
		sample_obs_noise: bool,
	) -> np.ndarray:
		"""Open-loop stochastic rollout (for training-time D_stsp_stoch selection and DSR metrics)."""
		device = self.A.device
		dtype = self.A.dtype
		x0_t = torch.as_tensor(x0, dtype=dtype, device=device).reshape(-1)
		if int(x0_t.numel()) == int(self.latent_dim):
			z_state = x0_t.clone()
		elif int(x0_t.numel()) == int(self.observed_dim):
			z_state = (x0_t.view(1, -1) @ self.embedding).reshape(-1)
			z_state[: self.observed_dim] = x0_t
		else:
			raise ValueError(
				f"x0 has dim {int(x0_t.numel())}, expected observed_dim={self.observed_dim} or latent_dim={self.latent_dim}"
			)

		A = self.A.reshape(-1)
		W = self.W
		h = self.h.reshape(-1)
		obs_std = self._sampling_obs_std().reshape(self.observed_dim)
		generator = torch.Generator(device=device)
		generator.manual_seed(int(seed))
		k_dim = int(self.gated_dim)
		m_dim = int(self.latent_dim)
		n_dim = int(self.observed_dim)

		observations_t: list[torch.Tensor] = []
		for _ in range(int(steps)):
			if k_dim > 0:
				dvec = torch.ones((m_dim,), device=device, dtype=dtype)
				dvec[-k_dim:] = (z_state[-k_dim:] > 0.0).to(dtype)
			else:
				dvec = torch.ones((m_dim,), device=device, dtype=dtype)

			z_state = z_state * A + (W @ (dvec * z_state)) + h
			if bool(sample_process_noise):
				z_state = self._apply_process_noise(z_state, generator=generator)

			x_state = z_state[:n_dim]
			if bool(sample_obs_noise):
				x_state = x_state + obs_std * torch.randn(
					(n_dim,),
					dtype=dtype,
					device=device,
					generator=generator,
				)
			observations_t.append(x_state)

		return torch.stack(observations_t, dim=0).detach().cpu().numpy().astype(np.float64)

	@torch.no_grad()
	def lyapunov_spectrum(
		self,
		x0: np.ndarray | torch.Tensor,
		*,
		steps: int,
		burn: int,
		seed: int,
		sample_process_noise: bool,
	) -> np.ndarray:
		"""Lyapunov spectrum of the shell-on map via online QR and in double precision."""
		steps = int(steps)
		burn = int(burn)
		if burn < 0 or steps <= burn:
			raise ValueError(f"Expected 0 <= burn < steps, got burn={burn}, steps={steps}")

		device = self.A.device
		dtype = self.A.dtype
		x0_t = torch.as_tensor(x0, dtype=dtype, device=device).reshape(-1)
		if int(x0_t.numel()) == int(self.latent_dim):
			z_state = x0_t.clone()
		elif int(x0_t.numel()) == int(self.observed_dim):
			z_state = (x0_t.view(1, -1) @ self.embedding).reshape(-1)
			z_state[: self.observed_dim] = x0_t
		else:
			raise ValueError(
				f"x0 has dim {int(x0_t.numel())}, expected observed_dim={self.observed_dim} or latent_dim={self.latent_dim}"
			)

		A = self.A.reshape(-1)
		W = self.W
		h = self.h.reshape(-1)
		generator = torch.Generator(device=device)
		generator.manual_seed(int(seed))
		k_dim = int(self.gated_dim)
		m_dim = int(self.latent_dim)
		A64 = A.to(torch.float64)
		W64 = W.to(torch.float64)
		q = torch.eye(m_dim, dtype=torch.float64, device=device)
		logs = torch.zeros(m_dim, dtype=torch.float64, device=device)

		for step in range(steps):
			dvec = torch.ones((m_dim,), device=device, dtype=dtype)
			if k_dim > 0:
				dvec[-k_dim:] = (z_state[-k_dim:] > 0.0).to(dtype)
			tangent = A64.view(m_dim, 1) * q + W64 @ (dvec.to(torch.float64).view(m_dim, 1) * q)
			q, r = torch.linalg.qr(tangent, mode="reduced")
			if step >= burn:
				logs += torch.log(torch.clamp(torch.abs(torch.diagonal(r)), min=1e-300))

			z_state = z_state * A + (W @ (dvec * z_state)) + h
			if bool(sample_process_noise):
				z_state = self._apply_process_noise(z_state, generator=generator)

		return (logs / float(steps - burn)).detach().cpu().numpy()

	@torch.no_grad()
	def rollout_observation_ensemble(
		self,
		x0: np.ndarray | torch.Tensor,
		*,
		length: int,
		n_ensemble: int,
		seed: int,
		sample_process_noise: bool = True,
		sample_obs_noise: bool = True,
	) -> np.ndarray:
		"""Ensemble of stochastic open-loop trajectories (for NLL and CRPS metrics)."""
		x0_t = torch.as_tensor(x0, dtype=self.A.dtype, device=self.A.device).reshape(1, -1)
		x0_batch = x0_t.expand(int(n_ensemble), -1).clone()
		z_state, _ = self.initialize_from_observation(x0_batch)
		obs_std = self._sampling_obs_std().view(1, -1)
		generator = torch.Generator(device=self.A.device)
		generator.manual_seed(int(seed))
		observations: list[torch.Tensor] = []
		for _ in range(int(length)):
			dvec = torch.ones_like(z_state)
			if self.gated_dim > 0:
				dvec[:, -self.gated_dim :] = (z_state[:, -self.gated_dim :] > 0.0).to(self.A.dtype)
			z_state = z_state * self.A.unsqueeze(0) + (z_state * dvec) @ self.W.t() + self.h.unsqueeze(0)
			if bool(sample_process_noise):
				z_state = self._apply_process_noise(z_state, generator=generator)
			x_state = z_state[:, : self.observed_dim]
			if bool(sample_obs_noise):
				x_state = x_state + obs_std * torch.randn(
					x_state.shape, dtype=self.A.dtype, device=self.A.device, generator=generator
				)
			observations.append(x_state)
		return torch.stack(observations, dim=1).detach().cpu().numpy().astype(np.float64)


__all__ = ["ALRNNShellLR"]
