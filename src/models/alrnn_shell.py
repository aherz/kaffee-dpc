"""AL-RNN stochastic shell implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from filtering.covariance import transport_covariance

from .alrnn import relu_transition_jacobian


class ALRNNShell(torch.nn.Module):
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
		train_core: bool = True,
		covariance_transport: str = "jacobian",
	):
		super().__init__()
		self.latent_dim = int(latent_dim)
		self.gated_dim = int(gated_dim)
		self.observed_dim = int(observed_dim)

		transport = str(covariance_transport).strip().lower()
		if transport not in {"jacobian", "identity"}:
			raise ValueError("covariance_transport must be 'jacobian' or 'identity'")
		self.covariance_transport = transport

		self.A = torch.nn.Parameter(A_init.detach().clone(), requires_grad=bool(train_core))
		self.W = torch.nn.Parameter(W_init.detach().clone(), requires_grad=bool(train_core))
		self.h = torch.nn.Parameter(h_init.detach().clone(), requires_grad=bool(train_core))
		self.register_buffer("embedding", embedding_init.detach().clone()) # Fixed initial state embedding matrix
		self.log_proc_std = torch.nn.Parameter(torch.log(proc_std_init.detach().clone().clamp(min=1e-6)))
		self.log_obs_std = torch.nn.Parameter(torch.log(obs_std_init.detach().clone().clamp(min=1e-6)))
		self._sampling_proc_std_exact: torch.Tensor | None = None
		self._sampling_obs_std_exact: torch.Tensor | None = None

	@classmethod
	def from_core_checkpoint(
		cls,
		checkpoint: dict[str, Any],
		*,
		proc_std_init: torch.Tensor,
		obs_std_init: torch.Tensor,
		train_core: bool,
		covariance_transport: str,
	) -> "ALRNNShell":
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
			train_core=train_core,
			covariance_transport=covariance_transport,
		)

	@property
	def proc_std(self) -> torch.Tensor:
		return torch.exp(self.log_proc_std)

	@property
	def obs_std(self) -> torch.Tensor:
		return torch.exp(self.log_obs_std)

	def set_sampling_std_exact(
		self,
		*,
		proc_std: torch.Tensor | None = None,
		obs_std: torch.Tensor | None = None,
	) -> None:
		self._sampling_proc_std_exact = None if proc_std is None else proc_std.detach().clone()
		self._sampling_obs_std_exact = None if obs_std is None else obs_std.detach().clone()

	def _sampling_proc_std(self) -> torch.Tensor:
		if self._sampling_proc_std_exact is not None:
			return self._sampling_proc_std_exact.to(device=self.A.device, dtype=self.A.dtype)
		return self.proc_std.to(device=self.A.device, dtype=self.A.dtype)

	def _sampling_obs_std(self) -> torch.Tensor:
		if self._sampling_obs_std_exact is not None:
			return self._sampling_obs_std_exact.to(device=self.A.device, dtype=self.A.dtype)
		return self.obs_std.to(device=self.A.device, dtype=self.A.dtype)

	@property
	def Gamma(self) -> torch.Tensor:
		return torch.diag(self.proc_std ** 2)

	@property
	def Lambda(self) -> torch.Tensor:
		return torch.diag(self.obs_std ** 2)

	def initialize_from_observation(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		z = (x0 @ self.embedding).detach()
		z[:, : self.observed_dim] = x0
		Sigma0 = self.Gamma.unsqueeze(0).expand(x0.shape[0], self.latent_dim, self.latent_dim).clone()
		return z, Sigma0

	def gated_state(self, z: torch.Tensor) -> torch.Tensor:
		gated = z.clone()
		if self.gated_dim > 0:
			gated[:, -self.gated_dim :] = torch.relu(gated[:, -self.gated_dim :])
		return gated

	def transition_mean(self, z: torch.Tensor) -> torch.Tensor:
		"""Return the AL-RNN transition."""
		return self.A.unsqueeze(0) * z + self.gated_state(z) @ self.W.t() + self.h.unsqueeze(0)

	def relu_derivative(self, z: torch.Tensor) -> torch.Tensor:
		"""Elementwise ReLU derivative: 1 on linear coordinates, 1_{z>0} on gated coordinates)."""
		dphi = torch.ones_like(z)
		if int(self.gated_dim) > 0:
			dphi[:, -self.gated_dim :] = (z[:, -self.gated_dim :] > 0.0).to(dtype=z.dtype)
		return dphi

	def transition_jacobian(self, z: torch.Tensor) -> torch.Tensor:
		return relu_transition_jacobian(self.A, self.W, z, gated_dim=self.gated_dim)

	def transition_and_jacobian(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return AL-RNN transition and ReLU-Jacobian."""
		z_pred = self.transition_mean(z)
		if int(self.gated_dim) <= 0:
			jacobian = torch.diag(self.A).unsqueeze(0).expand(z.shape[0], self.latent_dim, self.latent_dim)
			return z_pred, jacobian

		dphi = self.relu_derivative(z)
		diagonal = torch.diag_embed(self.A.unsqueeze(0).expand(z.shape[0], -1))
		jacobian = diagonal + self.W.unsqueeze(0) * dphi.unsqueeze(-2)
		return z_pred, jacobian

	def transition_and_covariance(
		self,
		z: torch.Tensor,
		Sigma: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return (J Sigma J^T, f(z)) without forming J explicitly."""
		if z.ndim != 2:
			raise ValueError(f"z must have shape (batch, M), got {tuple(z.shape)}")
		if Sigma.ndim != 3:
			raise ValueError(f"Sigma must have shape (batch, M, M), got {tuple(Sigma.shape)}")
		if Sigma.shape[0] != z.shape[0] or Sigma.shape[1:] != (self.latent_dim, self.latent_dim):
			raise ValueError(
				"Sigma dimensions must match z and latent_dim: "
				f"z={tuple(z.shape)}, Sigma={tuple(Sigma.shape)}, latent_dim={self.latent_dim}"
			)

		z_pred = self.transition_mean(z)
		A_rows = self.A.view(1, -1, 1)
		A_columns = self.A.view(1, 1, -1)
		if int(self.gated_dim) <= 0:
			return A_rows * Sigma * A_columns, z_pred

		dphi = self.relu_derivative(z)
		left_product = A_rows * Sigma + torch.matmul(
			self.W,
			dphi.unsqueeze(-1) * Sigma,
		)
		transported = left_product * A_columns + torch.matmul(
			left_product * dphi.unsqueeze(-2),
			self.W.t(),
		)
		return transported, z_pred

	def predict_covariance(self, Sigma: torch.Tensor, J_t: torch.Tensor | None) -> torch.Tensor:
		return transport_covariance(
			Sigma,
			self.Gamma.unsqueeze(0),
			jacobian=J_t,
			mode=self.covariance_transport,
		)

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
		"""Open-loop stochastic rollout (training-time D_stsp_stoch selection and DSR metrics)."""
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
		proc_std = self._sampling_proc_std().reshape(self.latent_dim)
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
				z_state = z_state + proc_std * torch.randn(
					(m_dim,),
					dtype=dtype,
					device=device,
					generator=generator,
				)

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
		"""Lyapunov spectrum of the shell-on map via online QR and in double precision.

		Each step applies the ReLU-Jacobian to the current tangent basis and
		re-orthonormalizes immediately. Additive process noise (the random dynamical system case) changes the trajectory on which the Jacobian is evaluated.
		"""
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
		proc_std = self._sampling_proc_std().reshape(self.latent_dim)
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
				z_state = z_state + proc_std * torch.randn(
					(m_dim,),
					dtype=dtype,
					device=device,
					generator=generator,
				)

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
		proc_std = self._sampling_proc_std().view(1, -1)
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
				z_state = z_state + proc_std * torch.randn(z_state.shape, dtype=self.A.dtype, device=self.A.device, generator=generator)
			x_state = z_state[:, : self.observed_dim]
			if bool(sample_obs_noise):
				x_state = x_state + obs_std * torch.randn(x_state.shape, dtype=self.A.dtype, device=self.A.device, generator=generator)
			observations.append(x_state)
		return torch.stack(observations, dim=1).detach().cpu().numpy().astype(np.float64)

__all__ = ["ALRNNShell"]

