"""Clipped shallow PLRNN (shPLRNN) implementation. The code follows the (Julia) implementation of Hess et al. (see https://arxiv.org/abs/2306.04406)."""

from __future__ import annotations

import math

import torch


def _uniform(shape: tuple[int, ...], generator: torch.Generator | None) -> torch.Tensor:
	radius = 1.0 / math.sqrt(float(shape[-1]))
	return (2.0 * torch.rand(shape, generator=generator) - 1.0) * radius


def _normalized_positive_definite(dim: int, generator: torch.Generator | None) -> torch.Tensor:
	raw = torch.randn((dim, dim), dtype=torch.float32, generator=generator)
	value = (raw.T @ raw) / float(dim) + torch.eye(dim, dtype=torch.float32)
	return value / torch.linalg.eigvalsh(value)[-1]


class ClippedShallowPLRNN(torch.nn.Module):

	def __init__(self, latent_dim: int, hidden_dim: int, *, generator: torch.Generator | None = None) -> None:
		super().__init__()
		self.latent_dim = int(latent_dim)
		self.hidden_dim = int(hidden_dim)
		# Draw order matches the pinned Julia implementation: A, h2, W1, W2.
		aw = _normalized_positive_definite(self.latent_dim, generator)
		h2 = _uniform((self.hidden_dim,), generator)
		w1 = _uniform((self.latent_dim, self.hidden_dim), generator)
		w2 = _uniform((self.hidden_dim, self.latent_dim), generator)
		self.A = torch.nn.Parameter(torch.diagonal(aw).clone())
		self.W1 = torch.nn.Parameter(w1)
		self.W2 = torch.nn.Parameter(w2)
		self.h1 = torch.nn.Parameter(torch.zeros(self.latent_dim))
		self.h2 = torch.nn.Parameter(h2)

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		"""clipped shallow PLRNN transition."""
		linear = torch.nn.functional.linear(z, self.W2)
		hidden = torch.relu(linear + self.h2) - torch.relu(linear)
		return self.A * z + torch.nn.functional.linear(hidden, self.W1) + self.h1

	def transition_jacobian(self, z: torch.Tensor) -> torch.Tensor:
		"""Jacobian of the clipped shallow PLRNN transition at latent state z."""
		if z.ndim != 1:
			raise ValueError("transition_jacobian expects one latent vector")
		linear = torch.nn.functional.linear(z, self.W2)
		gate = (linear > -self.h2).to(linear.dtype) - (linear > 0).to(linear.dtype)
		return torch.diag(self.A) + (self.W1 * gate.unsqueeze(0)) @ self.W2


class LinearObservation(torch.nn.Module):

	def __init__(self, observed_dim: int, latent_dim: int, *, generator: torch.Generator | None = None) -> None:
		super().__init__()
		self.observed_dim = int(observed_dim)
		self.latent_dim = int(latent_dim)
		self.B = torch.nn.Parameter(torch.empty(self.observed_dim, self.latent_dim))
		torch.nn.init.kaiming_normal_(self.B, generator=generator)

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		return torch.nn.functional.linear(z, self.B)

	def inverse(self, x: torch.Tensor) -> torch.Tensor:
		pinv = torch.linalg.pinv(self.B)
		return pinv @ x if x.ndim == 1 else x @ pinv.T
