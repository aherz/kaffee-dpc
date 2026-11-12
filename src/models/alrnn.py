"""Almost-linear RNN (AL-RNN) implementation. The code follows the AL-RNN paper's implementation (see https://arxiv.org/abs/2410.14240)."""

from __future__ import annotations

import math

import numpy as np
import torch


def relu_transition_jacobian(
	A: torch.Tensor,
	W: torch.Tensor,
	z: torch.Tensor,
	*,
	gated_dim: int,
) -> torch.Tensor:
	"""Jacobian of the AL-RNN transition at latent state z (1d)."""
	z = z.reshape(-1)
	m_dim = int(z.numel())
	dvec = torch.ones(m_dim, device=z.device, dtype=z.dtype)
	k_dim = int(gated_dim)
	if k_dim > 0:
		dvec = dvec.clone()
		dvec[-k_dim:] = (z[-k_dim:] > 0.0).to(dtype=z.dtype)
	return torch.diag(A.reshape(-1)) + W * dvec.view(1, m_dim)


class WindowedTimeSeriesDataset:
    def __init__(self, data: np.ndarray, *, sequence_length: int, batch_size: int):
        X = np.asarray(data, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D time-series array, got shape {X.shape}")
        if int(sequence_length) >= X.shape[0]:
            raise ValueError(
                f"sequence_length={sequence_length} must be smaller than the number of time steps {X.shape[0]}"
            )
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return int(self.X.shape[0] - self.sequence_length - 1)

    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(self) + 1, (self.batch_size,))
        offsets = torch.arange(self.sequence_length)
        x_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        y_idx = x_idx + 1
        return self.X[x_idx], self.X[y_idx]


class ALRNN(torch.nn.Module):
    def __init__(self, *, latent_dim: int, gated_dim: int, observed_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.gated_dim = int(gated_dim)
        self.observed_dim = int(observed_dim)
        self.A = torch.nn.Parameter(torch.diagonal(self._normalized_positive_definite(self.latent_dim), 0))
        self.W = torch.nn.Parameter(torch.randn(self.latent_dim, self.latent_dim) * 0.01)
        self.h = torch.nn.Parameter(torch.zeros(self.latent_dim))
        self.B = torch.nn.Parameter(torch.empty(self.observed_dim, self.latent_dim))
        bound = 1.0 / math.sqrt(self.observed_dim)
        torch.nn.init.uniform_(self.B, -bound, bound)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_activated = z.clone()
        if self.gated_dim > 0:
            z_activated[:, -self.gated_dim :] = torch.relu(z_activated[:, -self.gated_dim :])
        return self.A * z + z_activated @ self.W.t() + self.h

    def transition_and_jacobian(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """AL-RNN step and its analytical Jacobian."""
        if z.dim() == 1:
            z_next = self.forward(z.unsqueeze(0)).squeeze(0)
        else:
            z_next = self.forward(z)
        if z.dim() == 1:
            jacobian = relu_transition_jacobian(self.A, self.W, z, gated_dim=self.gated_dim)
        else:
            dvec = torch.ones_like(z)
            if self.gated_dim > 0:
                dvec[:, -self.gated_dim :] = (z[:, -self.gated_dim :] > 0.0).to(z.dtype)
            jacobian = self.W.unsqueeze(0) * dvec.unsqueeze(1) + torch.diag(self.A).unsqueeze(0)
        return z_next, jacobian

    @torch.no_grad()
    def lyapunov_spectrum(
        self,
        x0: np.ndarray | torch.Tensor,
        *,
        steps: int,
        burn: int,
    ) -> np.ndarray:
        """Lyapunov spectrum of the autonomous AL-RNN map, computed in double precision."""
        steps = int(steps)
        burn = int(burn)
        if burn < 0 or steps <= burn:
            raise ValueError(f"Expected 0 <= burn < steps, got burn={burn}, steps={steps}")

        device = self.A.device
        dtype = self.A.dtype
        x0_t = torch.as_tensor(x0, dtype=dtype, device=device).reshape(-1)
        if int(x0_t.numel()) != int(self.observed_dim):
            raise ValueError(
                f"x0 has dim {int(x0_t.numel())}, expected observed_dim={self.observed_dim}"
            )
        z_state = (x0_t.view(1, -1) @ self.B).reshape(-1)
        z_state[: self.observed_dim] = x0_t

        A = self.A.reshape(-1)
        W = self.W
        h = self.h.reshape(-1)
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

        return (logs / float(steps - burn)).detach().cpu().numpy()

    @staticmethod
    def _normalized_positive_definite(dim: int) -> torch.Tensor:
        R = np.random.randn(int(dim), int(dim)).astype(np.float32)
        K = (R.T @ R) / float(dim) + np.eye(int(dim), dtype=np.float32)
        eigvals = np.linalg.eigvals(K)
        lam_max = float(np.max(np.abs(eigvals)))
        return torch.tensor(K / lam_max, dtype=torch.float32)


def teacher_force(z: torch.Tensor, x: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    z = z.clone()
    observed_dim = min(int(z.shape[-1]), int(x.shape[-1]))
    z[:, :observed_dim] = float(alpha) * x[:, :observed_dim] + (1.0 - float(alpha)) * z[:, :observed_dim]
    return z


def rollout_itf(model: ALRNN, x: torch.Tensor, *, forcing_interval: int, alpha: float = 1.0) -> torch.Tensor:
    x_steps = x.permute(1, 0, 2)
    z = x_steps[0] @ model.B
    z = teacher_force(z, x_steps[0], alpha=1.0)
    trajectory: list[torch.Tensor] = []
    for step, x_t in enumerate(x_steps):
        if step > 0 and step % int(forcing_interval) == 0:
            z = teacher_force(z, x_t, alpha=alpha)
        z = model(z)
        trajectory.append(z)
    return torch.stack(trajectory, dim=1)


@torch.no_grad()
def rollout_free(model: ALRNN, x0: torch.Tensor, *, steps: int) -> torch.Tensor:
    z = x0 @ model.B
    z = teacher_force(z, x0, alpha=1.0)
    trajectory: list[torch.Tensor] = []
    for _ in range(int(steps)):
        z = model(z)
        trajectory.append(z)
    return torch.stack(trajectory, dim=1)
