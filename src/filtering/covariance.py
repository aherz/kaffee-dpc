"""Covariance (propagation) helpers. Holds utilities shared by filtered and
open-loop probabilistic objectives: Jacobian propagation, blind uncertainty controls, covariance symmetrization, and Joseph-form Kalman updates."""

from __future__ import annotations

from typing import Literal

import torch

CovarianceTransportMode = Literal["jacobian", "identity"]


def symmetrize_covariance(Sigma: torch.Tensor) -> torch.Tensor:
	"""Return the symmetric part of a covariance matrix or covariance batch."""
	return 0.5 * (Sigma + Sigma.transpose(-1, -2))


def add_covariance_jitter(Sigma: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
	"""Add isotropic diagonal jitter to Sigma (latent predictive covariance) and symmetrize the result."""

	stabilized = symmetrize_covariance(Sigma)
	if float(jitter) <= 0.0:
		return stabilized
	dim = stabilized.shape[-1]
	eye = torch.eye(dim, device=stabilized.device, dtype=stabilized.dtype)
	if stabilized.ndim > 2:
		eye = eye.expand(*stabilized.shape[:-2], dim, dim)
	return symmetrize_covariance(stabilized + float(jitter) * eye)


def jacobian_factor_predict(
	factor: torch.Tensor,
	jacobian: torch.Tensor,
	process_std: torch.Tensor,
) -> torch.Tensor:
	"""Return the predicted covariance square-root factor with S_pred S_pred^T = J (S S^T) J^T + diag(process_std^2).

	S is the covariance square-root factor. The returned factor is obtained from a truncated QR decomposition of [J S, diag(process_std)].
	"""

	work = torch.float64
	transformed = torch.matmul(jacobian.to(work), factor.to(work))
	noise = torch.diag(process_std.reshape(-1).to(device=transformed.device, dtype=work))
	if transformed.ndim == 3:
		noise = noise.expand(transformed.shape[0], -1, -1)
	wide = torch.cat((transformed, noise), dim=-1)
	R = torch.linalg.qr(wide.transpose(-1, -2), mode="reduced").R
	return R.transpose(-1, -2)


def transport_covariance(
	Sigma: torch.Tensor,
	Gamma: torch.Tensor,
	*,
	jacobian: torch.Tensor | None = None,
	mode: CovarianceTransportMode = "jacobian",
) -> torch.Tensor:
	"""Propagate Sigma (latent predictive covariance) under Jacobian or identity propagation and add Gamma (process noise covariance)."""

	normalized_mode = str(mode).strip().lower()
	if normalized_mode not in {"jacobian", "identity"}:
		raise ValueError("mode must be 'jacobian' or 'identity'")
	if normalized_mode == "jacobian":
		if jacobian is None:
			raise ValueError("Jacobian propagation requires a Jacobian matrix")
		transported = torch.matmul(jacobian, torch.matmul(Sigma, jacobian.transpose(-1, -2)))
	elif normalized_mode == "identity":
		transported = Sigma
	return symmetrize_covariance(transported + Gamma)


def joseph_update(
	Sigma_pred: torch.Tensor,
	K_t: torch.Tensor,
	B: torch.Tensor,
	Lambda: torch.Tensor,
) -> torch.Tensor:
	"""Apply the Joseph form covariance update. The Joseph form
		(I - K_t B) Sigma_pred (I - K_t B)^T + K_t Lambda K_t^T
	preserves symmetry and is numerically more stable/robust than the standard covariance update.
	"""

	state_dim = Sigma_pred.shape[-1]
	identity = torch.eye(state_dim, device=Sigma_pred.device, dtype=Sigma_pred.dtype)
	if Sigma_pred.ndim > 2:
		identity = identity.expand(*Sigma_pred.shape[:-2], state_dim, state_dim)
	ikh = identity - torch.matmul(K_t, B)
	updated = torch.matmul(ikh, torch.matmul(Sigma_pred, ikh.transpose(-1, -2)))
	updated = updated + torch.matmul(
		K_t,
		torch.matmul(Lambda, K_t.transpose(-1, -2)),
	)
	return symmetrize_covariance(updated)


def joseph_update_identity_observation(
	Sigma_pred: torch.Tensor,
	K_t: torch.Tensor,
	Lambda: torch.Tensor,
) -> torch.Tensor:
	"""Joseph update specialized to identity TF observation model B = [I_N | 0_(N x M)] without dense factors.

	All non-covariance matrix products have observation width N, reducing
	the correction from generic O(M^3) work to ``O(M^2 N + M N^2)``.
	"""
	if Sigma_pred.ndim < 2 or K_t.ndim < 2:
		raise ValueError("Sigma_pred and K_t must be matrices or batches of matrices")
	state_dim = int(Sigma_pred.shape[-1])
	observed_dim = int(K_t.shape[-1])
	if Sigma_pred.shape[-2] != state_dim:
		raise ValueError("Sigma_pred must be square")
	if K_t.shape[-2] != state_dim:
		raise ValueError("K_t state dimension must match Sigma_pred")
	if observed_dim > state_dim:
		raise ValueError("K_t observation dimension cannot exceed the state dimension")
	if Lambda.shape[-2:] != (observed_dim, observed_dim):
		raise ValueError("Lambda dimensions must match K_t observation dimension")

	observed_rows = Sigma_pred[..., :observed_dim, :]
	left = Sigma_pred - torch.matmul(K_t, observed_rows)
	left_observed_columns = left[..., :, :observed_dim]
	updated = left - torch.matmul(left_observed_columns, K_t.transpose(-1, -2))
	updated = updated + torch.matmul(
		K_t,
		torch.matmul(Lambda, K_t.transpose(-1, -2)),
	)
	return symmetrize_covariance(updated)


__all__ = [
	"CovarianceTransportMode",
	"add_covariance_jitter",
	"jacobian_factor_predict",
	"joseph_update",
	"joseph_update_identity_observation",
	"symmetrize_covariance",
	"transport_covariance",
]
