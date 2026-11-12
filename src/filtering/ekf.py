"""Extended Kalman filter (EKF) implementation and related utilities."""

from __future__ import annotations

import math

import torch

from .covariance import (
  add_covariance_jitter,
  joseph_update,
  joseph_update_identity_observation,
  symmetrize_covariance,
  transport_covariance,
)


def build_identity_observation_matrix(
  observed_dim: int,
  latent_dim: int,
  *,
  device: torch.device,
  dtype: torch.dtype,
) -> torch.Tensor:
  """Build the fixed observation map B = [I | 0] (identity teacher-forcing observation model) used by the AL-RNN shell."""
  B = torch.zeros(observed_dim, latent_dim, device=device, dtype=dtype)
  B[:, :observed_dim] = torch.eye(observed_dim, device=device, dtype=dtype)
  return B


def kalman_predict(
  z_mean: torch.Tensor,
  Sigma: torch.Tensor,
  *,
  Gamma: torch.Tensor,
  z_mean_pred: torch.Tensor | None = None,
  F_t: torch.Tensor | None = None,
  h_t: torch.Tensor | None = None,
  J_t: torch.Tensor | None = None,
  transported_covariance: torch.Tensor | None = None,
  covariance_transport: str = "jacobian",
) -> tuple[torch.Tensor, torch.Tensor]:
  """EKF prediction step.

  ``transported_covariance`` can provide J_t @ Sigma @ J_t.T directly,
   to avoid forming J_t.
  """

  if z_mean_pred is None:
    if F_t is None:
      raise ValueError("kalman_predict needs either z_mean_pred or F_t")
    z_mean_pred = torch.matmul(F_t, z_mean.unsqueeze(-1)).squeeze(-1)
    if h_t is not None:
      z_mean_pred = z_mean_pred + h_t

  effective_jacobian = J_t if J_t is not None else F_t
  if transported_covariance is not None:
    if J_t is not None:
      raise ValueError("kalman_predict accepts either J_t or transported_covariance, not both")
    if str(covariance_transport).strip().lower() != "jacobian":
      raise ValueError("transported_covariance is only valid for Jacobian covariance propagation")
    Sigma_pred = symmetrize_covariance(transported_covariance + Gamma)
  else:
    Sigma_pred = transport_covariance(
      Sigma,
      Gamma,
      jacobian=effective_jacobian,
      mode=covariance_transport,
    )
  return z_mean_pred, Sigma_pred


def innovation_statistics(
  z_mean_pred: torch.Tensor,
  Sigma_pred: torch.Tensor,
  x_t: torch.Tensor,
  *,
  B: torch.Tensor,
  Lambda: torch.Tensor,
  Omega_jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute innovations x_t - B z_{t|t-1} and their covariance Omega_t."""

  residual = x_t - torch.matmul(B, z_mean_pred.unsqueeze(-1)).squeeze(-1)
  Sigma_xy = torch.matmul(Sigma_pred, B.transpose(-1, -2))
  Omega_t = torch.matmul(B, Sigma_xy) + Lambda
  Omega_t = add_covariance_jitter(Omega_t, Omega_jitter)
  return residual, Omega_t


def innovation_statistics_identity_observation(
  z_mean_pred: torch.Tensor,
  Sigma_pred: torch.Tensor,
  x_t: torch.Tensor,
  *,
  Lambda: torch.Tensor,
  Omega_jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Innovation statistics specialized to the identity TF observation model B = [I_N | 0_(N x M)].

  Returns the residual, innovation covariance, and ``Sigma_pred @ B.T`` so the
  cross-covariance can be reused by the Kalman-gain calculation.
  """
  observed_dim = int(x_t.shape[-1])
  if observed_dim > int(z_mean_pred.shape[-1]):
    raise ValueError("observation dimension cannot exceed latent dimension")
  if Lambda.shape[-2:] != (observed_dim, observed_dim):
    raise ValueError("Lambda dimensions must match the observation dimension")

  residual = x_t - z_mean_pred[..., :observed_dim]
  Sigma_zx = Sigma_pred[..., :, :observed_dim]
  Omega_t = Sigma_pred[..., :observed_dim, :observed_dim] + Lambda
  Omega_t = add_covariance_jitter(Omega_t, Omega_jitter)
  return residual, Omega_t, Sigma_zx


def innovation_log_likelihood(x_residual: torch.Tensor, Omega_t: torch.Tensor) -> torch.Tensor:
  """Gaussian innovation log-likelihood."""

  obs_dim = x_residual.shape[-1]
  sign, logabsdet = torch.linalg.slogdet(Omega_t)
  if torch.any(sign <= 0):
    raise RuntimeError("Innovation log-likelihood requires positive-definite Omega_t")
  solved = torch.linalg.solve(Omega_t, x_residual.unsqueeze(-1)).squeeze(-1)
  quadratic = (x_residual * solved).sum(dim=-1)
  return -0.5 * (obs_dim * math.log(2.0 * math.pi) + logabsdet + quadratic)


def kalman_update(
  z_mean_pred: torch.Tensor,
  Sigma_pred: torch.Tensor,
  x_t: torch.Tensor,
  *,
  B: torch.Tensor,
  Lambda: torch.Tensor,
  Omega_jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """EKF update step; returns filtered mean, covariance, and innovationlog-likelihood."""

  residual, Omega_t = innovation_statistics(
    z_mean_pred,
    Sigma_pred,
    x_t,
    B=B,
    Lambda=Lambda,
    Omega_jitter=Omega_jitter,
  )
  Sigma_zx = torch.matmul(Sigma_pred, B.transpose(-1, -2)) # cross-covariance
  gain_t = torch.linalg.solve(Omega_t, Sigma_zx.transpose(-1, -2))
  K_t = gain_t.transpose(-1, -2)
  z_mean_filt = z_mean_pred + torch.matmul(K_t, residual.unsqueeze(-1)).squeeze(-1)
  Sigma_filt = joseph_update(
    Sigma_pred,
    K_t,
    B,
    Lambda,
  )
  log_likelihood = innovation_log_likelihood(residual, Omega_t)
  return z_mean_filt, Sigma_filt, log_likelihood


def kalman_update_identity_observation(
  z_mean_pred: torch.Tensor,
  Sigma_pred: torch.Tensor,
  x_t: torch.Tensor,
  *,
  Lambda: torch.Tensor,
  Omega_jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Exact EKF update specialized to the fixed map ``B = [I | 0]``."""
  residual, Omega_t, Sigma_zx = innovation_statistics_identity_observation(
    z_mean_pred,
    Sigma_pred,
    x_t,
    Lambda=Lambda,
    Omega_jitter=Omega_jitter,
  )
  gain_t = torch.linalg.solve(Omega_t, Sigma_zx.transpose(-1, -2))
  K_t = gain_t.transpose(-1, -2)
  z_mean_filt = z_mean_pred + torch.matmul(K_t, residual.unsqueeze(-1)).squeeze(-1)
  Sigma_filt = joseph_update_identity_observation(
    Sigma_pred,
    K_t,
    Lambda,
  )
  log_likelihood = innovation_log_likelihood(residual, Omega_t)
  return z_mean_filt, Sigma_filt, log_likelihood


__all__ = [
  "build_identity_observation_matrix",
  "innovation_log_likelihood",
  "innovation_statistics",
  "innovation_statistics_identity_observation",
  "kalman_predict",
  "kalman_update",
  "kalman_update_identity_observation",
]