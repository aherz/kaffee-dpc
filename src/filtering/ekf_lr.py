"""Rank-reduced (low-rank square-root) Kalman filter.

Python port of
https://github.com/schmidtjonathan/RRKF.jl/blob/main/src/algorithms/rankreduced_kalman.jl

Covariances are left square-root factors S with cov = S @ S.T (shape (..., M, r),
M = latent dimension).
"""

from __future__ import annotations

import math
import os

import torch

_RRKF_ENV = os.environ.get("KAFFEE_RRKF_FLOAT64", "").strip().lower()
if _RRKF_ENV in {"0", "false", "off", "no"}:
  RRKF_FLOAT64 = False
elif _RRKF_ENV in {"1", "true", "on", "yes"}:
  RRKF_FLOAT64 = True
else:
  RRKF_FLOAT64 = True


def _promote(*tensors: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
  if not RRKF_FLOAT64:
    return tensors
  return tuple(None if t is None else t.to(dtype=torch.float64) for t in tensors)


def _demote(dtype_in: torch.dtype, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
  if not RRKF_FLOAT64 or dtype_in == torch.float64:
    return tensors[0] if len(tensors) == 1 else tensors
  out = tuple(t.to(dtype=dtype_in) for t in tensors)
  return out[0] if len(out) == 1 else out


def rrkf_predict(
  z_mean: torch.Tensor,
  Sigma_sqrt: torch.Tensor,
  *,
  Gamma_sqrt: torch.Tensor,
  r: int,
  z_mean_pred: torch.Tensor | None = None,
  F_t: torch.Tensor | None = None,
  transported_sqrt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Rank-reduced predict; returns "(z_mean_pred, Sigma_sqrt_pred)"."""

  dtype_in = z_mean.dtype
  z_mean, Sigma_sqrt, Gamma_sqrt, z_mean_pred, F_t, transported_sqrt = _promote(
    z_mean, Sigma_sqrt, Gamma_sqrt, z_mean_pred, F_t, transported_sqrt
  )

  if z_mean_pred is None:
    if F_t is None:
      raise ValueError("rrkf_predict needs either z_mean_pred or F_t")
    z_mean_pred = torch.matmul(F_t, z_mean.unsqueeze(-1)).squeeze(-1)
  if transported_sqrt is not None and F_t is not None:
    raise ValueError("rrkf_predict accepts either F_t or transported_sqrt, not both")
  if transported_sqrt is None:
    if F_t is None:
      raise ValueError("rrkf_predict requires F_t or transported_sqrt for covariance propagation")
    transported_sqrt = torch.matmul(F_t, Sigma_sqrt)

  Gamma_sqrt = _maybe_expand_factor(Gamma_sqrt, transported_sqrt)
  stacked = torch.cat([transported_sqrt, Gamma_sqrt], dim=-1)
  return _demote(dtype_in, z_mean_pred, tsvd_factor(stacked, r))


def rrkf_update(
  z_mean_pred: torch.Tensor,
  Sigma_sqrt_pred: torch.Tensor,
  x_t: torch.Tensor,
  *,
  B: torch.Tensor,
  Lambda_sqrt: torch.Tensor,
  r: int,
  return_log_likelihood: bool = False,
) -> tuple[torch.Tensor, ...]:
  """Rank-reduced update; returns filtered mean and sqrt-covariance.

  If "return_log_likelihood=True", also returns the innovation log-likelihood
  from the same SVD factors as the update.
  """

  dtype_in = z_mean_pred.dtype
  z_mean_pred, Sigma_sqrt_pred, x_t, B, Lambda_sqrt = _promote(
    z_mean_pred, Sigma_sqrt_pred, x_t, B, Lambda_sqrt
  )
  if r <= int(B.shape[-2]):
    out = _rrkf_update_rank_le_obs(
      z_mean_pred,
      Sigma_sqrt_pred,
      B,
      Lambda_sqrt,
      x_t,
      return_log_likelihood=return_log_likelihood,
    )
  else:
    out = _rrkf_update_rank_gt_obs(
      z_mean_pred,
      Sigma_sqrt_pred,
      B,
      Lambda_sqrt,
      x_t,
      return_log_likelihood=return_log_likelihood,
    )
  return _demote(dtype_in, *out)


def rrkf_innovation_log_likelihood(
  z_mean_pred: torch.Tensor,
  Sigma_sqrt_pred: torch.Tensor,
  x_t: torch.Tensor,
  *,
  B: torch.Tensor,
  Lambda_sqrt: torch.Tensor,
) -> torch.Tensor:
  """Gaussian innovation log-likelihood under low-rank predictive covariance.

  Uses the RRKF square-root forms.
  """

  dtype_in = z_mean_pred.dtype
  z_mean_pred, Sigma_sqrt_pred, x_t, B, Lambda_sqrt = _promote(
    z_mean_pred, Sigma_sqrt_pred, x_t, B, Lambda_sqrt
  )
  r = int(Sigma_sqrt_pred.shape[-1])
  y_hat = torch.matmul(B, z_mean_pred.unsqueeze(-1)).squeeze(-1)
  C_Pi = torch.matmul(B, Sigma_sqrt_pred)
  Lambda_sqrt = _maybe_expand_factor(Lambda_sqrt, C_Pi)
  if r <= int(B.shape[-2]):
    whitened_cross = torch.linalg.solve(Lambda_sqrt, C_Pi)
    _U_S, s_S, Vh_S = torch.linalg.svd(whitened_cross.transpose(-1, -2), full_matrices=False)
    eps = torch.linalg.solve(Lambda_sqrt, x_t - y_hat)
    loglik = _loglik_rank_le_obs(
      eps=eps, s_S=s_S, Vh_S=Vh_S, Lambda_sqrt=Lambda_sqrt
    )
  else:
    U_S, s_S, _ = torch.linalg.svd(torch.cat([C_Pi, Lambda_sqrt], dim=-1), full_matrices=False)
    loglik = _loglik_rank_gt_obs(eps=x_t - y_hat, U_S=U_S, s_S=s_S)
  return _demote(dtype_in, loglik)


def _loglik_rank_le_obs(
  *,
  eps: torch.Tensor,
  s_S: torch.Tensor,
  Vh_S: torch.Tensor,
  Lambda_sqrt: torch.Tensor,
) -> torch.Tensor:
  """Innovation log-likelihood when r <= N."""

  obs_dim = eps.shape[-1]
  V_S = Vh_S.transpose(-1, -2)
  trm = torch.matmul(eps.unsqueeze(-2), V_S).squeeze(-2) * s_S
  inv_term = 1.0 / (s_S * s_S + 1.0)
  _, logdet_R = torch.linalg.slogdet(Lambda_sqrt)
  return (
    -0.5 * obs_dim * math.log(2.0 * math.pi)
    - logdet_R
    - 0.5 * torch.log(s_S * s_S + 1.0).sum(dim=-1)
    - 0.5 * (eps * eps).sum(dim=-1)
    + 0.5 * ((trm * inv_term) * trm).sum(dim=-1)
  )


def _loglik_rank_gt_obs(
  *,
  eps: torch.Tensor,
  U_S: torch.Tensor,
  s_S: torch.Tensor,
) -> torch.Tensor:
  """Innovation log-likelihood when r > N."""

  obs_dim = eps.shape[-1]
  tolerance = float(s_S.shape[-1]) * torch.finfo(s_S.dtype).eps
  inv = torch.where(s_S * s_S > tolerance, 1.0 / (s_S * s_S), torch.zeros_like(s_S))
  UTe = torch.matmul(U_S.transpose(-1, -2), eps.unsqueeze(-1)).squeeze(-1)
  quad = torch.matmul(
    eps.unsqueeze(-2),
    torch.matmul(U_S, (inv * UTe).unsqueeze(-1)),
  ).squeeze(-1).squeeze(-1)
  return (
    -0.5 * obs_dim * math.log(2.0 * math.pi)
    - 0.5 * quad
    - torch.log(s_S).sum(dim=-1)
  )


def _rrkf_update_rank_le_obs(
  z_mean_pred: torch.Tensor,
  Pi_sqrt: torch.Tensor,
  B: torch.Tensor,
  Lambda_sqrt: torch.Tensor,
  x_t: torch.Tensor,
  *,
  return_log_likelihood: bool,
) -> tuple[torch.Tensor, ...]:
  """Update when rank r is at most the observation dimension N."""

  y_hat = torch.matmul(B, z_mean_pred.unsqueeze(-1)).squeeze(-1)
  C_Pi = torch.matmul(B, Pi_sqrt)
  Lambda_sqrt = _maybe_expand_factor(Lambda_sqrt, C_Pi)
  whitened_cross = torch.linalg.solve(Lambda_sqrt, C_Pi)
  U_S, s_S, Vh_S = torch.linalg.svd(whitened_cross.transpose(-1, -2), full_matrices=False)

  I_plus_D2 = torch.diag_embed(1.0 + s_S * s_S)
  K_bar = torch.matmul(Pi_sqrt, _right_solve(U_S, I_plus_D2) * s_S.unsqueeze(-2))

  eps = torch.linalg.solve(Lambda_sqrt, x_t - y_hat)
  z_mean = z_mean_pred + torch.matmul(
    K_bar, torch.matmul(Vh_S, eps.unsqueeze(-1))
  ).squeeze(-1)

  scale = torch.sqrt(1.0 + s_S * s_S)
  Sigma_sqrt = _right_solve(torch.matmul(Pi_sqrt, U_S), torch.diag_embed(scale))
  if not return_log_likelihood:
    return z_mean, Sigma_sqrt
  loglik = _loglik_rank_le_obs(
    eps=eps, s_S=s_S, Vh_S=Vh_S, Lambda_sqrt=Lambda_sqrt
  )
  return z_mean, Sigma_sqrt, loglik


def _rrkf_update_rank_gt_obs(
  z_mean_pred: torch.Tensor,
  Pi_sqrt: torch.Tensor,
  B: torch.Tensor,
  Lambda_sqrt: torch.Tensor,
  x_t: torch.Tensor,
  *,
  return_log_likelihood: bool,
) -> tuple[torch.Tensor, ...]:
  """Update when rank r exceeds the observation dimension N."""

  r = int(Pi_sqrt.shape[-1])
  y_hat = torch.matmul(B, z_mean_pred.unsqueeze(-1)).squeeze(-1)
  C_Pi = torch.matmul(B, Pi_sqrt)
  Lambda_sqrt = _maybe_expand_factor(Lambda_sqrt, C_Pi)
  U_S, s_S, _ = torch.linalg.svd(torch.cat([C_Pi, Lambda_sqrt], dim=-1), full_matrices=False)

  K_tilde = _right_solve(torch.matmul(C_Pi.transpose(-1, -2), U_S), torch.diag_embed(s_S))
  K = torch.matmul(
    torch.matmul(Pi_sqrt, _right_solve(K_tilde, torch.diag_embed(s_S))),
    U_S.transpose(-1, -2),
  )

  eps = x_t - y_hat
  z_mean = z_mean_pred + torch.matmul(K, eps.unsqueeze(-1)).squeeze(-1)

  # Joseph-form square-root update step, then truncated SVD back to rank r.
  joseph = torch.cat(
    [Pi_sqrt - torch.matmul(K, C_Pi), torch.matmul(K, Lambda_sqrt)],
    dim=-1,
  )
  Sigma_sqrt = tsvd_factor(joseph, r)
  if not return_log_likelihood:
    return z_mean, Sigma_sqrt
  loglik = _loglik_rank_gt_obs(eps=eps, U_S=U_S, s_S=s_S)
  return z_mean, Sigma_sqrt, loglik


def tsvd(A: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Truncated SVD: "A ≈ U @ diag(s) @ V.T" with leading "r" factors."""

  U, s, Vh = torch.linalg.svd(A, full_matrices=False)
  rank = min(int(r), int(s.shape[-1]), int(U.shape[-1]), int(Vh.shape[-2]))
  return U[..., :rank], s[..., :rank], Vh[..., :rank, :].transpose(-1, -2)


def tsvd_factor(A: torch.Tensor, r: int) -> torch.Tensor:
  U, s, _V = tsvd(A, r)
  return U * s.unsqueeze(-2)


def cov_from_sqrt(Sigma_sqrt: torch.Tensor) -> torch.Tensor:
  return Sigma_sqrt @ Sigma_sqrt.transpose(-1, -2)


def _maybe_expand_factor(factor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
  if factor.ndim == like.ndim:
    return factor
  if factor.ndim + 1 == like.ndim:
    return factor.expand(*like.shape[:-2], *factor.shape[-2:])
  raise ValueError(f"Cannot broadcast factor of shape {tuple(factor.shape)} to {tuple(like.shape)}")


def _right_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
  return torch.linalg.solve(B.transpose(-1, -2), A.transpose(-1, -2)).transpose(-1, -2)


__all__ = [
  "RRKF_FLOAT64",
  "cov_from_sqrt",
  "rrkf_update",
  "rrkf_innovation_log_likelihood",
  "rrkf_predict",
  "tsvd",
  "tsvd_factor",
]
