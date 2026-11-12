"""G_tan metric, quantifying "dynamical groundedness" of local uncertainty via the alignment between model and oracle variances (i.e. covariance propagation).

Two variants are implemented:
G_tan: Spearman rank-correlation of pooled per-coordinate log variances. The model variance is the observed diagonal of the propagated posterior covariance J Sigma J^T (for "identity propagation" we just use Sigma). The oracle variance is diag(J_* Omega J_*^T) with ground truth Jacobians J_*.

G_tan_transport: same comparison after subtracting the log posterior baseline at the most recent assimilation time to isolate only the propagation effects. This is the version that we report in the paper. 
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data import lorenz63_drift, lorenz63_jacobian, lorenz96_drift, lorenz96_jacobian
from filtering.covariance import transport_covariance
from filtering.ekf import build_identity_observation_matrix
from filtering.ekf_lr import RRKF_FLOAT64, cov_from_sqrt, rrkf_predict, rrkf_update

try:
  from scipy.stats import spearmanr as _scipy_spearmanr

  _HAVE_SCIPY = True
except Exception:
  _HAVE_SCIPY = False


def rankdata(values: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=np.float64)
  order = np.argsort(values, kind="mergesort")
  ranks = np.empty_like(order, dtype=np.float64)
  ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
  _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
  if counts.max() > 1:
    sum_ranks = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sum_ranks, inverse, ranks)
    ranks = sum_ranks[inverse] / counts[inverse]
  return ranks


def spearman_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  if a.shape != b.shape:
    raise ValueError(f"a and b must have matching shape, got {a.shape} and {b.shape}")
  if _HAVE_SCIPY:
    rho, _ = _scipy_spearmanr(a, b)
    return float(rho)
  rank_a = rankdata(a)
  rank_b = rankdata(b)
  rank_a = rank_a - rank_a.mean()
  rank_b = rank_b - rank_b.mean()
  denom = float(np.sqrt((rank_a * rank_a).sum() * (rank_b * rank_b).sum()))
  if denom <= 0.0:
    return float("nan")
  return float((rank_a * rank_b).sum() / denom)


def gtan_from_log_variances(
  log_omega_model: np.ndarray,
  log_omega_oracle: np.ndarray,
  *,
  min_samples: int = 16,
) -> float:
  model = np.asarray(log_omega_model, dtype=np.float64).reshape(-1)
  oracle = np.asarray(log_omega_oracle, dtype=np.float64).reshape(-1)
  finite = np.isfinite(model) & np.isfinite(oracle)
  if int(finite.sum()) < int(min_samples):
    return float("nan")
  return spearman_rank_correlation(model[finite], oracle[finite])


def gtan_transport_from_log_variances(
  log_omega_model: np.ndarray,
  log_omega_oracle: np.ndarray,
  log_omega_post_baseline: np.ndarray,
  *,
  min_samples: int = 16,
) -> float:
  model = np.asarray(log_omega_model, dtype=np.float64) - np.asarray(log_omega_post_baseline, dtype=np.float64)
  oracle = np.asarray(log_omega_oracle, dtype=np.float64) - np.asarray(log_omega_post_baseline, dtype=np.float64)
  model = model.reshape(-1)
  oracle = oracle.reshape(-1)
  finite = np.isfinite(model) & np.isfinite(oracle)
  if int(finite.sum()) < int(min_samples):
    return float("nan")
  model = model[finite]
  oracle = oracle[finite]
  if float(np.std(model)) < 1.0e-5 or float(np.std(oracle)) < 1.0e-5:
    # We report anything less than 1e-5 std as 0 (constant/no covariance propagation)
    return 0.0
  return spearman_rank_correlation(model, oracle)


def gtan_from_variances(
  omega_model: np.ndarray,
  omega_oracle: np.ndarray,
  *,
  eps: float = 1.0e-8,
  min_samples: int = 16,
) -> float:
  return gtan_from_log_variances(
    np.log(float(eps) + np.clip(np.asarray(omega_model, dtype=np.float64), 0.0, None)),
    np.log(float(eps) + np.clip(np.asarray(omega_oracle, dtype=np.float64), 0.0, None)),
    min_samples=int(min_samples),
  )


def gtan_transport_from_variances(
  omega_model: np.ndarray,
  omega_oracle: np.ndarray,
  omega_post_baseline: np.ndarray,
  *,
  eps: float = 1.0e-8,
  min_samples: int = 16,
) -> float:
  return gtan_transport_from_log_variances(
    np.log(float(eps) + np.clip(np.asarray(omega_model, dtype=np.float64), 0.0, None)),
    np.log(float(eps) + np.clip(np.asarray(omega_oracle, dtype=np.float64), 0.0, None)),
    np.log(float(eps) + np.clip(np.asarray(omega_post_baseline, dtype=np.float64), 0.0, None)),
    min_samples=int(min_samples),
  )


def lorenz96_rk4_step_jacobian_raw(
  x_raw: np.ndarray,
  *,
  dt: float = 0.04,
  forcing: float = 8.0,
) -> np.ndarray:
  x = np.asarray(x_raw, dtype=np.float64).reshape(-1)
  dim = int(x.shape[0])
  identity = np.eye(dim, dtype=np.float64)

  k1 = lorenz96_drift(x, F=forcing)
  j1 = lorenz96_jacobian(x)

  x2 = x + 0.5 * float(dt) * k1
  k2 = lorenz96_drift(x2, F=forcing)
  j2 = lorenz96_jacobian(x2) @ (identity + 0.5 * float(dt) * j1)

  x3 = x + 0.5 * float(dt) * k2
  k3 = lorenz96_drift(x3, F=forcing)
  j3 = lorenz96_jacobian(x3) @ (identity + 0.5 * float(dt) * j2)

  x4 = x + float(dt) * k3
  j4 = lorenz96_jacobian(x4) @ (identity + float(dt) * j3)

  return identity + (float(dt) / 6.0) * (j1 + 2.0 * j2 + 2.0 * j3 + j4)


def lorenz96_rk4_step_jacobian_standardized(
  z_std: np.ndarray,
  *,
  mean: np.ndarray,
  std: np.ndarray,
  dt: float = 0.04,
  forcing: float = 8.0,
) -> np.ndarray:
  mean = np.asarray(mean, dtype=np.float64).reshape(-1)
  std = np.asarray(std, dtype=np.float64).reshape(-1)
  z_raw = np.asarray(z_std, dtype=np.float64).reshape(-1) * std + mean
  jacobian_raw = lorenz96_rk4_step_jacobian_raw(z_raw, dt=float(dt), forcing=float(forcing))
  return (1.0 / std)[:, None] * jacobian_raw * std[None, :]


def _ekf_transport_latent_covariance(
  cov: torch.Tensor,
  f_mat: torch.Tensor,
  *,
  gamma: torch.Tensor,
  covariance_transport: str,
) -> torch.Tensor:
  mode = str(covariance_transport).strip().lower()
  jacobian = None if mode == "identity" else f_mat
  return transport_covariance(
    cov.unsqueeze(0),
    gamma.unsqueeze(0),
    jacobian=None if jacobian is None else jacobian.unsqueeze(0),
    mode="identity" if mode == "identity" else "jacobian",
  ).squeeze(0)


def _observed_transported_variance(
  cov_det: torch.Tensor,
  f_mat: torch.Tensor,
  *,
  n_dim: int,
  covariance_transport: str,
) -> np.ndarray:
  """Observed variance of the propagated posterior without process noise."""
  mode = str(covariance_transport).strip().lower()
  if mode == "identity":
    scored = cov_det
  elif mode == "jacobian":
    scored = f_mat @ cov_det @ f_mat.T 
    scored = 0.5 * (scored + scored.T)
  return np.clip(torch.diag(scored[: int(n_dim), : int(n_dim)]).detach().cpu().numpy(), 0.0, None) # clip to avoid negative variances 


@dataclass(frozen=True)
class GroundingWindowResult:
  g_tan: float
  g_tan_transport: float
  n_samples: int


@torch.no_grad()
def _l96_rrkf_grounding_window_from_shell(
  shell: torch.nn.Module,
  obs: torch.Tensor,
  clean_np: np.ndarray,
  *,
  mean: np.ndarray,
  std: np.ndarray,
  dt: float,
  forcing: float,
  filter_warmup: int,
  eps: float,
  assim_every: int,
) -> GroundingWindowResult:
  """G_tan using the RRKF predictive covariance the low-rank shell trains with."""
  n_dim = int(shell.observed_dim)
  device = shell.A.device
  dtype = shell.A.dtype
  rank = int(shell.filter_rank)
  B = build_identity_observation_matrix(
    n_dim,
    int(shell.latent_dim),
    device=device,
    dtype=dtype,
  )
  transport_mode = str(shell.covariance_transport).strip().lower()
  transport_dtype = torch.float64 if RRKF_FLOAT64 else dtype
  k_assim = max(1, int(assim_every))

  z_mean, Sigma_sqrt = shell.initialize_from_observation(obs[0:1])
  det_sqrt = Sigma_sqrt
  p_obs_post = cov_from_sqrt(Sigma_sqrt[0])[:n_dim, :n_dim].detach().cpu().numpy().astype(np.float64)
  j_accum_np = np.eye(n_dim, dtype=np.float64)

  log_model_rows: list[np.ndarray] = []
  log_oracle_rows: list[np.ndarray] = []
  log_post_rows: list[np.ndarray] = []

  t_pred = int(obs.shape[0] - 1)
  for t in range(t_pred):
    if transport_mode == "identity":
      z_mean_pred = shell.transition_mean(z_mean)
      transported_sqrt = Sigma_sqrt
      det_transported = det_sqrt
    elif transport_mode == "jacobian":
      transported_sqrt, z_mean_pred = shell.transition_and_jacobian_factor(
        z_mean,
        Sigma_sqrt,
        transport_dtype=transport_dtype,
      )
      det_transported, _ = shell.transition_and_jacobian_factor(
        z_mean,
        det_sqrt,
        transport_dtype=transport_dtype,
      )
    else:
      raise ValueError(f"Unsupported covariance propagation: {shell.covariance_transport!r}")
    z_mean_pred, Sigma_sqrt_pred = rrkf_predict(
      z_mean,
      Sigma_sqrt,
      Gamma_sqrt=shell.Gamma_sqrt,
      r=rank,
      z_mean_pred=z_mean_pred,
      transported_sqrt=transported_sqrt,
    )
    p_det = cov_from_sqrt(det_transported[0])
    var_model = np.clip(torch.diag(p_det[:n_dim, :n_dim]).detach().cpu().numpy(), 0.0, None)

    j_star = lorenz96_rk4_step_jacobian_standardized(
      clean_np[t],
      mean=mean,
      std=std,
      dt=float(dt),
      forcing=float(forcing),
    )
    j_accum_new = j_star @ j_accum_np
    omega_oracle = j_accum_new @ p_obs_post @ j_accum_new.T
    var_oracle = np.clip(np.diag(omega_oracle), 0.0, None)

    log_model_rows.append(np.log(float(eps) + var_model))
    log_oracle_rows.append(np.log(float(eps) + var_oracle))
    log_post_rows.append(np.log(float(eps) + np.clip(np.diag(p_obs_post), 0.0, None)))

    if (t + 1) % k_assim == 0:
      z_mean, Sigma_sqrt = rrkf_update(
        z_mean_pred,
        Sigma_sqrt_pred,
        obs[t + 1 : t + 2],
        B=B,
        Lambda_sqrt=shell.Lambda_sqrt,
        r=rank,
        return_log_likelihood=False,
      )
      p_obs_post = cov_from_sqrt(Sigma_sqrt[0])[:n_dim, :n_dim].detach().cpu().numpy().astype(np.float64)
      det_sqrt = Sigma_sqrt
      j_accum_np = np.eye(n_dim, dtype=np.float64)
    else:
      z_mean, Sigma_sqrt = z_mean_pred, Sigma_sqrt_pred
      det_sqrt = det_transported
      j_accum_np = j_accum_new

  log_model = np.stack(log_model_rows, axis=0)
  log_oracle = np.stack(log_oracle_rows, axis=0)
  log_post = np.stack(log_post_rows, axis=0)

  warmup = int(filter_warmup)
  if warmup > 0:
    warmup_snapped = warmup
    if k_assim > 1 and warmup_snapped % k_assim != 0:
      warmup_snapped += k_assim - (warmup_snapped % k_assim)
    warmup_snapped = min(warmup_snapped, int(log_model.shape[0]))
    log_model = log_model[warmup_snapped:]
    log_oracle = log_oracle[warmup_snapped:]
    log_post = log_post[warmup_snapped:]

  g_tan = gtan_from_log_variances(log_model, log_oracle)
  g_tan_transport = gtan_transport_from_log_variances(log_model, log_oracle, log_post)
  finite = np.isfinite(log_model.reshape(-1)) & np.isfinite(log_oracle.reshape(-1))
  return GroundingWindowResult(
    g_tan=float(g_tan),
    g_tan_transport=float(g_tan_transport),
    n_samples=int(finite.sum()),
  )


@torch.no_grad()
def l96_ekf_grounding_window_from_shell(
  shell: torch.nn.Module,
  obs_window_std: np.ndarray | torch.Tensor,
  clean_window_std: np.ndarray | torch.Tensor,
  *,
  mean: np.ndarray,
  std: np.ndarray,
  dt: float = 0.04,
  forcing: float = 8.0,
  filter_warmup: int = 0,
  eps: float = 1.0e-8,
  assim_every: int = 1,
) -> GroundingWindowResult:
  """G_tan for ALRNN-transition map on a test window."""
  from models.alrnn_shell import ALRNNShell
  from models.alrnn_shell_lr import ALRNNShellLR

  if isinstance(shell, ALRNNShellLR):
    obs = torch.as_tensor(obs_window_std, dtype=torch.float32, device=shell.A.device)
    clean = torch.as_tensor(clean_window_std, dtype=torch.float32, device=shell.A.device)
    if obs.ndim != 2 or clean.shape != obs.shape:
      raise ValueError(f"obs/clean must share shape (T, N); got {obs.shape} vs {clean.shape}")
    return _l96_rrkf_grounding_window_from_shell(
      shell,
      obs,
      clean.detach().cpu().numpy().astype(np.float64),
      mean=np.asarray(mean, dtype=np.float64).reshape(-1),
      std=np.asarray(std, dtype=np.float64).reshape(-1),
      dt=float(dt),
      forcing=float(forcing),
      filter_warmup=int(filter_warmup),
      eps=float(eps),
      assim_every=int(assim_every),
    )

  if not isinstance(shell, ALRNNShell):
    raise TypeError(f"Expected ALRNNShell, got {type(shell)!r}")

  obs = torch.as_tensor(obs_window_std, dtype=torch.float32, device=shell.A.device)
  clean = torch.as_tensor(clean_window_std, dtype=torch.float32, device=shell.A.device)
  if obs.ndim != 2 or clean.shape != obs.shape:
    raise ValueError(f"obs/clean must share shape (T, N); got {obs.shape} vs {clean.shape}")

  m_dim = int(shell.latent_dim)
  n_dim = int(shell.observed_dim)
  device = shell.A.device
  dtype = shell.A.dtype
  eye_m = torch.eye(m_dim, device=device, dtype=dtype)
  h_mat = torch.zeros((n_dim, m_dim), device=device, dtype=dtype)
  h_mat[:, :n_dim] = torch.eye(n_dim, device=device, dtype=dtype)
  gamma = shell.Gamma.detach()
  r_mat = shell.Lambda.detach()

  mu_np = np.asarray(mean, dtype=np.float64).reshape(-1)
  sd_np = np.asarray(std, dtype=np.float64).reshape(-1)
  clean_np = clean.detach().cpu().numpy().astype(np.float64)
  k_assim = max(1, int(assim_every))

  z0 = obs[0]
  latent_mean = torch.matmul(z0, shell.embedding)
  latent_mean[:n_dim] = z0
  cov = gamma.clone()
  cov_det = cov.clone()
  p_obs_post_np = cov[:n_dim, :n_dim].detach().cpu().numpy().astype(np.float64)
  j_accum_np = np.eye(n_dim, dtype=np.float64)

  log_model_rows: list[np.ndarray] = []
  log_oracle_rows: list[np.ndarray] = []
  log_post_rows: list[np.ndarray] = []

  t_pred = int(obs.shape[0] - 1)
  for t in range(t_pred):
    f_mat = shell.transition_jacobian(latent_mean)
    m_pred = f_mat @ latent_mean + shell.h
    p_pred = _ekf_transport_latent_covariance(
      cov,
      f_mat,
      gamma=gamma,
      covariance_transport=str(shell.covariance_transport),
    )
    var_model = _observed_transported_variance(
      cov_det,
      f_mat,
      n_dim=n_dim,
      covariance_transport=str(shell.covariance_transport),
    )

    j_star = lorenz96_rk4_step_jacobian_standardized(
      clean_np[t],
      mean=mu_np,
      std=sd_np,
      dt=float(dt),
      forcing=float(forcing),
    )
    j_accum_new = j_star @ j_accum_np
    omega_oracle = j_accum_new @ p_obs_post_np @ j_accum_new.T
    var_oracle = np.clip(np.diag(omega_oracle), 0.0, None)

    log_model_rows.append(np.log(float(eps) + var_model))
    log_oracle_rows.append(np.log(float(eps) + var_oracle))
    log_post_rows.append(np.log(float(eps) + np.clip(np.diag(p_obs_post_np), 0.0, None)))

    if (t + 1) % k_assim == 0:
      omega_obs = p_pred[:n_dim, :n_dim] + r_mat
      y_t = obs[t + 1]
      gain = torch.linalg.solve(omega_obs, p_pred[:, :n_dim].T).T
      latent_mean = m_pred + gain @ (y_t - m_pred[:n_dim])
      cov = (eye_m - gain @ h_mat) @ p_pred @ (eye_m - gain @ h_mat).T + gain @ r_mat @ gain.T
      cov = 0.5 * (cov + cov.T)
      cov_det = cov.clone()
      p_obs_post_np = cov[:n_dim, :n_dim].detach().cpu().numpy().astype(np.float64)
      j_accum_np = np.eye(n_dim, dtype=np.float64)
    else:
      latent_mean = m_pred
      cov = p_pred
      if str(shell.covariance_transport).strip().lower() == "jacobian":
        cov_det = f_mat @ cov_det @ f_mat.T
        cov_det = 0.5 * (cov_det + cov_det.T)
      j_accum_np = j_accum_new

  log_model = np.stack(log_model_rows, axis=0)
  log_oracle = np.stack(log_oracle_rows, axis=0)
  log_post = np.stack(log_post_rows, axis=0)

  warmup = int(filter_warmup)
  if warmup > 0:
    warmup_snapped = warmup
    if k_assim > 1 and warmup_snapped % k_assim != 0:
      warmup_snapped += k_assim - (warmup_snapped % k_assim)
    warmup_snapped = min(warmup_snapped, int(log_model.shape[0]))
    log_model = log_model[warmup_snapped:]
    log_oracle = log_oracle[warmup_snapped:]
    log_post = log_post[warmup_snapped:]

  g_tan = gtan_from_log_variances(log_model, log_oracle)
  g_tan_transport = gtan_transport_from_log_variances(log_model, log_oracle, log_post)
  finite = np.isfinite(log_model.reshape(-1)) & np.isfinite(log_oracle.reshape(-1))
  return GroundingWindowResult(
    g_tan=float(g_tan),
    g_tan_transport=float(g_tan_transport),
    n_samples=int(finite.sum()),
  )


def lorenz63_rk4_step_jacobian_raw(
  x_raw: np.ndarray,
  *,
  dt: float = 0.01,
  sigma: float = 10.0,
  rho: float = 28.0,
  beta: float = 8.0 / 3.0,
) -> np.ndarray:
  x = np.asarray(x_raw, dtype=np.float64).reshape(3)
  identity = np.eye(3, dtype=np.float64)

  k1 = lorenz63_drift(x, sigma=sigma, rho=rho, beta=beta)
  j1 = lorenz63_jacobian(x, sigma=sigma, rho=rho, beta=beta)

  x2 = x + 0.5 * float(dt) * k1
  k2 = lorenz63_drift(x2, sigma=sigma, rho=rho, beta=beta)
  j2 = lorenz63_jacobian(x2, sigma=sigma, rho=rho, beta=beta) @ (identity + 0.5 * float(dt) * j1)

  x3 = x + 0.5 * float(dt) * k2
  k3 = lorenz63_drift(x3, sigma=sigma, rho=rho, beta=beta)
  j3 = lorenz63_jacobian(x3, sigma=sigma, rho=rho, beta=beta) @ (identity + 0.5 * float(dt) * j2)

  x4 = x + float(dt) * k3
  j4 = lorenz63_jacobian(x4, sigma=sigma, rho=rho, beta=beta) @ (identity + float(dt) * j3)

  return identity + (float(dt) / 6.0) * (j1 + 2.0 * j2 + 2.0 * j3 + j4)


def lorenz63_rk4_step_jacobian_standardized(
  z_std: np.ndarray,
  *,
  mean: np.ndarray,
  std: np.ndarray,
  dt: float = 0.01,
  sigma: float = 10.0,
  rho: float = 28.0,
  beta: float = 8.0 / 3.0,
) -> np.ndarray:
  mean = np.asarray(mean, dtype=np.float64).reshape(3)
  std = np.asarray(std, dtype=np.float64).reshape(3)
  z_raw = np.asarray(z_std, dtype=np.float64).reshape(3) * std + mean
  jacobian_raw = lorenz63_rk4_step_jacobian_raw(
    z_raw, dt=float(dt), sigma=float(sigma), rho=float(rho), beta=float(beta)
  )
  return (1.0 / std)[:, None] * jacobian_raw * std[None, :]


@torch.no_grad()
def l63_ekf_grounding_window_from_shell(
  shell: torch.nn.Module,
  obs_window_std: np.ndarray | torch.Tensor,
  clean_window_std: np.ndarray | torch.Tensor,
  *,
  mean: np.ndarray,
  std: np.ndarray,
  dt: float = 0.01,
  filter_warmup: int = 0,
  eps: float = 1.0e-8,
  assim_every: int = 1,
) -> GroundingWindowResult:
  """G_tan on Lorenz-63; the oracle uses the RK4 map Jacobian."""
  from models.alrnn_shell import ALRNNShell

  if not isinstance(shell, ALRNNShell):
    raise TypeError(f"Expected ALRNNShell, got {type(shell)!r}")

  obs = torch.as_tensor(obs_window_std, dtype=torch.float32, device=shell.A.device)
  clean = torch.as_tensor(clean_window_std, dtype=torch.float32, device=shell.A.device)
  if obs.ndim != 2 or clean.shape != obs.shape:
    raise ValueError(f"obs/clean must share shape (T, N); got {obs.shape} vs {clean.shape}")

  m_dim = int(shell.latent_dim)
  n_dim = int(shell.observed_dim)
  device = shell.A.device
  dtype = shell.A.dtype
  eye_m = torch.eye(m_dim, device=device, dtype=dtype)
  h_mat = torch.zeros((n_dim, m_dim), device=device, dtype=dtype)
  h_mat[:, :n_dim] = torch.eye(n_dim, device=device, dtype=dtype)
  gamma = shell.Gamma.detach()
  r_mat = shell.Lambda.detach()

  mu_np = np.asarray(mean, dtype=np.float64).reshape(-1)
  sd_np = np.asarray(std, dtype=np.float64).reshape(-1)
  clean_np = clean.detach().cpu().numpy().astype(np.float64)
  k_assim = max(1, int(assim_every))

  z0 = obs[0]
  latent_mean = torch.matmul(z0, shell.embedding)
  latent_mean[:n_dim] = z0
  cov = gamma.clone()
  cov_det = cov.clone()
  p_obs_post_np = cov[:n_dim, :n_dim].detach().cpu().numpy().astype(np.float64)
  j_accum_np = np.eye(n_dim, dtype=np.float64)

  log_model_rows: list[np.ndarray] = []
  log_oracle_rows: list[np.ndarray] = []
  log_post_rows: list[np.ndarray] = []

  t_pred = int(obs.shape[0] - 1)
  for t in range(t_pred):
    f_mat = shell.transition_jacobian(latent_mean)
    m_pred = f_mat @ latent_mean + shell.h
    p_pred = _ekf_transport_latent_covariance(
      cov,
      f_mat,
      gamma=gamma,
      covariance_transport=str(shell.covariance_transport),
    )
    var_model = _observed_transported_variance(
      cov_det,
      f_mat,
      n_dim=n_dim,
      covariance_transport=str(shell.covariance_transport),
    )

    j_star = lorenz63_rk4_step_jacobian_standardized(
      clean_np[t],
      mean=mu_np,
      std=sd_np,
      dt=float(dt),
    )
    j_accum_new = j_star @ j_accum_np
    omega_oracle = j_accum_new @ p_obs_post_np @ j_accum_new.T
    var_oracle = np.clip(np.diag(omega_oracle), 0.0, None)

    log_model_rows.append(np.log(float(eps) + var_model))
    log_oracle_rows.append(np.log(float(eps) + var_oracle))
    log_post_rows.append(np.log(float(eps) + np.clip(np.diag(p_obs_post_np), 0.0, None)))

    if (t + 1) % k_assim == 0:
      omega_obs = p_pred[:n_dim, :n_dim] + r_mat
      y_t = obs[t + 1]
      gain = torch.linalg.solve(omega_obs, p_pred[:, :n_dim].T).T
      latent_mean = m_pred + gain @ (y_t - m_pred[:n_dim])
      cov = (eye_m - gain @ h_mat) @ p_pred @ (eye_m - gain @ h_mat).T + gain @ r_mat @ gain.T
      cov = 0.5 * (cov + cov.T)
      cov_det = cov.clone()
      p_obs_post_np = cov[:n_dim, :n_dim].detach().cpu().numpy().astype(np.float64)
      j_accum_np = np.eye(n_dim, dtype=np.float64)
    else:
      latent_mean = m_pred
      cov = p_pred
      if str(shell.covariance_transport).strip().lower() == "jacobian":
        cov_det = f_mat @ cov_det @ f_mat.T
        cov_det = 0.5 * (cov_det + cov_det.T)
      j_accum_np = j_accum_new

  log_model = np.stack(log_model_rows, axis=0)
  log_oracle = np.stack(log_oracle_rows, axis=0)
  log_post = np.stack(log_post_rows, axis=0)

  warmup = int(filter_warmup)
  if warmup > 0:
    warmup_snapped = warmup
    if k_assim > 1 and warmup_snapped % k_assim != 0:
      warmup_snapped += k_assim - (warmup_snapped % k_assim)
    warmup_snapped = min(warmup_snapped, int(log_model.shape[0]))
    log_model = log_model[warmup_snapped:]
    log_oracle = log_oracle[warmup_snapped:]
    log_post = log_post[warmup_snapped:]

  g_tan = gtan_from_log_variances(log_model, log_oracle)
  g_tan_transport = gtan_transport_from_log_variances(log_model, log_oracle, log_post)
  finite = np.isfinite(log_model.reshape(-1)) & np.isfinite(log_oracle.reshape(-1))
  return GroundingWindowResult(
    g_tan=float(g_tan),
    g_tan_transport=float(g_tan_transport),
    n_samples=int(finite.sum()),
  )


__all__ = [
  "GroundingWindowResult",
  "gtan_from_log_variances",
  "gtan_from_variances",
  "gtan_transport_from_log_variances",
  "gtan_transport_from_variances",
  "l63_ekf_grounding_window_from_shell",
  "l96_ekf_grounding_window_from_shell",
  "lorenz63_rk4_step_jacobian_raw",
  "lorenz63_rk4_step_jacobian_standardized",
  "lorenz96_rk4_step_jacobian_raw",
  "lorenz96_rk4_step_jacobian_standardized",
  "rankdata",
  "spearman_rank_correlation",
]