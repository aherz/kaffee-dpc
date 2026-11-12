"""Lorenz-96 data-generation helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


def compute_standardizer(X: np.ndarray, eps: float = 1e-6) -> dict[str, np.ndarray]:
  X = np.asarray(X, dtype=np.float64)
  mean = X.mean(axis=0)
  std = X.std(axis=0)
  std = np.where(std < float(eps), 1.0, std)
  return {
    "mean": mean.astype(np.float32),
    "std": std.astype(np.float32),
  }


def standardize(X: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
  mean = np.asarray(stats["mean"], dtype=np.float32)
  std = np.asarray(stats["std"], dtype=np.float32)
  return ((np.asarray(X, dtype=np.float32) - mean) / std).astype(np.float32)


def unstandardize(Xz: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
  mean = np.asarray(stats["mean"], dtype=np.float32)
  std = np.asarray(stats["std"], dtype=np.float32)
  return (np.asarray(Xz, dtype=np.float32) * std + mean).astype(np.float32)


def standardize_dataset(X: np.ndarray, *, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  X = np.asarray(X, dtype=np.float32)
  mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
  std = X.reshape(-1, X.shape[-1]).std(axis=0)
  std = np.where(std < float(eps), 1.0, std)
  X_std = (X - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
  return X_std.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def add_observation_noise(
  X: np.ndarray,
  sigma_obs: float,
  *,
  per_dim_scale: np.ndarray,
  seed: int = 0,
) -> np.ndarray:
  X = np.asarray(X, dtype=np.float32)
  if float(sigma_obs) <= 0.0:
    return X
  rng = np.random.default_rng(int(seed))
  scale = np.asarray(per_dim_scale, dtype=np.float64).reshape(1, -1)
  noise = rng.standard_normal(X.shape) * (float(sigma_obs) * scale)
  return (X + noise.astype(np.float32)).astype(np.float32)


def _normalize_noise_space(value: object | None) -> str:
  text = str(value or "").strip().lower()
  if not text:
    return "raw"
  if text in ("raw", "unstandardized", "unstd"):
    return "raw"
  if text in ("standardized", "standardised", "std", "z", "zscore", "z-score"):
    return "standardized"
  raise ValueError(f"Unsupported noise space {value!r}; expected 'raw' or 'standardized'")


def _rk4_step(state: np.ndarray, dt: float, drift_fn) -> np.ndarray:
  k1 = drift_fn(state)
  k2 = drift_fn(state + 0.5 * dt * k1)
  k3 = drift_fn(state + 0.5 * dt * k2)
  k4 = drift_fn(state + dt * k3)
  return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def lorenz96_drift(x: np.ndarray, F: float = 8.0) -> np.ndarray: # We use forcing F = 8 here (standard value)
  x = np.asarray(x, dtype=np.float64).reshape(-1)
  N = int(x.shape[0])
  if N < 4:
    raise ValueError(f"Lorenz-96 requires N>=4; got N={N}")
  xm2 = np.roll(x, 2)
  xm1 = np.roll(x, 1)
  xp1 = np.roll(x, -1)
  return (xp1 - xm2) * xm1 - x + float(F)


def lorenz96_jacobian(x: np.ndarray) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64).reshape(-1)
  N = int(x.shape[0])
  if N < 4:
    raise ValueError(f"Lorenz-96 requires N>=4; got N={N}")
  J = np.zeros((N, N), dtype=np.float64)
  for i in range(N):
    im1 = (i - 1) % N
    im2 = (i - 2) % N
    ip1 = (i + 1) % N
    J[i, i] = -1.0
    J[i, im1] = float(x[ip1] - x[im2])
    J[i, ip1] = float(x[im1])
    J[i, im2] = float(-x[im1])
  return J


def _lorenz96_initial_state(N: int, F: float, seed: int) -> np.ndarray:
  rng = np.random.default_rng(int(seed))
  return float(F) * np.ones(int(N), dtype=np.float64) + 0.01 * rng.standard_normal(int(N))


def _solve_lorenz96_deterministic(total_T: int, dt: float, N: int, F: float, x0: np.ndarray) -> np.ndarray:
  t_eval = np.arange(int(total_T), dtype=np.float64) * float(dt)

  def drift(_t: float, state: np.ndarray) -> np.ndarray:
    return lorenz96_drift(state, F=float(F))

  solution = solve_ivp(
    drift,
    t_span=(t_eval[0], t_eval[-1]),
    y0=np.asarray(x0, dtype=np.float64),
    t_eval=t_eval,
    method="RK45",
    rtol=1e-6,
    atol=1e-9,
    first_step=float(dt),
    max_step=float(dt),
  )
  if not bool(solution.success):
    raise RuntimeError(f"solve_ivp failed for lorenz96 (success={solution.success}): {solution.message}")
  return np.asarray(solution.y.T, dtype=np.float64)


def normalize_lorenz96_proc_space(sigma_proc_space: str | None) -> str:
  text = str(sigma_proc_space or "raw").strip().lower()
  if text in {"standardized", "standardised", "std", "z", "zscore", "z-score", "zscored", "z-scored"}:
    return "standardized"
  if text == "raw":
    return "raw"
  raise ValueError(f"Unsupported Lorenz-96 sigma_proc_space={sigma_proc_space!r}; expected 'standardized' or 'raw'.")


def _generate_sequences(
  n_seq: int,
  T: int,
  *,
  dt: float,
  N: int,
  F: float,
  burn_in: int,
  sigma_proc: float,
  sigma_proc_space: str,
  rng: np.random.Generator,
) -> np.ndarray:
  sequences: list[np.ndarray] = []
  for _ in range(int(n_seq)):
    seq_seed = int(rng.integers(0, 2**31))
    sequences.append(
      make_lorenz96(
        int(T),
        dt=float(dt),
        N=int(N),
        F=float(F),
        burn_in=int(burn_in),
        seed=seq_seed,
        sigma_proc=float(sigma_proc),
        sigma_proc_space=str(sigma_proc_space),
      ).astype(np.float32)
    )
  return np.stack(sequences, axis=0)


def _apply_dataset_observation_noise(
  clean_raw: np.ndarray,
  sigma_obs: float,
  *,
  obs_sigma_space: str,
  rng: np.random.Generator,
  ref_std_per_dim: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | str]]:
  clean_raw = np.asarray(clean_raw, dtype=np.float32)
  n_dim = int(clean_raw.shape[-1])
  active_space = _normalize_noise_space(obs_sigma_space)
  inferred_ref_std = clean_raw.reshape(-1, n_dim).std(axis=0).astype(np.float64)
  if ref_std_per_dim is None:
    ref_std = inferred_ref_std
  else:
    ref_std = np.asarray(ref_std_per_dim, dtype=np.float64).reshape(-1)
    if ref_std.shape != (n_dim,):
      raise ValueError(f"ref_std_per_dim must have shape ({n_dim},), got {ref_std.shape}")

  if float(sigma_obs) <= 0.0:
    raw_per_dim = np.zeros((n_dim,), dtype=np.float64)
    observed_raw = clean_raw.copy()
  elif active_space == "standardized":
    raw_per_dim = float(sigma_obs) * ref_std
    observed_raw = clean_raw + (
      rng.standard_normal(size=clean_raw.shape).astype(np.float32)
      * raw_per_dim.reshape(1, 1, -1).astype(np.float32)
    )
  else:
    raw_per_dim = np.full((n_dim,), float(sigma_obs), dtype=np.float64)
    observed_raw = clean_raw + (
      rng.standard_normal(size=clean_raw.shape).astype(np.float32)
      * raw_per_dim.reshape(1, 1, -1).astype(np.float32)
    )

  return observed_raw.astype(np.float32), {
    "obs_sigma_space": active_space,
    "obs_sigma_raw_per_dim": raw_per_dim.astype(np.float64),
    "ref_std_per_dim": ref_std.astype(np.float64),
  }


def lorenz96_reference_std(
  T: int,
  *,
  dt: float = 0.04,
  N: int = 20,
  F: float = 8.0,
  burn_in: int = 1000,
  seed: int = 0,
) -> np.ndarray:
  x0 = _lorenz96_initial_state(int(N), float(F), int(seed))
  reference = _solve_lorenz96_deterministic(int(T) + int(burn_in), float(dt), int(N), float(F), x0)[int(burn_in) :]
  std = np.asarray(reference.std(axis=0), dtype=np.float64)
  return np.where(std < 1e-6, 1.0, std)


def _lorenz96_process_scale(
  T: int,
  *,
  dt: float,
  N: int,
  F: float,
  burn_in: int,
  seed: int,
  sigma_proc: float,
  sigma_proc_space: str,
) -> np.ndarray:
  sigma_proc = float(sigma_proc)
  if sigma_proc <= 0.0:
    return np.zeros(int(N), dtype=np.float64)
  if normalize_lorenz96_proc_space(sigma_proc_space) == "raw":
    return np.full(int(N), sigma_proc, dtype=np.float64)
  return sigma_proc * lorenz96_reference_std(
    T,
    dt=dt,
    N=N,
    F=F,
    burn_in=burn_in,
    seed=seed,
  )


def make_lorenz96(
  T: int,
  *,
  dt: float = 0.04,
  N: int = 20,
  F: float = 8.0,
  burn_in: int = 1000,
  seed: int = 0,
  sigma_proc: float = 0.0,
  sigma_proc_space: str = "raw",
  proc_scale_raw: np.ndarray | None = None,
) -> np.ndarray:
  T = int(T)
  burn_in = int(burn_in)
  dt = float(dt)
  N = int(N)
  F = float(F)
  rng = np.random.default_rng(int(seed))
  x0 = _lorenz96_initial_state(N, F, int(seed))

  total_T = T + burn_in
  if float(sigma_proc) <= 0.0:
    X = _solve_lorenz96_deterministic(total_T, dt, N, F, x0)
    return X[burn_in:].astype(np.float32)

  X = np.zeros((total_T, N), dtype=np.float64)
  X[0] = x0
  if proc_scale_raw is None:
    proc_scale = _lorenz96_process_scale(
      T,
      dt=dt,
      N=N,
      F=F,
      burn_in=burn_in,
      seed=seed,
      sigma_proc=sigma_proc,
      sigma_proc_space=sigma_proc_space,
    )
  else:
    proc_scale = np.asarray(proc_scale_raw, dtype=np.float64).reshape(-1)
    if proc_scale.shape != (N,) or not np.all(np.isfinite(proc_scale)):
      raise ValueError(f"proc_scale_raw must contain {N} finite values")
  sqrt_dt = math.sqrt(dt)
  drift_fn = lambda state: lorenz96_drift(state, F=F)
  for t in range(total_T - 1):
    drift = _rk4_step(X[t], dt, drift_fn)
    drift = drift + proc_scale * sqrt_dt * rng.standard_normal(N)
    X[t + 1] = drift
  return X[burn_in:].astype(np.float32)


def make_lorenz96_train_test(
  *,
  T_train: int,
  T_test: int,
  dt: float = 0.04,
  N: int = 20,
  F: float = 8.0,
  seed: int = 0,
  sigma_proc: float = 0.0,
  sigma_obs: float = 0.0,
  sigma_proc_space: str = "standardized",
  noise_on_test: bool = True,
) -> dict[str, Any]:
  rng = np.random.default_rng(int(seed))
  train_clean_raw = _generate_sequences(
    1,
    int(T_train),
    dt=float(dt),
    N=int(N),
    F=float(F),
    burn_in=1000,
    sigma_proc=float(sigma_proc),
    sigma_proc_space=str(sigma_proc_space),
    rng=rng,
  )
  train_obs_raw, train_obs_info = _apply_dataset_observation_noise(
    train_clean_raw,
    float(sigma_obs),
    obs_sigma_space="standardized",
    rng=rng,
  )
  train_obs, mean, std = standardize_dataset(train_obs_raw)

  test_clean_raw = _generate_sequences(
    1,
    int(T_test),
    dt=float(dt),
    N=int(N),
    F=float(F),
    burn_in=1000,
    sigma_proc=float(sigma_proc),
    sigma_proc_space=str(sigma_proc_space),
    rng=rng,
  )
  if bool(noise_on_test):
    test_obs_raw, _ = _apply_dataset_observation_noise(
      test_clean_raw,
      float(sigma_obs),
      obs_sigma_space="standardized",
      rng=rng,
      ref_std_per_dim=np.asarray(train_obs_info["ref_std_per_dim"], dtype=np.float64),
    )
  else:
    test_obs_raw = test_clean_raw.copy()

  stats = {
    "mean": mean.astype(np.float32),
    "std": std.astype(np.float32),
  }
  train_clean = standardize(train_clean_raw[0], stats)
  test_clean = standardize(test_clean_raw[0], stats)
  test_obs = standardize(test_obs_raw[0], stats)
  return {
    "train_obs": train_obs[0],
    "test_obs": test_obs,
    "train_clean": train_clean,
    "test_clean": test_clean,
    "train_obs_raw": train_obs_raw[0],
    "test_obs_raw": test_obs_raw[0],
    "train_clean_raw": train_clean_raw[0],
    "test_clean_raw": test_clean_raw[0],
    "dt": float(dt),
    "standardizer": stats,
    "data_info": {
      "system": "lorenz96",
      "setting": "both" if float(sigma_obs) > 0.0 and float(sigma_proc) > 0.0 else "original",
      "dt": float(dt),
      "proc_sigma": float(sigma_proc),
      "proc_sigma_space": str(sigma_proc_space),
      "obs_sigma": float(sigma_obs),
      "obs_sigma_space": str(train_obs_info["obs_sigma_space"]),
      "obs_sigma_raw_per_dim": np.asarray(train_obs_info["obs_sigma_raw_per_dim"], dtype=np.float64),
      "obs_sigma_ref_std_per_dim": np.asarray(train_obs_info["ref_std_per_dim"], dtype=np.float64),
      "T": int(T_train),
      "T_test": int(T_test),
      "burn_in": 1000,
      "n_seq": 1,
      "N": int(N),
      "seed": int(seed),
    },
  }


def load_lorenz96_dataset(
  dataset_path: str | Path,
  *,
  info_path: str | Path | None = None,
) -> dict[str, Any]:
  dataset_file = Path(dataset_path)
  bundle = np.load(dataset_file)

  required_keys = {
    "Z_raw",
    "X_raw",
    "X_std",
    "X_mu",
    "X_sd",
    "Z_test_raw",
    "X_test_raw",
  }
  missing = sorted(required_keys.difference(bundle.files))
  if missing:
    raise ValueError(f"Lorenz-96 dataset {dataset_file} is missing keys: {missing}")

  train_clean_raw = np.asarray(bundle["Z_raw"], dtype=np.float32)
  train_obs_raw = np.asarray(bundle["X_raw"], dtype=np.float32)
  train_obs = np.asarray(bundle["X_std"], dtype=np.float32)
  mean = np.asarray(bundle["X_mu"], dtype=np.float32).reshape(-1)
  std = np.asarray(bundle["X_sd"], dtype=np.float32).reshape(-1)
  test_clean_raw = np.asarray(bundle["Z_test_raw"], dtype=np.float32)
  test_obs_raw = np.asarray(bundle["X_test_raw"], dtype=np.float32)

  if train_clean_raw.ndim != 3 or train_clean_raw.shape[0] != 1:
    raise ValueError(f"Expected Z_raw shape (1, T, N), got {train_clean_raw.shape}")
  if train_obs_raw.shape != train_clean_raw.shape or train_obs.shape != train_clean_raw.shape:
    raise ValueError(
      "Expected X_raw and X_std to match Z_raw shape; got "
      f"Z_raw={train_clean_raw.shape}, X_raw={train_obs_raw.shape}, X_std={train_obs.shape}"
    )
  if test_clean_raw.ndim != 3 or test_clean_raw.shape[0] != 1:
    raise ValueError(f"Expected Z_test_raw shape (1, T_test, N), got {test_clean_raw.shape}")
  if test_obs_raw.shape != test_clean_raw.shape:
    raise ValueError(
      f"Expected X_test_raw to match Z_test_raw shape; got {test_obs_raw.shape} vs {test_clean_raw.shape}"
    )
  if mean.shape != std.shape or mean.shape[0] != train_clean_raw.shape[-1]:
    raise ValueError(
      f"Expected X_mu/X_sd shape ({train_clean_raw.shape[-1]},), got {mean.shape} / {std.shape}"
    )

  stats = {
    "mean": mean.astype(np.float32),
    "std": std.astype(np.float32),
  }
  train_clean = standardize(train_clean_raw[0], stats)
  test_clean = standardize(test_clean_raw[0], stats)
  test_obs = standardize(test_obs_raw[0], stats)

  if info_path is None:
    info_path = dataset_file.with_name("data_info.json")
  info_file = Path(info_path)
  if not info_file.is_file():
    raise FileNotFoundError(
      f"Missing Lorenz-96 data_info.json next to {dataset_file} (expected {info_file})"
    )
  with info_file.open("r", encoding="utf-8") as handle:
    data_info = dict(json.load(handle))
  data_info.setdefault("source", str(dataset_file))

  return {
    "train_obs": train_obs[0],
    "test_obs": test_obs,
    "train_clean": train_clean,
    "test_clean": test_clean,
    "train_obs_raw": train_obs_raw[0],
    "test_obs_raw": test_obs_raw[0],
    "train_clean_raw": train_clean_raw[0],
    "test_clean_raw": test_clean_raw[0],
    "standardizer": stats,
    "data_info": data_info,
  }

def _load_lorenz96_evaluation(
  data_path: str | Path,
  *,
  metadata_path: str | Path,
  standardizer: dict[str, np.ndarray],
  artifact_name: str,
) -> dict[str, Any]:
  data_file = Path(data_path)
  metadata_file = Path(metadata_path)
  metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

  with np.load(data_file) as bundle:
    required = {
      "clean_raw",
      "obs_raw",
      "clean_std",
      "obs_std",
      "process_seed",
      "observation_noise_seed",
    }
    missing = sorted(required.difference(bundle.files))
    if missing:
      raise ValueError(f"Lorenz-96 {artifact_name} {data_file} is missing keys: {missing}")
    clean_raw = np.asarray(bundle["clean_raw"], dtype=np.float32)
    obs_raw = np.asarray(bundle["obs_raw"], dtype=np.float32)
    clean_std = np.asarray(bundle["clean_std"], dtype=np.float32)
    obs_std = np.asarray(bundle["obs_std"], dtype=np.float32)
    process_seed = int(bundle["process_seed"])
    observation_seed = int(bundle["observation_noise_seed"])

  expected_shape = (int(metadata["length"]), int(metadata["state_dim"]))
  for name, values in {
    "clean_raw": clean_raw,
    "obs_raw": obs_raw,
    "clean_std": clean_std,
    "obs_std": obs_std,
  }.items():
    if values.shape != expected_shape:
      raise ValueError(f"Expected {name} shape {expected_shape}, got {values.shape}")
    if not np.all(np.isfinite(values)):
      raise ValueError(f"Lorenz-96 {artifact_name} contains non-finite values in {name}")
  if not np.allclose(standardize(clean_raw, standardizer), clean_std, atol=2e-5, rtol=2e-5):
    raise ValueError(f"Lorenz-96 {artifact_name} latent trajectory does not match the training standardizer")
  if not np.allclose(standardize(obs_raw, standardizer), obs_std, atol=2e-5, rtol=2e-5):
    raise ValueError(f"Lorenz-96 {artifact_name} observations do not match the training standardizer")

  return {
    "clean_raw": clean_raw,
    "obs_raw": obs_raw,
    "clean_std": clean_std,
    "obs_std": obs_std,
    "process_seed": process_seed,
    "observation_noise_seed": observation_seed,
    "metadata": metadata,
  }


def load_lorenz96_heldout_test(
  data_path: str | Path,
  *,
  metadata_path: str | Path,
  standardizer: dict[str, np.ndarray],
) -> dict[str, Any]:
  return _load_lorenz96_evaluation(
    data_path,
    metadata_path=metadata_path,
    standardizer=standardizer,
    artifact_name="held-out test",
  )


def load_lorenz96_stochastic_evaluation(
  data_path: str | Path,
  *,
  metadata_path: str | Path,
  standardizer: dict[str, np.ndarray],
) -> dict[str, Any]:
  return _load_lorenz96_evaluation(
    data_path,
    metadata_path=metadata_path,
    standardizer=standardizer,
    artifact_name="stochastic evaluation",
  )


__all__ = [
    "add_observation_noise",
    "compute_standardizer",
    "load_lorenz96_dataset",
    "load_lorenz96_heldout_test",
    "load_lorenz96_stochastic_evaluation",
    "lorenz96_drift",
    "lorenz96_jacobian",
    "lorenz96_reference_std",
    "make_lorenz96",
    "make_lorenz96_train_test",
    "standardize_dataset",
    "standardize",
    "unstandardize",
]