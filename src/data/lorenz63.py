"""Lorenz-63 data-generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

LORENZ63_SIGMA = 10.0
LORENZ63_RHO = 28.0
LORENZ63_BETA = 8.0 / 3.0
LORENZ63_STATE_DIM = 3


def lorenz63_drift(
	state: np.ndarray,
	*,
	sigma: float = LORENZ63_SIGMA,
	rho: float = LORENZ63_RHO,
	beta: float = LORENZ63_BETA,
) -> np.ndarray:
	x = np.asarray(state, dtype=np.float64)
	xx, yy, zz = x[..., 0], x[..., 1], x[..., 2]
	dx = sigma * (yy - xx)
	dy = xx * (rho - zz) - yy
	dz = xx * yy - beta * zz
	return np.stack((dx, dy, dz), axis=-1)


def lorenz63_jacobian(
	state: np.ndarray,
	*,
	sigma: float = LORENZ63_SIGMA,
	rho: float = LORENZ63_RHO,
	beta: float = LORENZ63_BETA,
) -> np.ndarray:
	x = np.asarray(state, dtype=np.float64).reshape(LORENZ63_STATE_DIM)
	return np.array(
		[
			[-sigma, sigma, 0.0],
			[rho - x[2], -1.0, -x[0]],
			[x[1], x[0], -beta],
		],
		dtype=np.float64,
	)


def lorenz63_rk4_step(
	state: np.ndarray,
	*,
	dt: float,
	sigma: float = LORENZ63_SIGMA,
	rho: float = LORENZ63_RHO,
	beta: float = LORENZ63_BETA,
) -> np.ndarray:
	x = np.asarray(state, dtype=np.float64)
	k1 = lorenz63_drift(x, sigma=sigma, rho=rho, beta=beta)
	k2 = lorenz63_drift(x + 0.5 * float(dt) * k1, sigma=sigma, rho=rho, beta=beta)
	k3 = lorenz63_drift(x + 0.5 * float(dt) * k2, sigma=sigma, rho=rho, beta=beta)
	k4 = lorenz63_drift(x + float(dt) * k3, sigma=sigma, rho=rho, beta=beta)
	return x + (float(dt) / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _require_series(bundle: Any, key: str, *, n_dim: int = LORENZ63_STATE_DIM) -> np.ndarray:
	if key not in bundle.files:
		raise ValueError(f"Lorenz-63 archive is missing {key!r}")
	series = np.asarray(bundle[key], dtype=np.float64)
	if series.ndim != 2 or int(series.shape[1]) != int(n_dim):
		raise ValueError(f"{key} must have shape (T, {n_dim}), got {series.shape}")
	return series


def load_lorenz63_dataset(path: str | Path) -> dict[str, Any]:
	resolved = Path(path)
	if not resolved.is_file():
		raise FileNotFoundError(f"Missing Lorenz-63 training dataset: {resolved}")
	with np.load(resolved) as bundle:
		mean = np.asarray(bundle["mean"], dtype=np.float64).reshape(-1)
		std = np.asarray(bundle["std"], dtype=np.float64).reshape(-1)
		if mean.shape != (LORENZ63_STATE_DIM,) or std.shape != (LORENZ63_STATE_DIM,) or np.any(std <= 0.0):
			raise ValueError(f"Lorenz-63 standardizer must be length-3 with positive std, got {mean}/{std}")
		out = {
			"train_obs": np.asarray(bundle["X_train_obs"], dtype=np.float32),
			"test_obs": np.asarray(bundle["X_test_obs"], dtype=np.float32),
			"train_clean": np.asarray(bundle["X_train_clean"], dtype=np.float32),
			"test_clean": np.asarray(bundle["X_test_clean"], dtype=np.float32),
			"train_obs_raw": np.asarray(bundle["X_train_obs_raw"], dtype=np.float32),
			"test_obs_raw": np.asarray(bundle["X_test_obs_raw"], dtype=np.float32),
			"train_clean_raw": np.asarray(bundle["X_train_clean_raw"], dtype=np.float32),
			"test_clean_raw": np.asarray(bundle["X_test_clean_raw"], dtype=np.float32),
			"standardizer": {"mean": mean, "std": std},
			"path": str(resolved),
		}
	for key in ("train_obs", "test_obs", "train_clean", "test_clean"):
		series = out[key]
		if series.ndim != 2 or int(series.shape[1]) != LORENZ63_STATE_DIM:
			raise ValueError(f"{key} must have shape (T, 3), got {series.shape}")
	return out


def load_lorenz63_evaluation(
	heldout_path: str | Path,
	stochastic_path: str | Path,
	reference_path: str | Path,
) -> dict[str, Any]:
	"""Load the held-out test, stochastic evaluation series, and Lyapunov reference."""
	with np.load(heldout_path) as heldout, np.load(stochastic_path) as stochastic, np.load(reference_path) as reference:
		test_obs_raw = _require_series(heldout, "X_test_obs_raw")
		test_clean_raw = _require_series(heldout, "X_test_clean_raw")
		stochastic_obs_raw = _require_series(stochastic, "X_stochastic_obs_raw")
		stochastic_clean_raw = _require_series(stochastic, "X_stochastic_clean_raw")
		test_core_raw = _require_series(reference, "X_test_core_raw")
		les_true = np.asarray(reference["les_true"], dtype=np.float64).reshape(-1)
	if test_obs_raw.shape != test_clean_raw.shape:
		raise ValueError(
			f"Held-out noisy and clean series must match, got {test_obs_raw.shape} vs {test_clean_raw.shape}"
		)
	if stochastic_obs_raw.shape != stochastic_clean_raw.shape:
		raise ValueError(
			"Stochastic noisy/clean trajectories must match, "
			f"got {stochastic_obs_raw.shape} vs {stochastic_clean_raw.shape}"
		)
	if les_true.size < 1 or not np.all(np.isfinite(les_true)):
		raise ValueError(f"les_true must be a finite spectrum, got {les_true}")
	return {
		"test_obs_raw": test_obs_raw,
		"test_clean_raw": test_clean_raw,
		"stochastic_obs_raw": stochastic_obs_raw,
		"stochastic_clean_raw": stochastic_clean_raw,
		"test_core_raw": test_core_raw,
		"les_true": les_true,
	}


def standardize_raw_series(raw: np.ndarray, standardizer: dict[str, Any]) -> np.ndarray:
	mean = np.asarray(standardizer["mean"], dtype=np.float64).reshape(1, -1)
	std = np.asarray(standardizer["std"], dtype=np.float64).reshape(1, -1)
	return (np.asarray(raw, dtype=np.float64) - mean) / std


def unstandardize_series(series_std: np.ndarray, standardizer: dict[str, Any]) -> np.ndarray:
	mean = np.asarray(standardizer["mean"], dtype=np.float64).reshape(1, -1)
	std = np.asarray(standardizer["std"], dtype=np.float64).reshape(1, -1)
	return np.asarray(series_std, dtype=np.float64) * std + mean
