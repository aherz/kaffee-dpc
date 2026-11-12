"""Metrics for predictive fidelity of the surrogate models. 

We use Gaussian negative log-likelihood (NLL) and continuous ranked probability score (CRPS) to evaluate the quality of probabilistic forecasts. CRPS assesses the quality of the predicted CDF against the observed value. Both metrics are averaged across forecast steps and coordinates. The Gaussian NLL is computed under a diagonal covariance assumption for simplicity, while the CRPS is computed using an ensemble of forecasts to approximate the marginal distribution. So both metrics quantify marginal calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PredictiveEvalConfig:
	forecast_horizon: int
	forecast_starts: int
	nll_starts: int
	forecast_ensemble: int
	obs_floor: float


def forecast_starts(total_steps: int, horizon: int, n_starts: int) -> np.ndarray:
	max_start = int(total_steps - horizon - 1)
	if max_start < 0:
		return np.asarray([], dtype=np.int64)
	if int(n_starts) >= max_start + 1:
		return np.arange(max_start + 1, dtype=np.int64)
	return np.unique(np.linspace(0, max_start, num=int(n_starts), dtype=np.int64))


def gaussian_nll_diag(target: np.ndarray, mean: np.ndarray, std: np.ndarray, *, floor: float) -> float:
	target = np.asarray(target, dtype=np.float64)
	mean = np.asarray(mean, dtype=np.float64)
	std = np.maximum(np.asarray(std, dtype=np.float64), float(floor))
	variance = std * std
	return float(np.mean(0.5 * np.log(2.0 * np.pi * variance) + 0.5 * ((target - mean) ** 2) / variance))


def ensemble_path_nll(target_path: np.ndarray, ensemble_path: np.ndarray, *, floor: float) -> float:
	target_path = np.asarray(target_path, dtype=np.float64)
	ensemble_path = np.asarray(ensemble_path, dtype=np.float64)
	if target_path.ndim != 2:
		raise ValueError(f"target_path must have shape (T, D), got {target_path.shape}")
	if ensemble_path.ndim != 3:
		raise ValueError(f"ensemble_path must have shape (E, T, D), got {ensemble_path.shape}")
	if ensemble_path.shape[1:] != target_path.shape:
		raise ValueError(
			f"ensemble_path trailing shape must match target_path, got {ensemble_path.shape[1:]} and {target_path.shape}"
		)
	scores: list[float] = []
	for step in range(int(target_path.shape[0])):
		members = ensemble_path[:, step, :]
		mean = np.mean(members, axis=0)
		if members.shape[0] > 1:
			std = np.std(members, axis=0, ddof=1)
		else:
			std = np.full_like(mean, float(floor))
		scores.append(gaussian_nll_diag(target_path[step], mean, std, floor=float(floor)))
	return float(np.mean(scores)) if scores else float("nan")


def marginal_crps_ensemble(target: np.ndarray, ensemble: np.ndarray) -> float:
	target = np.asarray(target, dtype=np.float64).reshape(-1)
	ensemble = np.asarray(ensemble, dtype=np.float64)
	if ensemble.ndim != 2:
		raise ValueError(f"ensemble must have shape (E, D), got {ensemble.shape}")
	if ensemble.shape[1] != target.shape[0] or ensemble.shape[0] < 2:
		return float("nan")
	diffs_to_truth = np.abs(ensemble - target.reshape(1, -1))
	diffs_pairwise = np.abs(ensemble[:, None, :] - ensemble[None, :, :])
	crps_per_dim = np.mean(diffs_to_truth, axis=0) - 0.5 * np.mean(diffs_pairwise, axis=(0, 1))
	return float(np.mean(crps_per_dim))


@torch.no_grad()
def rollout_probabilistic_ensemble(
	model: torch.nn.Module,
	x0: np.ndarray | torch.Tensor,
	*,
	length: int,
	n_ensemble: int,
	seed: int,
	sample_process_noise: bool = True,
	sample_obs_noise: bool = True,
) -> np.ndarray:
	"""Ensemble of open-loop forecast trajectories for NLL and CRPS.

	Each member uses the core (e.g. AL-RNN) step with the shell's process and observation noise.
	"""
	rollout = getattr(model, "rollout_observation_ensemble", None)
	if rollout is not None:
		return np.asarray(
			rollout(
				x0,
				length=int(length),
				n_ensemble=int(n_ensemble),
				seed=int(seed),
				sample_process_noise=bool(sample_process_noise),
				sample_obs_noise=bool(sample_obs_noise),
			),
			dtype=np.float64,
		)
	raise TypeError("rollout_probabilistic_ensemble requires a model with rollout_observation_ensemble")


__all__ = [
	"PredictiveEvalConfig",
	"ensemble_path_nll",
	"forecast_starts",
	"gaussian_nll_diag",
	"marginal_crps_ensemble",
	"rollout_probabilistic_ensemble",
]