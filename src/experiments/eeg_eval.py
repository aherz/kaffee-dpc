"""Score EEG cores and shells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import norm

from filtering.ekf import innovation_statistics, kalman_predict, kalman_update
from metrics import state_space_divergence
from metrics.power_spectrum_error import power_spectrum_error
from models.shplrnn import ClippedShallowPLRNN, LinearObservation
from models.shplrnn_shell import ShallowPLRNNShell, core_rollout, stochastic_rollout


ROLLOUT_SEEDS = (12345, 12346, 12347, 12348)


def _evaluation(config: dict[str, Any]) -> dict[str, Any]:
	evaluation = dict(config["evaluation"])
	if int(evaluation["nll_horizon_steps"]) != 1:
		raise ValueError("EEG scoring is the one-step Kalman filter")
	return evaluation


def _write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_suffix(".json.tmp")
	temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
	temporary.replace(path)


@torch.no_grad()
def score_eeg_shell(
	shell: ShallowPLRNNShell,
	data: np.ndarray,
	*,
	subject: str,
	seed: int,
	method: str,
	config: dict[str, Any],
) -> dict[str, Any]:
	evaluation = _evaluation(config)
	history = int(evaluation["filter_history_steps"])
	jitter = float(evaluation["innovation_jitter"])
	burn = int(evaluation["rollout_burn_steps"])
	score_steps = int(evaluation["rollout_score_steps"])
	mc_samples = int(evaluation.get("mc_samples", 1000))
	observations_np = np.asarray(data)
	if observations_np.shape != (score_steps, int(config["data"]["observed_dim"])):
		raise ValueError(
			f"EEG scoring expected {(score_steps, int(config['data']['observed_dim']))}, got {observations_np.shape}"
		)
	observations = torch.as_tensor(observations_np, dtype=shell.core.A.dtype, device=shell.core.A.device)
	z_mean, sigma = shell.initialize(observations[0:1])
	observation_matrix = shell.observation.B
	transport = str(shell.covariance_transport).strip().lower()
	nll_terms: list[float] = []
	residuals: list[np.ndarray] = []
	standard_deviations: list[np.ndarray] = []
	for step in range(observations.shape[0] - 1):
		z_pred, jacobian = shell.transition_and_jacobian(z_mean)
		z_pred, sigma_pred = kalman_predict(
			z_mean,
			sigma,
			Gamma=shell.process_covariance,
			z_mean_pred=z_pred,
			J_t=jacobian,
			covariance_transport=transport,
		)
		target = observations[step + 1 : step + 2]
		residual, omega = innovation_statistics(
			z_pred,
			sigma_pred,
			target,
			B=observation_matrix,
			Lambda=shell.observation_covariance,
			Omega_jitter=jitter,
		)
		z_mean, sigma, log_likelihood = kalman_update(
			z_pred,
			sigma_pred,
			target,
			B=observation_matrix,
			Lambda=shell.observation_covariance,
			Omega_jitter=jitter,
		)
		if step + 1 >= history:
			nll_terms.append(float((-log_likelihood).detach().reshape(-1).cpu()[0]))
			residuals.append(residual.detach().cpu().numpy().astype(np.float64).reshape(-1))
			standard_deviations.append(
				torch.sqrt(torch.clamp(torch.diagonal(omega, dim1=-2, dim2=-1), min=1e-12))
				.detach()
				.cpu()
				.numpy()
				.astype(np.float64)
				.reshape(-1)
			)
	if len(nll_terms) != observations.shape[0] - history:
		raise RuntimeError("EEG score target count is wrong")
	residual_np = np.stack(residuals)
	std = np.stack(standard_deviations)
	z_score = residual_np / np.maximum(std, 1e-12)
	crps = std * (z_score * (2.0 * norm.cdf(z_score) - 1.0) + 2.0 * norm.pdf(z_score) - 1.0 / np.sqrt(np.pi))
	reference = np.asarray(observations_np, dtype=np.float64)
	state_scores: list[float] = []
	spectral_scores: list[float] = []
	for rollout_seed in ROLLOUT_SEEDS:
		generated = stochastic_rollout(
			shell,
			observations_np[0],
			burn_steps=burn,
			score_steps=score_steps,
			seed=int(rollout_seed),
		)
		state_scores.append(
			float(state_space_divergence(generated, reference, sigma2=1.0, max_T=score_steps, mc_samples=mc_samples, seed=0))
		)
		spectral_scores.append(float(power_spectrum_error(reference, generated, smoothing=20)))
	noise_free = core_rollout(
		shell.core,
		shell.observation,
		observations_np[0],
		burn_steps=burn,
		score_steps=score_steps,
	)
	return {
		"subject": subject,
		"seed": int(seed),
		"method": method,
		"NLL": float(np.mean(nll_terms) / observations_np.shape[1]),
		"CRPS": float(np.mean(crps)),
		"scoring_update": "kalman",
		"nll_horizon_steps": 1,
		"rollout_burn_steps": burn,
		"rollout_score_steps": score_steps,
		"D_stsp_stoch_repeats": state_scores,
		"D_H_stoch_repeats": spectral_scores,
		"D_stsp_core": float(
			state_space_divergence(noise_free, reference, sigma2=1.0, max_T=score_steps, mc_samples=mc_samples, seed=0)
		),
		"D_H_core": float(power_spectrum_error(reference, noise_free, smoothing=20)),
		"core_rollout_device": str(shell.core.A.device),
	}


def write_shell_eval(path: Path, payload: dict[str, Any]) -> None:
	_write_json(Path(path), payload)


@torch.no_grad()
def score_eeg_core(
	core: ClippedShallowPLRNN,
	observation: LinearObservation,
	data: np.ndarray,
	*,
	config: dict[str, Any],
) -> dict[str, Any]:
	evaluation = _evaluation(config)
	burn = int(evaluation["rollout_burn_steps"])
	score_steps = int(evaluation["rollout_score_steps"])
	mc_samples = int(evaluation.get("mc_samples", 1000))
	reference = np.asarray(data, dtype=np.float64)
	if reference.shape != (score_steps, int(config["data"]["observed_dim"])):
		raise ValueError(f"EEG core scoring expected length {score_steps}, got {reference.shape}")
	generated = core_rollout(core, observation, reference[0], burn_steps=burn, score_steps=score_steps)
	return {
		"device": str(core.A.device),
		"D_H": float(power_spectrum_error(reference, generated, smoothing=20)),
		"D_stsp": float(
			state_space_divergence(generated, reference, sigma2=1.0, max_T=score_steps, mc_samples=mc_samples, seed=0)
		),
		"compared_steps": int(score_steps),
		"discarded_steps": int(burn),
	}


def write_core_metrics(path: Path, payload: dict[str, Any]) -> None:
	_write_json(Path(path), payload)
