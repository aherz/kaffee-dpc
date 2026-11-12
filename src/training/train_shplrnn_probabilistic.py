"""Probabilistic shell training for clipped shPLRNNs."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from filtering.ekf import innovation_log_likelihood, innovation_statistics, kalman_predict, kalman_update
from metrics import state_space_divergence
from models.shplrnn import LinearObservation, ClippedShallowPLRNN
from models.shplrnn_shell import METHOD_SPECS, ShallowPLRNNShell, stochastic_rollout


@torch.no_grad()
def _initial_noise(
	core: ClippedShallowPLRNN,
	observation: LinearObservation,
	train: np.ndarray,
	*,
	observation_fraction: float,
	process_fraction: float,
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
	data = np.asarray(train, dtype=np.float32)
	population = data.shape[0] - 1
	rng = np.random.default_rng(0)
	starts = np.sort(rng.choice(population, size=min(4096, population), replace=False))
	residuals = []
	for start in starts:
		x = torch.as_tensor(data[start], dtype=torch.float32, device=device).unsqueeze(0)
		target = torch.as_tensor(data[start + 1], dtype=torch.float32, device=device).unsqueeze(0)
		predicted = observation(core(observation.inverse(x)))
		residuals.append((target - predicted).squeeze(0))
	variance = np.var(torch.stack(residuals).cpu().numpy().astype(np.float64), axis=0, ddof=0, dtype=np.float64)
	obs_fraction, proc_fraction = float(observation_fraction), float(process_fraction)
	if not np.isclose(obs_fraction + proc_fraction, 1.0, atol=1e-12, rtol=0):
		raise ValueError("Initial noise fractions must sum to one")
	b = observation.B.detach().cpu().numpy().astype(np.float64)
	proc_variance = proc_fraction * np.sum(variance, dtype=np.float64) / np.sum(b * b, dtype=np.float64)
	proc_std = np.full(core.latent_dim, np.sqrt(proc_variance), dtype=np.float64).astype(np.float32)
	obs_std = np.sqrt(obs_fraction * variance).astype(np.float32)
	if min(float(proc_std.min()), float(obs_std.min())) < 1e-3:
		raise ValueError("Initial noise standard deviation is below 1e-3")
	return torch.from_numpy(proc_std), torch.from_numpy(obs_std)


def _sample_windows(data: np.ndarray, *, total_steps: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
	max_start = data.shape[0] - int(total_steps) - 1
	starts = rng.integers(0, max_start, size=int(batch_size))
	return np.stack([data[start : start + int(total_steps) + 1] for start in starts]).astype(np.float32)


def _stabilize_open_loop(omega: torch.Tensor, *, floor: float) -> torch.Tensor:
	symmetric = 0.5 * (omega + omega.transpose(-1, -2))
	work = symmetric.to(torch.float64)
	minimum = torch.linalg.eigvalsh(work)[..., 0]
	diagonal_scale = torch.amax(torch.abs(torch.diagonal(work, dim1=-2, dim2=-1)), dim=-1)
	target = torch.clamp(1e-6 * diagonal_scale, min=float(floor))
	shift = torch.clamp(target - minimum, min=0).to(symmetric.dtype)
	eye = torch.eye(omega.shape[-1], device=omega.device, dtype=omega.dtype)
	return symmetric + shift[..., None, None] * eye


def gaussian_window_nll(shell: ShallowPLRNNShell, windows: torch.Tensor, *, burn_in: int, objective: str, jitter: float = 1e-4) -> torch.Tensor:
	z_mean, sigma = shell.initialize(windows[:, 0])
	b = shell.observation.B
	total = torch.zeros((), device=windows.device, dtype=windows.dtype)
	for step in range(windows.shape[1] - 1):
		z_pred, jacobian = shell.transition_and_jacobian(z_mean)
		z_pred, sigma_pred = kalman_predict(
			z_mean,
			sigma,
			Gamma=shell.process_covariance,
			z_mean_pred=z_pred,
			J_t=jacobian,
			covariance_transport=shell.covariance_transport,
		)
		x_next = windows[:, step + 1]
		if objective == "filtered":
			z_mean, sigma, log_likelihood = kalman_update(
				z_pred,
				sigma_pred,
				x_next,
				B=b,
				Lambda=shell.observation_covariance,
				Omega_jitter=jitter,
			)
		elif objective == "open_loop":
			residual, omega = innovation_statistics(
				z_pred,
				sigma_pred,
				x_next,
				B=b,
				Lambda=shell.observation_covariance,
				Omega_jitter=jitter,
			)
			omega = _stabilize_open_loop(omega, floor=jitter)
			log_likelihood = innovation_log_likelihood(residual, omega)
			z_mean, sigma = z_pred, sigma_pred
		else:
			raise ValueError(f"Unknown objective: {objective}")
		if step >= int(burn_in):
			total = total - log_likelihood.sum()
	return total


def train_shplrnn_probabilistic(
	train: np.ndarray,
	*,
	config: dict[str, Any],
	method: str,
	seed: int,
	core_dir: str | Path,
	output_dir: str | Path,
	device: torch.device | str = "cpu",
	smoke: bool = False,
) -> dict[str, Any]:
	np.random.seed(int(10_000 + seed))
	torch.manual_seed(int(10_000 + seed))
	device = torch.device(device)
	model_config = dict(config["model"])
	training = copy.deepcopy(config["shell_training"])
	core_state = torch.load(Path(core_dir) / "core.pt", map_location="cpu", weights_only=True)
	observation_state = torch.load(Path(core_dir) / "observation.pt", map_location="cpu", weights_only=True)
	core = ClippedShallowPLRNN(int(model_config["latent_dim"]), int(model_config["hidden_dim"]))
	observation = LinearObservation(int(config["data"]["observed_dim"]), int(model_config["latent_dim"]))
	core.load_state_dict(core_state); observation.load_state_dict(observation_state)
	core = core.to(device).eval(); observation = observation.to(device).eval()
	proc_std, obs_std = _initial_noise(
		core,
		observation,
		train,
		observation_fraction=float(training["initial_observation_variance_fraction"]),
		process_fraction=float(training["initial_process_variance_fraction"]),
		device=device,
	)
	shell = ShallowPLRNNShell.from_states(
		core_state,
		observation_state,
		latent_dim=int(model_config["latent_dim"]),
		hidden_dim=int(model_config["hidden_dim"]),
		observed_dim=int(config["data"]["observed_dim"]),
		proc_std=proc_std,
		obs_std=obs_std,
		method=METHOD_SPECS[method],
	).to(device)
	core_parameters = [value for value in shell.core.parameters() if value.requires_grad]
	groups = []
	if core_parameters:
		groups.append({"params": core_parameters, "lr": float(training["core_lr"]), "name": "core"})
	groups.append({"params": [shell.log_proc_std, shell.log_obs_std], "lr": float(training["noise_lr"]), "name": "noise"})
	optimizer = torch.optim.Adam(groups, weight_decay=float(training["weight_decay"]))
	schedule_steps = int(training["schedule_total_iterations"])
	minimum = float(training["minimum_lr_factor"])
	scheduler = torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		lr_lambda=lambda step: minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * min(float(step) / schedule_steps, 1.0))),
	)
	selector = dict(training["selector"])
	selector_steps = min(int(selector["rollout_steps"]), train.shape[0])
	selector_burn = int(np.floor(float(selector["burn_fraction"]) * selector_steps))
	train_rng = np.random.default_rng(1000 + int(seed))
	iterations = 1 if smoke else int(training["iterations"])
	trace_every = 1 if smoke else int(training["checkpoint_every_iterations"])
	best_score, best_iteration, best_bundle = float("inf"), -1, None
	trace = []
	for iteration in range(1, iterations + 1):
		shell.train()
		windows = torch.as_tensor(
			_sample_windows(
				train,
				total_steps=int(training["window_total_steps"]),
				batch_size=int(training["batch_size"]),
				rng=train_rng,
			),
			device=device,
		)
		optimizer.zero_grad(set_to_none=True)
		total_nll = gaussian_window_nll(
			shell,
			windows,
			burn_in=int(training["covariance_burn_in_steps"]),
			objective=METHOD_SPECS[method].objective,
			jitter=float(training["innovation_jitter"]),
		)
		loss = total_nll / (int(training["batch_size"]) * int(training["scored_steps"]))
		loss.backward()
		trainable = [parameter for group in optimizer.param_groups for parameter in group["params"]]
		torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip"]), error_if_nonfinite=True)
		optimizer.step(); scheduler.step()
		if iteration == 1 or iteration % trace_every == 0 or iteration == iterations:
			shell.eval()
			generated = stochastic_rollout(
				shell,
				train[0],
				burn_steps=selector_burn,
				score_steps=selector_steps,
				seed=int(selector["rollout_seed_base"]) + int(seed),
			)
			score = state_space_divergence(
				generated,
				train[:selector_steps],
				sigma2=float(selector["sigma2"]),
				max_T=selector_steps,
				mc_samples=int(selector["mc_samples"]),
				seed=int(selector["metric_seed_base"]) + int(seed),
			)
			trace.append({"iteration": iteration, "train_nll_per_step": float(loss.item()), "training_reference_d_stsp": score})
			if np.isfinite(score) and score <= best_score:
				best_score, best_iteration = float(score), int(iteration)
				best_bundle = shell.clone_bundle()
	if best_bundle is None:
		raise RuntimeError("No finite shell checkpoint candidate")
	output = Path(output_dir)
	output.mkdir(parents=True, exist_ok=True)
	torch.save(best_bundle, output / "shell_bundle.pt")
	summary = {
		"seed": int(seed),
		"method": method,
		"iterations": int(iterations),
		"selected_iteration": best_iteration,
		"selected_training_d_stsp": best_score,
		"smoke": bool(smoke),
		"trace": trace,
	}
	(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
	return summary
