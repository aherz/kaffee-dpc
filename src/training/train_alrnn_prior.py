"""Training for ALRNN-cores ("deterministic priors") via BPTT with identity teacher forcing (ITF)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from metrics import power_spectrum_error, state_space_divergence_binning
from models import ALRNN, WindowedTimeSeriesDataset, rollout_free, rollout_itf


DEFAULT_SELECTION_EVAL_EVERY = 100 # Interval for model selection/early stopping
DEFAULT_SELECTION_ROLLOUT_STEPS = 10_000 # Trajectory length for eval of model selection criteria

def _seed_everything(seed: int) -> None:
	np.random.seed(int(seed))
	torch.manual_seed(int(seed))
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(int(seed))

@torch.no_grad()
def _teacher_forced_test_mse(
	model: ALRNN,
	test_obs: np.ndarray,
	*,
	forcing_interval: int,
	observed_dim: int,
	device: torch.device,
) -> float:
	x_obs_prev = torch.as_tensor(test_obs[:-1], dtype=torch.float32, device=device).unsqueeze(0)
	x_obs_next = torch.as_tensor(test_obs[1:], dtype=torch.float32, device=device).unsqueeze(0)
	x_obs_pred = rollout_itf(model, x_obs_prev, forcing_interval=int(forcing_interval))
	return float(torch.mean((x_obs_pred[:, :, :observed_dim] - x_obs_next) ** 2).item())


@torch.no_grad()
def _free_rollout_test_mse(
	model: ALRNN,
	test_obs: np.ndarray,
	*,
	observed_dim: int,
	horizon: int,
	device: torch.device,
) -> float:
	horizon = min(int(horizon), int(test_obs.shape[0] - 1))
	x_obs_init = torch.as_tensor(test_obs[:1], dtype=torch.float32, device=device)
	x_obs_target = torch.as_tensor(test_obs[1 : horizon + 1], dtype=torch.float32, device=device).unsqueeze(0)
	x_obs_pred = rollout_free(model, x_obs_init, steps=horizon)
	return float(torch.mean((x_obs_pred[:, :, :observed_dim] - x_obs_target) ** 2).item())


@torch.no_grad()
def _selection_metrics(
	model: ALRNN,
	train_obs: np.ndarray,
	train_clean_raw: np.ndarray,
	*,
	mean: np.ndarray,
	std: np.ndarray,
	observed_dim: int,
	rollout_steps: int,
	state_space_sigma2: float,
	state_space_mc_samples: int,
	state_space_density_batch_size: int,
	pse_smoothing: int,
	device: torch.device,
) -> dict[str, float]:
	rollout_steps = min(int(rollout_steps), int(train_clean_raw.shape[0]))
	x0_std = torch.as_tensor(train_obs[:1], dtype=torch.float32, device=device)
	z_gen = rollout_free(model, x0_std, steps=rollout_steps)[0].detach().cpu().numpy()

	mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
	std = np.asarray(std, dtype=np.float64).reshape(1, -1)
	reference_raw = np.asarray(train_clean_raw[:rollout_steps], dtype=np.float64)
	generated_std = np.asarray(z_gen[: reference_raw.shape[0], :observed_dim], dtype=np.float64)
	generated_raw = generated_std * std + mean

	if int(observed_dim) > 5:
		reference_state_space = (reference_raw - mean) / std
		generated_state_space = generated_std
	else:
		reference_state_space = reference_raw
		generated_state_space = generated_raw

	d_stsp = state_space_divergence_binning(
		generated_state_space,
		reference_state_space,
		sigma2=float(state_space_sigma2),
		max_T=int(rollout_steps),
		mc_samples=int(state_space_mc_samples),
		batch_size=int(state_space_density_batch_size),
	)
	hellinger_ps = power_spectrum_error(
		reference_raw,
		generated_raw,
		smoothing=int(pse_smoothing),
	)
	return {
		"D_stsp": float(d_stsp),
		"HellingerPS": float(hellinger_ps),
	}


def train_alrnn_prior(
	dataset: dict[str, Any],
	protocol: dict[str, Any],
	*,
	seed: int,
	device: torch.device,
	max_epochs: int | None = None,
	max_batches_per_epoch: int | None = None,
	eval_every: int | None = None,
	free_rollout_horizon: int = 256,
) -> tuple[ALRNN, dict[str, Any]]:
	
	_seed_everything(int(seed))

	model_cfg = dict(protocol["model"])
	training_cfg = dict(protocol["training"])
	eval_cfg = dict(protocol.get("evaluation", {}))

	train_obs = np.asarray(dataset["train_obs"], dtype=np.float32)
	test_obs = np.asarray(dataset["test_obs"], dtype=np.float32)
	train_clean_raw = np.asarray(dataset["train_clean_raw"], dtype=np.float64)
	mean = np.asarray(dataset["standardizer"]["mean"], dtype=np.float64)
	std = np.asarray(dataset["standardizer"]["std"], dtype=np.float64)

	model = ALRNN(
		latent_dim=int(model_cfg["latent_dim"]),
		gated_dim=int(model_cfg["gated_dim"]),
		observed_dim=int(model_cfg["observed_dim"]),
	).to(device)

	epochs = int(max_epochs) if max_epochs is not None else int(training_cfg["epochs"])
	batches_per_epoch = (
		int(max_batches_per_epoch) if max_batches_per_epoch is not None else int(training_cfg["batches_per_epoch"])
	)
	eval_every = int(eval_every) if eval_every is not None else int(DEFAULT_SELECTION_EVAL_EVERY)
	selection_rollout_steps = min(
		int(DEFAULT_SELECTION_ROLLOUT_STEPS),
		int(train_clean_raw.shape[0]),
	)

	dataset_windows = WindowedTimeSeriesDataset(
		train_obs,
		sequence_length=int(training_cfg["sequence_length"]),
		batch_size=int(training_cfg["batch_size"]),
	)

	optimizer_name = str(training_cfg.get("optimizer", "radam")).strip().lower()
	if optimizer_name == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["lr_start"]))
	elif optimizer_name == "radam":
		optimizer = torch.optim.RAdam(model.parameters(), lr=float(training_cfg["lr_start"]))
	else:
		raise ValueError(f"Unsupported core optimizer: {training_cfg.get('optimizer')!r}")
	gamma = (float(training_cfg["lr_end"]) / float(training_cfg["lr_start"])) ** (1.0 / max(1, epochs))
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
	loss_fn = torch.nn.MSELoss()
	forcing_interval = int(model_cfg["teacher_forcing_interval"])
	observed_dim = int(model_cfg["observed_dim"])

	best_state = copy.deepcopy(model.state_dict())
	best_metric = float("inf")
	best_epoch: int | None = None
	history: list[dict[str, float | int]] = []
	losses: list[float] = []

	for epoch in range(epochs):
		model.train()
		epoch_losses: list[float] = []
		for _ in range(batches_per_epoch):
			x_batch, y_batch = dataset_windows.sample_batch()
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)

			optimizer.zero_grad()
			preds = rollout_itf(model, x_batch, forcing_interval=forcing_interval)
			loss = loss_fn(preds[:, :, :observed_dim], y_batch)
			loss.backward()
			optimizer.step()
			epoch_losses.append(float(loss.item()))

		scheduler.step()
		avg_loss = float(np.mean(epoch_losses))
		losses.append(avg_loss)

		if epoch % eval_every == 0 or epoch == epochs - 1:
			model.eval()
			selection_metrics = _selection_metrics(
				model,
				train_obs,
				train_clean_raw,
				mean=mean,
				std=std,
				observed_dim=observed_dim,
				rollout_steps=selection_rollout_steps,
				state_space_sigma2=float(eval_cfg.get("state_space_sigma2", 1.0)),
				state_space_mc_samples=int(eval_cfg.get("state_space_mc_samples", 1000)),
				state_space_density_batch_size=int(eval_cfg.get("state_space_density_batch_size", 512)),
				pse_smoothing=int(eval_cfg.get("pse_smoothing", 20)),
				device=device,
			)
			tf_mse = _teacher_forced_test_mse(
				model,
				test_obs,
				forcing_interval=forcing_interval,
				observed_dim=observed_dim,
				device=device,
			)
			free_mse = _free_rollout_test_mse(
				model,
				test_obs,
				observed_dim=observed_dim,
				horizon=free_rollout_horizon,
				device=device,
			)
			row = {
				"epoch": int(epoch),
				"loss": avg_loss,
				"D_stsp": float(selection_metrics["D_stsp"]),
				"HellingerPS": float(selection_metrics["HellingerPS"]),
				"teacher_forced_test_mse": float(tf_mse),
				"free_rollout_test_mse": float(free_mse),
			}
			history.append(row)

			sel_metric = selection_metrics["D_stsp"]
			if not np.isfinite(sel_metric):
				sel_metric = selection_metrics["HellingerPS"]
			if np.isfinite(sel_metric) and float(sel_metric) < best_metric:
				best_metric = float(sel_metric)
				best_epoch = int(epoch)
				best_state = copy.deepcopy(model.state_dict())

	model.load_state_dict(best_state)
	final_selection_metrics = _selection_metrics(
		model,
		train_obs,
		train_clean_raw,
		mean=mean,
		std=std,
		observed_dim=observed_dim,
		rollout_steps=selection_rollout_steps,
		state_space_sigma2=float(eval_cfg.get("state_space_sigma2", 1.0)),
		state_space_mc_samples=int(eval_cfg.get("state_space_mc_samples", 1000)),
		state_space_density_batch_size=int(eval_cfg.get("state_space_density_batch_size", 512)),
		pse_smoothing=int(eval_cfg.get("pse_smoothing", 20)),
		device=device,
	)
	final_tf_mse = _teacher_forced_test_mse(
		model,
		test_obs,
		forcing_interval=forcing_interval,
		observed_dim=observed_dim,
		device=device,
	)
	final_free_mse = _free_rollout_test_mse(
		model,
		test_obs,
		observed_dim=observed_dim,
		horizon=free_rollout_horizon,
		device=device,
	)

	if best_epoch is None:
		best_epoch = int(epochs - 1)
		best_metric = float(final_selection_metrics["D_stsp"])
		if not np.isfinite(best_metric):
			best_metric = float(final_selection_metrics["HellingerPS"])

	return model, {
		"selection_metric": "D_stsp",
		"selection_value": float(best_metric),
		"selection_epoch": int(best_epoch),
		"final_metrics": {
			"D_stsp": float(final_selection_metrics["D_stsp"]),
			"HellingerPS": float(final_selection_metrics["HellingerPS"]),
			"teacher_forced_test_mse": float(final_tf_mse),
			"free_rollout_test_mse": float(final_free_mse),
		},
		"history": history,
		"loss_tail": losses[-10:],
		"epochs": int(epochs),
		"batches_per_epoch": int(batches_per_epoch),
		"eval_every": int(eval_every),
	}


__all__ = ["train_alrnn_prior"]