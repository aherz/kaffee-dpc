"""Create datasets for DynaMix in-context adaptation/evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class DynaMixWindowDataset:
	context: torch.Tensor
	windows_train: list[torch.Tensor]
	windows_held: list[torch.Tensor]
	train_starts: list[int]
	held_starts: list[int]
	mean: torch.Tensor
	std: torch.Tensor
	raw_train: np.ndarray
	raw_test: np.ndarray


def build_dynamix_window_dataset(
	x_raw: np.ndarray,
	model: torch.nn.Module,
	*,
	context_length: int,
	window_length: int,
	stride: int,
	device: torch.device | str,
	covariance_burn_in_steps: int = 8,
) -> DynaMixWindowDataset:
	"""Build (in-context) DynaMix windows.

	Moments are mean and standard deviation of the noisy context, x[:context_length].
	Each window holds one initial observation, then scores the window length after a covariance burn-in phase.
	"""
	from dynamix.model.preprocessing import DataPreprocessor

	x = np.asarray(x_raw, dtype=np.float32)
	if x.ndim != 2:
		raise ValueError(f"x_raw must have shape (T, N), got {x.shape}")
	dev = torch.device(device)
	dtype = next(model.parameters()).dtype
	total_t, _n_dim = x.shape
	raw_window_length = int(window_length) + int(covariance_burn_in_steps) + 1
	train_end = min(max(int(context_length), 1), int(total_t))
	mean = x[:train_end].mean(axis=0)
	std = x[:train_end].std(axis=0)
	std = np.where(std < 1.0e-6, 1.0, std)
	x_std = ((x - mean) / std).astype(np.float32)
	mean_t = torch.as_tensor(mean, dtype=dtype, device=dev)
	std_t = torch.as_tensor(std, dtype=dtype, device=dev)

	context_raw = torch.as_tensor(x_std[: int(context_length)], dtype=dtype, device=dev).unsqueeze(1)
	preprocessor = DataPreprocessor(standardize=False, preprocessing_method="pos_embedding")
	context, _init_cond = preprocessor.preprocess(context_raw, int(model.N))

	raw_warmup_steps = max(16, int(covariance_burn_in_steps) * 2)
	train_min_start = raw_warmup_steps
	train_max_start = max(train_min_start, int(context_length) - raw_window_length)
	train_starts = list(range(train_min_start, train_max_start, int(stride)))
	held_min_start = int(context_length)
	held_max_start = max(held_min_start, int(total_t) - raw_window_length)
	held_starts = list(range(held_min_start, held_max_start, int(stride)))

	def _collect(starts: list[int]) -> tuple[list[torch.Tensor], list[int]]:
		windows: list[torch.Tensor] = []
		kept_starts: list[int] = []
		for start in starts:
			win = torch.as_tensor(x_std[start : start + raw_window_length], dtype=dtype, device=dev)
			if int(win.shape[0]) != raw_window_length:
				continue
			windows.append(win)
			kept_starts.append(int(start))
		return windows, kept_starts

	windows_train, kept_train_starts = _collect(train_starts)
	windows_held, kept_held_starts = _collect(held_starts)
	return DynaMixWindowDataset(
		context=context,
		windows_train=windows_train,
		windows_held=windows_held,
		train_starts=kept_train_starts,
		held_starts=kept_held_starts,
		mean=mean_t,
		std=std_t,
		raw_train=x[:train_end],
		raw_test=x[train_end:],
	)


def estimate_dynamix_tf_residual_noise_init(
	x_raw: np.ndarray,
	model: torch.nn.Module,
	*,
	context_length: int,
	device: torch.device | str,
	obs_variance_fraction: float = 0.9,
	proc_variance_fraction: float = 0.1,
	fallback_log_sigma: float = -1.0,
	zero_shot_warmup: int = 100,
) -> dict[str, Any]:
	"""Estimate initial noise scales from teacher-forced residuals."""
	from dynamix.model.preprocessing import DataPreprocessor

	total_fraction = float(obs_variance_fraction) + float(proc_variance_fraction)
	if total_fraction <= 0.0:
		raise ValueError("obs_variance_fraction + proc_variance_fraction must be positive")
	obs_fraction = float(obs_variance_fraction) / total_fraction
	proc_fraction = float(proc_variance_fraction) / total_fraction

	x = np.asarray(x_raw, dtype=np.float32)
	if x.ndim != 2:
		raise ValueError(f"x_raw must have shape (T, N), got {x.shape}")
	dev = torch.device(device)
	dtype = next(model.parameters()).dtype
	t_total, n_dim = x.shape
	moments_end = int(min(context_length, t_total))
	residual_start = int(max(0, min(int(zero_shot_warmup), int(context_length) - 2)))
	residual_stop = int(min(int(context_length) - 1, t_total - 1))
	mean = x[:moments_end].mean(axis=0)
	std = x[:moments_end].std(axis=0)
	std = np.where(std < 1.0e-6, 1.0, std)
	x_std = ((x - mean) / std).astype(np.float32)

	context_raw = torch.as_tensor(x_std[: int(context_length)], dtype=dtype, device=dev).unsqueeze(1)
	preprocessor = DataPreprocessor(standardize=False, preprocessing_method="pos_embedding")
	context, init_cond = preprocessor.preprocess(context_raw, int(model.N))
	precomputed_cnn = model.precompute_cnn(context)
	residuals: list[torch.Tensor] = []
	was_training = model.training
	model.eval()
	with torch.no_grad():
		z = model.B.data.T @ init_cond.T
		z[:n_dim, :] = init_cond[:, :n_dim].T
		for pos in range(residual_start, residual_stop):
			obs_t = torch.as_tensor(x_std[pos], dtype=dtype, device=dev)
			z[:n_dim, 0] = obs_t
			z_pred = model.step(z, context, precomputed_cnn=precomputed_cnn)
			target_next = torch.as_tensor(x_std[pos + 1], dtype=dtype, device=dev)
			residuals.append((z_pred[:n_dim, 0] - target_next).detach().cpu())
			z = z_pred
	if was_training:
		model.train()

	fallback_var = math.exp(float(fallback_log_sigma)) ** 2 # If no residuals available, use this variance
	if residuals:
		residual_tensor = torch.stack(residuals, dim=0)
		residual_var = float(residual_tensor.pow(2).mean().item())
		residual_std_per_dim = residual_tensor.pow(2).mean(dim=0).sqrt().tolist()
		source = "teacher_forced_residual"
		n_steps = int(residual_tensor.shape[0])
	else:
		residual_var = float(fallback_var)
		residual_std_per_dim = [math.sqrt(fallback_var) for _ in range(n_dim)]
		source = "fallback_default"
		n_steps = 0
	residual_std = float(math.sqrt(max(residual_var, 1.0e-12)))
	sigma_proc = float(math.sqrt(max(proc_fraction * residual_var, 1.0e-12)))
	sigma_obs = float(math.sqrt(max(obs_fraction * residual_var, 1.0e-12)))
	return {
		"source": source,
		"space": "standardized_observation",
		"n_steps": n_steps,
		"context_length": int(context_length),
		"train_end": int(moments_end),
		"zero_shot": True,
		"residual_start": int(residual_start),
		"residual_stop": int(residual_stop),
		"residual_std": residual_std,
		"residual_var": float(residual_var),
		"residual_std_per_dim": [float(v) for v in residual_std_per_dim],
		"proc_variance_fraction": proc_fraction,
		"obs_variance_fraction": obs_fraction,
		"sigma_proc": sigma_proc,
		"sigma_obs": sigma_obs,
		"log_sigma_proc": float(math.log(max(sigma_proc, 1.0e-12))),
		"log_sigma_obs": float(math.log(max(sigma_obs, 1.0e-12))),
	}


__all__ = [
	"DynaMixWindowDataset",
	"build_dynamix_window_dataset",
	"estimate_dynamix_tf_residual_noise_init",
]
