"""DynaMix evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metrics import (
	power_spectrum_error,
	state_space_divergence,
)
from models import DynaMixShell
from data.dynamix_windows import build_dynamix_window_dataset

QOI_ROLLOUT_STEPS = 55_000 
QOI_BURN_IN_STEPS = 1_000
QOI_EVAL_STEPS = 50_000
POWER_SPECTRUM_SMOOTHING = 20


@dataclass(frozen=True)
class DynaMixContextState:
	context: torch.Tensor
	precomputed_cnn: torch.Tensor
	z0: torch.Tensor
	mean: np.ndarray
	std: np.ndarray


def compute_standardization_stats(x_train_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
	x_train = np.asarray(x_train_raw, dtype=np.float64)
	if x_train.ndim != 2:
		raise ValueError(f"x_train_raw must have shape (T, N), got {x_train.shape}")
	train_end = int(x_train.shape[0])
	if train_end < 1:
		raise ValueError("x_train_raw must contain at least one observation")
	mean = np.asarray(x_train.mean(axis=0), dtype=np.float64)
	std = np.asarray(x_train.std(axis=0), dtype=np.float64)
	std = np.where(std < 1.0e-6, 1.0, std)
	return mean, std, train_end


def core_standardization_trajectory(
	x_full: np.ndarray,
	*,
	train_total_t: int,
	context_length: int,
) -> np.ndarray:
	"""Slice used to fit core-rollout mean/std for every method, including dynamix_base."""
	n = max(int(train_total_t), int(context_length) + 2)
	x = np.asarray(x_full)
	if int(x.shape[0]) < n:
		raise ValueError(f"Need {n} steps for core standardization, got {int(x.shape[0])}")
	return x[:n]


def prepare_dynamix_context_state(
	model: torch.nn.Module,
	context_raw: np.ndarray,
	*,
	mean: np.ndarray,
	std: np.ndarray,
	device: torch.device | str,
) -> DynaMixContextState:
	"""Fixed context and initial latent state embedding for DynaMix rollouts."""
	from dynamix.model.preprocessing import DataPreprocessor

	dev = torch.device(device)
	dtype = next(model.parameters()).dtype
	mu = np.asarray(mean, dtype=np.float64).reshape(1, -1)
	sd = np.asarray(std, dtype=np.float64).reshape(1, -1)
	context_std = ((np.asarray(context_raw, dtype=np.float64) - mu) / sd).astype(np.float32)
	context_tensor = torch.as_tensor(context_std, dtype=dtype, device=dev).unsqueeze(1)

	preprocessor = DataPreprocessor(standardize=False, preprocessing_method="pos_embedding")
	context_embedded, init_cond = preprocessor.preprocess(context_tensor, int(model.N))

	z0 = model.B.data.T @ init_cond.T
	z0[: int(model.N), :] = init_cond[:, : int(model.N)].T
	return DynaMixContextState(
		context=context_embedded,
		precomputed_cnn=model.precompute_cnn(context_embedded),
		z0=z0,
		mean=np.asarray(mean, dtype=np.float64).reshape(-1),
		std=np.asarray(std, dtype=np.float64).reshape(-1),
	)


@torch.no_grad()
def rollout_model_raw(
	model: torch.nn.Module,
	*,
	context_raw: np.ndarray,
	mean: np.ndarray,
	std: np.ndarray,
	horizon: int,
	device: torch.device | str,
	z_start: torch.Tensor | None = None,
	include_initial: bool = False,
) -> dict[str, np.ndarray]:
	"""Autonomous (shell-off) DynaMix rollout in raw observation coordinates.

	Default launches from the context encoder state and records one step per
	horizon index. "z_start" overrides that latent state. "include_initial=True"
	writes destandardized "z[:N]" at index 0, e.g. to have a shared ground-truth start against the ODE reference.
	"""
	model.eval()
	state = prepare_dynamix_context_state(model, context_raw, mean=mean, std=std, device=device)
	if z_start is None:
		z = state.z0.clone()
	else:
		z = z_start.to(device=state.z0.device, dtype=state.z0.dtype)
		if z.ndim == 1:
			z = z.unsqueeze(1)
		elif z.ndim == 2 and int(z.shape[1]) != 1 and int(z.shape[0]) == 1:
			z = z.T.contiguous()
	horizon_i = int(horizon)
	forecast = np.empty((horizon_i, int(model.N)), dtype=np.float32)
	latent = np.empty((horizon_i, int(model.M)), dtype=np.float32)

	def _emit(step_idx: int, z_col: torch.Tensor) -> None:
		z_np = z_col[:, 0].detach().cpu().numpy().astype(np.float64)
		latent[step_idx] = z_np.astype(np.float32)
		forecast[step_idx] = (z_np[: int(model.N)] * state.std + state.mean).astype(np.float32)

	start = 0
	if bool(include_initial):
		_emit(0, z)
		start = 1
	for step_idx in range(start, horizon_i):
		z = model.step(z, state.context, precomputed_cnn=state.precomputed_cnn)
		_emit(step_idx, z)

	return {"forecast": forecast, "latent": latent}


def compute_dynamix_core_metrics(
	forecast_raw: np.ndarray,
	reference_raw: np.ndarray,
	*,
	burn_in_steps: int = QOI_BURN_IN_STEPS,
	eval_steps: int = QOI_EVAL_STEPS,
) -> dict[str, float]:
	"""D_H and D_stsp in destandardized (physical) coordinates.

	D_stsp is histogram KL with bins from the raw reference min/max. D_H z-scores each series before the spectral Hellinger distance. Forecasts are destandardized and scored against the raw ODE attractor.
	"""
	burn = int(burn_in_steps)
	n_eval = int(eval_steps)
	pred_eval = np.asarray(forecast_raw, dtype=np.float64)[burn : burn + n_eval]
	ref_eval = np.asarray(reference_raw, dtype=np.float64)[burn : burn + n_eval]
	if pred_eval.shape != ref_eval.shape:
		raise ValueError(f"forecast/reference eval windows must match, got {pred_eval.shape} and {ref_eval.shape}")
	return {
		"D_stsp": float(state_space_divergence(pred_eval, ref_eval)),
		"D_H": float(power_spectrum_error(ref_eval, pred_eval, smoothing=POWER_SPECTRUM_SMOOTHING)),
	}


def evaluate_dynamix_core_rollout(
	model: torch.nn.Module,
	*,
	x_train_raw: np.ndarray,
	x_reference_raw: np.ndarray,
	context_length: int,
	device: torch.device | str,
	rollout_steps: int = QOI_ROLLOUT_STEPS,
	burn_in_steps: int = QOI_BURN_IN_STEPS,
	eval_steps: int = QOI_EVAL_STEPS,
	rollout_dir: Path | None = None,
	system_name: str | None = None,
	method_key: str | None = None,
) -> dict[str, Any]:
	"""Evaluate shell-off (deterministic)DSR metrics for one system/method.

	"x_train_raw" is the noisy context shared for every method. Its mean and standard deviation are used for the model's standardization, and that same series is the rollout context. Forecasts are
	destandardized and scored against the clean reference after the context.
	"""
	mean, std, train_end = compute_standardization_stats(x_train_raw)
	context_len = int(context_length)
	reference = np.asarray(x_reference_raw, dtype=np.float64)
	context = np.asarray(x_train_raw[:context_len], dtype=np.float32)
	reference_rollout = reference[context_len : context_len + int(rollout_steps)]
	if int(reference_rollout.shape[0]) < int(rollout_steps):
		raise ValueError(
			f"Reference trajectory is too short: need {int(rollout_steps)} post-context steps, "
			f"got {int(reference_rollout.shape[0])}"
		)
	torch.manual_seed(0)
	rollout = rollout_model_raw(
		model,
		context_raw=context,
		mean=mean,
		std=std,
		horizon=int(rollout_steps),
		device=device,
	)
	metrics = compute_dynamix_core_metrics(
		rollout["forecast"],
		reference_rollout,
		burn_in_steps=int(burn_in_steps),
		eval_steps=int(eval_steps),
	)
	out: dict[str, Any] = {
		**metrics,
		"D_stsp_core": float(metrics["D_stsp"]),
		"D_H_core": float(metrics["D_H"]),
		"standardization_train_end": int(train_end),
		"qoi_rollout_steps": int(rollout_steps),
		"qoi_burn_in_steps": int(burn_in_steps),
		"qoi_eval_steps": int(eval_steps),
	}
	if rollout_dir is not None and system_name is not None and method_key is not None:
		rollout_path = Path(rollout_dir)
		rollout_path.mkdir(parents=True, exist_ok=True)
		file_path = rollout_path / f"rollout_{system_name}_{method_key}.npz"
		np.savez_compressed(
			file_path,
			forecast=rollout["forecast"],
			reference=reference_rollout.astype(np.float32),
			burn_in_steps=np.asarray(burn_in_steps, dtype=np.int64),
			eval_steps=np.asarray(eval_steps, dtype=np.int64),
		)
		out["rollout_file"] = str(file_path)
	return out


@torch.no_grad()
def evaluate_held_nll(
	shell: DynaMixShell,
	*,
	z_inits: list[torch.Tensor],
	context: torch.Tensor,
	obs_windows: list[torch.Tensor],
	n_ensemble: int = 256,
	scored_steps: int | None = None,
	burn_in_steps: int = 0,
) -> float:
	"""Mean open-loop ensemble Gaussian NLL over a short forecast horizon."""
	if len(z_inits) != len(obs_windows):
		raise ValueError("z_inits and obs_windows must have the same length")
	windows = list(obs_windows)
	if scored_steps is not None:
		horizon = int(scored_steps)
		if horizon < 1:
			raise ValueError(f"scored_steps must be >= 1, got {scored_steps}")
		trimmed: list[torch.Tensor] = []
		need = horizon + 1
		for obs in windows:
			if int(obs.shape[0]) < need:
				raise ValueError(
					f"held window length {int(obs.shape[0])} is shorter than initial observation plus NLL horizon {horizon}"
				)
			trimmed.append(obs[:need])
		windows = trimmed
	contexts = [context] * len(z_inits)
	nll = shell.batched_windowed_nll(
		z_inits,
		contexts,
		windows,
		n_ensemble=int(n_ensemble),
		burn_in_steps=int(burn_in_steps),
	)
	return float(nll.detach().cpu().item())


def _select_held_windows(
	windows: list[torch.Tensor],
	*,
	n_starts: int,
) -> tuple[list[torch.Tensor], np.ndarray]:
	n_windows = len(windows)
	if n_windows == 0:
		return [], np.zeros((0,), dtype=np.int64)
	if int(n_starts) < n_windows:
		idx = np.unique(np.linspace(0, n_windows - 1, num=int(n_starts), dtype=np.int64))
		return [windows[int(i)] for i in idx], idx
	idx = np.arange(n_windows, dtype=np.int64)
	return list(windows), idx


@torch.no_grad()
def score_held_nll_horizon(
	shell: DynaMixShell,
	*,
	x_train_noisy: Any,
	context_length: int,
	nll_horizon: int,
	device: torch.device | str,
	n_ensemble: int = 256,
	n_starts: int = 100,
) -> tuple[float, int]:
	"""Lead-averaged ensemble NLL on held-out windows.

	Each window starts at an encoded noisy observation. Leads/horizons 1,...,nll_horizon are
	scored against the noisy reference.
	"""
	shell.eval()
	nll_dataset = build_dynamix_window_dataset(
		x_train_noisy,
		shell.model,
		context_length=int(context_length),
		window_length=int(nll_horizon),
		stride=int(nll_horizon),
		covariance_burn_in_steps=0,
		device=device,
	)
	obs_windows, _idx = _select_held_windows(nll_dataset.windows_held, n_starts=int(n_starts))
	if not obs_windows:
		return float("nan"), 0
	z_inits = [shell.latent_column_from_observation(obs[0]) for obs in obs_windows]
	torch.manual_seed(0)
	nll = evaluate_held_nll(
		shell,
		z_inits=z_inits,
		context=nll_dataset.context,
		obs_windows=obs_windows,
		n_ensemble=int(n_ensemble),
		scored_steps=int(nll_horizon),
		burn_in_steps=0,
	)
	return float(nll), int(len(z_inits))


@torch.no_grad()
def evaluate_rmse_horizon(
	shell: DynaMixShell,
	*,
	z_inits: list[torch.Tensor],
	context: torch.Tensor,
	obs_windows: list[torch.Tensor],
	horizon: int = 10,
	n_ensemble: int = 256,
	n_eval_windows: int = 30,
	target_windows: list[torch.Tensor] | None = None,
) -> float:
	"""Ensemble-mean RMSE of the nll_horizon-th transition.
	"""
	if int(horizon) < 1:
		raise ValueError(f"horizon must be >= 1, got {horizon}")
	shell.eval()
	targets = list(obs_windows if target_windows is None else target_windows)
	n_eval = min(int(n_eval_windows), len(z_inits), len(obs_windows), len(targets))
	if n_eval <= 0:
		return float("nan")
	kept_z: list[torch.Tensor] = []
	kept_y: list[torch.Tensor] = []
	for z0, target in zip(z_inits[:n_eval], targets[:n_eval], strict=True):
		if int(target.shape[0]) <= int(horizon):
			continue
		kept_z.append(z0)
		kept_y.append(target[int(horizon)])
	if not kept_z:
		return float("nan")
	z_row = torch.stack([z[:, 0] if z.ndim == 2 else z for z in kept_z], dim=0)
	y_true = torch.stack(kept_y, dim=0)
	b = int(z_row.shape[0])
	m_dim = int(z_row.shape[1])
	n_dim = int(shell.observed_dim)
	torch.manual_seed(0)
	z = (
		z_row.transpose(0, 1)
		.unsqueeze(-1)
		.expand(m_dim, b, int(n_ensemble))
		.reshape(m_dim, b * int(n_ensemble))
		.contiguous()
	)
	ctx = context.expand(-1, b * int(n_ensemble), -1)
	cnn = shell.model.precompute_cnn(ctx)
	sigma_q = shell.log_proc_std.exp().unsqueeze(1)
	for _step in range(int(horizon)):
		z = shell.model.step(z, ctx, precomputed_cnn=cnn)
		z = z + sigma_q * torch.randn_like(z)
	mu = z[:n_dim].reshape(n_dim, b, int(n_ensemble)).mean(dim=-1).transpose(0, 1)
	per_window = (mu - y_true).pow(2).mean(dim=-1).sqrt()
	return float(per_window.mean().detach().cpu().item())


__all__ = [
	"POWER_SPECTRUM_SMOOTHING",
	"QOI_BURN_IN_STEPS",
	"QOI_EVAL_STEPS",
	"QOI_ROLLOUT_STEPS",
	"compute_dynamix_core_metrics",
	"compute_standardization_stats",
	"core_standardization_trajectory",
	"evaluate_held_nll",
	"evaluate_rmse_horizon",
	"evaluate_dynamix_core_rollout",
	"prepare_dynamix_context_state",
	"rollout_model_raw",
	"score_held_nll_horizon",
]
