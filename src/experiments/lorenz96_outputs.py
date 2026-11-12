"""Lorenz-96 evaluation helpers: dynamical and predictive metrics."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metrics import (
	PredictiveEvalConfig,
	ensemble_path_nll,
	forecast_starts,
	l96_ekf_grounding_window_from_shell,
	lorenz96_lyapunov_spectrum_from_samples,
	lorenz96_rds_lyapunov_spectrum,
	lyap_error_dict,
	marginal_crps_ensemble,
	power_spectrum_error,
	positive_lyap_sum,
	robust_se_of_median,
	rollout_probabilistic_ensemble,
	state_space_divergence,
)
from models import ALRNN, ALRNNShell, ALRNNShellLR, rollout_free
from training import BASELINE_SPECS

from .lorenz96_main import prepare_lorenz96_main_bundle

_REPO_ROOT = Path(__file__).resolve().parents[2]
L96_CORE_CHECKPOINT_NAME = "alrnn_core.pt"
SHELL_CHECKPOINT_NAME = "alrnn_shell.pt"


# Transport-only G_tan: Spearman on log-variance expansion minus the
# log posterior baseline at the last assimilation time.
GTAN_TRANSPORT_FIELD = "G_tan_transport"
GTAN_TRANSPORT_SUMMARY_FIELD = "G_tan_transport_median"

DYNAMICAL_NUMERIC_FIELDS = (
	"D_stsp_core",
	"D_H_core",
	"signed_LE_error_core",
	"abs_signed_LE_error_core",
	"h_lambda_core",
	"signed_h_lambda_error_core",
	"abs_signed_h_lambda_error_core",
	"D_stsp_stoch",
	"signed_LE_error_stoch",
	"abs_signed_LE_error_stoch",
	"h_lambda_stoch",
	"signed_h_lambda_error_stoch",
	"abs_signed_h_lambda_error_stoch",
	"G_tan",
	"G_tan_transport",
)

PREDICTIVE_NUMERIC_FIELDS = (
	"NLL",
	"CRPS",
)


def _resolve_repo_path(path: str | Path | None) -> Path | None:
	if path is None:
		return None
	resolved = Path(path)
	if resolved.is_absolute():
		return resolved
	return _REPO_ROOT / resolved


def _load_json(path: str | Path) -> dict[str, Any]:
	with Path(path).open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _standardize_raw_series(raw: np.ndarray, standardizer: dict[str, Any]) -> np.ndarray:
	mean = np.asarray(standardizer["mean"], dtype=np.float64).reshape(1, -1)
	std = np.asarray(standardizer["std"], dtype=np.float64).reshape(1, -1)
	return (np.asarray(raw, dtype=np.float64) - mean) / std


def _raw_series(x_std: np.ndarray, standardizer: dict[str, Any]) -> np.ndarray:
	mean = np.asarray(standardizer["mean"], dtype=np.float64).reshape(1, -1)
	std = np.asarray(standardizer["std"], dtype=np.float64).reshape(1, -1)
	return np.asarray(x_std, dtype=np.float64) * std + mean


@dataclass(frozen=True)
class OutputMethodSpec:
	key: str
	label: str


PROBABILISTIC_METHODS = [
	OutputMethodSpec("kaffee", "KAFFEE"),
	OutputMethodSpec("frozen_core_shell", "Frozen-core shell EKF innovation NLL"),
	OutputMethodSpec("filtered_no_propagation", "Filtered innovation NLL, no covariance propagation"),
	OutputMethodSpec("open_loop_tied", "Open-loop Gaussian NLL (tied)"),
	OutputMethodSpec("open_loop_no_propagation", "Open-loop Gaussian NLL (no propagation)"),
]


def _predictive_eval_config(config: dict[str, Any]) -> PredictiveEvalConfig:
	evaluation = dict(config["evaluation"])
	return PredictiveEvalConfig(
		forecast_horizon=int(evaluation["main_forecast_horizon_steps"]),
		forecast_starts=int(evaluation.get("predictive_forecast_starts", evaluation.get("forecast_anchor_count", 10))),
		nll_starts=int(evaluation["predictive_nll_starts"]),
		forecast_ensemble=int(evaluation["predictive_forecast_ensemble"]),
		obs_floor=float(evaluation["predictive_obs_floor"]),
	)


def _predictive_ensemble_seed(*, seed: int, start: int) -> int:
	return int(np.random.SeedSequence([int(seed), int(start)]).generate_state(1)[0])


def _dynamical_eval_config(config: dict[str, Any], *, smoke: bool) -> dict[str, int | float]:
	evaluation = dict(config["evaluation"])
	smoke_scale = 0.05 if bool(smoke) else 1.0

	def _scaled(name: str, default: int) -> int:
		value = int(evaluation.get(name, default))
		if bool(smoke):
			return max(2, int(round(value * smoke_scale)))
		return value

	return {
		"core_start_index": int(evaluation.get("core_start_index", 500)),
		"stochastic_start_index": int(evaluation.get("stochastic_start_index", 500)),
		"core_state_space_rollout_steps": _scaled("state_space_rollout_steps", 50_000),
		"state_space_burn_steps": _scaled("state_space_burn_steps", 1000),
		"lyapunov_burn_steps": _scaled("lyapunov_burn_steps", 1000),
		"lyapunov_rollout_steps": _scaled("lyapunov_rollout_steps", 30_000),
		"state_space_mc_samples": int(evaluation.get("state_space_mc_samples", 1000)),
		"state_space_density_batch_size": int(evaluation.get("state_space_density_batch_size", 512)),
		"state_space_sigma2": float(evaluation.get("state_space_sigma2", 1.0)),
		"pse_smoothing": int(evaluation.get("pse_smoothing", 20)),
		"gtan_window_start": int(evaluation.get("gtan_window_start", 200)),
		"gtan_warmup_steps": _scaled("gtan_warmup_steps", 200),
		"gtan_window_steps": _scaled("gtan_window_steps", 4000),
	}


def _load_reference_bundle(
	config: dict[str, Any],
	*,
	field: str,
	total_steps: int,
	standardizer: dict[str, Any],
) -> dict[str, np.ndarray]:
	reference_path = _resolve_repo_path(dict(config.get("evaluation", {})).get("reference_path"))
	if reference_path is None:
		raise ValueError("lorenz96_main evaluation.reference_path must be set")
	if not reference_path.is_file():
		raise FileNotFoundError(f"Missing Lorenz-96 evaluation references: {reference_path}")
	with np.load(reference_path) as bundle:
		if field not in bundle.files:
			raise ValueError(f"Evaluation reference {reference_path} is missing {field!r}")
		raw = np.asarray(bundle[field], dtype=np.float64)
	if raw.ndim != 2:
		raise ValueError(f"Expected {field} to have shape (T, N), got {raw.shape}")
	if int(raw.shape[0]) < int(total_steps):
		raise ValueError(f"Reference {field} has {raw.shape[0]} steps, need at least {total_steps}")
	raw = raw[: int(total_steps)]
	return {
		"clean_raw": raw,
		"clean_std": _standardize_raw_series(raw, standardizer),
		"obs_raw": raw,
		"obs_std": _standardize_raw_series(raw, standardizer),
		"standardizer": standardizer,
	}


def _lyapunov_segment(
	series: np.ndarray,
	*,
	start_index: int,
	burn_steps: int,
	rollout_steps: int,
) -> np.ndarray:
	start = int(start_index)
	burn = int(burn_steps)
	rollout = int(rollout_steps)
	stop = start + burn + rollout + 1
	series = np.asarray(series, dtype=np.float64)
	if series.ndim != 2:
		raise ValueError(f"Lyapunov reference must have shape (T, N), got {series.shape}")
	if start < 0 or burn < 0 or rollout <= 10 or stop > int(series.shape[0]):
		raise ValueError(
			f"Invalid Lyapunov window start={start}, burn={burn}, rollout={rollout}, "
			f"reference_length={series.shape[0]}"
		)
	return series[start:stop]


def _reference_lorenz96_spectrum(
	reference_raw: np.ndarray,
	*,
	start_index: int,
	burn_steps: int,
	rollout_steps: int,
	dt: float,
) -> np.ndarray:
	"""Lorenz-96 Lyapunov spectrum (per unit time) via variational RK4 along the reference."""
	segment = _lyapunov_segment(
		reference_raw,
		start_index=int(start_index),
		burn_steps=int(burn_steps),
		rollout_steps=int(rollout_steps),
	)
	return lorenz96_lyapunov_spectrum_from_samples(
		segment,
		dt=float(dt),
		burn_in=int(burn_steps),
		max_T=int(burn_steps) + int(rollout_steps),
		qr_every=1,
	)


def _process_noisy_lorenz96_spectrum(
	reference_raw: np.ndarray,
	*,
	start_index: int,
	burn_steps: int,
	rollout_steps: int,
	dt: float,
	forcing: float,
) -> np.ndarray:
	"""Process-noisy (Random Dynamical System, RDS) Lorenz-96 spectrum (per unit time). Observation noise is excluded."""
	segment = _lyapunov_segment(
		reference_raw,
		start_index=int(start_index),
		burn_steps=int(burn_steps),
		rollout_steps=int(rollout_steps),
	)
	return lorenz96_rds_lyapunov_spectrum(
		segment,
		dt=float(dt),
		burn_in=int(burn_steps),
		max_T=int(burn_steps) + int(rollout_steps),
		forcing=float(forcing),
	)


def _load_core_checkpoint(core_root: Path, seed: int) -> dict[str, Any]:
	path = Path(core_root) / f"seed{int(seed):02d}" / L96_CORE_CHECKPOINT_NAME
	if not path.exists():
		raise FileNotFoundError(f"Missing deterministic core checkpoint: {path}")
	return torch.load(path, map_location="cpu", weights_only=False)


def _load_alrnn_core(checkpoint: dict[str, Any], device: torch.device) -> ALRNN:
	model = ALRNN(
		latent_dim=int(checkpoint["M"]),
		gated_dim=int(checkpoint.get("P", 50)),
		observed_dim=int(checkpoint["N"]),
	)
	model.load_state_dict(checkpoint["state_dict"])
	return model.to(device).eval()


def _core_metric_block(
	generated_std: np.ndarray,
	reference_std: np.ndarray,
	standardizer: dict[str, Any],
	spectrum_per_step: np.ndarray,
	*,
	dt: float,
	ref_spectrum_per_unit_time: np.ndarray | float,
	sigma2: float,
	mc_samples: int,
	density_batch_size: int,
	pse_smoothing: int,
) -> dict[str, float]:
	generated_std = np.asarray(generated_std, dtype=np.float64)
	reference_std = np.asarray(reference_std, dtype=np.float64)
	le_block = lyap_error_dict(
		spectrum_per_step,
		ref_spectrum_per_unit_time,
		dt=float(dt),
		suffix="_core",
	)
	seg_len = min(int(generated_std.shape[0]), int(reference_std.shape[0]))
	gen_score_std = generated_std[:seg_len]
	ref_score_std = reference_std[:seg_len]
	gen_score_raw = _raw_series(gen_score_std, standardizer)
	ref_score_raw = _raw_series(ref_score_std, standardizer)
	d_stsp = float(
		state_space_divergence(
			gen_score_std,
			ref_score_std,
			sigma2=float(sigma2),
			max_T=int(seg_len),
			mc_samples=int(mc_samples),
			batch_size=int(density_batch_size),
		)
	)
	pse = float(power_spectrum_error(ref_score_raw, gen_score_raw, smoothing=int(pse_smoothing)))
	return {"D_stsp_core": d_stsp, "D_H_core": pse, **le_block}


def _shell_rollout(
	model: ALRNNShell,
	x0_std: np.ndarray,
	*,
	steps: int,
	seed: int,
	sample_process_noise: bool,
	sample_obs_noise: bool,
) -> np.ndarray:
	return np.asarray(
		model.rollout_observation_sequence(
			x0_std,
			steps=int(steps),
			seed=int(seed),
			sample_process_noise=bool(sample_process_noise),
			sample_obs_noise=bool(sample_obs_noise),
		),
		dtype=np.float64,
	)


def _stochastic_metric_block(
	model: ALRNNShell,
	reference_std: np.ndarray,
	process_state0_std: np.ndarray,
	*,
	dt: float,
	ref_spectrum_per_unit_time: np.ndarray,
	sigma2: float,
	mc_samples: int,
	density_batch_size: int,
	burn_steps: int,
	rollout_steps: int,
	lyapunov_rollout_steps: int,
	lyapunov_burn_steps: int,
	seed: int,
	start_index: int,
) -> dict[str, float]:
	x0_std = np.asarray(reference_std[int(start_index)], dtype=np.float64)
	total_steps = int(burn_steps) + int(rollout_steps)
	rollout_std = _shell_rollout(
		model,
		x0_std,
		steps=total_steps,
		seed=int(seed),
		sample_process_noise=True,
		sample_obs_noise=True,
	)
	gen_score_std = np.asarray(rollout_std[int(burn_steps) : int(burn_steps) + int(rollout_steps)], dtype=np.float64)
	ref_score_std = np.asarray(reference_std[int(start_index) : int(start_index) + int(rollout_steps)], dtype=np.float64)
	seg_len = min(int(gen_score_std.shape[0]), int(ref_score_std.shape[0]))
	spectrum_per_step = model.lyapunov_spectrum(
		process_state0_std,
		steps=int(lyapunov_burn_steps) + int(lyapunov_rollout_steps),
		burn=int(lyapunov_burn_steps),
		seed=int(seed),
		sample_process_noise=True,
	)
	le_block = lyap_error_dict(
		spectrum_per_step,
		ref_spectrum_per_unit_time,
		dt=float(dt),
		suffix="_stoch",
	)
	d_stsp = float(
		state_space_divergence(
			gen_score_std[:seg_len],
			ref_score_std[:seg_len],
			sigma2=float(sigma2),
			max_T=int(seg_len),
			mc_samples=int(mc_samples),
			batch_size=int(density_batch_size),
		)
	)
	return {"D_stsp_stoch": d_stsp, **le_block}


@torch.no_grad()
def _evaluate_grounding(
	model: ALRNNShell,
	test_obs_std: np.ndarray,
	test_clean_std: np.ndarray,
	*,
	dt: float,
	forcing: float,
	standardizer: dict[str, Any],
	window_start: int,
	window_length: int,
	filter_warmup: int,
	assim_every: int = 1, # Assimilation interval (H)
) -> dict[str, float]:

	test_obs_std = np.asarray(test_obs_std, dtype=np.float32)
	test_clean_std = np.asarray(test_clean_std, dtype=np.float32)
	if test_obs_std.ndim != 2 or test_clean_std.shape != test_obs_std.shape:
		raise ValueError(
			f"test_obs_std and test_clean_std must share shape (T, N); "
			f"got {test_obs_std.shape} vs {test_clean_std.shape}"
		)

	wstart = int(window_start)
	wlen = int(window_length)
	warmup = int(filter_warmup)
	slice_start = wstart - warmup
	slice_end = wstart + wlen + 1
	if slice_start < 0:
		raise ValueError(f"gtan_window_start ({wstart}) must be >= gtan_warmup_steps ({warmup})")
	if slice_end > int(test_obs_std.shape[0]):
		raise ValueError(
			f"G_tan window end {slice_end} exceeds test trajectory length {test_obs_std.shape[0]}"
		)

	mean = np.asarray(standardizer["mean"], dtype=np.float64).reshape(-1)
	std = np.asarray(standardizer["std"], dtype=np.float64).reshape(-1)
	result = l96_ekf_grounding_window_from_shell(
		model,
		test_obs_std[slice_start:slice_end],
		test_clean_std[slice_start:slice_end],
		mean=mean,
		std=std,
		dt=float(dt),
		forcing=float(forcing),
		filter_warmup=warmup,
		assim_every=int(assim_every),
	)
	return {
		"G_tan": float(result.g_tan),
		"G_tan_transport": float(result.g_tan_transport),
		"G_tan_n_samples": int(result.n_samples),
	}


def _load_probabilistic_checkpoint(probabilistic_root: Path, seed: int, method_key: str) -> dict[str, Any]:
	path = Path(probabilistic_root) / f"seed{int(seed):02d}" / method_key / SHELL_CHECKPOINT_NAME
	return torch.load(path, map_location="cpu", weights_only=False)


def _load_probabilistic_model(
	checkpoint: dict[str, Any],
	*,
	method_key: str,
	config: dict[str, Any],
	device: torch.device,
) -> ALRNNShell | ALRNNShellLR:
	state = checkpoint["state_dict"]
	spec = BASELINE_SPECS[method_key]
	model_cfg = dict(config["model"])
	if "proc_noise_sqrt" in state:
		factor = state["proc_noise_sqrt"].to(device=device, dtype=torch.float32)
		obs_std = torch.exp(state["log_obs_std"]).to(device=device, dtype=torch.float32)
		marginal_proc = torch.sqrt((factor * factor).sum(dim=-1).clamp_min(1e-12))
		model: ALRNNShell | ALRNNShellLR = ALRNNShellLR(
			latent_dim=int(state["A"].numel()),
			gated_dim=int(model_cfg.get("gated_dim", 50)),
			observed_dim=int(state["embedding"].shape[0]),
			A_init=state["A"].to(device=device, dtype=torch.float32),
			W_init=state["W"].to(device=device, dtype=torch.float32),
			h_init=state["h"].to(device=device, dtype=torch.float32),
			embedding_init=state["embedding"].to(device=device, dtype=torch.float32),
			proc_std_init=marginal_proc,
			obs_std_init=obs_std,
			filter_rank=int(factor.shape[-1]),
			proc_noise_sqrt_init=factor,
			train_core=bool(spec.train_core),
			covariance_transport=str(spec.covariance_transport),
		).to(device)
		model.load_state_dict(state)
		if "obs_std_exact" in checkpoint:
			model.set_sampling_std_exact(
				obs_std=torch.as_tensor(checkpoint["obs_std_exact"], dtype=torch.float32, device=device),
			)
		return model

	proc_std = torch.exp(state["log_proc_std"]).to(device=device, dtype=torch.float32)
	obs_std = torch.exp(state["log_obs_std"]).to(device=device, dtype=torch.float32)
	model = ALRNNShell(
		latent_dim=int(state["A"].numel()),
		gated_dim=int(model_cfg.get("gated_dim", 50)),
		observed_dim=int(state["embedding"].shape[0]),
		A_init=state["A"].to(device=device, dtype=torch.float32),
		W_init=state["W"].to(device=device, dtype=torch.float32),
		h_init=state["h"].to(device=device, dtype=torch.float32),
		embedding_init=state["embedding"].to(device=device, dtype=torch.float32),
		proc_std_init=proc_std,
		obs_std_init=obs_std,
		train_core=bool(spec.train_core),
		covariance_transport=str(spec.covariance_transport),
	).to(device)
	model.load_state_dict(state)
	if "proc_std_exact" in checkpoint and "obs_std_exact" in checkpoint:
		model.set_sampling_std_exact(
			proc_std=torch.as_tensor(checkpoint["proc_std_exact"], dtype=torch.float32, device=device),
			obs_std=torch.as_tensor(checkpoint["obs_std_exact"], dtype=torch.float32, device=device),
		)
	return model


@torch.no_grad()
def _evaluate_probabilistic_predictive(
	model: ALRNNShell,
	test_obs_std: np.ndarray,
	test_obs_raw: np.ndarray,
	standardizer: dict[str, Any],
	*,
	predictive: PredictiveEvalConfig,
	seed: int,
) -> dict[str, float]:
	nll_values: list[float] = []
	crps_values: list[float] = []

	starts_path = forecast_starts(int(test_obs_raw.shape[0]), int(predictive.forecast_horizon), int(predictive.nll_starts))
	for start in starts_path:
		ensemble_std = rollout_probabilistic_ensemble(
			model,
			test_obs_std[int(start)],
			length=int(predictive.forecast_horizon),
			n_ensemble=int(predictive.forecast_ensemble),
			seed=_predictive_ensemble_seed(seed=int(seed), start=int(start)),
			sample_process_noise=True,
			sample_obs_noise=True,
		)
		ensemble_raw = _raw_series(ensemble_std, standardizer)
		target = np.asarray(
			test_obs_raw[int(start) + 1 : int(start) + int(predictive.forecast_horizon) + 1],
			dtype=np.float64,
		)
		nll_values.append(ensemble_path_nll(target, ensemble_raw, floor=float(predictive.obs_floor)))

	starts_forecast = forecast_starts(
		int(test_obs_raw.shape[0]),
		int(predictive.forecast_horizon),
		int(predictive.forecast_starts),
	)
	for start in starts_forecast:
		ensemble_std = rollout_probabilistic_ensemble(
			model,
			test_obs_std[int(start)],
			length=int(predictive.forecast_horizon),
			n_ensemble=int(predictive.forecast_ensemble),
			seed=_predictive_ensemble_seed(seed=int(seed), start=int(start)),
			sample_process_noise=True,
			sample_obs_noise=True,
		)
		ensemble_raw = _raw_series(ensemble_std, standardizer)
		target = np.asarray(test_obs_raw[int(start) + int(predictive.forecast_horizon)], dtype=np.float64)
		crps_values.append(
			float(
				marginal_crps_ensemble(
					target,
					ensemble_raw[:, -1, :],
				)
			)
		)

	return {
		"NLL": float(np.mean(nll_values)) if nll_values else float("nan"),
		"CRPS": float(np.mean(crps_values)) if crps_values else float("nan"),
	}


def _median_row(rows: list[dict[str, Any]], field: str) -> float:
	values = [float(row[field]) for row in rows if field in row and np.isfinite(float(row[field]))]
	return float(np.median(values)) if values else float("nan")


def aggregate_lorenz96_predictive_results(
	config: dict[str, Any],
	probabilistic_root: Path,
	*,
	seeds: list[int],
	smoke: bool,
	device: torch.device,
	on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
	prepared = prepare_lorenz96_main_bundle(config, smoke=bool(smoke))
	dataset = prepared["dataset"]
	heldout_test = prepared["heldout_test"]
	test_obs_std = np.asarray(heldout_test["obs_std"], dtype=np.float32)
	test_obs_raw = np.asarray(heldout_test["obs_raw"], dtype=np.float64)
	standardizer = dict(dataset["standardizer"])
	predictive = _predictive_eval_config(config)

	per_seed_rows: list[dict[str, Any]] = []
	for seed in seeds:
		for method in PROBABILISTIC_METHODS:
			method_key = method.key
			method_dir = Path(probabilistic_root) / f"seed{int(seed):02d}" / method_key
			checkpoint_path = method_dir / SHELL_CHECKPOINT_NAME
			if not checkpoint_path.exists():
				continue
			checkpoint = _load_probabilistic_checkpoint(probabilistic_root, int(seed), method_key)
			model = _load_probabilistic_model(
				checkpoint,
				method_key=method_key,
				config=config,
				device=device,
			)
			metrics = _evaluate_probabilistic_predictive(
				model,
				test_obs_std,
				test_obs_raw,
				standardizer,
				predictive=predictive,
				seed=int(seed),
			)
			spec = BASELINE_SPECS[method_key]
			per_seed_rows.append(
				{
					"seed": int(seed),
					"method_key": method_key,
					"method_label": spec.label,
					"description": spec.description,
					**metrics,
				}
			)
			if on_progress is not None:
				on_progress(f"seed {int(seed):02d} {method_key} NLL/CRPS")

	summary_rows: list[dict[str, Any]] = []
	for method in PROBABILISTIC_METHODS:
		rows = [row for row in per_seed_rows if row["method_key"] == method.key]
		if not rows:
			continue
		summary: dict[str, Any] = {
			"method_key": method.key,
			"method_label": rows[0]["method_label"],
			"description": rows[0]["description"],
			"seed_count": int(len(rows)),
		}
		for field in PREDICTIVE_NUMERIC_FIELDS:
			vals = [
				float(row[field])
				for row in rows
				if field in row and np.isfinite(float(row[field]))
			]
			summary[f"{field}_n_finite"] = len(vals)
			summary[f"{field}_median"] = float(np.median(vals)) if vals else float("nan")
			summary[f"{field}_mean"] = float(np.mean(vals)) if vals else float("nan")
			summary[f"{field}_std"] = float(np.std(vals)) if vals else float("nan")
		summary_rows.append(summary)

	return {"per_seed_rows": per_seed_rows, "summary_rows": summary_rows}


def compute_lorenz96_dynamical_metrics(
	config: dict[str, Any],
	core_root: Path,
	probabilistic_root: Path,
	*,
	seeds: list[int],
	smoke: bool,
	device: torch.device,
	on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
	prepared = prepare_lorenz96_main_bundle(config, smoke=bool(smoke))
	dataset = prepared["dataset"]
	heldout_test = prepared["heldout_test"]
	stochastic_evaluation = prepared["stochastic_evaluation"]
	standardizer = dict(dataset["standardizer"])
	system = dict(config["system"])
	dynamical = _dynamical_eval_config(config, smoke=bool(smoke))
	dt = float(system["dt"])
	forcing = float(system["forcing"])

	core_start_index = int(dynamical["core_start_index"])
	stochastic_start_index = int(dynamical["stochastic_start_index"])
	state_space_rollout_steps = int(dynamical["core_state_space_rollout_steps"])
	state_space_burn_steps = int(dynamical["state_space_burn_steps"])
	lyapunov_burn_steps = int(dynamical["lyapunov_burn_steps"])
	lyapunov_rollout_steps = int(dynamical["lyapunov_rollout_steps"])
	lyapunov_steps = int(lyapunov_burn_steps) + int(lyapunov_rollout_steps)

	core_reference_total_steps = max(
		int(core_start_index) + int(state_space_rollout_steps),
		int(core_start_index) + int(lyapunov_steps) + 1,
	)
	core_reference_bundle = _load_reference_bundle(
		config,
		field="core_reference_raw",
		total_steps=int(core_reference_total_steps),
		standardizer=standardizer,
	)
	core_reference_raw = np.asarray(core_reference_bundle["clean_raw"], dtype=np.float64)
	core_reference_std = np.asarray(core_reference_bundle["clean_std"], dtype=np.float64)

	long_obs_std = np.asarray(stochastic_evaluation["obs_std"], dtype=np.float32)

	ref_spectrum_per_unit_time = _reference_lorenz96_spectrum(
		core_reference_raw,
		start_index=int(core_start_index),
		burn_steps=int(lyapunov_burn_steps),
		rollout_steps=int(lyapunov_rollout_steps),
		dt=dt,
	)
	if on_progress is not None:
		on_progress("L96 core reference spectrum")
	stoch_ref_spectrum = _process_noisy_lorenz96_spectrum(
		np.asarray(stochastic_evaluation["clean_raw"], dtype=np.float64),
		start_index=int(stochastic_start_index),
		burn_steps=int(lyapunov_burn_steps),
		rollout_steps=int(lyapunov_rollout_steps),
		dt=dt,
		forcing=forcing,
	)
	if on_progress is not None:
		on_progress("process-noisy RDS spectrum")
	process_state0_std = np.asarray(
		stochastic_evaluation["clean_std"][int(stochastic_start_index)],
		dtype=np.float64,
	)

	per_seed_rows: list[dict[str, Any]] = []
	for seed in seeds:
		core_checkpoint = _load_core_checkpoint(Path(core_root), int(seed))
		core_model = _load_alrnn_core(core_checkpoint, device)

		prior_standardizer = {
			"mean": np.asarray(core_checkpoint["mean"], dtype=np.float64).reshape(-1),
			"std": np.asarray(core_checkpoint["std"], dtype=np.float64).reshape(-1),
		}
		prior_reference_std = _standardize_raw_series(core_reference_raw, prior_standardizer)
		prior_rollout_std = (
			rollout_free(
				core_model,
				torch.as_tensor(
					prior_reference_std[int(core_start_index) : int(core_start_index) + 1],
					dtype=torch.float32,
					device=device,
				),
				steps=int(state_space_burn_steps) + int(state_space_rollout_steps),
			)[0, :, : int(core_checkpoint["N"])]
			.detach()
			.cpu()
			.numpy()
			.astype(np.float64)
		)
		prior_spectrum = core_model.lyapunov_spectrum(
			prior_reference_std[int(core_start_index)],
			steps=int(lyapunov_steps),
			burn=int(lyapunov_burn_steps),
		)
		prior_core = _core_metric_block(
			prior_rollout_std[int(state_space_burn_steps) : int(state_space_burn_steps) + int(state_space_rollout_steps)],
			prior_reference_std[int(core_start_index) : int(core_start_index) + int(state_space_rollout_steps)],
			prior_standardizer,
			prior_spectrum,
			dt=dt,
			ref_spectrum_per_unit_time=ref_spectrum_per_unit_time,
			sigma2=float(dynamical["state_space_sigma2"]),
			mc_samples=int(dynamical["state_space_mc_samples"]),
			density_batch_size=int(dynamical["state_space_density_batch_size"]),
			pse_smoothing=int(dynamical["pse_smoothing"]),
		)
		stoch_nan = {
			field: float("nan")
			for field in DYNAMICAL_NUMERIC_FIELDS
			if field.endswith("_stoch") or field.startswith("G_tan")
		}
		per_seed_rows.append(
			{
				"seed": int(seed),
				"method_key": "alrnn_core",
				"method_label": "AL-RNN (no shell)",
				"description": "Deterministic AL-RNN core (no shell).",
				**prior_core,
				**stoch_nan,
			}
		)
		if on_progress is not None:
			on_progress(f"seed {int(seed):02d} alrnn_core")

		for method in PROBABILISTIC_METHODS:
			method_key = method.key
			checkpoint_path = Path(probabilistic_root) / f"seed{int(seed):02d}" / method_key / SHELL_CHECKPOINT_NAME
			if not checkpoint_path.exists():
				continue
			checkpoint = _load_probabilistic_checkpoint(probabilistic_root, int(seed), method_key)
			shell = _load_probabilistic_model(
				checkpoint,
				method_key=method_key,
				config=config,
				device=device,
			)
			spec = BASELINE_SPECS[method_key]

			if method_key == "frozen_core_shell":
				core_rollout_std = prior_rollout_std
				core_reference_for_row = prior_reference_std
				core_spectrum = prior_spectrum
				core_standardizer = prior_standardizer
			else:
				shell_reference_std = _standardize_raw_series(core_reference_raw, standardizer)
				core_rollout_std = _shell_rollout(
					shell,
					shell_reference_std[int(core_start_index)],
					steps=int(state_space_burn_steps) + int(state_space_rollout_steps),
					seed=int(seed),
					sample_process_noise=False,
					sample_obs_noise=False,
				)
				core_spectrum = shell.lyapunov_spectrum(
					shell_reference_std[int(core_start_index)],
					steps=int(lyapunov_steps),
					burn=int(lyapunov_burn_steps),
					seed=int(seed),
					sample_process_noise=False,
				)
				core_reference_for_row = shell_reference_std
				core_standardizer = standardizer

			core_metrics = _core_metric_block(
				core_rollout_std[int(state_space_burn_steps) : int(state_space_burn_steps) + int(state_space_rollout_steps)],
				core_reference_for_row[int(core_start_index) : int(core_start_index) + int(state_space_rollout_steps)],
				core_standardizer,
				core_spectrum,
				dt=dt,
				ref_spectrum_per_unit_time=ref_spectrum_per_unit_time,
				sigma2=float(dynamical["state_space_sigma2"]),
				mc_samples=int(dynamical["state_space_mc_samples"]),
				density_batch_size=int(dynamical["state_space_density_batch_size"]),
				pse_smoothing=int(dynamical["pse_smoothing"]),
			)
			stoch_metrics = _stochastic_metric_block(
				shell,
				long_obs_std,
				process_state0_std,
				dt=dt,
				ref_spectrum_per_unit_time=stoch_ref_spectrum,
				sigma2=float(dynamical["state_space_sigma2"]),
				mc_samples=int(dynamical["state_space_mc_samples"]),
				density_batch_size=int(dynamical["state_space_density_batch_size"]),
				burn_steps=int(state_space_burn_steps),
				rollout_steps=int(state_space_rollout_steps),
				lyapunov_rollout_steps=int(lyapunov_rollout_steps),
				lyapunov_burn_steps=int(lyapunov_burn_steps),
				seed=int(seed),
				start_index=int(stochastic_start_index),
			)
			gtan_metrics = _evaluate_grounding(
				shell.cpu(),
				np.asarray(heldout_test["obs_std"], dtype=np.float32),
				np.asarray(heldout_test["clean_std"], dtype=np.float32),
				dt=dt,
				forcing=forcing,
				standardizer=standardizer,
				window_start=int(dynamical["gtan_window_start"]),
				window_length=int(dynamical["gtan_window_steps"]),
				filter_warmup=int(dynamical["gtan_warmup_steps"]),
			)
			per_seed_rows.append(
				{
					"seed": int(seed),
					"method_key": method_key,
					"method_label": spec.label,
					"description": spec.description,
					**core_metrics,
					**stoch_metrics,
					**gtan_metrics,
				}
			)
			if on_progress is not None:
				on_progress(f"seed {int(seed):02d} {method_key}")

	summary_rows: list[dict[str, Any]] = []
	for method_key in ["alrnn_core", *[m.key for m in PROBABILISTIC_METHODS]]:
		rows = [row for row in per_seed_rows if row["method_key"] == method_key]
		if not rows:
			continue
		summary: dict[str, Any] = {
			"method_key": method_key,
			"method_label": rows[0]["method_label"],
			"description": rows[0]["description"],
			"seed_count": int(len(rows)),
		}
		for field in DYNAMICAL_NUMERIC_FIELDS:
			vals = [float(row[field]) for row in rows if field in row and np.isfinite(float(row[field]))]
			summary[f"{field}_n_finite"] = len(vals)
			summary[f"{field}_median"] = _median_row(rows, field)
			summary[f"{field}_mean"] = float(np.mean(vals)) if vals else float("nan")
			summary[f"{field}_std"] = float(np.std(vals)) if vals else float("nan")
			se = robust_se_of_median(vals)
			summary[f"{field}_robust_se"] = float(se) if se is not None else float("nan")
		summary_rows.append(summary)

	return {
		"per_seed_rows": per_seed_rows,
		"summary_rows": summary_rows,
		"metric_protocol": {
			"lyapunov_burn_steps": int(lyapunov_burn_steps),
			"lyapunov_rollout_steps": int(lyapunov_rollout_steps),
			"lambda1_core_reference": float(ref_spectrum_per_unit_time[0]),
			"h_lambda_core_reference": float(positive_lyap_sum(ref_spectrum_per_unit_time)),
			"lambda1_stoch_reference": float(stoch_ref_spectrum[0]),
			"h_lambda_stoch_reference": float(positive_lyap_sum(stoch_ref_spectrum)),
		},
	}


def aggregate_lorenz96_main_results(
	config: dict[str, Any],
	core_root: Path,
	probabilistic_root: Path,
	*,
	seeds: list[int],
	smoke: bool,
	device: torch.device,
	on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
	dynamical = compute_lorenz96_dynamical_metrics(
		config,
		core_root,
		probabilistic_root,
		seeds=seeds,
		smoke=smoke,
		device=device,
		on_progress=on_progress,
	)
	predictive = aggregate_lorenz96_predictive_results(
		config,
		probabilistic_root,
		seeds=seeds,
		smoke=smoke,
		device=device,
		on_progress=on_progress,
	)

	merged: dict[tuple[int, str], dict[str, Any]] = {}
	for row in dynamical["per_seed_rows"]:
		merged[(int(row["seed"]), str(row["method_key"]))] = dict(row)
	for row in predictive["per_seed_rows"]:
		key = (int(row["seed"]), str(row["method_key"]))
		if key in merged:
			merged[key].update(row)
		else:
			merged[key] = dict(row)
	per_seed_rows = sorted(merged.values(), key=lambda item: (item["method_key"], int(item["seed"])))

	summary_by_method: dict[str, dict[str, Any]] = {}
	for row in dynamical["summary_rows"]:
		summary_by_method[str(row["method_key"])] = dict(row)
	for row in predictive["summary_rows"]:
		key = str(row["method_key"])
		if key in summary_by_method:
			summary_by_method[key].update(row)
		else:
			summary_by_method[key] = dict(row)
	summary_rows = list(summary_by_method.values())

	return {
		"per_seed_rows": per_seed_rows,
		"summary_rows": summary_rows,
		"metric_protocol": dynamical["metric_protocol"],
	}


__all__ = [
	"GTAN_TRANSPORT_FIELD",
	"GTAN_TRANSPORT_SUMMARY_FIELD",
	"aggregate_lorenz96_main_results",
	"aggregate_lorenz96_predictive_results",
	"compute_lorenz96_dynamical_metrics",
]
