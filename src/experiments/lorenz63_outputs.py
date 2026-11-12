"""Lorenz-63 evaluation helpers: dynamical and predictive metrics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.lorenz63 import (
	standardize_raw_series,
	unstandardize_series,
)
from experiments.lorenz63_main import load_lorenz63_eval_bundle, prepare_lorenz63_main_bundle
from metrics import (
	PredictiveEvalConfig,
	ensemble_path_nll,
	forecast_starts,
	l63_ekf_grounding_window_from_shell,
	lorenz63_lyapunov_spectrum_from_samples,
	lorenz63_rds_lyapunov_spectrum,
	lyap_error_dict,
	marginal_crps_ensemble,
	power_spectrum_error,
	robust_se_of_median,
	rollout_probabilistic_ensemble,
	state_space_divergence,
)
from models import ALRNN, ALRNNShell, rollout_free
from training import BASELINE_SPECS

CORE_CHECKPOINT_NAME = "alrnn_core.pt"
SHELL_CHECKPOINT_NAME = "alrnn_shell.pt"

L63_METHODS = (
	"kaffee",
	"frozen_core_shell",
	"filtered_no_propagation",
	"open_loop_tied",
	"open_loop_no_propagation",
)

METRIC_COLS = (
	"NLL",
	"CRPS",
	"G_tan_transport",
	"D_stsp_stoch",
	"abs_signed_h_lambda_error_stoch",
	"D_stsp_core",
	"abs_signed_h_lambda_error_core",
	"D_H_core",
)

PER_SEED_FIELDS = (
	"baseline",
	"label",
	"seed",
	"NLL",
	"NLL_n_starts",
	"CRPS",
	"CRPS_n_starts",
	"predictive_ensemble_size",
	"forecast_horizon",
	"predictive_test_steps",
	"G_tan",
	"G_tan_transport",
	"G_tan_n_samples",
	"D_stsp_stoch",
	"D_stsp_stoch_n_steps",
	"D_stsp_core",
	"D_stsp_core_n_steps",
	"D_H_core",
	"state_space_burn_steps",
	"lyapunov_n_jacobians",
	"signed_LE_error_stoch",
	"abs_signed_LE_error_stoch",
	"h_lambda_stoch",
	"signed_h_lambda_error_stoch",
	"abs_signed_h_lambda_error_stoch",
	"signed_LE_error_core",
	"abs_signed_LE_error_core",
	"h_lambda_core",
	"signed_h_lambda_error_core",
	"abs_signed_h_lambda_error_core",
)


def _predictive_eval_config(config: dict[str, Any], *, smoke: bool) -> PredictiveEvalConfig:
	evaluation = dict(config["evaluation"])
	horizon = int(evaluation["main_forecast_horizon_steps"])
	nll_starts = int(evaluation["predictive_nll_starts"])
	forecast_n = int(evaluation["predictive_forecast_starts"])
	ensemble = int(evaluation["predictive_forecast_ensemble"])
	if bool(smoke):
		nll_starts = max(2, min(nll_starts, 4))
		forecast_n = max(2, min(forecast_n, 4))
		ensemble = max(8, min(ensemble, 16))
	return PredictiveEvalConfig(
		forecast_horizon=horizon,
		forecast_starts=forecast_n,
		nll_starts=nll_starts,
		forecast_ensemble=ensemble,
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
		"state_space_rollout_steps": _scaled("state_space_rollout_steps", 50_000),
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


def _core_standardizer(checkpoint: dict[str, Any]) -> dict[str, np.ndarray]:
	if "standardizer" in checkpoint:
		mean = np.asarray(checkpoint["standardizer"]["mean"], dtype=np.float64).reshape(-1)
		std = np.asarray(checkpoint["standardizer"]["std"], dtype=np.float64).reshape(-1)
	else:
		mean = np.asarray(checkpoint["mean"], dtype=np.float64).reshape(-1)
		std = np.asarray(checkpoint["std"], dtype=np.float64).reshape(-1)
	if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0.0):
		raise ValueError(f"Lorenz-63 core standardizer must be length-3 with positive std, got {mean}/{std}")
	return {"mean": mean, "std": std}


def _require_standardizer_match(checkpoint: dict[str, Any], dataset: dict[str, Any], *, seed: int) -> None:
	core = _core_standardizer(checkpoint)
	data = dataset["standardizer"]
	for field in ("mean", "std"):
		left = np.asarray(core[field], dtype=np.float64).reshape(-1)
		right = np.asarray(data[field], dtype=np.float64).reshape(-1)
		if left.shape != right.shape or not np.array_equal(left, right):
			raise ValueError(
				f"Seed {int(seed)} core-checkpoint and training-npz standardizer {field!r} do not match"
			)


def _load_core_checkpoint(core_root: Path, seed: int) -> dict[str, Any]:
	path = Path(core_root) / f"seed{int(seed):02d}" / CORE_CHECKPOINT_NAME
	if not path.exists():
		raise FileNotFoundError(f"Missing Lorenz-63 core checkpoint: {path}")
	return torch.load(path, map_location="cpu", weights_only=False)


def _load_shell_checkpoint(shell_root: Path, seed: int, baseline: str) -> dict[str, Any]:
	path = Path(shell_root) / f"seed{int(seed):02d}" / baseline / SHELL_CHECKPOINT_NAME
	if not path.exists():
		raise FileNotFoundError(f"Missing Lorenz-63 shell checkpoint: {path}")
	return torch.load(path, map_location="cpu", weights_only=False)


def _load_shell_model(
	checkpoint: dict[str, Any],
	*,
	baseline: str,
	config: dict[str, Any],
	device: torch.device,
) -> ALRNNShell:
	state = checkpoint["state_dict"]
	spec = BASELINE_SPECS[baseline]
	model = ALRNNShell(
		latent_dim=int(state["A"].numel()),
		gated_dim=int(config["model"]["gated_dim"]),
		observed_dim=int(state["embedding"].shape[0]),
		A_init=state["A"].to(device=device, dtype=torch.float32),
		W_init=state["W"].to(device=device, dtype=torch.float32),
		h_init=state["h"].to(device=device, dtype=torch.float32),
		embedding_init=state["embedding"].to(device=device, dtype=torch.float32),
		proc_std_init=torch.exp(state["log_proc_std"]).to(device=device, dtype=torch.float32),
		obs_std_init=torch.exp(state["log_obs_std"]).to(device=device, dtype=torch.float32),
		train_core=bool(spec.train_core),
		covariance_transport=str(spec.covariance_transport),
	).to(device)
	model.load_state_dict(state)
	if "proc_std_exact" in checkpoint and "obs_std_exact" in checkpoint:
		model.set_sampling_std_exact(
			proc_std=torch.as_tensor(checkpoint["proc_std_exact"], dtype=torch.float32, device=device),
			obs_std=torch.as_tensor(checkpoint["obs_std_exact"], dtype=torch.float32, device=device),
		)
	return model.eval()


def _core_from_shell(shell: ALRNNShell, device: torch.device) -> ALRNN:
	core = ALRNN(
		latent_dim=int(shell.latent_dim),
		gated_dim=int(shell.gated_dim),
		observed_dim=int(shell.observed_dim),
	).to(device)
	core.A.data.copy_(shell.A.detach())
	core.W.data.copy_(shell.W.detach())
	core.h.data.copy_(shell.h.detach())
	core.B.data.copy_(shell.embedding.detach())
	return core.eval()


@torch.no_grad()
def _l63_window(raw: np.ndarray, *, start: int, burn: int, roll: int) -> np.ndarray:
	stop = int(start) + int(burn) + int(roll) + 1
	series = np.asarray(raw, dtype=np.float64)
	if series.ndim != 2 or int(series.shape[1]) != 3:
		raise ValueError(f"Lorenz-63 Lyapunov reference must have shape (T, 3), got {series.shape}")
	if int(start) < 0 or stop > int(series.shape[0]):
		raise ValueError(
			f"Lyapunov window [{int(start)}:{stop}] does not fit a trajectory of length {series.shape[0]}"
		)
	return series[int(start) : stop]


def _l63_clean_spectrum(
	raw: np.ndarray,
	*,
	start: int,
	burn: int,
	roll: int,
	dt: float,
) -> np.ndarray:
	"""Deterministic Lorenz-63 spectrum along the clean latent reference."""
	return lorenz63_lyapunov_spectrum_from_samples(
		_l63_window(raw, start=start, burn=burn, roll=roll),
		dt=float(dt),
		burn_in=int(burn),
		max_T=int(burn) + int(roll),
	)


def _l63_process_noisy_spectrum(
	raw: np.ndarray,
	*,
	start: int,
	burn: int,
	roll: int,
	dt: float,
) -> np.ndarray:
	"""Process-noisy Lorenz-63 spectrum along the latent trajectory. Observation noise is excluded."""
	return lorenz63_rds_lyapunov_spectrum(
		_l63_window(raw, start=start, burn=burn, roll=roll),
		dt=float(dt),
		burn_in=int(burn),
		max_T=int(burn) + int(roll),
	)


def _evaluate_one(
	*,
	config: dict[str, Any],
	seed: int,
	baseline: str,
	shell: ALRNNShell,
	standardizer: dict[str, Any],
	evaluation: dict[str, Any],
	predictive: PredictiveEvalConfig,
	dynamical: dict[str, int | float],
	device: torch.device,
	smoke: bool,
) -> dict[str, Any]:
	system = dict(config["system"])
	dt = float(system["dt"])
	predictive_steps = int(system["T_test"]) if not smoke else min(int(system["T_test"]), 256)
	test_obs_raw = np.asarray(evaluation["test_obs_raw"], dtype=np.float64)
	test_clean_raw = np.asarray(evaluation["test_clean_raw"], dtype=np.float64)
	if int(test_obs_raw.shape[0]) < predictive_steps:
		raise ValueError(
			f"Predictive trajectory has {test_obs_raw.shape[0]} steps; need {predictive_steps}"
		)
	test_obs = standardize_raw_series(test_obs_raw, standardizer).astype(np.float32)
	test_clean = standardize_raw_series(test_clean_raw, standardizer).astype(np.float32)
	predictive_obs = test_obs[:predictive_steps]
	predictive_obs_raw = test_obs_raw[:predictive_steps]

	horizon = int(predictive.forecast_horizon)
	nll_starts = forecast_starts(predictive_steps, horizon, int(predictive.nll_starts))
	nll_vals: list[float] = []
	for start in nll_starts:
		ensemble_std = rollout_probabilistic_ensemble(
			shell,
			predictive_obs[int(start)],
			length=horizon,
			n_ensemble=int(predictive.forecast_ensemble),
			seed=_predictive_ensemble_seed(seed=int(seed), start=int(start)),
			sample_process_noise=True,
			sample_obs_noise=True,
		)
		ensemble_raw = unstandardize_series(ensemble_std, standardizer)
		target = predictive_obs_raw[int(start) + 1 : int(start) + horizon + 1]
		nll_vals.append(float(ensemble_path_nll(target, ensemble_raw, floor=float(predictive.obs_floor))))

	crps_starts = forecast_starts(predictive_steps, horizon, int(predictive.forecast_starts))
	crps_vals: list[float] = []
	for start in crps_starts:
		ensemble_std = rollout_probabilistic_ensemble(
			shell,
			predictive_obs[int(start)],
			length=horizon,
			n_ensemble=int(predictive.forecast_ensemble),
			seed=_predictive_ensemble_seed(seed=int(seed), start=int(start)),
			sample_process_noise=True,
			sample_obs_noise=True,
		)
		ensemble_raw = unstandardize_series(ensemble_std, standardizer)
		target = predictive_obs_raw[int(start) + horizon]
		crps_vals.append(float(marginal_crps_ensemble(target, ensemble_raw[:, -1, :])))

	gtan_start = int(dynamical["gtan_window_start"])
	gtan_warm = int(dynamical["gtan_warmup_steps"])
	gtan_len = int(dynamical["gtan_window_steps"])
	slice_start = gtan_start - gtan_warm
	slice_end = gtan_start + gtan_len + 1
	if slice_start < 0 or slice_end > int(test_obs.shape[0]):
		raise ValueError(
			f"G_tan slice [{slice_start}:{slice_end}] is invalid for trajectory length {test_obs.shape[0]}"
		)
	gtan_res = l63_ekf_grounding_window_from_shell(
		shell,
		test_obs[slice_start:slice_end],
		test_clean[slice_start:slice_end],
		mean=standardizer["mean"],
		std=standardizer["std"],
		dt=dt,
		filter_warmup=gtan_warm,
	)

	burn = int(dynamical["state_space_burn_steps"])
	roll = int(dynamical["state_space_rollout_steps"])
	lyap_burn = int(dynamical["lyapunov_burn_steps"])
	lyap_roll = int(dynamical["lyapunov_rollout_steps"])
	if lyap_burn + lyap_roll > burn + roll:
		raise ValueError("Lyapunov rollout exceeds the generated state-space rollout")

	stoch_start = int(dynamical["stochastic_start_index"])
	stochastic_obs = standardize_raw_series(evaluation["stochastic_obs_raw"], standardizer)
	if int(stochastic_obs.shape[0]) < stoch_start + roll:
		raise ValueError(
			f"Stochastic reference has {stochastic_obs.shape[0]} steps; need {stoch_start + roll}"
		)
	latent0 = np.asarray(
		standardize_raw_series(evaluation["stochastic_clean_raw"][stoch_start : stoch_start + 1], standardizer)[0],
		dtype=np.float32,
	)
	x0 = np.asarray(stochastic_obs[stoch_start], dtype=np.float32)
	gen_full = np.asarray(
		shell.rollout_observation_sequence(
			x0,
			steps=burn + roll,
			seed=int(seed),
			sample_process_noise=True,
			sample_obs_noise=True,
		),
		dtype=np.float64,
	)
	gen_stoch = gen_full[burn : burn + roll]
	ref_stoch = np.asarray(stochastic_obs[stoch_start : stoch_start + roll], dtype=np.float64)
	d_stsp_stoch = float(
		state_space_divergence(
			gen_stoch,
			ref_stoch,
			sigma2=float(dynamical["state_space_sigma2"]),
			max_T=roll,
			mc_samples=int(dynamical["state_space_mc_samples"]),
			batch_size=int(dynamical["state_space_density_batch_size"]),
		)
	)
	clean_ref = np.asarray(evaluation["l63_clean_lyapunov"], dtype=np.float64)
	stoch_ref = np.asarray(evaluation["l63_stoch_lyapunov"], dtype=np.float64)
	stoch_spectrum = shell.lyapunov_spectrum(
		latent0,
		steps=lyap_burn + lyap_roll,
		burn=lyap_burn,
		seed=int(seed),
		sample_process_noise=True,
	)
	stoch_le = lyap_error_dict(stoch_spectrum, stoch_ref, dt=dt, suffix="_stoch")

	core_model = _core_from_shell(shell, device)
	core_start = int(dynamical["core_start_index"])
	test_core = standardize_raw_series(evaluation["test_core_raw"], standardizer)
	if int(test_core.shape[0]) < core_start + roll:
		raise ValueError(f"Core reference has {test_core.shape[0]} steps; need {core_start + roll}")
	core_full = (
		rollout_free(
			core_model,
			torch.as_tensor(test_core[core_start : core_start + 1], dtype=torch.float32, device=device),
			steps=burn + roll,
		)[0, :, : int(core_model.observed_dim)]
		.detach()
		.cpu()
		.numpy()
		.astype(np.float64)
	)
	gen_core = core_full[burn : burn + roll]
	ref_core = np.asarray(test_core[core_start : core_start + roll], dtype=np.float64)
	d_stsp_core = float(
		state_space_divergence(
			gen_core,
			ref_core,
			sigma2=float(dynamical["state_space_sigma2"]),
			max_T=roll,
			mc_samples=int(dynamical["state_space_mc_samples"]),
			batch_size=int(dynamical["state_space_density_batch_size"]),
		)
	)
	d_h = float(
		power_spectrum_error(
			unstandardize_series(ref_core, standardizer),
			unstandardize_series(gen_core, standardizer),
			smoothing=int(dynamical["pse_smoothing"]),
		)
	)
	core_spectrum = core_model.lyapunov_spectrum(
		test_core[core_start],
		steps=lyap_burn + lyap_roll,
		burn=lyap_burn,
	)
	core_le = lyap_error_dict(core_spectrum, clean_ref, dt=dt, suffix="_core")

	return {
		"baseline": baseline,
		"label": BASELINE_SPECS[baseline].label,
		"seed": int(seed),
		"NLL": float(np.mean(nll_vals)),
		"NLL_n_starts": int(len(nll_vals)),
		"CRPS": float(np.mean(crps_vals)),
		"CRPS_n_starts": int(len(crps_vals)),
		"predictive_ensemble_size": int(predictive.forecast_ensemble),
		"forecast_horizon": horizon,
		"predictive_test_steps": predictive_steps,
		"G_tan": float(gtan_res.g_tan),
		"G_tan_transport": float(gtan_res.g_tan_transport),
		"G_tan_n_samples": int(gtan_res.n_samples),
		"D_stsp_stoch": d_stsp_stoch,
		"D_stsp_stoch_n_steps": int(roll),
		"D_stsp_core": d_stsp_core,
		"D_stsp_core_n_steps": int(roll),
		"D_H_core": d_h,
		"state_space_burn_steps": int(burn),
		"lyapunov_n_jacobians": int(lyap_burn + lyap_roll),
		**stoch_le,
		**core_le,
	}


def _aggregate_rows(
	rows: list[dict[str, Any]],
	*,
	baselines: list[str],
	seeds: list[int],
) -> list[dict[str, Any]]:
	expected = [int(seed) for seed in seeds]
	expected_set = set(expected)
	agg: list[dict[str, Any]] = []
	for baseline in baselines:
		group = [row for row in rows if str(row["baseline"]) == str(baseline)]
		observed = [int(row["seed"]) for row in group]
		if len(observed) != len(set(observed)):
			raise ValueError(f"Duplicate Lorenz-63 rows for baseline {baseline!r}: {observed}")
		if set(observed) != expected_set:
			raise ValueError(
				f"Baseline {baseline!r} has seeds {sorted(set(observed))}; expected {sorted(expected_set)}"
			)
		out: dict[str, Any] = {
			"baseline": baseline,
			"label": str(group[0]["label"]),
			"n": int(len(observed)),
			"n_expected": int(len(expected)),
		}
		for col in METRIC_COLS:
			values = [float(row[col]) for row in group if np.isfinite(float(row[col]))]
			out[f"{col}_n"] = int(len(values))
			if len(values) != len(expected):
				raise ValueError(
					f"Baseline {baseline!r} metric {col!r} has {len(values)} finite values; "
					f"expected {len(expected)}"
				)
			se = robust_se_of_median(values)
			out[f"{col}_median"] = float(np.median(values))
			out[f"{col}_se"] = float(se) if se is not None else float("nan")
		agg.append(out)
	return agg


def aggregate_lorenz63_main_results(
	config: dict[str, Any],
	core_root: Path,
	shell_root: Path,
	*,
	seeds: list[int],
	baseline_keys: list[str] | None = None,
	smoke: bool,
	device: torch.device,
	on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
	baselines = list(baseline_keys or config.get("baselines") or L63_METHODS)
	unknown = [key for key in baselines if key not in L63_METHODS]
	if unknown:
		raise ValueError(f"Lorenz-63 appendix supports internal methods only, got {unknown}")
	evaluation = load_lorenz63_eval_bundle(config)
	predictive = _predictive_eval_config(config, smoke=bool(smoke))
	dynamical = _dynamical_eval_config(config, smoke=bool(smoke))
	dt = float(config["system"]["dt"])
	evaluation["l63_clean_lyapunov"] = _l63_clean_spectrum(
		evaluation["test_core_raw"],
		start=int(dynamical["core_start_index"]),
		burn=int(dynamical["lyapunov_burn_steps"]),
		roll=int(dynamical["lyapunov_rollout_steps"]),
		dt=dt,
	)
	evaluation["l63_stoch_lyapunov"] = _l63_process_noisy_spectrum(
		evaluation["stochastic_clean_raw"],
		start=int(dynamical["stochastic_start_index"]),
		burn=int(dynamical["lyapunov_burn_steps"]),
		roll=int(dynamical["lyapunov_rollout_steps"]),
		dt=dt,
	)
	per_seed_rows: list[dict[str, Any]] = []

	for seed in seeds:
		prepared = prepare_lorenz63_main_bundle(config, smoke=bool(smoke))
		core_checkpoint = _load_core_checkpoint(core_root, int(seed))
		_require_standardizer_match(core_checkpoint, prepared["dataset"], seed=int(seed))
		standardizer = _core_standardizer(core_checkpoint)
		for baseline in baselines:
			checkpoint = _load_shell_checkpoint(shell_root, int(seed), baseline)
			shell = _load_shell_model(
				checkpoint,
				baseline=baseline,
				config=config,
				device=device,
			)
			row = _evaluate_one(
				config=config,
				seed=int(seed),
				baseline=baseline,
				shell=shell,
				standardizer=standardizer,
				evaluation=evaluation,
				predictive=predictive,
				dynamical=dynamical,
				device=device,
				smoke=bool(smoke),
			)
			per_seed_rows.append({key: row[key] for key in PER_SEED_FIELDS})
			if on_progress is not None:
				on_progress(f"seed {int(seed):02d} {baseline}")

	summary_rows = _aggregate_rows(per_seed_rows, baselines=baselines, seeds=list(seeds))
	clean_ref = np.asarray(evaluation["l63_clean_lyapunov"], dtype=np.float64)
	stoch_ref = np.asarray(evaluation["l63_stoch_lyapunov"], dtype=np.float64)
	return {
		"per_seed_rows": per_seed_rows,
		"summary_rows": summary_rows,
		"metric_protocol": {
			"forecast_horizon": int(predictive.forecast_horizon),
			"predictive_nll_starts": int(predictive.nll_starts),
			"predictive_forecast_starts": int(predictive.forecast_starts),
			"predictive_forecast_ensemble": int(predictive.forecast_ensemble),
			"lyapunov_burn_steps": int(dynamical["lyapunov_burn_steps"]),
			"lyapunov_rollout_steps": int(dynamical["lyapunov_rollout_steps"]),
			"lyapunov_reference_core": "variational RK4 along eval_references.npz:X_test_core_raw",
			"lyapunov_reference_stoch": "RK4-map Jacobian along stochastic_evaluation.npz:X_stochastic_clean_raw",
			"lambda1_core_reference": float(clean_ref[0]),
			"h_lambda_core_reference": float(np.sum(clean_ref[clean_ref > 0.0])),
			"lambda1_stoch_reference": float(stoch_ref[0]),
			"h_lambda_stoch_reference": float(np.sum(stoch_ref[stoch_ref > 0.0])),
		},
	}


__all__ = [
	"L63_METHODS",
	"METRIC_COLS",
	"aggregate_lorenz63_main_results",
]
