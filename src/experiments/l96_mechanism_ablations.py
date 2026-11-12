"""KAFFEE mechanism ablations on Lorenz-96: Update-SG and GTF-alpha.

Update-SG stops gradients through the Kalman update. GTF-alpha replaces the learned gain by a scalar "forcing strength" K = alpha B^T to mimic the fixed "gain" of generalized teacher forcing (GTF, see https://arxiv.org/abs/2306.04406), on alpha in {0, 0.1, 0.25, 0.5, 1}. Training and scoring use the main Lorenz-96 experiment protocol. 

Retrain from "models/lorenz96_main_cores" via
"scripts/run_l96_mechanism_ablations.py".
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.lorenz96_main import prepare_lorenz96_main_bundle
from experiments.lorenz96_outputs import (
	_core_metric_block,
	_dynamical_eval_config,
	_evaluate_grounding,
	_evaluate_probabilistic_predictive,
	_load_reference_bundle,
	_predictive_eval_config,
	_process_noisy_lorenz96_spectrum,
	_reference_lorenz96_spectrum,
	_shell_rollout,
	_standardize_raw_series,
	_stochastic_metric_block,
)
from experiments.lorenz96_pipeline import write_csv, write_json
from filtering.covariance import joseph_update_identity_observation
from filtering.ekf import (
	innovation_log_likelihood,
	innovation_statistics_identity_observation,
	kalman_predict,
	kalman_update_identity_observation,
)
from metrics import robust_se_of_median
from models import ALRNNShell
from training.train_alrnn_probabilistic import (
	_SELECTOR_TRACE_VAL_WINDOW_SEED_BASE,
	_SELECTOR_TRACE_VAL_WINDOWS,
	_initialize_shell_statistics,
	_make_optimizer,
	_make_scheduler,
	_prepare_fixed_val_windows,
	_sample_train_windows,
	_seed_everything,
	_select_best_trace_row,
	_selector_d_stsp_stoch,
	_trace_iteration,
	_trace_row_for_selection,
	_trace_sort_key,
)

METHOD_UPDATE_SG = "update_sg"
GTF_ALPHA_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
SHELL_CHECKPOINT_NAME = "alrnn_shell.pt"
SUMMARY_METRIC_FIELDS: tuple[str, ...] = (
	"NLL",
	"CRPS",
	"G_tan_transport",
	"D_stsp_stoch",
	"abs_signed_h_lambda_error_stoch",
	"D_stsp_core",
	"abs_signed_h_lambda_error_core",
	"D_H_core",
)


def gtf_method_key(alpha: float) -> str:
	value = float(alpha)
	if value == 0.0:
		return "gtf_alpha_0"
	if value == 1.0:
		return "gtf_alpha_1"
	return f"gtf_alpha_{format(value, 'g')}"


def parse_gtf_alpha(method_key: str) -> float | None:
	text = str(method_key).strip()
	prefix = "gtf_alpha_"
	if not text.startswith(prefix):
		return None
	try:
		alpha = float(text[len(prefix) :])
	except ValueError:
		return None
	for allowed in GTF_ALPHA_GRID:
		if abs(alpha - float(allowed)) <= 1.0e-12:
			return float(allowed)
	return None


def canonicalize_method_key(method_key: str) -> str:
	text = str(method_key).strip()
	if text == METHOD_UPDATE_SG:
		return METHOD_UPDATE_SG
	alpha = parse_gtf_alpha(text)
	if alpha is None:
		raise ValueError(f"Unknown method: {method_key!r}")
	return gtf_method_key(alpha)


def method_label(method_key: str) -> str:
	key = canonicalize_method_key(method_key)
	if key == METHOD_UPDATE_SG:
		return "Update-SG"
	alpha = parse_gtf_alpha(key)
	assert alpha is not None
	return f"GTF-alpha {alpha:g}"


def ablation_method_keys() -> list[str]:
	return [METHOD_UPDATE_SG, *[gtf_method_key(alpha) for alpha in GTF_ALPHA_GRID]]


def _fixed_gain_identity_update(
	z_mean_pred: torch.Tensor,
	Sigma_pred: torch.Tensor,
	x_t: torch.Tensor,
	*,
	Lambda: torch.Tensor,
	alpha: float,
	Omega_jitter: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	residual, Omega_t, _sigma_zx = innovation_statistics_identity_observation(
		z_mean_pred,
		Sigma_pred,
		x_t,
		Lambda=Lambda,
		Omega_jitter=Omega_jitter,
	)
	observed_dim = int(x_t.shape[-1])
	gain = z_mean_pred.new_zeros(*z_mean_pred.shape[:-1], int(z_mean_pred.shape[-1]), observed_dim)
	gain[..., :observed_dim, :] = float(alpha) * torch.eye(
		observed_dim, device=z_mean_pred.device, dtype=z_mean_pred.dtype
	)
	z_mean_filt = z_mean_pred + torch.matmul(gain, residual.unsqueeze(-1)).squeeze(-1)
	Sigma_filt = joseph_update_identity_observation(Sigma_pred, gain, Lambda)
	return z_mean_filt, Sigma_filt, innovation_log_likelihood(residual, Omega_t)


def _filtered_nll(
	model: ALRNNShell,
	x_windows: torch.Tensor,
	*,
	method_key: str,
	burn_in_steps: int,
	innovation_jitter: float = 1e-6,
) -> torch.Tensor:
	if x_windows.ndim != 3:
		raise ValueError("x_windows must have shape (batch, time, observed_dim)")
	z_mean, Sigma = model.initialize_from_observation(x_windows[:, 0, :])
	Gamma = model.Gamma.to(device=z_mean.device, dtype=z_mean.dtype)
	Lambda = model.Lambda.to(device=z_mean.device, dtype=z_mean.dtype)
	total_nll = torch.zeros((), device=z_mean.device, dtype=z_mean.dtype)
	total_steps = x_windows.shape[1] - 1
	alpha = parse_gtf_alpha(method_key)
	for step in range(total_steps):
		transported_covariance, z_mean_pred = model.transition_and_covariance(z_mean, Sigma)
		z_mean_pred, Sigma_pred = kalman_predict(
			z_mean,
			Sigma,
			Gamma=Gamma,
			z_mean_pred=z_mean_pred,
			transported_covariance=transported_covariance,
			covariance_transport="jacobian",
		)
		x_t = x_windows[:, step + 1, :]
		if method_key == METHOD_UPDATE_SG:
			z_post, Sigma_post, log_likelihood = kalman_update_identity_observation(
				z_mean_pred,
				Sigma_pred,
				x_t,
				Lambda=Lambda,
				Omega_jitter=innovation_jitter,
			)
			z_mean = z_post.detach() + (z_mean_pred - z_mean_pred.detach())
			Sigma = Sigma_post.detach() + (Sigma_pred - Sigma_pred.detach())
		else:
			assert alpha is not None
			z_mean, Sigma, log_likelihood = _fixed_gain_identity_update(
				z_mean_pred,
				Sigma_pred,
				x_t,
				Lambda=Lambda,
				alpha=float(alpha),
				Omega_jitter=innovation_jitter,
			)
		if step >= int(burn_in_steps):
			total_nll = total_nll - log_likelihood.sum()
	return total_nll


@torch.no_grad()
def _heldout_val_nll_per_step(
	model: ALRNNShell,
	val_windows: torch.Tensor,
	*,
	method_key: str,
	burn_in_steps: int,
	scored_steps: int,
) -> float:
	batch = int(val_windows.shape[0])
	if batch < 1:
		return float("nan")
	total_nll = _filtered_nll(
		model,
		val_windows,
		method_key=method_key,
		burn_in_steps=int(burn_in_steps),
	)
	return float(total_nll.item()) / (float(batch) * float(scored_steps))


def run_isolated_shell(
	checkpoint: dict[str, Any],
	dataset: dict[str, Any],
	protocol: dict[str, Any],
	*,
	method_key: str,
	seed: int,
	device: torch.device,
	max_iterations: int | None = None,
	batch_size_override: int | None = None,
	eval_every: int | None = None,
) -> dict[str, Any]:
	method_key = canonicalize_method_key(method_key)
	_seed_everything(int(seed))
	probabilistic = dict(protocol["probabilistic_training"])
	evaluation = dict(protocol.get("evaluation", {}))
	x_obs_train = np.asarray(dataset["train_obs"], dtype=np.float32)
	x_obs_test = np.asarray(dataset["test_obs"], dtype=np.float32)
	total_steps = int(probabilistic["window_total_steps"])
	burn_in_steps = int(probabilistic["covariance_burn_in_steps"])
	scored_steps = int(probabilistic["scored_steps"])
	if total_steps - burn_in_steps != scored_steps:
		raise ValueError("Probabilistic window settings are inconsistent")

	batch_size = (
		int(batch_size_override)
		if batch_size_override is not None
		else int(protocol["training"]["batch_size"])
	)
	shell_init = _initialize_shell_statistics(checkpoint, dataset, protocol, device=device)
	iterations = int(max_iterations) if max_iterations is not None else int(probabilistic["iterations"])
	optimizer_name = str(probabilistic.get("optimizer", "adam"))
	weight_decay = float(probabilistic.get("weight_decay", probabilistic.get("wd", 0.0)))
	scheduler_name = str(probabilistic.get("scheduler", "constant"))
	eta_min = probabilistic.get("eta_min")
	gamma = probabilistic.get("gamma")
	gradient_clip = float(probabilistic.get("gradient_clip", 0.0))
	lr = float(probabilistic.get("lr", 0.005))
	trace_every = int(eval_every) if eval_every is not None else int(probabilistic.get("trace_every", 1))
	val_window_seed = int(_SELECTOR_TRACE_VAL_WINDOW_SEED_BASE) + int(seed)
	selector_burn = int(probabilistic.get("selector_burn", evaluation.get("core_start_index", 500)))
	selector_roll = int(probabilistic.get("selector_roll", 10_000))
	selector_sigma2 = float(probabilistic.get("selector_sigma2", evaluation.get("state_space_sigma2", 1.0)))
	selector_mc_samples = int(
		probabilistic.get("selector_mc_samples", evaluation.get("state_space_mc_samples", 1000))
	)
	selector_chunk = int(
		probabilistic.get("selector_chunk", evaluation.get("state_space_density_batch_size", 512))
	)
	loss_normalizer = float(batch_size) * float(scored_steps)

	_seed_everything(42 + int(seed))
	model = ALRNNShell.from_core_checkpoint(
		checkpoint,
		proc_std_init=torch.as_tensor(shell_init["gamma_diag_std"], dtype=torch.float32, device=device),
		obs_std_init=torch.as_tensor(shell_init["lambda_diag_std"], dtype=torch.float32, device=device),
		train_core=True,
		covariance_transport="jacobian",
	).to(device)
	selector_seed = 50_000 + int(seed)
	trainable = [p for p in model.parameters() if p.requires_grad]
	optimizer = _make_optimizer(trainable, name=optimizer_name, lr=lr, weight_decay=weight_decay)
	scheduler = _make_scheduler(
		optimizer,
		name=scheduler_name,
		n_iters=iterations,
		eta_min=float(eta_min) if eta_min is not None else None,
		gamma=float(gamma) if gamma is not None else None,
	)
	train_rng = np.random.default_rng(44 + int(seed))
	val_windows_t = torch.as_tensor(
		_prepare_fixed_val_windows(
			x_obs_test,
			n_windows=int(_SELECTOR_TRACE_VAL_WINDOWS),
			total_steps=total_steps,
			seed=val_window_seed,
		),
		dtype=torch.float32,
		device=device,
	)
	trace_rows: list[dict[str, float | int]] = []
	train_nll_history: list[float] = []
	best_state_dict: dict[str, torch.Tensor] | None = None
	best_trace_row: dict[str, float] | None = None

	for iteration in range(iterations):
		model.train()
		x_windows = torch.as_tensor(
			_sample_train_windows(
				x_obs_train,
				total_steps=total_steps,
				batch_size=batch_size,
				rng=train_rng,
			),
			dtype=torch.float32,
			device=device,
		)
		optimizer.zero_grad(set_to_none=True)
		total_nll = _filtered_nll(
			model,
			x_windows,
			method_key=method_key,
			burn_in_steps=burn_in_steps,
		)
		loss = total_nll / loss_normalizer
		loss.backward()
		if gradient_clip > 0.0:
			torch.nn.utils.clip_grad_norm_(trainable, float(gradient_clip))
		optimizer.step()
		scheduler.step()

		train_nll_per_step = float(total_nll.item()) / loss_normalizer
		train_nll_history.append(train_nll_per_step)
		should_trace = (
			iteration == 0
			or (iteration + 1) == int(iterations)
			or ((iteration + 1) % int(trace_every) == 0)
		)
		if should_trace:
			model.eval()
			val_nll = _heldout_val_nll_per_step(
				model,
				val_windows_t,
				method_key=method_key,
				burn_in_steps=burn_in_steps,
				scored_steps=scored_steps,
			)
			row: dict[str, float | int] = {
				"iter": int(iteration + 1),
				"iteration": int(iteration + 1),
				"train_nll": float(train_nll_per_step),
				"train_nll_per_step": float(train_nll_per_step),
				"val_nll": float(val_nll),
				"D_stsp_stoch": _selector_d_stsp_stoch(
					model,
					x_obs_test,
					device=device,
					seed=selector_seed,
					burn=selector_burn,
					roll=selector_roll,
					sigma2=selector_sigma2,
					mc_samples=selector_mc_samples,
					density_batch_size=selector_chunk,
				),
				"Gamma_diag_mean": float(torch.diag(model.Gamma).mean().detach().item()),
				"Lambda_diag_mean": float(torch.diag(model.Lambda).mean().detach().item()),
				"lr": float(optimizer.param_groups[0]["lr"]),
			}
			trace_rows.append(row)
			candidate = _trace_row_for_selection(row)
			if best_trace_row is None or _trace_sort_key(candidate) < _trace_sort_key(best_trace_row):
				best_trace_row = candidate
				best_state_dict = copy.deepcopy(model.state_dict())
		if not np.isfinite(train_nll_per_step):
			break

	if best_state_dict is not None:
		model.load_state_dict(best_state_dict)
	final_row = trace_rows[-1] if trace_rows else {"iter": int(iterations), "train_nll_per_step": float("nan")}
	selected_row = _select_best_trace_row(trace_rows)
	if selected_row is None:
		selected_row = final_row
	return {
		"method_key": method_key,
		"method_label": method_label(method_key),
		"model": model,
		"history": trace_rows,
		"train_nll_history": train_nll_history,
		"selection_metric": "D_stsp_stoch",
		"selection_value": float(selected_row.get("D_stsp_stoch", float("nan"))),
		"selected_iteration": _trace_iteration(selected_row),
		"selected_val_nll": float(selected_row.get("val_nll", float("nan"))),
		"final_metrics": {
			"innovation_nll_per_step": float(final_row.get("train_nll_per_step", float("nan"))),
			"final_iteration": _trace_iteration(final_row),
			"final_train_nll_per_step": float(final_row.get("train_nll_per_step", float("nan"))),
			"selected_train_nll_per_step": float(selected_row.get("train_nll_per_step", float("nan"))),
			"selected_iteration": _trace_iteration(selected_row),
			"selected_D_stsp_stoch": float(selected_row.get("D_stsp_stoch", float("nan"))),
			"selected_val_nll": float(selected_row.get("val_nll", float("nan"))),
			"final_val_nll": float(final_row.get("val_nll", float("nan"))),
		},
		"shell_init": {
			"x_residual_std": np.asarray(shell_init["x_residual_std"], dtype=np.float32).tolist(),
			"lambda_diag_std": np.asarray(shell_init["lambda_diag_std"], dtype=np.float32).tolist(),
			"gamma_scalar_std": float(shell_init["gamma_scalar_std"]),
		},
		"train_config": {
			"optimizer": optimizer_name,
			"lr": float(lr),
			"iterations": int(iterations),
			"weight_decay": float(weight_decay),
			"scheduler": scheduler_name,
			"eta_min": float(eta_min) if eta_min is not None else None,
			"gradient_clip": float(gradient_clip),
			"shell_init_policy": str(shell_init["shell_init_policy"]),
			"trace_every": int(trace_every),
			"checkpoint_policy": "D_stsp_stoch",
			"selector_burn": int(selector_burn),
			"selector_roll": int(selector_roll),
			"gtf_alpha": parse_gtf_alpha(method_key),
			"use_update_sg": bool(method_key == METHOD_UPDATE_SG),
		},
	}


def load_isolated_shell(
	checkpoint: dict[str, Any],
	*,
	config: dict[str, Any],
	device: torch.device,
) -> ALRNNShell:
	state = checkpoint["state_dict"]
	model_cfg = dict(config["model"])
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
		train_core=True,
		covariance_transport="jacobian",
	).to(device)
	model.load_state_dict(state)
	if "proc_std_exact" in checkpoint and "obs_std_exact" in checkpoint:
		model.set_sampling_std_exact(
			proc_std=torch.as_tensor(checkpoint["proc_std_exact"], dtype=torch.float32, device=device),
			obs_std=torch.as_tensor(checkpoint["obs_std_exact"], dtype=torch.float32, device=device),
		)
	return model


def evaluate_isolated_shell(
	config: dict[str, Any],
	checkpoint: dict[str, Any],
	*,
	method_key: str,
	seed: int,
	device: torch.device,
	smoke: bool = False,
) -> dict[str, Any]:
	method_key = canonicalize_method_key(method_key)
	prepared = prepare_lorenz96_main_bundle(config, smoke=bool(smoke))
	heldout_test = prepared["heldout_test"]
	stochastic_evaluation = prepared["stochastic_evaluation"]
	standardizer = dict(prepared["dataset"]["standardizer"])
	system = dict(config["system"])
	dynamical = _dynamical_eval_config(config, smoke=bool(smoke))
	predictive = _predictive_eval_config(config)
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
	ref_spectrum_per_unit_time = _reference_lorenz96_spectrum(
		core_reference_raw,
		start_index=int(core_start_index),
		burn_steps=int(lyapunov_burn_steps),
		rollout_steps=int(lyapunov_rollout_steps),
		dt=dt,
	)
	stoch_ref_spectrum = _process_noisy_lorenz96_spectrum(
		np.asarray(stochastic_evaluation["clean_raw"], dtype=np.float64),
		start_index=int(stochastic_start_index),
		burn_steps=int(lyapunov_burn_steps),
		rollout_steps=int(lyapunov_rollout_steps),
		dt=dt,
		forcing=forcing,
	)
	shell = load_isolated_shell(checkpoint, config=config, device=device)
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
	core_metrics = _core_metric_block(
		core_rollout_std[int(state_space_burn_steps) : int(state_space_burn_steps) + int(state_space_rollout_steps)],
		shell_reference_std[int(core_start_index) : int(core_start_index) + int(state_space_rollout_steps)],
		standardizer,
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
		np.asarray(stochastic_evaluation["obs_std"], dtype=np.float32),
		np.asarray(stochastic_evaluation["clean_std"][int(stochastic_start_index)], dtype=np.float64),
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
	predictive_metrics = _evaluate_probabilistic_predictive(
		shell.to(device),
		np.asarray(heldout_test["obs_std"], dtype=np.float32),
		np.asarray(heldout_test["obs_raw"], dtype=np.float64),
		standardizer,
		predictive=predictive,
		seed=int(seed),
	)
	return {
		"seed": int(seed),
		"method_key": method_key,
		"method_label": method_label(method_key),
		**core_metrics,
		**stoch_metrics,
		**gtan_metrics,
		**predictive_metrics,
	}


def _scalar_eval_row(payload: dict[str, Any]) -> dict[str, Any]:
	method_key = canonicalize_method_key(str(payload["method_key"]))
	row: dict[str, Any] = {
		"seed": int(payload["seed"]),
		"method_key": method_key,
		"method_label": method_label(method_key),
	}
	for field in SUMMARY_METRIC_FIELDS + ("G_tan", "signed_h_lambda_error_core", "signed_h_lambda_error_stoch"):
		if field in payload:
			row[field] = float(payload[field])
	return row


def load_eval_rows(eval_root: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for path in sorted(Path(eval_root).rglob("eval.json")):
		payload = json.loads(path.read_text(encoding="utf-8"))
		try:
			row = _scalar_eval_row(payload)
		except ValueError:
			continue
		rows.append(row)
	if not rows:
		raise FileNotFoundError(f"No eval.json files under {eval_root}")
	rows.sort(key=lambda row: (str(row["method_key"]), int(row["seed"])))
	return rows


def summarize_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	summaries: list[dict[str, Any]] = []
	by_method: dict[str, list[dict[str, Any]]] = {}
	for row in rows:
		by_method.setdefault(str(row["method_key"]), []).append(row)
	for method_key in ablation_method_keys():
		method_rows = by_method.get(method_key, [])
		out: dict[str, Any] = {
			"method_key": method_key,
			"method_label": method_label(method_key),
			"n": len(method_rows),
		}
		for field in SUMMARY_METRIC_FIELDS:
			values = [float(row[field]) for row in method_rows]
			finite = [v for v in values if np.isfinite(v)]
			out[f"{field}_n"] = len(finite)
			out[f"{field}_median"] = float(np.median(finite)) if finite else float("nan")
			se = robust_se_of_median(finite)
			out[f"{field}_se"] = float(se) if se is not None else float("nan")
		summaries.append(out)
	return summaries


def write_ablation_tables(eval_root: Path, output_root: Path) -> dict[str, Path]:
	rows = load_eval_rows(eval_root)
	summaries = summarize_eval_rows(rows)
	tables = Path(output_root) / "tables"
	per_seed = tables / "per_seed.csv"
	summary_csv = tables / "summary_medians.csv"
	write_csv(per_seed, rows)
	write_csv(summary_csv, summaries)
	eval_root_stored = str(eval_root)
	repo_root = Path(__file__).resolve().parents[2]
	try:
		eval_root_stored = str(Path(eval_root).resolve().relative_to(repo_root))
	except ValueError:
		pass
	write_json(
		tables / "summary_medians.json",
		{
			"methods": ablation_method_keys(),
			"n": {row["method_key"]: int(row["n"]) for row in summaries},
			"eval_root": eval_root_stored,
		},
	)
	return {"per_seed": per_seed, "summary": summary_csv}
