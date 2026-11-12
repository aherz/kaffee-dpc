"""Generate artifacts needed for the main Lorenz-96 figure in the paper and the figure itself.

Writes figure inputs under `figures/figures_inputs/lorenz96_main_figure/`:

- `budget_traces/budget{N}/per_seed_rows.csv` — weights after N Adam steps (N=0 is the initial shell)
- `trace_summary.csv` — median +- robust SE traces
- `lyapunov_spectrum/` — ground-truth and per-seed reconstructed Lyapunov spectra
- `reassim/` — periodic data-assimilation rollout
- `meta.json` — path and time step of the reassimilation rollout

`notebooks/l96_main_figure.ipynb` plots the figure from these artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.lines import Line2D

from filtering.covariance import transport_covariance
from experiments.lorenz96_main import prepare_lorenz96_main_bundle

_paper_style_spec = importlib.util.spec_from_file_location(
	"paper_figure_style",
	Path(__file__).resolve().parents[2] / "figures" / "paper_style.py",
)
_paper_style = importlib.util.module_from_spec(_paper_style_spec)
_paper_style_spec.loader.exec_module(_paper_style)
TWO_ROW_FIGSIZE = _paper_style.TWO_ROW_FIGSIZE
paper_points = _paper_style.points
paper_rc = _paper_style.rc_params
from experiments.lorenz96_outputs import (
	L96_CORE_CHECKPOINT_NAME,
	SHELL_CHECKPOINT_NAME,
	_evaluate_probabilistic_predictive,
	_core_metric_block,
	_dynamical_eval_config,
	_load_core_checkpoint,
	_load_alrnn_core,
	_load_probabilistic_checkpoint,
	_load_probabilistic_model,
	_load_reference_bundle,
	_predictive_eval_config,
	_process_noisy_lorenz96_spectrum,
	_reference_lorenz96_spectrum,
	_shell_rollout,
	_stochastic_metric_block,
	rollout_free,
)
from metrics import robust_se_of_median
from models import ALRNNShell
from training import BASELINE_SPECS
from training.train_alrnn_probabilistic import (
	_initialize_shell_statistics,
	_make_optimizer,
	_make_scheduler,
	_objective_total_nll,
	_sample_train_windows,
	_seed_everything,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

TRACE_METHOD_KEYS = ("open_loop_tied", "kaffee")
LYAP_METHOD_KEYS = (
	"frozen_core_shell",
	"open_loop_no_propagation",
	"open_loop_tied",
	"filtered_no_propagation",
	"kaffee",
)
BUDGETS = (0, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150)
TRACE_METHOD_REMAP = {
	"Open-loop Gaussian NLL (tied)": "Predictive NLL control",
	"KAFFEE": "KAFFEE",
}


REASSIM_ROLLOUT_DEFAULTS = {
	"seed": 0,
	"launch_abs": 16320,  # start with locally unstable episode
	"H": 50,
	"warmup": 120,
	"stride": 24,
	"dim": 0,
	"dt": 0.04,
}


@dataclass(frozen=True)
class Lorenz96MainFigurePaths:
	repo_root: Path
	output_root: Path
	budget_root: Path
	lyap_dir: Path
	reassim_dir: Path
	trace_summary_csv: Path
	meta_json: Path


def resolve_paths(repo_root: Path | str | None = None, output_root: Path | str | None = None) -> Lorenz96MainFigurePaths:
	root = Path(repo_root or _REPO_ROOT).resolve()
	out = Path(output_root or root / "figures" / "figures_inputs" / "lorenz96_main_figure").resolve()
	return Lorenz96MainFigurePaths(
		repo_root=root,
		output_root=out,
		budget_root=out / "budget_traces",
		lyap_dir=out / "lyapunov_spectrum",
		reassim_dir=out / "reassim",
		trace_summary_csv=out / "trace_summary.csv",
		meta_json=out / "meta.json",
	)


def _write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _paper_method_label(method_key: str) -> str:
	return BASELINE_SPECS[method_key].label


def _robust_median_se(values: np.ndarray) -> tuple[float, float]:
	arr = np.asarray(values, dtype=np.float64)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return float("nan"), float("nan")
	se = robust_se_of_median(arr)
	return float(np.median(arr)), float(se if se is not None else np.nan)


def _summarize_budget_rows(df: pd.DataFrame, method_label: str, budget: int) -> dict[str, Any]:
	row: dict[str, Any] = {"method": method_label, "budget": int(budget)}
	for metric in ("NLL", "D_stsp_core", "D_stsp_stoch", "abs_signed_h_lambda_error_core"):
		if metric not in df.columns:
			continue
		med, se = _robust_median_se(df[metric].to_numpy())
		row[f"{metric}_median"] = med
		row[f"{metric}_robust_se"] = se
	return row


@torch.no_grad()
def _evaluate_shell_row(
	config: dict[str, Any],
	*,
	core_checkpoint: dict[str, Any],
	shell: ALRNNShell,
	method_key: str,
	seed: int,
	device: torch.device,
	smoke: bool,
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
	stoch_ref_spectrum = _process_noisy_lorenz96_spectrum(
		np.asarray(stochastic_evaluation["clean_raw"], dtype=np.float64),
		start_index=int(stochastic_start_index),
		burn_steps=int(lyapunov_burn_steps),
		rollout_steps=int(lyapunov_rollout_steps),
		dt=dt,
		forcing=forcing,
	)
	process_state0_std = np.asarray(
		stochastic_evaluation["clean_std"][int(stochastic_start_index)],
		dtype=np.float64,
	)

	core_model = _load_alrnn_core(core_checkpoint, device)
	prior_standardizer = {
		"mean": np.asarray(core_checkpoint["mean"], dtype=np.float64).reshape(-1),
		"std": np.asarray(core_checkpoint["std"], dtype=np.float64).reshape(-1),
	}
	prior_reference_std = _standardize_raw_series(core_reference_raw, prior_standardizer)

	if method_key == "frozen_core_shell":
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
		core_rollout_std = prior_rollout_std
		core_reference_for_row = prior_reference_std
		core_spectrum = core_model.lyapunov_spectrum(
			prior_reference_std[int(core_start_index)],
			steps=int(lyapunov_steps),
			burn=int(lyapunov_burn_steps),
		)
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

	predictive = _predictive_eval_config(config)
	test_obs_std = np.asarray(heldout_test["obs_std"], dtype=np.float32)
	test_obs_raw = np.asarray(heldout_test["obs_raw"], dtype=np.float64)
	predictive_metrics = _evaluate_probabilistic_predictive(
		shell,
		test_obs_std,
		test_obs_raw,
		standardizer,
		predictive=predictive,
		seed=int(seed),
	)

	return {
		"seed": int(seed),
		"method": _paper_method_label(method_key),
		"baseline_key": method_key,
		**predictive_metrics,
		**core_metrics,
		**stoch_metrics,
	}


def _standardize_raw_series(raw: np.ndarray, standardizer: dict[str, Any]) -> np.ndarray:
	mean = np.asarray(standardizer["mean"], dtype=np.float64).reshape(1, -1)
	std = np.asarray(standardizer["std"], dtype=np.float64).reshape(1, -1)
	return (np.asarray(raw, dtype=np.float64) - mean) / std


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
	return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _train_shell_snapshots(
	config: dict[str, Any],
	*,
	core_root: Path,
	method_key: str,
	seed: int,
	budgets: tuple[int, ...],
	device: torch.device,
	smoke: bool,
) -> dict[int, dict[str, torch.Tensor]]:
	"""Train one shell and keep the weights at the stored updates (budgets).

	Update 0 is the pretrained core plus the initial noise shell.
	"""
	_seed_everything(int(seed))
	prepared = prepare_lorenz96_main_bundle(config, smoke=bool(smoke))
	protocol = prepared["protocol"]
	dataset = prepared["dataset"]
	core_checkpoint = _load_core_checkpoint(core_root, int(seed))
	shell_init = _initialize_shell_statistics(core_checkpoint, dataset, protocol, device=device)
	probabilistic = dict(protocol["probabilistic_training"])
	spec = BASELINE_SPECS[method_key]
	iterations = int(probabilistic["iterations"])
	total_steps = int(probabilistic["window_total_steps"])
	burn_in_steps = int(probabilistic["covariance_burn_in_steps"])
	scored_steps = int(probabilistic["scored_steps"])
	batch_size = int(protocol["training"]["batch_size"])
	loss_normalizer = float(batch_size) * float(scored_steps)
	gradient_clip = float(probabilistic.get("gradient_clip", 0.0))
	eta_min = probabilistic.get("eta_min")
	_seed_everything(42 + int(seed))
	model = ALRNNShell.from_core_checkpoint(
		core_checkpoint,
		proc_std_init=torch.as_tensor(shell_init["gamma_diag_std"], dtype=torch.float32, device=device),
		obs_std_init=torch.as_tensor(shell_init["lambda_diag_std"], dtype=torch.float32, device=device),
		train_core=bool(spec.train_core),
		covariance_transport=str(spec.covariance_transport),
	).to(device)
	wanted = {int(budget) for budget in budgets}
	snapshots: dict[int, dict[str, torch.Tensor]] = {}
	if 0 in wanted:
		snapshots[0] = _cpu_state(model)
	trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
	optimizer = _make_optimizer(
		trainable,
		name=str(probabilistic.get("optimizer", "adam")),
		lr=float(probabilistic.get("lr", 0.005)),
		weight_decay=float(probabilistic.get("weight_decay", probabilistic.get("wd", 0.0))),
	)
	scheduler = _make_scheduler(
		optimizer,
		name=str(probabilistic.get("scheduler", "constant")),
		n_iters=iterations,
		eta_min=float(eta_min) if eta_min is not None else None,
		gamma=float(probabilistic["gamma"]) if probabilistic.get("gamma") is not None else None,
	)
	train_rng = np.random.default_rng(44 + int(seed))
	x_obs_train = np.asarray(dataset["train_obs"], dtype=np.float32)
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
		total_nll = _objective_total_nll(
			model,
			x_windows,
			objective=spec.objective,
			burn_in_steps=burn_in_steps,
		)
		(total_nll / loss_normalizer).backward()
		if gradient_clip > 0.0:
			torch.nn.utils.clip_grad_norm_(trainable, float(gradient_clip))
		optimizer.step()
		scheduler.step()
		step = int(iteration + 1)
		if step in wanted:
			snapshots[step] = _cpu_state(model)
	missing = sorted(wanted.difference(snapshots))
	if missing:
		raise RuntimeError(f"Missing snapshots {missing} for {method_key} seed {seed}")
	return snapshots


def _train_and_evaluate_trajectory(
	config: dict[str, Any],
	*,
	core_root: Path,
	method_key: str,
	seed: int,
	budgets: tuple[int, ...],
	device: torch.device,
	smoke: bool,
) -> list[dict[str, Any]]:
	"""Score one training run at the weights of each plotted update."""
	snapshots = _train_shell_snapshots(
		config,
		core_root=core_root,
		method_key=method_key,
		seed=int(seed),
		budgets=budgets,
		device=device,
		smoke=bool(smoke),
	)
	core_checkpoint = _load_core_checkpoint(core_root, int(seed))
	prepared = prepare_lorenz96_main_bundle(config, smoke=bool(smoke))
	shell_init = _initialize_shell_statistics(
		core_checkpoint,
		prepared["dataset"],
		prepared["protocol"],
		device=device,
	)
	spec = BASELINE_SPECS[method_key]
	_seed_everything(42 + int(seed))
	model = ALRNNShell.from_core_checkpoint(
		core_checkpoint,
		proc_std_init=torch.as_tensor(shell_init["gamma_diag_std"], dtype=torch.float32, device=device),
		obs_std_init=torch.as_tensor(shell_init["lambda_diag_std"], dtype=torch.float32, device=device),
		train_core=bool(spec.train_core),
		covariance_transport=str(spec.covariance_transport),
	).to(device)
	rows: list[dict[str, Any]] = []
	for budget in budgets:
		model.load_state_dict(snapshots[int(budget)])
		row = _evaluate_shell_row(
			config,
			core_checkpoint=core_checkpoint,
			shell=model,
			method_key=method_key,
			seed=int(seed),
			device=device,
			smoke=bool(smoke),
		)
		row["budget"] = int(budget)
		row["scored_iteration"] = int(budget)
		rows.append(row)
	return rows


def compute_trajectory_traces(
	config: dict[str, Any],
	paths: Lorenz96MainFigurePaths,
	*,
	core_root: Path,
	shell_root: Path | None,
	seeds: list[int],
	device: torch.device,
	smoke: bool = False,
	budgets: tuple[int, ...] = BUDGETS,
	skip_existing: bool = False,
) -> None:
	"""Score trajectory traces from one training run per method and seed."""
	del shell_root
	paths.budget_root.mkdir(parents=True, exist_ok=True)
	budget_rows: dict[int, list[dict[str, Any]]] = {int(b): [] for b in budgets}
	for seed in seeds:
		for method_key in TRACE_METHOD_KEYS:
			print(
				f"[trajectory trace] seed={seed:02d} method={method_key} "
				f"full_train_iters={config['probabilistic_training']['iterations']}",
				flush=True,
			)
			for row in _train_and_evaluate_trajectory(
				config,
				core_root=core_root,
				method_key=method_key,
				seed=int(seed),
				budgets=budgets,
				device=device,
				smoke=bool(smoke),
			):
				budget_rows[int(row["budget"])].append(row)

	for budget, rows in budget_rows.items():
		budget_dir = paths.budget_root / f"budget{int(budget):03d}"
		budget_dir.mkdir(parents=True, exist_ok=True)
		out_csv = budget_dir / "per_seed_rows.csv"
		if bool(skip_existing) and out_csv.is_file():
			continue
		pd.DataFrame(rows).to_csv(out_csv, index=False)


def build_trace_summary(paths: Lorenz96MainFigurePaths) -> pd.DataFrame:
	rows: list[dict[str, Any]] = []
	for budget_dir in sorted(paths.budget_root.glob("budget*")):
		try:
			budget_int = int(budget_dir.name.replace("budget", ""))
		except ValueError:
			continue
		per_seed = budget_dir / "per_seed_rows.csv"
		if not per_seed.is_file():
			continue
		df = pd.read_csv(per_seed)
		for source_name, target_label in TRACE_METHOD_REMAP.items():
			sub = df[df["method"] == source_name]
			if sub.empty:
				continue
			rows.append(_summarize_budget_rows(sub, target_label, budget_int))
	if not rows:
		raise FileNotFoundError(f"No budget trace rows under {paths.budget_root}")
	summary = (
		pd.DataFrame(rows)
		.drop_duplicates(["method", "budget"], keep="last")
		.sort_values(["method", "budget"], kind="mergesort")
		.reset_index(drop=True)
	)
	summary.to_csv(paths.trace_summary_csv, index=False)
	return summary


@torch.no_grad()
def _core_lyapunov_spectrum(
	config: dict[str, Any],
	*,
	core_checkpoint: dict[str, Any],
	shell: ALRNNShell,
	method_key: str,
	seed: int,
	device: torch.device,
	smoke: bool,
) -> np.ndarray:
	dynamical = _dynamical_eval_config(config, smoke=bool(smoke))
	dt = float(config["system"]["dt"])
	core_start_index = int(dynamical["core_start_index"])
	burn = int(dynamical["lyapunov_burn_steps"])
	roll = int(dynamical["lyapunov_rollout_steps"])
	prepared = prepare_lorenz96_main_bundle(config, smoke=bool(smoke))
	standardizer = dict(prepared["dataset"]["standardizer"])
	core_reference_bundle = _load_reference_bundle(
		config,
		field="core_reference_raw",
		total_steps=int(core_start_index) + int(burn) + int(roll) + 1,
		standardizer=standardizer,
	)
	core_reference_std = np.asarray(core_reference_bundle["clean_std"], dtype=np.float64)
	x0 = core_reference_std[int(core_start_index)]
	if method_key == "frozen_core_shell":
		core_model = _load_alrnn_core(core_checkpoint, device)
		spec_per_step = core_model.lyapunov_spectrum(x0, steps=int(burn) + int(roll), burn=int(burn))
	else:
		spec_per_step = shell.lyapunov_spectrum(
			x0,
			steps=int(burn) + int(roll),
			burn=int(burn),
			seed=int(seed),
			sample_process_noise=False,
		)
	return (spec_per_step / float(dt)).astype(np.float64)


def compute_lyapunov_spectra(
	config: dict[str, Any],
	paths: Lorenz96MainFigurePaths,
	*,
	core_root: Path,
	shell_root: Path,
	seeds: list[int],
	device: torch.device,
	smoke: bool = False,
	skip_existing: bool = False,
) -> None:
	paths.lyap_dir.mkdir(parents=True, exist_ok=True)
	prepared = prepare_lorenz96_main_bundle(config, smoke=bool(smoke))
	standardizer = dict(prepared["dataset"]["standardizer"])
	dynamical = _dynamical_eval_config(config, smoke=bool(smoke))
	dt = float(config["system"]["dt"])
	burn = int(dynamical["lyapunov_burn_steps"])
	core_reference_bundle = _load_reference_bundle(
		config,
		field="core_reference_raw",
		total_steps=int(dynamical["core_start_index"]) + int(burn) + int(dynamical["lyapunov_rollout_steps"]) + 1,
		standardizer=standardizer,
	)
	gt_spec = _reference_lorenz96_spectrum(
		np.asarray(core_reference_bundle["clean_raw"], dtype=np.float64),
		start_index=int(dynamical["core_start_index"]),
		burn_steps=int(burn),
		rollout_steps=int(dynamical["lyapunov_rollout_steps"]),
		dt=dt,
	)
	gt_path = paths.lyap_dir / "ground_truth_spectrum.npz"
	if not (bool(skip_existing) and gt_path.is_file()):
		np.savez(gt_path, lambdas_cont=np.sort(gt_spec)[::-1])

	for seed in seeds:
		seed_path = paths.lyap_dir / f"seed{int(seed):02d}.npz"
		if bool(skip_existing) and seed_path.is_file():
			continue
		method_order: list[str] = []
		spectra: list[np.ndarray] = []
		core_checkpoint = _load_core_checkpoint(core_root, int(seed))
		for method_key in LYAP_METHOD_KEYS:
			if method_key == "frozen_core_shell":
				frozen_ckpt = _load_probabilistic_checkpoint(shell_root, int(seed), method_key)
				shell = _load_probabilistic_model(
					frozen_ckpt,
					method_key=method_key,
					config=config,
					device=device,
				)
			else:
				ckpt_path = shell_root / f"seed{int(seed):02d}" / method_key / SHELL_CHECKPOINT_NAME
				if not ckpt_path.is_file():
					continue
				shell = _load_probabilistic_model(
					_load_probabilistic_checkpoint(shell_root, int(seed), method_key),
					method_key=method_key,
					config=config,
					device=device,
				)
			print(f"[lyap] seed={seed:02d} method={method_key}", flush=True)
			spec = _core_lyapunov_spectrum(
				config,
				core_checkpoint=core_checkpoint,
				shell=shell,
				method_key=method_key,
				seed=int(seed),
				device=device,
				smoke=bool(smoke),
			)
			method_order.append(_paper_method_label(method_key))
			spectra.append(np.sort(spec)[::-1])
		np.savez(
			seed_path,
			method_order=np.asarray(method_order, dtype=object),
			lambdas_cont_core=np.stack(spectra, axis=0),
			lambdas_cont=np.stack(spectra, axis=0),
			dt=float(dt),
		)


@torch.no_grad()
def _reassim_rollout(
	shell: ALRNNShell,
	obs_window_std: torch.Tensor,
	*,
	launch_idx: int,
	H: int,
	stride: int,
) -> dict[str, np.ndarray]:
	n_dim = int(shell.observed_dim)
	m_dim = int(shell.latent_dim)
	device = shell.A.device
	dtype = shell.A.dtype
	eye_m = torch.eye(m_dim, device=device, dtype=dtype)
	h_mat = torch.zeros((n_dim, m_dim), device=device, dtype=dtype)
	h_mat[:, :n_dim] = torch.eye(n_dim, device=device, dtype=dtype)
	gamma = shell.Gamma.detach()
	r_mat = shell.Lambda.detach()

	def _analysis(mean: torch.Tensor, cov: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		f_mat = shell.transition_jacobian(mean.unsqueeze(0)).squeeze(0)
		m_pred = f_mat @ mean + shell.h
		p_pred = transport_covariance(
			cov.unsqueeze(0),
			gamma.unsqueeze(0),
			jacobian=f_mat.unsqueeze(0),
			mode=str(shell.covariance_transport),
		).squeeze(0)
		p_pred = 0.5 * (p_pred + p_pred.T)
		omega_obs = p_pred[:n_dim, :n_dim] + r_mat
		gain = torch.linalg.solve(omega_obs, p_pred[:, :n_dim].T).T
		new_mean = m_pred + gain @ (y - m_pred[:n_dim])
		new_cov = (eye_m - gain @ h_mat) @ p_pred @ (eye_m - gain @ h_mat).T + gain @ r_mat @ gain.T
		return new_mean, 0.5 * (new_cov + new_cov.T)

	def _predict_only(mean: torch.Tensor, cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		f_mat = shell.transition_jacobian(mean.unsqueeze(0)).squeeze(0)
		m_pred = f_mat @ mean + shell.h
		p_pred = transport_covariance(
			cov.unsqueeze(0),
			gamma.unsqueeze(0),
			jacobian=f_mat.unsqueeze(0),
			mode=str(shell.covariance_transport),
		).squeeze(0)
		return m_pred, 0.5 * (p_pred + p_pred.T)

	z0 = obs_window_std[0]
	mean = torch.matmul(z0, shell.embedding)
	mean[:n_dim] = z0
	cov = gamma.clone()

	for t in range(int(launch_idx)):
		mean, cov = _analysis(mean, cov, obs_window_std[t + 1])

	m_traj = np.zeros((H + 1, n_dim), dtype=np.float64)
	var_diag = np.zeros((H + 1, n_dim), dtype=np.float64)
	assim_flags = np.zeros(H + 1, dtype=np.bool_)
	m_traj[0] = mean[:n_dim].detach().cpu().numpy()
	var_diag[0] = torch.diag(cov[:n_dim, :n_dim]).detach().cpu().numpy()
	assim_flags[0] = True

	do_stride = int(stride) > 0
	for h in range(int(H)):
		if do_stride and ((h + 1) % int(stride) == 0):
			mean, cov = _analysis(mean, cov, obs_window_std[int(launch_idx) + h + 1])
			assim_flags[h + 1] = True
		else:
			mean, cov = _predict_only(mean, cov)
		m_traj[h + 1] = mean[:n_dim].detach().cpu().numpy()
		var_diag[h + 1] = torch.diag(cov[:n_dim, :n_dim]).detach().cpu().numpy()
	return {"m_traj": m_traj, "var_diag": var_diag, "assim_flags": assim_flags}


def compute_reassim_rollout(
	config: dict[str, Any],
	paths: Lorenz96MainFigurePaths,
	*,
	shell_root: Path,
	device: torch.device,
	rollout_options: dict[str, Any] | None = None,
	skip_existing: bool = False,
) -> Path:
	paths.reassim_dir.mkdir(parents=True, exist_ok=True)
	knobs = {**REASSIM_ROLLOUT_DEFAULTS, **(rollout_options or {})}
	seed = int(knobs["seed"])
	launch_abs = int(knobs["launch_abs"])
	H = int(knobs["H"])
	warmup = int(knobs["warmup"])
	stride = int(knobs["stride"])
	out_path = (
		paths.reassim_dir
		/ f"reassim_rollout_seed{seed:02d}_launch{launch_abs:05d}_H{H}_stride{stride:03d}.npz"
	)
	if bool(skip_existing) and out_path.is_file():
		return out_path

	prepared = prepare_lorenz96_main_bundle(config, smoke=False)
	heldout_test = prepared["heldout_test"]
	test_obs_std = np.asarray(heldout_test["obs_std"], dtype=np.float32)
	test_clean_std = np.asarray(heldout_test["clean_std"], dtype=np.float32)
	slice_start = int(launch_abs) - int(warmup)
	slice_end = int(launch_abs) + int(H) + 1
	if slice_start < 0 or slice_end > int(test_obs_std.shape[0]):
		raise ValueError(
			f"rollout window [{slice_start}:{slice_end}] out of range for test length {test_obs_std.shape[0]}"
		)
	obs = torch.as_tensor(test_obs_std[slice_start:slice_end], dtype=torch.float32, device=device)
	z_clean = test_clean_std[slice_start:slice_end]
	launch_idx = int(warmup)
	z_post = z_clean[launch_idx : launch_idx + int(H) + 1]

	results: dict[str, dict[str, np.ndarray]] = {}
	for method_key, alias in (("kaffee", "kaffee"), ("filtered_no_propagation", "nt")):
		ckpt = _load_probabilistic_checkpoint(shell_root, int(seed), method_key)
		shell = _load_probabilistic_model(ckpt, method_key=method_key, config=config, device=device)
		results[alias] = _reassim_rollout(shell, obs, launch_idx=launch_idx, H=int(H), stride=int(stride))

	np.savez(
		out_path,
		seed=int(seed),
		launch_abs=int(launch_abs),
		H=int(H),
		warmup=int(warmup),
		stride=int(stride),
		dim=int(knobs["dim"]),
		z_clean_post_launch=z_post,
		assim_flags=results["kaffee"]["assim_flags"],
		kaffee_m=results["kaffee"]["m_traj"],
		kaffee_var_diag=results["kaffee"]["var_diag"],
		nt_m=results["nt"]["m_traj"],
		nt_var_diag=results["nt"]["var_diag"],
	)
	return out_path


def run_l96_main_figure_data(
	config: dict[str, Any],
	*,
	paths: Lorenz96MainFigurePaths,
	core_root: Path,
	shell_root: Path,
	seeds: list[int],
	device: torch.device,
	smoke: bool = False,
	skip_existing: bool = False,
	rollout_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
	seeds = [int(seed) for seed in seeds]
	if not seeds:
		raise ValueError("At least one seed is required")
	for seed in seeds:
		core_path = Path(core_root) / f"seed{int(seed):02d}" / L96_CORE_CHECKPOINT_NAME
		if not core_path.is_file():
			raise FileNotFoundError(core_path)
		for method_key in LYAP_METHOD_KEYS:
			shell_path = Path(shell_root) / f"seed{int(seed):02d}" / method_key / SHELL_CHECKPOINT_NAME
			if not shell_path.is_file():
				raise FileNotFoundError(shell_path)
	paths.output_root.mkdir(parents=True, exist_ok=True)
	trace_budgets = BUDGETS if not smoke else tuple(range(1, 6))
	compute_trajectory_traces(
		config,
		paths,
		core_root=core_root,
		shell_root=shell_root,
		seeds=seeds,
		device=device,
		smoke=bool(smoke),
		budgets=trace_budgets,
		skip_existing=bool(skip_existing),
	)
	trace_summary = build_trace_summary(paths)
	compute_lyapunov_spectra(
		config,
		paths,
		core_root=core_root,
		shell_root=shell_root,
		seeds=seeds,
		device=device,
		smoke=bool(smoke),
		skip_existing=bool(skip_existing),
	)
	reassim_path = compute_reassim_rollout(
		config,
		paths,
		shell_root=shell_root,
		device=device,
		rollout_options=rollout_options,
		skip_existing=bool(skip_existing),
	)
	try:
		reassim_reference = str(reassim_path.relative_to(paths.repo_root))
	except ValueError:
		reassim_reference = reassim_path.name
	rollout = {**REASSIM_ROLLOUT_DEFAULTS, **(rollout_options or {})}
	meta = {
		"reassim_rollout_npz": reassim_reference,
		"reassim_rollout": {
			"dim": int(rollout["dim"]),
			"stride": int(rollout["stride"]),
			"dt": float(rollout["dt"]),
		},
	}
	_write_json(paths.meta_json, meta)
	return {"trace_summary": trace_summary, "meta": meta}


def load_lyap_payload(lyap_dir: Path) -> dict[str, Any]:
	gt = np.load(lyap_dir / "ground_truth_spectrum.npz")["lambdas_cont"]
	lam_gt = np.sort(np.asarray(gt, dtype=np.float64))[::-1]
	per_method: dict[str, list[np.ndarray]] = {}
	for seed_path in sorted(lyap_dir.glob("seed*.npz")):
		seed_data = np.load(seed_path, allow_pickle=True)
		methods = [str(m) for m in seed_data["method_order"]]
		spectra = np.asarray(seed_data["lambdas_cont_core"], dtype=np.float64)
		for idx, method_name in enumerate(methods):
			per_method.setdefault(method_name, []).append(np.sort(spectra[idx])[::-1])
	medians, se, ks = {}, {}, {}
	for method_name, stacks in per_method.items():
		arr = np.stack(stacks, axis=0)
		medians[method_name] = np.nanmedian(arr, axis=0)
		se[method_name] = np.asarray(
			[robust_se_of_median(arr[:, index]) or 0.0 for index in range(arr.shape[1])],
			dtype=np.float64,
		)
		ks[method_name] = np.maximum(arr, 0.0).sum(axis=1)
	return {
		"medians": medians,
		"se": se,
		"ks": ks,
		"lam_gt": lam_gt,
		"ks_gt": float(np.maximum(lam_gt, 0.0).sum()),
	}


def load_roll_payload(npz_path: Path) -> dict[str, Any]:
	data = np.load(npz_path, allow_pickle=True)
	return {
		"z_clean_post_launch": np.asarray(data["z_clean_post_launch"], dtype=np.float64),
		"kaffee_m": np.asarray(data["kaffee_m"], dtype=np.float64),
		"nt_m": np.asarray(data["nt_m"], dtype=np.float64),
		"kaffee_var_diag": np.asarray(data["kaffee_var_diag"], dtype=np.float64),
		"nt_var_diag": np.asarray(data["nt_var_diag"], dtype=np.float64),
		"assim_flags": np.asarray(data["assim_flags"], dtype=bool),
	}


# Plotting

def _despine(ax) -> None:
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.yaxis.set_ticks_position("left")
	ax.xaxis.set_ticks_position("bottom")


DEFAULT_FIGSIZE = TWO_ROW_FIGSIZE
PAPER_POINTS = paper_points(DEFAULT_FIGSIZE[0])
PAPER_LABEL = PAPER_POINTS["label"]
PAPER_TICK = PAPER_POINTS["tick"]
PAPER_LEGEND = PAPER_POINTS["legend"]
PAPER_PANEL = PAPER_POINTS["panel"]

TRACE_STYLE = {
	**paper_rc(DEFAULT_FIGSIZE[0]),
	"axes.grid": False,
}
TRACE_COLORS = {
	"Predictive NLL control": "#D55E00",
	"KAFFEE": "#009E73",
}
TRACE_METHOD_ORDER = ("Predictive NLL control", "KAFFEE")
TRACE_METHOD_LABELS = {
	"Predictive NLL control": "OL-tied",
	"KAFFEE": "KAFFEE",
}
TRACE_LEGEND_LOC = "center right"
TRACE_PANEL_SPECS = (
	("NLL", r"NLL"),
	("D_stsp_core", r"$D_{\mathrm{stsp}}^{\mathrm{core}}$"),
	("abs_signed_h_lambda_error_core", r"$|\Delta h_{\lambda}^{\mathrm{core}}|$"),
	("D_stsp_stoch", r"$D_{\mathrm{stsp}}^{\mathrm{stoch}}$"),
)
TRACE_PANEL_YLIMS: dict[str, tuple[float, float] | None] = {
	"NLL": (1.30, 2.05),
	"D_stsp_core": (0.0, 4.0),
	"abs_signed_h_lambda_error_core": (0.0, 6.0),
	"D_stsp_stoch": (0.2, 2.1),
}
TRACE_PANEL_YTICKS: dict[str, list[float] | None] = {
	"NLL": [1.4, 1.6, 1.8, 2.0],
	"D_stsp_core": [0, 1, 2, 3, 4],
	"abs_signed_h_lambda_error_core": [0, 2, 4, 6],
	"D_stsp_stoch": [0.5, 1.0, 1.5, 2.0],
}
LYAP_METHOD_ORDER = (
	"Frozen-core shell EKF innovation NLL",
	"Open-loop Gaussian NLL (no propagation)",
	"Open-loop Gaussian NLL (tied)",
	"Filtered innovation NLL, no covariance propagation",
	"KAFFEE",
)
TABLE1_PALETTE = {
	"Truth": "#000000",
	"KAFFEE": "#009E73",
	"No-transport": "#CC79A7",
	"OL-tied": "#D55E00",
	"OL-no-transport": "#1f77b4",
	"Prior": "#7f7f7f",
}
RAW_TO_PALETTE_KEY = {
	"Frozen-core shell EKF innovation NLL": "Prior",
	"Open-loop Gaussian NLL (no propagation)": "OL-no-transport",
	"Open-loop Gaussian NLL (tied)": "OL-tied",
	"Filtered innovation NLL, no covariance propagation": "No-transport",
	"KAFFEE": "KAFFEE",
}
METHOD_DISPLAY_LABELS = {
	"KAFFEE": "KAFFEE",
	"No-transport": "No-propagation",
	"OL-tied": "OL-tied",
	"OL-no-transport": "OL-no-prop.",
	"Prior": "Prior",
}
PANEL_LABELS = tuple(f"({chr(ord('a') + i)})" for i in range(7))
TOP_ROW_PANEL_LABEL_Y = 1.07
DEFAULT_FIGURE_DPI = 360
FIGURE_MARGINS = dict(left=0.055, right=0.995, bottom=0.105, top=0.975)
OUTER_WIDTH_RATIOS = (1.07, 1.0)
OUTER_WSPACE = 0.30
TRACE_GRID_WSPACE = 0.42
TRACE_GRID_HSPACE = 0.40
HERO_GRID_HEIGHT_RATIOS = (1.30, 1.30)
HERO_GRID_WIDTH_RATIOS = (0.92, 1.35)
HERO_GRID_HSPACE = 0.95
HERO_GRID_WSPACE = 0.28
HERO_CURVE_WIDTH = 1.5
LYAP_LINE_STYLES = {
	"KAFFEE": dict(ls="-", lw=HERO_CURVE_WIDTH, marker="o", ms=4.2, mew=0.6, zorder=7),
	"No-transport": dict(ls="-", lw=HERO_CURVE_WIDTH, marker="o", ms=4.2, mew=0.6, zorder=6),
	"OL-tied": dict(ls="-", lw=HERO_CURVE_WIDTH, marker="o", ms=4.2, mew=0.6, zorder=5),
	"OL-no-transport": dict(ls="-", lw=HERO_CURVE_WIDTH, marker="o", ms=4.2, mew=0.6, zorder=4),
	"Prior": dict(ls="-", lw=HERO_CURVE_WIDTH, marker="o", ms=4.2, mew=0.6, zorder=3),
}
LYAP_ERR_DODGE = 0.12
LYAP_SPECTRUM_YLIM = (-1.0, 2.05)
LYAP_SPECTRUM_YTICKS = (-1, 0, 1, 2)


@dataclass(frozen=True)
class Lorenz96MainFigureData:
	"""Loaded artifacts for the Lorenz-96 main figure."""

	trace_summary: pd.DataFrame
	lyap: dict[str, Any]
	rollout: dict[str, Any]
	rollout_dim: int
	rollout_stride: int
	dt: float


def load_l96_main_figure_data(
	paths: Lorenz96MainFigurePaths,
	*,
	rollout_dim: int | None = None,
) -> Lorenz96MainFigureData:
	meta = json.loads(paths.meta_json.read_text(encoding="utf-8"))
	rollout_meta = meta.get("reassim_rollout", {})
	rollout_npz = paths.repo_root / str(meta["reassim_rollout_npz"])
	return Lorenz96MainFigureData(
		trace_summary=pd.read_csv(paths.trace_summary_csv),
		lyap=load_lyap_payload(paths.lyap_dir),
		rollout=load_roll_payload(rollout_npz),
		rollout_dim=int(rollout_dim if rollout_dim is not None else rollout_meta.get("dim", 0)),
		rollout_stride=int(rollout_meta.get("stride", REASSIM_ROLLOUT_DEFAULTS["stride"])),
		dt=float(rollout_meta.get("dt", REASSIM_ROLLOUT_DEFAULTS["dt"])),
	)


def make_l96_main_figure_layout(
	*,
	figsize: tuple[float, float] = DEFAULT_FIGSIZE,
):
	"""Create the main L96 figure layout. Returns (fig, axes)."""
	import matplotlib.pyplot as plt

	fig = plt.figure(figsize=figsize, constrained_layout=False)
	outer = fig.add_gridspec(
		1,
		2,
		width_ratios=list(OUTER_WIDTH_RATIOS),
		wspace=OUTER_WSPACE,
		**FIGURE_MARGINS,
	)
	trace_gs = outer[0, 0].subgridspec(2, 2, wspace=TRACE_GRID_WSPACE, hspace=TRACE_GRID_HSPACE)
	trace_axes = np.array([[fig.add_subplot(trace_gs[i, j]) for j in range(2)] for i in range(2)])
	hero_gs = outer[0, 1].subgridspec(
		2,
		2,
		height_ratios=list(HERO_GRID_HEIGHT_RATIOS),
		width_ratios=list(HERO_GRID_WIDTH_RATIOS),
		hspace=HERO_GRID_HSPACE,
		wspace=HERO_GRID_WSPACE,
	)
	axes = {
		"trace": trace_axes,
		"lyap_spectrum": fig.add_subplot(hero_gs[0, 0]),
		"ks_entropy": fig.add_subplot(hero_gs[0, 1]),
		"reassim_rollout": fig.add_subplot(hero_gs[1, :]),
	}
	return fig, axes


def plot_trace_panels(
	trace_axes: np.ndarray,
	summary_df: pd.DataFrame,
	*,
	show_panel_labels: bool = True,
	show_trace_legend: bool = True,
	method_order: tuple[str, ...] = TRACE_METHOD_ORDER,
	colors: dict[str, str] | None = None,
	panel_ylims: dict[str, tuple[float, float] | None] | None = None,
	panel_yticks: dict[str, list[float] | None] | None = None,
) -> None:
	"""Draw optimization-trace panels (a–d)."""
	palette = TRACE_COLORS if colors is None else colors
	ylims = TRACE_PANEL_YLIMS if panel_ylims is None else panel_ylims
	yticks_map = TRACE_PANEL_YTICKS if panel_yticks is None else panel_yticks
	axes_flat = list(np.asarray(trace_axes).ravel())
	budget_values = np.asarray(summary_df["budget"], dtype=np.float64)
	finite_budgets = budget_values[np.isfinite(budget_values)]
	bmax = float(np.max(finite_budgets)) if finite_budgets.size else 150.0
	if bmax <= 160.0:
		xticks, xlim_max = [0, 25, 50, 100, 150], 155.0
	elif bmax <= 220.0:
		xticks, xlim_max = [0, 100, 200], bmax + 5.0
	else:
		xticks, xlim_max = [0, 100, 200, 300], max(305.0, bmax + 5.0)

	for panel_idx, (metric, ylabel) in enumerate(TRACE_PANEL_SPECS):
		ax = axes_flat[panel_idx]
		for method in method_order:
			frame = summary_df.loc[summary_df["method"] == method].sort_values("budget")
			if frame.empty:
				continue
			med_col, se_col = f"{metric}_median", f"{metric}_robust_se"
			if med_col not in frame.columns and metric == "NLL":
				med_col, se_col = "NLL_1step_median", "NLL_1step_robust_se"
			ax.errorbar(
				np.asarray(frame["budget"], dtype=np.float64),
				np.asarray(frame[med_col], dtype=np.float64),
				yerr=np.asarray(frame[se_col], dtype=np.float64),
				color=palette[method],
				lw=3.0,
				marker="o",
				ms=5.0,
				elinewidth=2.0,
				capsize=3.0,
				capthick=2.0,
				label=TRACE_METHOD_LABELS.get(method, method) if panel_idx == 0 else "_nolegend_",
			)
		ax.set_xlim(-2.0, xlim_max)
		ax.set_xticks(xticks)
		ax.set_xlabel("")
		ylim = ylims.get(metric)
		if ylim is not None:
			ax.set_ylim(*ylim)
		yticks = yticks_map.get(metric)
		if yticks is not None:
			ax.set_yticks(yticks)
		ax.set_ylabel(ylabel, labelpad=0.4 if metric in {"D_stsp_core", "D_stsp_stoch"} else 2.0, fontsize=PAPER_LABEL)
		ax.yaxis.set_label_position("left")
		ax.yaxis.tick_left()
		ax.spines["left"].set_visible(True)
		ax.spines["right"].set_visible(False)
		ax.tick_params(axis="both", width=0.8, length=3.0, pad=1.1, labelsize=PAPER_TICK)
		ax.margins(x=0.02)
		if show_panel_labels:
			ax.text(
				0.035,
				TOP_ROW_PANEL_LABEL_Y if panel_idx < 2 else 1.005,
				PANEL_LABELS[panel_idx],
				transform=ax.transAxes,
				fontsize=PAPER_PANEL,
				fontweight="bold",
				va="bottom",
				ha="left",
			)
		if panel_idx < 2:
			ax.tick_params(axis="x", labelbottom=False)
		_despine(ax)
	if show_trace_legend:
		axes_flat[0].legend(
			loc=TRACE_LEGEND_LOC,
			frameon=False,
			fontsize=PAPER_LEGEND,
			handlelength=1.35,
			borderaxespad=0.10,
			labelspacing=0.24,
			handletextpad=0.35,
		)


def plot_lyapunov_spectrum_panel(
	ax,
	lyap_payload: dict[str, Any],
	*,
	method_order: tuple[str, ...] = LYAP_METHOD_ORDER,
	n_indices: int = 7,
	show_panel_label: bool = True,
	palette: dict[str, str] | None = None,
) -> None:
	"""Draw the Lyapunov spectrum panel (e)."""
	colors = TABLE1_PALETTE if palette is None else palette
	order_raw = [m for m in method_order if m in lyap_payload["medians"]]
	x = np.arange(1, int(n_indices) + 1)
	offsets = (np.arange(len(order_raw)) - (len(order_raw) - 1) / 2.0) * LYAP_ERR_DODGE
	for offset, method_raw in zip(offsets, order_raw):
		key = RAW_TO_PALETTE_KEY[method_raw]
		color = colors[key]
		style = LYAP_LINE_STYLES[key]
		med = lyap_payload["medians"][method_raw][: int(n_indices)]
		se = lyap_payload["se"][method_raw][: int(n_indices)]
		ax.errorbar(
			x + offset,
			med,
			yerr=se,
			color=color,
			lw=style["lw"],
			ls=style["ls"],
			marker=style["marker"],
			ms=style["ms"],
			mew=style["mew"],
			elinewidth=0.8,
			capsize=1.6,
			capthick=0.7,
			zorder=style["zorder"],
			label=METHOD_DISPLAY_LABELS[key],
		)
	ax.plot(
		x,
		lyap_payload["lam_gt"][: int(n_indices)],
		color=colors["Truth"],
		lw=HERO_CURVE_WIDTH,
		ls="-",
		marker="o",
		ms=4.2,
		markerfacecolor="white",
		markeredgecolor=colors["Truth"],
		markeredgewidth=1.0,
		zorder=9,
		label="truth",
	)
	ax.axhline(0, color="black", lw=0.5, alpha=0.5, zorder=1)
	ax.set_xlabel(r"Lyapunov index $i$", fontsize=PAPER_LABEL, labelpad=1)
	ax.set_ylabel(r"$\lambda_i$", fontsize=PAPER_LABEL, labelpad=1)
	ax.set_xticks(x)
	ax.tick_params(labelsize=PAPER_TICK)
	ax.set_ylim(*LYAP_SPECTRUM_YLIM)
	ax.set_yticks(list(LYAP_SPECTRUM_YTICKS))
	_despine(ax)
	if show_panel_label:
		ax.text(
			-0.15,
			TOP_ROW_PANEL_LABEL_Y,
			PANEL_LABELS[4],
			transform=ax.transAxes,
			fontsize=PAPER_PANEL,
			fontweight="bold",
			va="bottom",
			ha="left",
		)


def plot_ks_entropy_panel(
	ax,
	lyap_payload: dict[str, Any],
	*,
	method_order: tuple[str, ...] = LYAP_METHOD_ORDER,
	show_panel_label: bool = True,
	palette: dict[str, str] | None = None,
) -> None:
	"""Draw the KS entropy bar panel (f)."""
	colors = TABLE1_PALETTE if palette is None else palette
	order_raw = [m for m in method_order if m in lyap_payload["medians"]]
	vals, errs, cols = [], [], []
	for method_raw in order_raw:
		key = RAW_TO_PALETTE_KEY[method_raw]
		ks = lyap_payload["ks"][method_raw]
		med = float(np.nanmedian(ks))
		vals.append(med)
		se = robust_se_of_median(ks)
		errs.append(float(se if se is not None else 0.0))
		cols.append(colors[key])
	xp = np.arange(len(vals))
	ax.bar(
		xp,
		vals,
		yerr=errs,
		color=cols,
		edgecolor="black",
		linewidth=0.4,
		ecolor="black",
		error_kw={"lw": 0.7, "capsize": 1.6},
	)
	ax.axhline(lyap_payload["ks_gt"], color="black", ls=":", lw=0.95, zorder=3)
	ax.annotate(
		"truth",
		xy=(0.99, float(lyap_payload["ks_gt"])),
		xycoords=ax.get_yaxis_transform(),
		xytext=(0, 1),
		textcoords="offset points",
		ha="right",
		va="bottom",
		fontsize=PAPER_LABEL,
		color="black",
		annotation_clip=False,
		zorder=4,
	)
	ax.set_xticks(xp)
	ax.set_ylabel(r"$h_\lambda$", fontsize=PAPER_LABEL)
	ax.set_ylim(0, max(float(lyap_payload["ks_gt"]) * 1.32, max(vals) * 1.15))
	_despine(ax)
	tick_labels = [
		METHOD_DISPLAY_LABELS[RAW_TO_PALETTE_KEY[method_raw]] for method_raw in order_raw
	]
	ax.set_xticklabels(
		tick_labels,
		rotation=36,
		ha="right",
		va="top",
		fontsize=PAPER_TICK,
		rotation_mode="anchor",
	)
	for tick_label in ax.get_xticklabels():
		tick_label.set_clip_on(False)
	ax.tick_params(axis="x", pad=2, length=3.0)
	if show_panel_label:
		ax.text(
			-0.22,
			TOP_ROW_PANEL_LABEL_Y,
			PANEL_LABELS[5],
			transform=ax.transAxes,
			fontsize=PAPER_PANEL,
			fontweight="bold",
			va="bottom",
			ha="left",
		)


def plot_reassim_rollout_panel(
	ax,
	lyap_payload: dict[str, Any],
	rollout_payload: dict[str, Any],
	*,
	dim: int,
	stride: int,
	dt: float,
	uncertainty_sigma: float = 2.0,
	show_panel_label: bool = True,
	palette: dict[str, str] | None = None,
) -> None:
	"""Draw the periodic-assimilation rollout panel (g)."""
	colors = TABLE1_PALETTE if palette is None else palette
	z_post = rollout_payload["z_clean_post_launch"]
	km, nm = rollout_payload["kaffee_m"], rollout_payload["nt_m"]
	kv, nv = rollout_payload["kaffee_var_diag"], rollout_payload["nt_var_diag"]
	H = int(z_post.shape[0]) - 1
	lyap_lambda_max = float(np.max(lyap_payload["lam_gt"]))
	x_g = np.arange(H + 1, dtype=np.float64) * float(dt) * lyap_lambda_max
	sig = float(uncertainty_sigma)
	k_sig, n_sig = np.sqrt(kv[:, int(dim)]), np.sqrt(nv[:, int(dim)])
	ax.fill_between(
		x_g,
		nm[:, int(dim)] - sig * n_sig,
		nm[:, int(dim)] + sig * n_sig,
		color=colors["No-transport"],
		alpha=0.22,
		zorder=2,
	)
	ax.fill_between(
		x_g,
		km[:, int(dim)] - sig * k_sig,
		km[:, int(dim)] + sig * k_sig,
		color=colors["KAFFEE"],
		alpha=0.22,
		zorder=3,
	)
	ax.plot(
		x_g,
		nm[:, int(dim)],
		color=colors["No-transport"],
		lw=HERO_CURVE_WIDTH,
		marker="s",
		ms=2.6,
		mew=0,
		zorder=4,
	)
	ax.plot(
		x_g,
		km[:, int(dim)],
		color=colors["KAFFEE"],
		lw=HERO_CURVE_WIDTH,
		marker="o",
		ms=2.8,
		mew=0,
		zorder=5,
	)
	ax.plot(x_g, z_post[:, int(dim)], color=colors["Truth"], lw=HERO_CURVE_WIDTH, zorder=7)
	assim_k = np.array([], dtype=int)
	if int(stride) > 0 and rollout_payload.get("assim_flags") is not None:
		assim_k = np.flatnonzero(np.asarray(rollout_payload["assim_flags"], dtype=bool)[1:]) + 1
	if assim_k.size > 0:
		for k in assim_k:
			ax.axvline(x_g[k], color="black", lw=1.1, ls="--", alpha=0.95, zorder=8.2)
	marker_kw = dict(
		s=25,
		facecolors="white",
		edgecolors=colors["Truth"],
		linewidths=1.1,
		zorder=9,
	)
	ax.scatter([x_g[0]], [z_post[0, int(dim)]], **marker_kw)
	if assim_k.size > 0:
		ax.scatter(x_g[assim_k], z_post[assim_k, int(dim)], **marker_kw)
	ax.set_xlabel(r"horizon ($\tau_\lambda$)", fontsize=PAPER_LABEL)
	ax.set_ylabel(r"$z_0$", fontsize=PAPER_LABEL)
	ylo, yhi = ax.get_ylim()
	ax.set_ylim(ylo, yhi + 0.12 * (yhi - ylo))
	if assim_k.size > 0:
		ax.legend(
			[Line2D([0], [0], color="black", lw=1.1, ls="--")],
			["assimilation"],
			loc="upper left",
			fontsize=PAPER_LEGEND,
			frameon=False,
			handlelength=1.5,
			borderpad=0.2,
		)
	_despine(ax)
	if show_panel_label:
		ax.text(
			-0.055,
			1.005,
			PANEL_LABELS[6],
			transform=ax.transAxes,
			fontsize=PAPER_PANEL,
			fontweight="bold",
			va="bottom",
			ha="left",
		)


def _spectrum_legend_handles() -> tuple[list, list[str]]:
	"""Solid legend samples, one per spectrum color."""
	order = (
		("Truth", "truth"),
		("Prior", METHOD_DISPLAY_LABELS["Prior"]),
		("OL-no-transport", METHOD_DISPLAY_LABELS["OL-no-transport"]),
		("OL-tied", METHOD_DISPLAY_LABELS["OL-tied"]),
		("No-transport", METHOD_DISPLAY_LABELS["No-transport"]),
		("KAFFEE", METHOD_DISPLAY_LABELS["KAFFEE"]),
	)
	handles: list = []
	labels: list[str] = []
	for key, label in order:
		color = TABLE1_PALETTE[key]
		face = "white" if key == "Truth" else color
		edge_width = 1.0 if key == "Truth" else 0.4
		line = Line2D(
			[0],
			[0],
			color=color,
			lw=1.8,
			ls="-",
			marker="o",
			ms=5.0,
			markerfacecolor=face,
			markeredgecolor=color,
			markeredgewidth=edge_width,
		)
		if key == "Truth":
			handles.append(line)
		else:
			bar = LineCollection([[(0.0, -1.0), (0.0, 1.0)]], colors=[color], linewidths=[0.9])
			cap = Line2D([0], [0], color=color, lw=0.6, markersize=3.5)
			handles.append(ErrorbarContainer((line, (cap,), (bar,)), has_xerr=False, has_yerr=True))
		labels.append(label)
	return handles, labels


def _match_panel_label_height(fig, source_ax, target_ax, source_text: str, target_text: str) -> None:
	"""Put target_text at the same vertical position as source_text."""
	source = next((text for text in source_ax.texts if text.get_text() == source_text), None)
	target = next((text for text in target_ax.texts if text.get_text() == target_text), None)
	if source is None or target is None:
		return
	source.set_va("bottom")
	target.set_va("bottom")
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	dy = source.get_window_extent(renderer).y0 - target.get_window_extent(renderer).y0
	x, y = target.get_position()
	x_disp, y_disp = target.get_transform().transform((x, y))
	_, y_new = target.get_transform().inverted().transform((x_disp, y_disp + dy))
	target.set_position((x, float(y_new)))


def _legend_above_spectrum(fig, trace_axes, spectrum_ax, entropy_ax) -> None:
	"""Keep (e) and (f) top-aligned with (a), and put one legend row above the figure."""
	ref = trace_axes[0, 0].get_position()
	for ax in (spectrum_ax, entropy_ax):
		pos = ax.get_position()
		ax.set_position([pos.x0, ref.y1 - pos.height, pos.width, pos.height])
	handles, labels = _spectrum_legend_handles()
	top = ref.y1
	fig.legend(
		handles,
		labels,
		loc="lower center",
		bbox_to_anchor=(0.5, top + 0.085),
		bbox_transform=fig.transFigure,
		ncol=len(labels),
		frameon=False,
		fontsize=PAPER_LABEL,
		handlelength=1.8,
		handletextpad=0.45,
		columnspacing=0.85,
		borderaxespad=0.0,
		handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0.32)},
	)


def draw_l96_main_figure(
	fig,
	axes: dict[str, Any],
	data: Lorenz96MainFigureData,
	*,
	show_panel_labels: bool = True,
	trace_method_order: tuple[str, ...] = TRACE_METHOD_ORDER,
	lyap_method_order: tuple[str, ...] = LYAP_METHOD_ORDER,
	trace_colors: dict[str, str] | None = None,
	palette: dict[str, str] | None = None,
	n_lyap_indices: int = 7,
	uncertainty_sigma: float = 2.0,
	panel_ylims: dict[str, tuple[float, float] | None] | None = None,
	panel_yticks: dict[str, list[float] | None] | None = None,
) -> None:
	"""Draw all seven panels onto a layout from :func:`make_l96_main_figure_layout`."""
	plot_trace_panels(
		axes["trace"],
		data.trace_summary,
		show_panel_labels=show_panel_labels,
		show_trace_legend=False,
		method_order=trace_method_order,
		colors=trace_colors,
		panel_ylims=panel_ylims,
		panel_yticks=panel_yticks,
	)
	plot_lyapunov_spectrum_panel(
		axes["lyap_spectrum"],
		data.lyap,
		show_panel_label=show_panel_labels,
		method_order=lyap_method_order,
		n_indices=n_lyap_indices,
		palette=palette,
	)
	plot_ks_entropy_panel(
		axes["ks_entropy"],
		data.lyap,
		show_panel_label=show_panel_labels,
		method_order=lyap_method_order,
		palette=palette,
	)
	plot_reassim_rollout_panel(
		axes["reassim_rollout"],
		data.lyap,
		data.rollout,
		dim=int(data.rollout_dim),
		stride=int(data.rollout_stride),
		dt=float(data.dt),
		uncertainty_sigma=uncertainty_sigma,
		show_panel_label=show_panel_labels,
		palette=palette,
	)
	_legend_above_spectrum(fig, axes["trace"], axes["lyap_spectrum"], axes["ks_entropy"])
	bottom_left = axes["trace"][1, 0].get_position()
	bottom_right = axes["trace"][1, 1].get_position()
	fig.text(
		0.5 * (bottom_left.x0 + bottom_right.x1),
		bottom_left.y0 - 0.055,
		"update",
		ha="center",
		va="top",
		fontsize=PAPER_LABEL,
	)
	if show_panel_labels:
		_match_panel_label_height(fig, axes["trace"][1, 0], axes["reassim_rollout"], "(c)", "(g)")


def save_l96_main_figure(
	fig,
	figure_dir: Path,
	*,
	stem: str = "l96_results",
	dpi: int = DEFAULT_FIGURE_DPI,
) -> Path:
	"""Write the composite figure as PDF and PNG."""
	import matplotlib.pyplot as plt

	figure_dir = Path(figure_dir)
	figure_dir.mkdir(parents=True, exist_ok=True)
	out_stem = figure_dir / stem
	fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
	fig.savefig(out_stem.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight", pad_inches=0.01)
	plt.close(fig)
	return out_stem.with_suffix(".pdf")


def plot_l96_main_figure(
	paths: Lorenz96MainFigurePaths,
	*,
	figure_dir: Path,
	rollout_dim: int | None = None,
	figsize: tuple[float, float] = DEFAULT_FIGSIZE,
	dpi: int = DEFAULT_FIGURE_DPI,
	**draw_kwargs: Any,
) -> Path:
	"""Load artifacts, render the paper figure, and save PDF/PNG."""
	import matplotlib.pyplot as plt

	data = load_l96_main_figure_data(paths, rollout_dim=rollout_dim)
	with plt.rc_context(TRACE_STYLE):
		fig, axes = make_l96_main_figure_layout(figsize=figsize)
		draw_l96_main_figure(fig, axes, data, **draw_kwargs)
	return save_l96_main_figure(fig, figure_dir, dpi=dpi)


__all__ = [
	"BUDGETS",
	"DEFAULT_FIGSIZE",
	"DEFAULT_FIGURE_DPI",
	"Lorenz96MainFigureData",
	"Lorenz96MainFigurePaths",
	"LYAP_METHOD_ORDER",
	"PANEL_LABELS",
	"REASSIM_ROLLOUT_DEFAULTS",
	"TABLE1_PALETTE",
	"TRACE_METHOD_ORDER",
	"TRACE_PANEL_SPECS",
	"TRACE_STYLE",
	"build_trace_summary",
	"compute_trajectory_traces",
	"compute_lyapunov_spectra",
	"compute_reassim_rollout",
	"draw_l96_main_figure",
	"load_l96_main_figure_data",
	"load_lyap_payload",
	"load_roll_payload",
	"make_l96_main_figure_layout",
	"plot_ks_entropy_panel",
	"plot_l96_main_figure",
	"plot_lyapunov_spectrum_panel",
	"plot_reassim_rollout_panel",
	"plot_trace_panels",
	"resolve_paths",
	"run_l96_main_figure_data",
	"save_l96_main_figure",
]
