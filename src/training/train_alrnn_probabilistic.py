"""Probabilistic shell training for AL-RNN ablations."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from filtering import filtered_innovation_nll_batched, open_loop_gaussian_nll_batched
from metrics.state_space_divergence import state_space_divergence
from models import ALRNN, ALRNNShell, ALRNNShellLR


@dataclass(frozen=True)
class BaselineSpec:
	key: str
	label: str
	objective: str
	covariance_transport: str
	train_core: bool
	description: str
	seed_offset: int = 0 


# Fixed held-out windows for validation NLL optimization traces.
_SELECTOR_TRACE_VAL_WINDOWS = 24
_SELECTOR_TRACE_VAL_WINDOW_SEED_BASE = 1000

BASELINE_SPECS: dict[str, BaselineSpec] = {
	"kaffee": BaselineSpec(
		"kaffee",
		"KAFFEE",
		"filtered",
		"jacobian",
		True,
		"Filtered innovation NLL.",
		seed_offset=0,
	),
	"frozen_core_shell": BaselineSpec(
		"frozen_core_shell",
		"Frozen-core shell EKF innovation NLL",
		"filtered",
		"jacobian",
		False,
		"Filtered innovation NLL with frozen deterministic core.",
		seed_offset=20_000,
	),
	"filtered_no_propagation": BaselineSpec(
		"filtered_no_propagation",
		"Filtered innovation NLL, no covariance propagation",
		"filtered",
		"identity",
		True,
		"Filtered innovation NLL with identity covariance propagation.",
		seed_offset=30_000,
	),
	"open_loop_tied": BaselineSpec(
		"open_loop_tied",
		"Open-loop Gaussian NLL (tied)",
		"open_loop",
		"jacobian",
		True,
		"Open-loop Gaussian NLL with Jacobian covariance propagation.",
		seed_offset=40_000,
	),
	"open_loop_no_propagation": BaselineSpec(
		"open_loop_no_propagation",
		"Open-loop Gaussian NLL (no propagation)",
		"open_loop",
		"identity",
		True,
		"Open-loop Gaussian NLL with identity covariance propagation.",
		seed_offset=60_000,
	),
}


def _seed_everything(seed: int) -> None:
	np.random.seed(int(seed))
	torch.manual_seed(int(seed))
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(int(seed))


def _window_start_high(n_steps: int, total_steps: int) -> int:
	return max(1, int(n_steps) - int(total_steps) - 1)


def _sample_train_windows(
	x_obs: np.ndarray,
	*,
	total_steps: int,
	batch_size: int,
	rng: np.random.Generator,
) -> np.ndarray:
	x_obs = np.asarray(x_obs, dtype=np.float32)
	if x_obs.ndim != 2:
		raise ValueError(f"Expected 2D observations, got shape {x_obs.shape}")
	if int(x_obs.shape[0]) <= int(total_steps):
		raise ValueError(f"Need at least {total_steps + 1} observations, got {x_obs.shape[0]}")
	starts = rng.integers(0, _window_start_high(int(x_obs.shape[0]), int(total_steps)), size=int(batch_size))
	return np.stack(
		[x_obs[int(start) : int(start) + int(total_steps) + 1] for start in starts],
		axis=0,
	).astype(np.float32)


def _observation_series_batch(obs: np.ndarray) -> np.ndarray:
	values = np.asarray(obs, dtype=np.float32)
	if values.ndim == 2:
		return values[np.newaxis, ...]
	if values.ndim == 3:
		return values
	raise ValueError(f"Expected observations with ndim 2 or 3, got shape {values.shape}")


def _trace_iteration(row: dict[str, float | int]) -> int:
	return int(row.get("iter", row.get("iteration", 0)))


def _prepare_fixed_val_windows(
	x_val_std: np.ndarray,
	*,
	n_windows: int,
	total_steps: int,
	seed: int,
) -> np.ndarray:
	"""Fixed held-out windows for validation NLL optimization traces."""
	x_val = np.asarray(x_val_std, dtype=np.float32)
	if x_val.ndim == 2:
		x_val = x_val[np.newaxis, ...]
	if x_val.ndim != 3:
		raise ValueError(f"Expected val observations with ndim 2 or 3, got shape {x_val.shape}")
	s_count, t_count, _ = x_val.shape
	n_windows = int(n_windows)
	if n_windows < 1:
		raise ValueError("val_windows must be >= 1")
	if int(t_count) <= int(total_steps):
		raise ValueError(
			f"Need at least {int(total_steps) + 1} validation time steps, got {int(t_count)}"
		)
	rng = np.random.default_rng(int(seed))
	s_idx = rng.integers(0, int(s_count), size=n_windows)
	t0_idx = rng.integers(
		0,
		max(1, int(t_count) - int(total_steps) - 1),
		size=n_windows,
	)
	return np.stack(
		[
			x_val[int(s_idx[i]), int(t0_idx[i]) : int(t0_idx[i]) + int(total_steps) + 1, :]
			for i in range(n_windows)
		],
		axis=0,
	).astype(np.float32)


def _trace_sort_key(row: dict[str, float]) -> tuple[float, float, float]:
	"""Checkpoint = lowest D_stsp_stoch on traced steps; ties: take latest iteration."""
	d_stsp = float(row.get("D_stsp_stoch", float("nan")))
	it = float(_trace_iteration(row))
	missing_rank = 0.0 if np.isfinite(d_stsp) else 1.0
	metric = d_stsp if np.isfinite(d_stsp) else float("inf")
	return (missing_rank, metric, -it)


def _trace_row_for_selection(row: dict[str, float | int]) -> dict[str, float]:
	out = {str(key): float(value) for key, value in row.items() if key not in {"iteration"}}
	out["iter"] = float(_trace_iteration(row))
	if "val_nll" in row:
		out["val_nll"] = float(row["val_nll"])
	if "D_stsp_stoch" in row:
		out["D_stsp_stoch"] = float(row["D_stsp_stoch"])
	return out


def _select_best_trace_row(trace_rows: list[dict[str, float | int]]) -> dict[str, float | int] | None:
	if not trace_rows:
		return None
	return min(trace_rows, key=lambda row: _trace_sort_key(_trace_row_for_selection(row)))


def select_best_trace_row_up_to(
	trace_rows: list[dict[str, float | int]],
	*,
	max_iteration: int,
) -> dict[str, float | int] | None:
	"""Best D_stsp_stoch checkpoint among traced iterations <= max_iteration."""
	cutoff = int(max_iteration)
	candidates = [row for row in trace_rows if int(_trace_iteration(row)) <= cutoff]
	return _select_best_trace_row(candidates)


@torch.no_grad()
def _heldout_val_nll_per_step(
	model: ALRNNShell | ALRNNShellLR,
	val_windows: torch.Tensor,
	*,
	objective: str,
	burn_in_steps: int,
	scored_steps: int,
	innovation_jitter: float = 1e-6,
	use_jacobian_factor: bool = False,
) -> float:
	batch = int(val_windows.shape[0])
	if batch < 1:
		return float("nan")
	total_nll = _objective_total_nll(
		model,
		val_windows,
		objective=objective,
		burn_in_steps=int(burn_in_steps),
		innovation_jitter=float(innovation_jitter),
		use_jacobian_factor=bool(use_jacobian_factor),
	)
	return float(total_nll.item()) / (float(batch) * float(scored_steps))


@torch.no_grad()
def _selector_d_stsp_stoch(
	model: ALRNNShell | ALRNNShellLR,
	test_obs_std: np.ndarray,
	*,
	device: torch.device,
	seed: int,
	burn: int,
	roll: int,
	sigma2: float,
	mc_samples: int,
	density_batch_size: int,
) -> float:
	"""Training-time model selection (observable stochastic D_stsp)."""
	series = np.asarray(_observation_series_batch(test_obs_std), dtype=np.float64)
	if series.shape[0] < 1:
		return float("nan")
	burn = int(max(0, burn))
	roll = int(max(1, roll))
	ref_std = series[0, burn : burn + roll, :]
	if ref_std.shape[0] < 1:
		return float("nan")

	x0_std = np.asarray(series[0, burn, :], dtype=np.float64)
	model = model.to(device)
	model.eval()
	gen_std = np.asarray(
		model.rollout_observation_sequence(
			x0_std,
			steps=int(roll),
			seed=int(seed),
			sample_process_noise=True,
			sample_obs_noise=True,
		),
		dtype=np.float64,
	)
	seg_len = int(min(ref_std.shape[0], gen_std.shape[0]))
	return float(
		state_space_divergence(
			gen_std[:seg_len],
			ref_std[:seg_len],
			sigma2=float(sigma2),
			max_T=int(seg_len),
			mc_samples=int(mc_samples),
			batch_size=int(density_batch_size),
		)
	)


def _make_core_model(checkpoint: dict[str, Any], *, device: torch.device) -> ALRNN:
	model = ALRNN(
		latent_dim=int(checkpoint["M"]),
		gated_dim=int(checkpoint["P"]),
		observed_dim=int(checkpoint["N"]),
	)
	model.load_state_dict(checkpoint["state_dict"])
	return model.to(device).eval()


@torch.no_grad()
def _teacher_force_residual_std(
	model: ALRNN,
	x_train: np.ndarray,
	*,
	device: torch.device,
	n_interleave: int,
	std_floor: float,
) -> np.ndarray:
	x_train = np.asarray(x_train, dtype=np.float32)
	if x_train.ndim == 3:
		obs = x_train.reshape(-1, int(x_train.shape[-1]))
	elif x_train.ndim == 2:
		obs = x_train
	else:
		raise ValueError(f"Expected train observations with ndim 2 or 3, got {x_train.shape}")
	if obs.shape[0] < 2:
		raise ValueError("Need at least two training observations for TF-residual shell init.")

	x_prev = torch.as_tensor(obs[:-1], dtype=torch.float32, device=device)
	x_next = torch.as_tensor(obs[1:], dtype=torch.float32, device=device)
	n_dim = int(model.observed_dim)
	k_dim = int(model.gated_dim)
	n_interleave = int(max(1, n_interleave))

	z = x_prev[0] @ model.B
	z[:n_dim] = x_prev[0]
	residuals: list[torch.Tensor] = []
	for t in range(int(x_prev.shape[0])):
		if t > 0 and (t % n_interleave) == 0:
			z = z.clone()
			z[:n_dim] = x_prev[t]
		z_unactivated = z
		z_gated = z.clone()
		if k_dim > 0:
			z_gated[-k_dim:] = torch.relu(z_gated[-k_dim:])
		z = model.A * z_unactivated + z_gated @ model.W.t() + model.h
		residuals.append((x_next[t] - z[:n_dim]).detach())

	resid = torch.stack(residuals, dim=0).cpu().numpy().astype(np.float64)
	return np.maximum(resid.std(axis=0), float(std_floor)).astype(np.float32)


@torch.no_grad()
def _initialize_shell_statistics(
	checkpoint: dict[str, Any],
	dataset: dict[str, Any],
	protocol: dict[str, Any],
	*,
	device: torch.device,
) -> dict[str, Any]:
	model = _make_core_model(checkpoint, device=device)
	x_obs = np.asarray(dataset["train_obs"], dtype=np.float32)
	probabilistic = dict(protocol["probabilistic_training"])
	n_interleave = int(protocol["model"].get("teacher_forcing_interval", 20))
	resid_std = _teacher_force_residual_std(
		model,
		x_obs,
		device=device,
		n_interleave=n_interleave,
		std_floor=1e-6, # Jitter to avoid zero stds
	)

	obs_frac = float(probabilistic.get("tfresid_obs_var_frac", 0.9))
	proc_frac = float(probabilistic.get("tfresid_proc_var_frac", 0.1))
	obs_std = np.sqrt(obs_frac) * resid_std
	proc_scalar = float(np.sqrt(proc_frac) * np.mean(resid_std))
	proc_std = np.full((int(checkpoint["M"]),), proc_scalar, dtype=np.float32)
	return {
		"shell_init_policy": "tf_residual_90_10",
		"x_residual_std": resid_std,
		"lambda_diag_std": obs_std.astype(np.float32),
		"gamma_diag_std": proc_std,
		"gamma_scalar_std": proc_scalar,
	}


def _objective_total_nll(
	model: ALRNNShell | ALRNNShellLR,
	x_windows: torch.Tensor,
	*,
	objective: str,
	burn_in_steps: int,
	innovation_jitter: float = 1e-6,
	use_jacobian_factor: bool = False,
) -> torch.Tensor:
	objective_name = str(objective).strip().lower()
	if objective_name in frozenset({"filtered", "innovation", "innovation_nll"}):
		return filtered_innovation_nll_batched(
			model,
			x_windows,
			burn_in_steps=int(burn_in_steps),
			innovation_jitter=float(innovation_jitter),
			use_jacobian_factor=bool(use_jacobian_factor),
		)
	if objective_name in frozenset({"openloop", "open_loop", "open_loop_gaussian_nll"}):
		return open_loop_gaussian_nll_batched(
			model,
			x_windows,
			burn_in_steps=int(burn_in_steps),
			innovation_jitter=float(innovation_jitter),
			use_jacobian_factor=bool(use_jacobian_factor),
		)
	raise ValueError(f"Unsupported probabilistic objective: {objective!r}")


def _make_optimizer(
	params: Any,
	*,
	name: str,
	lr: float,
	weight_decay: float,
) -> torch.optim.Optimizer:
	name = str(name).strip().lower()
	if name == "adam":
		return torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
	if name == "radam":
		return torch.optim.RAdam(params, lr=float(lr), weight_decay=float(weight_decay))
	raise ValueError(f"Unsupported optimizer: {name!r}")


def _make_scheduler(
	optimizer: torch.optim.Optimizer,
	*,
	name: str,
	n_iters: int,
	eta_min: float | None,
	gamma: float | None,
) -> torch.optim.lr_scheduler.LRScheduler:
	name = str(name or "constant").strip().lower()
	if name == "constant":
		return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
	if name == "cosine":
		eta = float(eta_min) if eta_min is not None else 0.0
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(n_iters), eta_min=eta)
	if name == "exponential":
		g = float(gamma) if gamma is not None else 1.0
		return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=g)
	raise ValueError(f"Unsupported scheduler: {name!r}")


def run_alrnn_probabilistic_ablation(
	checkpoint: dict[str, Any],
	dataset: dict[str, Any],
	protocol: dict[str, Any],
	*,
	baseline_keys: list[str],
	seed: int,
	device: torch.device,
	max_iterations: int | None = None,
	batch_size_override: int | None = None,
	eval_every: int | None = None,
	save_trace_snapshots: bool = False,
	innovation_jitter: float = 1e-6,
	use_jacobian_factor: bool = False,
) -> dict[str, Any]:
	innovation_jitter = float(innovation_jitter)
	use_jacobian_factor = bool(use_jacobian_factor)
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

	filter_name = str(probabilistic.get("filter", "ekf")).strip().lower()
	if filter_name not in {"ekf", "rrkf"}:
		raise ValueError(f"Unsupported probabilistic_training.filter: {filter_name!r}")
	filter_rank_raw = probabilistic.get("filter_rank")
	filter_rank = None if filter_rank_raw is None else int(filter_rank_raw)
	if filter_name == "rrkf" and (filter_rank is None or filter_rank < 1):
		raise ValueError("probabilistic_training.filter_rank is required when filter is rrkf")

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
	val_windows = int(_SELECTOR_TRACE_VAL_WINDOWS)
	val_window_seed = int(_SELECTOR_TRACE_VAL_WINDOW_SEED_BASE) + int(seed)
	checkpoint_selection_metric = "D_stsp_stoch"
	selector_burn = int(probabilistic.get("selector_burn", evaluation.get("core_start_index", 500)))
	selector_roll = int(probabilistic.get("selector_roll", 10_000))
	selector_sigma2 = float(probabilistic.get("selector_sigma2", evaluation.get("state_space_sigma2", 1.0)))
	selector_mc_samples = int(
		probabilistic.get("selector_mc_samples", evaluation.get("state_space_mc_samples", 1000))
	)
	selector_chunk = int(
		probabilistic.get("selector_chunk", evaluation.get("state_space_density_batch_size", 512))
	)

	results: dict[str, Any] = {}
	loss_normalizer = float(batch_size) * float(scored_steps)

	for baseline_key in baseline_keys:
		spec = BASELINE_SPECS[baseline_key]
		_seed_everything(42 + int(seed))
		shell_kwargs = {
			"proc_std_init": torch.as_tensor(shell_init["gamma_diag_std"], dtype=torch.float32, device=device),
			"obs_std_init": torch.as_tensor(shell_init["lambda_diag_std"], dtype=torch.float32, device=device),
			"train_core": bool(spec.train_core),
			"covariance_transport": str(spec.covariance_transport),
		}
		if filter_name == "rrkf":
			model = ALRNNShellLR.from_core_checkpoint(
				checkpoint,
				filter_rank=int(filter_rank),
				**shell_kwargs,
			).to(device)
		else:
			model = ALRNNShell.from_core_checkpoint(checkpoint, **shell_kwargs).to(device)

		selector_seed = 50_000 + int(seed) + int(spec.seed_offset)

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
		val_windows_np = _prepare_fixed_val_windows(
			x_obs_test,
			n_windows=val_windows,
			total_steps=total_steps,
			seed=val_window_seed,
		)
		val_windows_t = torch.as_tensor(val_windows_np, dtype=torch.float32, device=device)
		trace_rows: list[dict[str, float | int]] = []
		trace_snapshots: dict[int, dict[str, torch.Tensor]] = {}
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
			total_nll = _objective_total_nll(
				model,
				x_windows,
				objective=spec.objective,
				burn_in_steps=burn_in_steps,
				innovation_jitter=innovation_jitter,
				use_jacobian_factor=use_jacobian_factor,
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
					objective=spec.objective,
					burn_in_steps=burn_in_steps,
					scored_steps=scored_steps,
					innovation_jitter=innovation_jitter,
					use_jacobian_factor=use_jacobian_factor,
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
				if bool(save_trace_snapshots):
					trace_snapshots[int(iteration + 1)] = copy.deepcopy(model.state_dict())
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
		selection_row = selected_row
		selected_val_nll = float(selection_row.get("val_nll", float("nan")))
		baseline_out: dict[str, Any] = {
			"spec": spec,
			"model": model,
			"history": trace_rows,
			"train_nll_history": train_nll_history,
			"selection_metric": checkpoint_selection_metric,
			"selection_value": float(selection_row.get("D_stsp_stoch", float("nan"))),
			"selected_iteration": _trace_iteration(selection_row),
			"selected_val_nll": selected_val_nll,
			"final_metrics": {
				"innovation_nll_per_step": float(final_row.get("train_nll_per_step", float("nan"))),
				"final_iteration": _trace_iteration(final_row),
				"final_train_nll_per_step": float(final_row.get("train_nll_per_step", float("nan"))),
				"selected_train_nll_per_step": float(selection_row.get("train_nll_per_step", float("nan"))),
				"selected_iteration": _trace_iteration(selection_row),
				"selected_D_stsp_stoch": float(selection_row.get("D_stsp_stoch", float("nan"))),
				"selected_val_nll": selected_val_nll,
				"final_val_nll": float(final_row.get("val_nll", float("nan"))),
			},
			"shell_init": {
				"x_residual_std": np.asarray(shell_init["x_residual_std"], dtype=np.float32).tolist(),
				"lambda_diag_std": np.asarray(shell_init["lambda_diag_std"], dtype=np.float32).tolist(),
				"gamma_scalar_std": float(shell_init["gamma_scalar_std"]),
			},
			"train_config": {
				"optimizer": optimizer_name,
				"weight_decay": float(weight_decay),
				"scheduler": scheduler_name,
				"eta_min": float(eta_min) if eta_min is not None else None,
				"gamma": float(gamma) if gamma is not None else None,
				"shell_init_policy": str(shell_init["shell_init_policy"]),
				"trace_every": int(trace_every),
				"checkpoint_policy": checkpoint_selection_metric,
				"selector_burn": int(selector_burn),
				"selector_roll": int(selector_roll),
				"selector_sigma2": float(selector_sigma2),
				"selector_mc_samples": int(selector_mc_samples),
				"selector_chunk": int(selector_chunk),
				"filter": str(filter_name),
				"filter_rank": None if filter_rank is None else int(filter_rank),
			},
		}
		if bool(save_trace_snapshots):
			baseline_out["trace_snapshots"] = trace_snapshots
		results[baseline_key] = baseline_out

	return {
		"baselines": results,
		"seed": int(seed),
		"window_total_steps": int(total_steps),
		"covariance_burn_in_steps": int(burn_in_steps),
		"scored_steps": int(scored_steps),
	}


__all__ = [
	"BASELINE_SPECS",
	"BaselineSpec",
	"run_alrnn_probabilistic_ablation",
	"select_best_trace_row_up_to",
]
