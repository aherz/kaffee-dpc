"""DynaMix experiment pipeline: training, evaluation, and results export."""

from __future__ import annotations

import json
import copy
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from data import (
	DYNAMIX_ODE_SPECS,
	add_dynamix_standardized_obs_noise,
	build_dynamix_window_dataset,
	dynamix_nll_horizon_steps,
	dynamix_required_reference_steps,
	dynamix_scored_window_length,
	estimate_dynamix_tf_residual_noise_init,
	generate_dynamix_ode_system,
)
from experiments.dynamix_eval import (
	evaluate_dynamix_core_rollout,
	evaluate_rmse_horizon,
	score_held_nll_horizon,
)
from experiments.dynamix_main import prepare_dynamix_bundle, matched_adaptation_train_total_t
from experiments.dynamix_outputs import (
	BASELINE_LABELS,
	SUMMARY_FIELD_MAP,
	aggregate_dynamix_main_results,
	expected_reporting_n_jobs,
)
from experiments.lorenz96_pipeline import (
	StageProgress,
	load_config,
	parse_train_devices,
	seed_list,
	write_csv,
	write_json,
)
from models import DynaMixShell, load_dynamix_base
from models.dynamix_shell import load_dynamix_shell_checkpoint
from training.dynamix_adaptation import train_dynamix_shell


class RunMode(str, Enum):
	USE_CHECKPOINTS = "use-checkpoints"
	ADAPT = "adapt"


@dataclass
class DynamixRunPaths:
	repo_root: Path
	config_path: Path
	output_root: Path
	base_checkpoint_root: Path
	adaptation_source_root: Path
	summary_csv: Path
	per_seed_output: Path
	meta_output: Path
	adaptation_root: Path


def resolve_paths(
	repo_root: Path,
	*,
	config: Path,
	output_root: Path,
	adaptation_source_root: Path,
	summary_csv: Path | None,
	per_seed_output: Path | None,
	meta_output: Path | None,
) -> DynamixRunPaths:
	def _res(p: Path) -> Path:
		return (repo_root / p).resolve() if not p.is_absolute() else p.resolve()

	repo = repo_root.resolve()
	out = _res(output_root)
	tables = out / "tables"
	return DynamixRunPaths(
		repo_root=repo,
		config_path=_res(config),
		output_root=out,
		base_checkpoint_root=repo / "models" / "dynamix_base",
		adaptation_source_root=_res(adaptation_source_root),
		summary_csv=_res(summary_csv) if summary_csv else tables / "main_dynamix_ablation.csv",
		per_seed_output=_res(per_seed_output) if per_seed_output else tables / "main_dynamix_ablation_per_seed.csv",
		meta_output=_res(meta_output) if meta_output else tables / "main_dynamix_ablation.json",
		adaptation_root=out / "adaptations",
	)


def _format_metric(value: Any) -> str:
	if value is None:
		return ""
	try:
		text = float(value)
	except (TypeError, ValueError):
		return ""
	if not (text == text):
		return ""
	return f"{text:.6g}"


def paper_summary_rows(
	summary_rows: list[dict[str, Any]],
	*,
	required_system_count: int,
	required_seed_count: int,
	required_n_jobs: int | None = None,
) -> list[dict[str, str]]:
	n_jobs_required = (
		int(required_n_jobs)
		if required_n_jobs is not None
		else int(required_system_count) * int(required_seed_count)
	)
	rows: list[dict[str, str]] = []
	for summary in summary_rows:
		method_key = str(summary["method_key"])
		row: dict[str, str] = {
			"Method": BASELINE_LABELS.get(method_key, method_key),
		}
		for source_field, paper_column in SUMMARY_FIELD_MAP:
			count_field = f"{source_field.removesuffix('_median')}_n_systems"
			complete = (
				int(summary.get("n_systems", 0)) == int(required_system_count)
				and int(summary.get("n_jobs", 0)) == int(n_jobs_required)
				and int(summary.get(count_field, 0)) == int(required_system_count)
			)
			value = summary.get(source_field, float("nan")) if complete else float("nan")
			row[paper_column] = _format_metric(value)
		rows.append(row)
	return rows


def run_dynamix_main(
	*,
	paths: DynamixRunPaths,
	config: dict[str, Any],
	mode: RunMode,
	systems: list[str],
	seeds: list[int],
	baseline_keys: list[str],
	device: torch.device,
	smoke: bool,
	train_devices: list[str] | None = None,
	cpu_threads: int = 2,
) -> dict[str, Any]:
	if smoke:
		systems = list(systems[:2])
	paths.output_root.mkdir(parents=True, exist_ok=True)
	prepared = prepare_dynamix_bundle(config, smoke=bool(smoke))
	protocol = prepared["protocol"]
	aggregation_config = {**config, "baselines": list(baseline_keys)}

	if mode is RunMode.ADAPT:
		_train_dynamix_adaptations(
			paths=paths,
			protocol=protocol,
			systems=systems,
			seeds=seeds,
			baseline_keys=baseline_keys,
			device=device,
			train_devices=train_devices,
			cpu_threads=int(cpu_threads),
			smoke=bool(smoke),
			config_path=paths.config_path,
			progress=StageProgress(),
		)
		adaptation_eval_root = paths.adaptation_root
		aggregated = aggregate_dynamix_main_results(
			aggregation_config,
			adaptation_eval_root,
			systems=systems,
			seeds=seeds,
		)

	else:
		shell_root = dynamix_shell_root(paths.repo_root, paths.adaptation_source_root)
		_score_saved_dynamix_shells(
			paths=paths,
			protocol=protocol,
			shell_root=shell_root,
			systems=systems,
			seeds=seeds,
			baseline_keys=baseline_keys,
			device=device,
		)
		aggregated = aggregate_dynamix_main_results(
			aggregation_config,
			paths.adaptation_root,
			systems=systems,
			seeds=seeds,
		)
	write_csv(paths.per_seed_output, aggregated["per_seed_rows"])
	if paths.per_seed_output.name.endswith("_per_seed.csv"):
		per_system_name = paths.per_seed_output.name[: -len("_per_seed.csv")] + "_per_system.csv"
	else:
		per_system_name = paths.per_seed_output.stem + "_per_system.csv"
	per_system_output = paths.per_seed_output.with_name(per_system_name)
	write_csv(per_system_output, aggregated["per_system_rows"])
	discovered = {
		str(system): frozenset(int(s) for s in seeds)
		for system, seeds in dict(aggregated.get("seed_exclusions") or {}).items()
	}
	write_csv(
		paths.summary_csv,
		paper_summary_rows(
			aggregated["summary_rows"],
			required_system_count=len(systems),
			required_seed_count=len(seeds),
			required_n_jobs=expected_reporting_n_jobs(systems, seeds, discovered),
		),
	)
	meta = {
		"systems": systems,
		"seed_list": seeds,
		"seed_exclusions": dict(aggregated.get("seed_exclusions") or {}),
		"summary_rows": aggregated["summary_rows"],
	}
	paths.meta_output.parent.mkdir(parents=True, exist_ok=True)
	write_json(paths.meta_output, meta)
	aggregated["summary_output"] = str(paths.summary_csv)
	return aggregated


def _baseline_objective(baseline_key: str) -> tuple[str, str, bool]:
	if baseline_key == "frozen_core_shell":
		return "filtered", "jacobian", False
	if baseline_key == "open_loop_no_propagation":
		return "open_loop", "identity", True
	if baseline_key == "open_loop_tied":
		return "open_loop", "jacobian", True
	if baseline_key == "filtered_no_propagation":
		return "filtered", "identity", True
	if baseline_key == "kaffee_dynamix":
		return "filtered", "jacobian", True
	raise KeyError(f"No DynaMix training objective for baseline {baseline_key!r}")


def dynamix_shell_root(repo_root: Path, adaptation_root: Path) -> Path:
	requested = (Path(repo_root) / adaptation_root).resolve()
	if not requested.is_dir():
		raise FileNotFoundError(f"DynaMix checkpoint directory does not exist: {requested}")
	return requested


def _checkpoint_path(root: Path, baseline_key: str, system: str, seed: int) -> Path:
	seed_dir = f"seed{int(seed):03d}"
	direct = root / baseline_key / system / seed_dir / "model.safetensors"
	legacy = root / "checkpoints" / baseline_key / system / seed_dir / "model.safetensors"
	if not direct.is_file() and legacy.is_file():
		return legacy
	return direct


def _result_path(root: Path, baseline_key: str, system: str, seed: int) -> Path:
	return root / "results" / baseline_key / f"result_{baseline_key}_{system}_seed{int(seed)}.json"


def _is_non_pd_omega_error(exc: BaseException) -> bool:
	return isinstance(exc, RuntimeError) and "positive-definite Omega_t" in str(exc)


def _job_timing(device: torch.device, training_start: float) -> dict[str, Any]:
	if device.type == "cuda":
		torch.cuda.synchronize(device)
	return {
		"wall_seconds": float(time.perf_counter() - training_start),
		"cuda_peak_allocated_bytes": (
			int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
		),
	}


def _metric_block(
	*,
	held_nll: float,
	d_stsp: float,
	d_h: float,
	rmse: float,
	timing: dict[str, Any] | None,
) -> dict[str, Any]:
	return {
		"held_nll": float(held_nll),
		"D_stsp": float(d_stsp),
		"D_H": float(d_h),
		"multihorizon": {"10": {"rmse": {"mean": float(rmse)}}},
		"timing": timing,
	}


def _failed_adaptation_metrics(*, timing: dict[str, Any] | None) -> dict[str, Any]:
	"""If a run fails, we record this (NaN values), and we do not save the checkpoint."""
	return _metric_block(
		held_nll=float("nan"),
		d_stsp=float("nan"),
		d_h=float("nan"),
		rmse=float("nan"),
		timing=timing,
	)


def _save_shell(path: Path, shell: DynaMixShell) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	save_file({key: value.detach().cpu().contiguous() for key, value in shell.state_dict().items()}, str(path))


def _run_one_dynamix_job(
	*,
	repo_root: Path,
	protocol: dict[str, Any],
	adaptation_root: Path,
	baseline_key: str,
	system: str,
	seed: int,
	device: torch.device,
	saved_shell_root: Path | None = None,
) -> dict[str, Any]:
	adapt_cfg = dict(protocol["adaptation"])
	eval_cfg = dict(protocol["evaluation"])
	schedule = dict(protocol["adaptation_schedules"][baseline_key])
	context_length = int(adapt_cfg["context_length"])
	cov_burn = int(adapt_cfg["covariance_burn_in_steps"])
	train_total_t = int(schedule.get("train_total_t", 0))
	if bool(protocol.get("smoke", False)):
		reference_steps = context_length + int(eval_cfg["qoi_rollout_steps"])
	else:
		reference_steps = max(
			dynamix_required_reference_steps(context_length),
			context_length + int(eval_cfg["qoi_rollout_steps"]),
		)
	required_steps = max(
		int(train_total_t),
		int(reference_steps),
		int(matched_adaptation_train_total_t(protocol)),
	)
	x_full, dt = generate_dynamix_ode_system(system, required_steps, seed=int(seed))
	x_reference = x_full[:reference_steps]
	series_t = max(int(train_total_t), int(matched_adaptation_train_total_t(protocol)), int(context_length) + 2)
	x_train_clean = x_full[:series_t]
	x_train_noisy = add_dynamix_standardized_obs_noise(
		x_train_clean,
		sigma=float(protocol["data_noise"]["observation_std_standardized"]),
		seed=int(seed),
		context_length=context_length,
	)
	noisy_context = x_train_noisy[:context_length]
	reference_lambda1 = float(DYNAMIX_ODE_SPECS[str(system).strip().lower()].lambda1_ref)
	scored_window = dynamix_scored_window_length(
		reference_lambda1,
		dt,
		k=float(adapt_cfg["lyapunov_time_window_k"]),
		lo=int(adapt_cfg["scored_window_min"]),
		hi=int(adapt_cfg["scored_window_max"]),
	)
	nll_horizon = dynamix_nll_horizon_steps(
		reference_lambda1,
		dt,
		k=float(eval_cfg["nll_horizon_tau_lambda"]),
		lo=int(adapt_cfg["scored_window_min"]),
		hi=int(adapt_cfg["scored_window_max"]),
	)
	base_model = load_dynamix_base(
		device=device,
		repo_root=repo_root,
	)
	base_result_path = _result_path(adaptation_root, "dynamix_base", system, int(seed))
	baseline_block: dict[str, Any] | None = None
	if baseline_key != "dynamix_base" and base_result_path.is_file():
		base_payload = json.loads(base_result_path.read_text(encoding="utf-8"))
		expected_identity = ("dynamix_base", str(system), int(seed))
		actual_identity = (
			str(base_payload.get("baseline_key")),
			str(base_payload.get("system")),
			int(base_payload.get("seed", -1)),
		)
		if actual_identity != expected_identity:
			raise ValueError(f"Incompatible DynaMix base result: {base_result_path}")
		baseline_block = dict(base_payload.get("baseline") or base_payload.get("dynamix_base") or {})
	if baseline_block is None:
		baseline_dyn = evaluate_dynamix_core_rollout(
			base_model,
			x_train_raw=noisy_context,
			x_reference_raw=x_reference,
			context_length=context_length,
			device=device,
			rollout_steps=int(eval_cfg["qoi_rollout_steps"]),
			burn_in_steps=int(eval_cfg["qoi_burn_in_steps"]),
			eval_steps=int(eval_cfg["qoi_eval_steps"]),
		)
		baseline_block = {
			"D_stsp": float(baseline_dyn["D_stsp"]),
			"D_H": float(baseline_dyn["D_H"]),
		}
	result: dict[str, Any] = {
		"system": system,
		"seed": int(seed),
		"baseline_key": baseline_key,
		"baseline": {"D_stsp": float(baseline_block["D_stsp"])},
	}
	if baseline_key == "dynamix_base":
		result[baseline_key] = _metric_block(
			held_nll=float("nan"),
			d_stsp=float(baseline_block["D_stsp"]),
			d_h=float(baseline_block["D_H"]),
			rmse=float("nan"),
			timing=None,
		)
		return result

	dataset = build_dynamix_window_dataset(
		x_train_noisy,
		base_model,
		context_length=context_length,
		window_length=scored_window,
		stride=scored_window,
		covariance_burn_in_steps=cov_burn,
		device=device,
	)
	noise_init = estimate_dynamix_tf_residual_noise_init(
		x_train_noisy,
		base_model,
		context_length=context_length,
		device=device,
	)
	objective, transport, train_core = _baseline_objective(baseline_key)
	model = copy.deepcopy(base_model)
	timing = None
	if saved_shell_root is not None:
		ckpt = _checkpoint_path(saved_shell_root, baseline_key, system, int(seed))
		if not ckpt.is_file():
			result[baseline_key] = _failed_adaptation_metrics(timing=None)
			return result
		shell = load_dynamix_shell_checkpoint(ckpt, model=model, trainable_shell=False).to(device)
	else:
		shell = DynaMixShell(
			model,
			log_proc_std=float(noise_init["log_sigma_proc"]),
			log_obs_std=float(noise_init["log_sigma_obs"]),
			trainable=True,
		).to(device)
		if device.type == "cuda":
			torch.cuda.synchronize(device)
			torch.cuda.reset_peak_memory_stats(device)
		training_start = time.perf_counter()
		try:
			train_dynamix_shell(
				shell,
				dataset,
				objective=objective,
				transport=transport,
				train_core=bool(train_core),
				n_iters=int(adapt_cfg["iterations"]),
				batch_size=int(adapt_cfg["batch_size"]),
				lr=float(adapt_cfg["lr"]),
				seed=int(seed),
				covariance_burn_in_steps=cov_burn,
				n_ensemble_train=64,
				gradient_clip=float(adapt_cfg["gradient_clip"]),
				optimizer=str(adapt_cfg.get("optimizer", "Adam")),
			)
		except RuntimeError as exc:
			if not _is_non_pd_omega_error(exc):
				raise
			print(
				f"non-finite Omega_t {baseline_key}:{system}:seed{seed}",
				flush=True,
			)
			result[baseline_key] = _failed_adaptation_metrics(
				timing=_job_timing(device, training_start),
			)
			return result
		timing = _job_timing(device, training_start)
		ckpt = _checkpoint_path(adaptation_root, baseline_key, system, int(seed))
		_save_shell(ckpt, shell)
	held_nll, _n_windows_nll = score_held_nll_horizon(
		shell,
		x_train_noisy=x_train_noisy,
		context_length=context_length,
		nll_horizon=int(nll_horizon),
		device=device,
		n_starts=int(eval_cfg["nll_n_starts"]),
	)
	n_rmse = min(30, len(dataset.windows_held))
	rmse_windows = dataset.windows_held[:n_rmse]
	rmse_starts = dataset.held_starts[:n_rmse]
	rmse_horizon = int(eval_cfg["rmse_forecast_horizon_steps"])
	mean_np = dataset.mean.detach().cpu().numpy().astype(np.float64)
	std_np = dataset.std.detach().cpu().numpy().astype(np.float64)
	clean_std = ((np.asarray(x_train_clean, dtype=np.float64) - mean_np) / std_np).astype(np.float32)
	target_windows = [
		torch.as_tensor(
			clean_std[start : start + rmse_horizon + 1],
			dtype=rmse_windows[0].dtype,
			device=rmse_windows[0].device,
		)
		for start in rmse_starts
	]
	rmse_z = [shell.latent_column_from_observation(obs[0]).detach() for obs in rmse_windows]
	rmse_10 = evaluate_rmse_horizon(
		shell,
		z_inits=rmse_z,
		context=dataset.context,
		obs_windows=rmse_windows,
		target_windows=target_windows,
		horizon=rmse_horizon,
		n_ensemble=256,
		n_eval_windows=30,
	)
	dyn = evaluate_dynamix_core_rollout(
		shell.model,
		x_train_raw=noisy_context,
		x_reference_raw=x_reference,
		context_length=context_length,
		device=device,
		rollout_steps=int(eval_cfg["qoi_rollout_steps"]),
		burn_in_steps=int(eval_cfg["qoi_burn_in_steps"]),
		eval_steps=int(eval_cfg["qoi_eval_steps"]),
	)
	result[baseline_key] = _metric_block(
		held_nll=float(held_nll),
		d_stsp=float(dyn["D_stsp"]),
		d_h=float(dyn["D_H"]),
		rmse=float(rmse_10),
		timing=timing,
	)
	return result


def _score_saved_dynamix_shells(
	*,
	paths: DynamixRunPaths,
	protocol: dict[str, Any],
	shell_root: Path,
	systems: list[str],
	seeds: list[int],
	baseline_keys: list[str],
	device: torch.device,
) -> None:
	jobs = [(b, s, int(seed)) for b in baseline_keys for s in systems for seed in seeds]
	for baseline_key, system, seed in jobs:
		out = _result_path(paths.adaptation_root, baseline_key, system, seed)
		out.parent.mkdir(parents=True, exist_ok=True)
		result = _run_one_dynamix_job(
			repo_root=paths.repo_root,
			protocol=protocol,
			adaptation_root=paths.adaptation_root,
			baseline_key=baseline_key,
			system=system,
			seed=int(seed),
			device=device,
			saved_shell_root=None if baseline_key == "dynamix_base" else shell_root,
		)
		if baseline_key != "dynamix_base":
			checkpoint = _checkpoint_path(shell_root, baseline_key, system, seed)
			result["checkpoint_path"] = os.path.relpath(checkpoint, out.parent)
		write_json(out, result)


def _train_dynamix_adaptations(
	*,
	paths: DynamixRunPaths,
	protocol: dict[str, Any],
	systems: list[str],
	seeds: list[int],
	baseline_keys: list[str],
	device: torch.device,
	train_devices: list[str] | None,
	cpu_threads: int,
	smoke: bool,
	config_path: Path,
	progress: StageProgress,
) -> None:
	jobs = [(b, s, int(seed)) for b in baseline_keys for s in systems for seed in seeds]
	progress.start("Train/evaluate DynaMix baselines", total=len(jobs))
	devices = list(train_devices or [str(device)])
	worker = paths.repo_root / "scripts" / "dynamix_train_one_job.py"
	config_rel = (
		config_path.relative_to(paths.repo_root) if config_path.is_relative_to(paths.repo_root) else config_path
	)
	adapt_rel = (
		paths.adaptation_root.relative_to(paths.repo_root)
		if paths.adaptation_root.is_relative_to(paths.repo_root)
		else paths.adaptation_root
	)

	def run_job(job: tuple[str, str, int], device_name: str) -> tuple[tuple[str, str, int], str]:
		baseline_key, system, seed = job
		out = _result_path(paths.adaptation_root, baseline_key, system, seed)
		if out.exists():
			return job, "skip"
		if len(devices) <= 1:
			out.parent.mkdir(parents=True, exist_ok=True)
			result = _run_one_dynamix_job(
				repo_root=paths.repo_root,
				protocol=protocol,
				adaptation_root=paths.adaptation_root,
				baseline_key=baseline_key,
				system=system,
				seed=int(seed),
				device=torch.device(device_name),
			)
			write_json(out, result)
			return job, "done"
		cmd = [
			sys.executable,
			str(worker),
			"--config",
			str(config_rel),
			"--baseline",
			baseline_key,
			"--system",
			system,
			"--seed",
			str(int(seed)),
			"--adaptation-root",
			str(adapt_rel),
			"--device",
			device_name,
			"--cpu-threads",
			str(int(cpu_threads)),
		]
		if smoke:
			cmd.append("--smoke")
		subprocess.run(cmd, check=True, cwd=str(paths.repo_root))
		return job, "done"

	if len(devices) <= 1:
		for job in jobs:
			(baseline_key, system, seed), status = run_job(job, devices[0])
			progress.update(message=f"{baseline_key}:{system}:seed{seed} ({status})")
	else:
		assignments = [jobs[index:: len(devices)] for index in range(len(devices))]

		def run_device(device_name: str, device_jobs: list[tuple[str, str, int]]) -> None:
			for job in device_jobs:
				(baseline_key, system, seed), status = run_job(job, device_name)
				progress.update(message=f"{baseline_key}:{system}:seed{seed} ({status})")

		with ThreadPoolExecutor(max_workers=len(devices)) as pool:
			futures = [
				pool.submit(run_device, device, device_jobs)
				for device, device_jobs in zip(devices, assignments)
				if device_jobs
			]
			for future in futures:
				future.result()
	progress.end()


__all__ = [
	"DynamixRunPaths",
	"RunMode",
	"_failed_adaptation_metrics",
	"_is_non_pd_omega_error",
	"_run_one_dynamix_job",
	"load_config",
	"paper_summary_rows",
	"parse_train_devices",
	"resolve_paths",
	"dynamix_shell_root",
	"run_dynamix_main",
	"seed_list",
]
