"""Lorenz-63 appendix pipeline: core/shell training, evaluation, and table export."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor

from experiments.lorenz63_main import prepare_lorenz63_main_bundle, train_lorenz63_core
from experiments.lorenz63_outputs import L63_METHODS, aggregate_lorenz63_main_results
from experiments.lorenz96_pipeline import (
	RunMode,
	StageProgress,
	parse_train_devices,
	repo_relative_path,
	seed_list,
	write_csv,
	write_json,
)
from training import run_alrnn_probabilistic_ablation

CORE_CHECKPOINT_NAME = "alrnn_core.pt"
SHELL_CHECKPOINT_NAME = "alrnn_shell.pt"


@dataclass
class Lorenz63RunPaths:
	repo_root: Path
	config_path: Path
	output_root: Path
	core_source_root: Path
	shell_source_root: Path
	summary_csv: Path
	per_seed_output: Path
	meta_output: Path
	core_root: Path
	shell_root: Path


def resolve_paths(
	repo_root: Path,
	*,
	config: Path,
	output_root: Path,
	core_source_root: Path,
	shell_source_root: Path,
	summary_csv: Path | None,
	per_seed_output: Path | None,
	meta_output: Path | None,
) -> Lorenz63RunPaths:
	def _res(p: Path) -> Path:
		return (repo_root / p).resolve() if not p.is_absolute() else p.resolve()

	out = _res(output_root)
	tables = out / "tables"
	return Lorenz63RunPaths(
		repo_root=repo_root.resolve(),
		config_path=_res(config),
		output_root=out,
		core_source_root=_res(core_source_root),
		shell_source_root=_res(shell_source_root),
		summary_csv=_res(summary_csv) if summary_csv else tables / "main_l63_ablation.csv",
		per_seed_output=_res(per_seed_output) if per_seed_output else tables / "main_l63_ablation_per_seed.csv",
		meta_output=_res(meta_output) if meta_output else tables / "main_l63_ablation.json",
		core_root=out / "cores",
		shell_root=out / "shells",
	)


def load_config(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _core_checkpoint_path(core_root: Path, seed: int) -> Path:
	return core_root / f"seed{int(seed):02d}" / CORE_CHECKPOINT_NAME


def _shell_checkpoint_path(shell_root: Path, seed: int, baseline_key: str) -> Path:
	return shell_root / f"seed{int(seed):02d}" / baseline_key / SHELL_CHECKPOINT_NAME


def _verify_core_checkpoints(core_root: Path, seeds: list[int]) -> None:
	for seed in seeds:
		path = _core_checkpoint_path(core_root, int(seed))
		if not path.exists():
			raise FileNotFoundError(f"Missing core checkpoint: {path}")


def _save_core_checkpoint(
	path: Path,
	model: torch.nn.Module,
	dataset: dict[str, Any],
	protocol: dict[str, Any],
	train_info: dict[str, Any],
	*,
	seed: int,
	config: dict[str, Any],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"state_dict": model.state_dict(),
			"M": int(protocol["model"]["latent_dim"]),
			"P": int(protocol["model"]["gated_dim"]),
			"N": int(protocol["model"]["observed_dim"]),
			"mean": dataset["standardizer"]["mean"].reshape(1, -1),
			"std": dataset["standardizer"]["std"].reshape(1, -1),
			"seed": int(seed),
			"data_seed": int(protocol["seed_sweep"].get("data_seed", 0)),
			"teacher_forcing_interval": int(protocol["model"]["teacher_forcing_interval"]),
			"n_interleave": int(protocol["model"]["teacher_forcing_interval"]),
			"sigma_obs": float(protocol["system"]["sigma_obs"]),
			"sigma_proc": float(protocol["system"]["sigma_proc"]),
			"selection_metric": train_info["selection_metric"],
			"selection_value": float(train_info["selection_value"]),
			"selection_epoch": int(train_info["selection_epoch"]),
			"config_json": json.dumps(config, sort_keys=True),
		},
		path,
	)


def _train_cores_sequential(
	config: dict[str, Any],
	*,
	core_root: Path,
	seeds: list[int],
	device: torch.device,
	smoke: bool,
	progress: StageProgress,
) -> None:
	progress.start("Train deterministic cores", total=len(seeds))
	for seed in seeds:
		if _core_checkpoint_path(core_root, int(seed)).exists():
			progress.update(message=f"seed {seed:02d} (skip)")
			continue
		prepared = prepare_lorenz63_main_bundle(config, smoke=bool(smoke))
		model, train_info = train_lorenz63_core(
			prepared["dataset"],
			prepared["protocol"],
			seed=int(seed),
			device=device,
			max_epochs=4 if smoke else None,
			max_batches_per_epoch=2 if smoke else None,
			eval_every=1 if smoke else None,
			free_rollout_horizon=64 if smoke else 256,
		)
		_save_core_checkpoint(
			_core_checkpoint_path(core_root, int(seed)),
			model,
			prepared["dataset"],
			prepared["protocol"],
			train_info,
			seed=int(seed),
			config=config,
		)
		progress.update(message=f"seed {seed:02d}")
	progress.end()


def _train_cores_parallel(
	paths: Lorenz63RunPaths,
	seeds: list[int],
	*,
	train_devices: list[str],
	smoke: bool,
	cpu_threads: int,
	progress: StageProgress,
) -> None:
	pending = [int(seed) for seed in seeds if not _core_checkpoint_path(paths.core_root, int(seed)).exists()]
	skipped = len(seeds) - len(pending)
	if skipped:
		print(f"Skipping {skipped} already-trained core seeds", flush=True)
	progress.start("Train deterministic cores", total=len(seeds))
	if skipped:
		progress.update(skipped)
	if not pending:
		progress.end()
		return

	worker = paths.repo_root / "scripts" / "l63_train_one_core.py"
	config_rel = (
		paths.config_path.relative_to(paths.repo_root)
		if paths.config_path.is_relative_to(paths.repo_root)
		else paths.config_path
	)
	core_rel = (
		paths.core_root.relative_to(paths.repo_root)
		if paths.core_root.is_relative_to(paths.repo_root)
		else paths.core_root
	)

	def _run_seed(seed: int, device_str: str) -> None:
		cmd = [
			sys.executable,
			str(worker),
			"--config",
			str(config_rel),
			"--seed",
			str(int(seed)),
			"--out-root",
			str(core_rel),
			"--device",
			device_str,
			"--cpu-threads",
			str(int(cpu_threads)),
		]
		if smoke:
			cmd.append("--smoke")
		subprocess.run(cmd, check=True, cwd=str(paths.repo_root))

	assignments = [pending[index :: len(train_devices)] for index in range(len(train_devices))]

	def _run_device(device_str: str, device_seeds: list[int]) -> None:
		for seed in device_seeds:
			_run_seed(seed, device_str)
			progress.update(message=f"seed {seed:02d}")

	with ThreadPoolExecutor(max_workers=len(train_devices)) as pool:
		futures = [
			pool.submit(_run_device, device, device_seeds)
			for device, device_seeds in zip(train_devices, assignments)
			if device_seeds
		]
		for future in futures:
			future.result()
	progress.end()


def _save_shell_checkpoint(path: Path, model: torch.nn.Module, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save({"state_dict": model.state_dict(), **payload}, path)


def _train_one_shell_job(
	config: dict[str, Any],
	*,
	core_root: Path,
	shell_root: Path,
	seed: int,
	baseline_key: str,
	device: torch.device,
	smoke: bool,
) -> None:
	prepared = prepare_lorenz63_main_bundle(config, smoke=bool(smoke))
	checkpoint = torch.load(
		_core_checkpoint_path(core_root, int(seed)),
		map_location="cpu",
		weights_only=False,
	)
	results = run_alrnn_probabilistic_ablation(
		checkpoint,
		prepared["dataset"],
		prepared["protocol"],
		baseline_keys=[baseline_key],
		seed=int(seed),
		device=device,
		use_jacobian_factor=True,
		max_iterations=4 if smoke else None,
		batch_size_override=2 if smoke else None,
		eval_every=1 if smoke else None,
	)
	bundle = results["baselines"][baseline_key]
	model = bundle["model"]
	run_dir = shell_root / f"seed{int(seed):02d}" / baseline_key
	run_dir.mkdir(parents=True, exist_ok=True)
	_save_shell_checkpoint(
		run_dir / SHELL_CHECKPOINT_NAME,
		model,
		{
			"seed": int(seed),
			"baseline_key": baseline_key,
			"history": bundle["history"],
			"train_nll_history": bundle.get("train_nll_history"),
			"final_metrics": bundle["final_metrics"],
			"train_config": bundle["train_config"],
			"proc_std_exact": model._sampling_proc_std().detach().cpu().numpy(),
			"obs_std_exact": model._sampling_obs_std().detach().cpu().numpy(),
		},
	)
	trace_rows = list(bundle.get("history", []))
	if trace_rows:
		pd.DataFrame(trace_rows).to_csv(run_dir / "trace.csv", index=False)
	train_nll_hist = list(bundle.get("train_nll_history", []))
	if train_nll_hist:
		pd.DataFrame(
			{
				"iter": np.arange(1, len(train_nll_hist) + 1, dtype=int),
				"train_nll": np.asarray(train_nll_hist, dtype=np.float64),
			}
		).to_csv(run_dir / "loss.csv", index=False)
	write_json(
		run_dir / "summary.json",
		{
			"seed": int(seed),
			"baseline_key": baseline_key,
			"baseline_label": bundle["spec"].label,
			"selection_metric": bundle["selection_metric"],
			"selection_value": bundle["selection_value"],
			"selected_iteration": bundle.get("selected_iteration"),
			"smoke": bool(smoke),
		},
	)


def _train_shells_sequential(
	config: dict[str, Any],
	*,
	core_root: Path,
	shell_root: Path,
	seeds: list[int],
	baseline_keys: list[str],
	device: torch.device,
	smoke: bool,
	progress: StageProgress,
) -> None:
	jobs = [(int(s), str(b)) for s in seeds for b in baseline_keys]
	pending = [job for job in jobs if not _shell_checkpoint_path(shell_root, job[0], job[1]).exists()]
	skipped = len(jobs) - len(pending)
	if skipped:
		print(f"Skipping {skipped} already-trained shell jobs", flush=True)
	progress.start("Train probabilistic shells", total=len(jobs))
	if skipped:
		progress.update(skipped)
	for seed, baseline_key in pending:
		_train_one_shell_job(
			config,
			core_root=core_root,
			shell_root=shell_root,
			seed=seed,
			baseline_key=baseline_key,
			device=device,
			smoke=smoke,
		)
		progress.update(message=f"seed {seed:02d} {baseline_key}")
	progress.end()


def _train_shells_parallel(
	paths: Lorenz63RunPaths,
	seeds: list[int],
	baseline_keys: list[str],
	*,
	train_devices: list[str],
	smoke: bool,
	cpu_threads: int,
	progress: StageProgress,
	core_root: Path,
) -> None:
	jobs = [(int(s), str(b)) for s in seeds for b in baseline_keys]
	pending = [job for job in jobs if not _shell_checkpoint_path(paths.shell_root, job[0], job[1]).exists()]
	skipped = len(jobs) - len(pending)
	if skipped:
		print(f"Skipping {skipped} already-trained shell jobs", flush=True)
	progress.start("Train probabilistic shells", total=len(jobs))
	if skipped:
		progress.update(skipped)
	if not pending:
		progress.end()
		return

	worker = paths.repo_root / "scripts" / "l63_train_one_shell.py"
	config_rel = (
		paths.config_path.relative_to(paths.repo_root)
		if paths.config_path.is_relative_to(paths.repo_root)
		else paths.config_path
	)
	core_rel = core_root.relative_to(paths.repo_root) if core_root.is_relative_to(paths.repo_root) else core_root
	shell_rel = (
		paths.shell_root.relative_to(paths.repo_root)
		if paths.shell_root.is_relative_to(paths.repo_root)
		else paths.shell_root
	)

	def _run_job(seed: int, baseline_key: str, device_str: str) -> None:
		cmd = [
			sys.executable,
			str(worker),
			"--config",
			str(config_rel),
			"--seed",
			str(int(seed)),
			"--baseline",
			baseline_key,
			"--core-root",
			str(core_rel),
			"--out-root",
			str(shell_rel),
			"--device",
			device_str,
			"--cpu-threads",
			str(int(cpu_threads)),
		]
		if smoke:
			cmd.append("--smoke")
		subprocess.run(cmd, check=True, cwd=str(paths.repo_root))

	assignments = [pending[index :: len(train_devices)] for index in range(len(train_devices))]

	def _run_device(device_str: str, device_jobs: list[tuple[int, str]]) -> None:
		for seed, baseline_key in device_jobs:
			_run_job(seed, baseline_key, device_str)
			progress.update(message=f"seed {seed:02d} {baseline_key}")

	with ThreadPoolExecutor(max_workers=len(train_devices)) as pool:
		futures = [
			pool.submit(_run_device, device, device_jobs)
			for device, device_jobs in zip(train_devices, assignments)
			if device_jobs
		]
		for future in futures:
			future.result()
	progress.end()


def run_lorenz63_main(
	*,
	paths: Lorenz63RunPaths,
	config: dict[str, Any],
	mode: RunMode,
	seeds: list[int],
	baseline_keys: list[str],
	device: torch.device,
	smoke: bool,
	train_devices: list[str] | None = None,
	cpu_threads: int = 2,
) -> dict[str, Any]:
	unknown = [key for key in baseline_keys if key not in L63_METHODS]
	if unknown:
		raise ValueError(f"Lorenz-63 appendix supports internal methods only, got {unknown}")
	progress = StageProgress()
	paths.output_root.mkdir(parents=True, exist_ok=True)
	workers = list(train_devices or parse_train_devices(None, fallback_device=device))
	parallel_train = len(workers) > 1

	if mode is RunMode.FROM_SCRATCH:
		paths.core_root.mkdir(parents=True, exist_ok=True)
		if parallel_train:
			_train_cores_parallel(
				paths,
				seeds,
				train_devices=workers,
				smoke=smoke,
				cpu_threads=cpu_threads,
				progress=progress,
			)
		else:
			_train_cores_sequential(
				config,
				core_root=paths.core_root,
				seeds=seeds,
				device=device,
				smoke=smoke,
				progress=progress,
			)
		core_eval_root = paths.core_root
	else:
		core_eval_root = paths.core_source_root
		_verify_core_checkpoints(core_eval_root, seeds)
		print(f"Using core checkpoints from {core_eval_root}", flush=True)

	if mode is RunMode.USE_CHECKPOINTS:
		shell_eval_root = paths.shell_source_root
		if not shell_eval_root.is_dir():
			raise FileNotFoundError(f"Missing shell checkpoints: {shell_eval_root}")
		print(f"Using shell checkpoints from {shell_eval_root}", flush=True)
	else:
		shell_eval_root = paths.shell_root
		paths.shell_root.mkdir(parents=True, exist_ok=True)
		if parallel_train:
			_train_shells_parallel(
				paths,
				seeds,
				baseline_keys,
				train_devices=workers,
				smoke=smoke,
				cpu_threads=cpu_threads,
				progress=progress,
				core_root=core_eval_root,
			)
		else:
			_train_shells_sequential(
				config,
				core_root=core_eval_root,
				shell_root=paths.shell_root,
				seeds=seeds,
				baseline_keys=baseline_keys,
				device=device,
				smoke=smoke,
				progress=progress,
			)

	progress.start("Evaluating metrics", total=len(seeds) * len(baseline_keys))
	aggregated = aggregate_lorenz63_main_results(
		config,
		core_eval_root,
		shell_eval_root,
		seeds=seeds,
		baseline_keys=baseline_keys,
		smoke=bool(smoke),
		device=device,
		on_progress=lambda message: progress.update(message=message),
	)
	progress.end()

	write_csv(paths.per_seed_output, aggregated["per_seed_rows"])
	write_csv(paths.summary_csv, aggregated["summary_rows"])
	meta = {
		"name": "lorenz63_main",
		"mode": mode.value,
		"n_seeds": int(len(seeds)),
		"seeds": list(seeds),
		"core_eval_root": repo_relative_path(paths.repo_root, core_eval_root),
		"shell_eval_root": repo_relative_path(paths.repo_root, shell_eval_root),
		"metric_protocol": aggregated["metric_protocol"],
		"metric_cols": list(aggregated["summary_rows"][0].keys()) if aggregated["summary_rows"] else [],
		"rows": aggregated["summary_rows"],
	}
	write_json(paths.meta_output, meta)
	aggregated["summary_output"] = str(paths.summary_csv)
	return aggregated


__all__ = [
	"Lorenz63RunPaths",
	"RunMode",
	"load_config",
	"parse_train_devices",
	"resolve_paths",
	"run_lorenz63_main",
	"seed_list",
]
