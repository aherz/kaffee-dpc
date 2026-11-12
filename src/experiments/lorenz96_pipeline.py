"""Lorenz-96 main ablation pipeline: deterministic core/stochastic shell training, evaluation and results export."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from experiments.lorenz96_main import prepare_lorenz96_main_bundle, train_lorenz96_core
from experiments.lorenz96_outputs import GTAN_TRANSPORT_SUMMARY_FIELD, aggregate_lorenz96_main_results
from training import run_alrnn_probabilistic_ablation

CORE_CHECKPOINT_NAME = "alrnn_core.pt"
SHELL_CHECKPOINT_NAME = "alrnn_shell.pt"

PAPER_SUMMARY_FIELD_MAP: tuple[tuple[str, str], ...] = (
	("NLL_median", "NLL"),
	("CRPS_median", "CRPS"),
	(GTAN_TRANSPORT_SUMMARY_FIELD, "G_tan"),
	("D_stsp_stoch_median", "D_stsp_stoch"),
	("abs_signed_h_lambda_error_stoch_median", "|Delta h_lambda^stoch|"),
	("D_stsp_core_median", "D_stsp_core"),
	("abs_signed_h_lambda_error_core_median", "|Delta h_lambda^core|"),
	("D_H_core_median", "D_H^core"),
)


class RunMode(str, Enum):
	USE_CHECKPOINTS = "use-checkpoints"
	TRAIN_SHELLS = "train-shells"
	FROM_SCRATCH = "from-scratch"


@dataclass
class Lorenz96RunPaths:
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


class StageProgress:
	"""Console progress."""

	def __init__(self) -> None:
		self._pipeline_t0 = time.time()
		self._stage_t0 = self._pipeline_t0
		self._stage_name = ""
		self._pbar: tqdm | None = None

	def start(self, name: str, *, total: int | None = None) -> None:
		self.end()
		self._stage_name = str(name)
		self._stage_t0 = time.time()
		elapsed = self._stage_t0 - self._pipeline_t0
		print(f"\n[{self._format_duration(elapsed)}] === {self._stage_name} ===", flush=True)
		if total is not None and int(total) > 0:
			self._pbar = tqdm(total=int(total), unit="step", leave=True)

	def update(self, n: int = 1, *, message: str | None = None) -> None:
		if self._pbar is not None:
			self._pbar.update(int(n))
			if message:
				self._pbar.set_postfix_str(str(message)[:60])
		elif message:
			print(f"  {message}", flush=True)

	def end(self) -> None:
		if self._pbar is not None:
			self._pbar.close()
			self._pbar = None
		if self._stage_name:
			dt = time.time() - self._stage_t0
			print(f"--- finished {self._stage_name} in {self._format_duration(dt)} ---", flush=True)
			self._stage_name = ""

	@staticmethod
	def _format_duration(seconds: float) -> str:
		seconds = max(0.0, float(seconds))
		m, s = divmod(int(seconds), 60)
		h, m = divmod(m, 60)
		if h:
			return f"{h:d}h{m:02d}m"
		if m:
			return f"{m:d}m{s:02d}s"
		return f"{s:d}s"


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
) -> Lorenz96RunPaths:
	def _res(p: Path) -> Path:
		return (repo_root / p).resolve() if not p.is_absolute() else p.resolve()

	out = _res(output_root)
	tables = out / "tables"
	return Lorenz96RunPaths(
		repo_root=repo_root.resolve(),
		config_path=_res(config),
		output_root=out,
		core_source_root=_res(core_source_root),
		shell_source_root=_res(shell_source_root),
		summary_csv=_res(summary_csv) if summary_csv else tables / "main_l96_ablation.csv",
		per_seed_output=_res(per_seed_output) if per_seed_output else tables / "main_l96_ablation_per_seed.csv",
		meta_output=_res(meta_output) if meta_output else tables / "main_l96_ablation.json",
		core_root=out / "cores",
		shell_root=out / "shells",
	)


def repo_relative_path(repo_root: Path, path: Path) -> str:
	"""Path string for metadata."""
	resolved = Path(path).resolve()
	try:
		return str(resolved.relative_to(Path(repo_root).resolve()))
	except ValueError:
		return str(resolved)


def load_config(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def seed_list(
	config: dict[str, Any],
	*,
	smoke: bool,
	max_seeds: int | None,
	seed: int | None,
) -> list[int]:
	if seed is not None:
		return [int(seed)]
	sweep = dict(config["seed_sweep"])
	count = int(sweep.get("count", 20))
	if smoke:
		count = min(count, 2)
	if max_seeds is not None:
		count = min(count, int(max_seeds))
	start = int(sweep.get("start_seed", 0))
	return list(range(start, start + count))


def parse_train_devices(gpus_arg: str | None, *, fallback_device: torch.device) -> list[str]:
	"""Parse `--gpus` into worker device strings (`0,1`, `cuda:0`, `all`, `cpu`).

	This excludes duplicate aliases (`0` and `cuda:0`) and CUDA indices that are not
	present so a sweep cannot silently share a GPU.
	"""
	if gpus_arg is None:
		if fallback_device.type == "cuda":
			index = 0 if fallback_device.index is None else int(fallback_device.index)
			return [f"cuda:{index}"]
		return [str(fallback_device)]

	text = str(gpus_arg).strip().lower()
	if text == "all":
		if not torch.cuda.is_available():
			raise RuntimeError("--gpus all requires CUDA")
		return [f"cuda:{index}" for index in range(int(torch.cuda.device_count()))]

	devices: list[str] = []
	for part in text.split(","):
		token = part.strip()
		if not token:
			continue
		if token.isdigit():
			device_index = int(token)
			devices.append(f"cuda:{device_index}")
		elif token.startswith("cuda"):
			parsed = torch.device(token)
			device_index = 0 if parsed.index is None else int(parsed.index)
			devices.append(f"cuda:{device_index}")
		elif token == "cpu":
			devices.append("cpu")
		else:
			raise ValueError(f"Unsupported GPU id in --gpus: {token!r}")
	if not devices:
		raise ValueError("--gpus was provided but no devices were parsed")
	if len(set(devices)) != len(devices):
		raise ValueError(f"--gpus contains duplicate device aliases: {devices}")
	for device_name in devices:
		if device_name.startswith("cuda:") and int(device_name.split(":", 1)[1]) >= torch.cuda.device_count():
			raise ValueError(f"CUDA device is not available: {device_name}")
	return devices


def _core_checkpoint_path(core_root: Path, seed: int) -> Path:
	return core_root / f"seed{int(seed):02d}" / CORE_CHECKPOINT_NAME


def _shell_checkpoint_path(shell_root: Path, seed: int, baseline_key: str) -> Path:
	return shell_root / f"seed{int(seed):02d}" / baseline_key / SHELL_CHECKPOINT_NAME


def _verify_core_checkpoints(core_root: Path, seeds: list[int]) -> None:
	for seed in seeds:
		path = _core_checkpoint_path(core_root, int(seed))
		if not path.exists():
			raise FileNotFoundError(f"Missing core checkpoint: {path}")


def _train_cores_sequential(
	config: dict[str, Any],
	dataset: dict[str, Any],
	protocol: dict[str, Any],
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
		run_dir = core_root / f"seed{int(seed):02d}"
		run_dir.mkdir(parents=True, exist_ok=True)
		model, train_info = train_lorenz96_core(
			dataset,
			protocol,
			seed=int(seed),
			device=device,
			max_epochs=4 if smoke else None,
			max_batches_per_epoch=2 if smoke else None,
			eval_every=1 if smoke else None,
			free_rollout_horizon=64 if smoke else 256,
		)
		checkpoint = {
			"state_dict": model.state_dict(),
			"M": int(protocol["model"]["latent_dim"]),
			"P": int(protocol["model"]["gated_dim"]),
			"N": int(protocol["model"]["observed_dim"]),
			"mean": dataset["standardizer"]["mean"].reshape(1, -1),
			"std": dataset["standardizer"]["std"].reshape(1, -1),
			"seed": int(seed),
			"data_seed": int(protocol["seed_sweep"].get("data_seed", 0)),
			"teacher_forcing_interval": int(protocol["model"]["teacher_forcing_interval"]),
			"selection_metric": train_info["selection_metric"],
			"selection_value": float(train_info["selection_value"]),
			"selection_epoch": int(train_info["selection_epoch"]),
			"config_json": json.dumps(config, sort_keys=True),
		}
		torch.save(checkpoint, run_dir / CORE_CHECKPOINT_NAME)
		progress.update(message=f"seed {seed:02d}")
	progress.end()


def _train_cores_parallel(
	paths: Lorenz96RunPaths,
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

	worker = paths.repo_root / "scripts" / "l96_train_one_core.py"
	config_rel = paths.config_path.relative_to(paths.repo_root) if paths.config_path.is_relative_to(paths.repo_root) else paths.config_path
	core_rel = paths.core_root.relative_to(paths.repo_root) if paths.core_root.is_relative_to(paths.repo_root) else paths.core_root

	def _run_seed(seed: int, device_str: str) -> int:
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
		return int(seed)

	assignments = [pending[index:: len(train_devices)] for index in range(len(train_devices))]

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
	dataset: dict[str, Any],
	protocol: dict[str, Any],
	*,
	core_root: Path,
	shell_root: Path,
	seed: int,
	baseline_key: str,
	device: torch.device,
	smoke: bool,
) -> None:
	checkpoint = torch.load(
		_core_checkpoint_path(core_root, int(seed)),
		map_location="cpu",
		weights_only=False,
	)
	results = run_alrnn_probabilistic_ablation(
		checkpoint,
		dataset,
		protocol,
		baseline_keys=[baseline_key],
		seed=int(seed),
		device=device,
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
			"filter": bundle["train_config"].get("filter"),
			"filter_rank": bundle["train_config"].get("filter_rank"),
			"smoke": bool(smoke),
		},
	)


def _train_shells_sequential(
	config: dict[str, Any],
	dataset: dict[str, Any],
	protocol: dict[str, Any],
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
			dataset,
			protocol,
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
	paths: Lorenz96RunPaths,
	seeds: list[int],
	baseline_keys: list[str],
	*,
	train_devices: list[str],
	smoke: bool,
	cpu_threads: int,
	progress: StageProgress,
	core_root: Path,
	filter_rank: int | None = None,
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

	worker = paths.repo_root / "scripts" / "l96_train_one_shell.py"
	config_rel = paths.config_path.relative_to(paths.repo_root) if paths.config_path.is_relative_to(paths.repo_root) else paths.config_path
	core_rel = core_root.relative_to(paths.repo_root) if core_root.is_relative_to(paths.repo_root) else core_root
	shell_rel = paths.shell_root.relative_to(paths.repo_root) if paths.shell_root.is_relative_to(paths.repo_root) else paths.shell_root

	def _run_job(seed: int, baseline_key: str, device_str: str) -> tuple[int, str]:
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
		if filter_rank is not None:
			cmd.extend(["--filter-rank", str(int(filter_rank))])
		subprocess.run(cmd, check=True, cwd=str(paths.repo_root))
		return int(seed), baseline_key

	print(f"Training {len(pending)} shell jobs on {len(train_devices)} GPU worker(s): {train_devices}", flush=True)
	assignments = [pending[index:: len(train_devices)] for index in range(len(train_devices))]

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


def _format_metric(value: Any) -> str:
	text = float(value)
	if not np.isfinite(text):
		return ""
	return f"{text:.6g}"


def paper_summary_rows(
	summary_rows: list[dict[str, Any]],
	*,
	required_seed_count: int,
) -> list[dict[str, str]]:
	rows: list[dict[str, str]] = []
	for summary in summary_rows:
		method_key = str(summary["method_key"])
		label = "AL-RNN (no shell)" if method_key == "alrnn_core" else str(summary.get("method_label", method_key))
		row: dict[str, str] = {"Model / Objective": label}
		for source_field, paper_column in PAPER_SUMMARY_FIELD_MAP:
			count_field = f"{source_field.removesuffix('_median')}_n_finite"
			complete = (
				int(summary.get("seed_count", 0)) == int(required_seed_count)
				and int(summary.get(count_field, 0)) == int(required_seed_count)
			)
			value = summary.get(source_field, float("nan")) if complete else float("nan")
			if source_field.startswith("NLL") or source_field.startswith("CRPS"):
				row[paper_column] = "" if method_key == "alrnn_core" else _format_metric(value)
			else:
				row[paper_column] = _format_metric(value)
		rows.append(row)
	return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	if not rows:
		return
	fieldnames: list[str] = []
	for row in rows:
		for key in row:
			if key not in fieldnames:
				fieldnames.append(key)
	path.parent.mkdir(parents=True, exist_ok=True)
	import csv

	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run_lorenz96_main(
	*,
	paths: Lorenz96RunPaths,
	config: dict[str, Any],
	mode: RunMode,
	seeds: list[int],
	baseline_keys: list[str],
	device: torch.device,
	smoke: bool,
	train_devices: list[str] | None = None,
	cpu_threads: int = 2,
) -> dict[str, Any]:
	progress = StageProgress()
	paths.output_root.mkdir(parents=True, exist_ok=True)

	prepared = prepare_lorenz96_main_bundle(config, smoke=bool(smoke))
	dataset = prepared["dataset"]
	protocol = prepared["protocol"]

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
				dataset,
				protocol,
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
			prob = dict(protocol.get("probabilistic_training", {}))
			rank_override = (
				int(prob["filter_rank"])
				if str(prob.get("filter", "ekf")).strip().lower() == "rrkf"
				else None
			)
			_train_shells_parallel(
				paths,
				seeds,
				baseline_keys,
				train_devices=workers,
				smoke=smoke,
				cpu_threads=cpu_threads,
				progress=progress,
				core_root=core_eval_root,
				filter_rank=rank_override,
			)
		else:
			_train_shells_sequential(
				config,
				dataset,
				protocol,
				core_root=core_eval_root,
				shell_root=paths.shell_root,
				seeds=seeds,
				baseline_keys=baseline_keys,
				device=device,
				smoke=smoke,
				progress=progress,
			)

	n_seeds = len(seeds)
	n_shells = len(baseline_keys)
	# 3-axis eval: 2 reference spectra, then dynamical/DSR (core + shells) and NLL/CRPS (shells).
	progress.start("Evaluating metrics", total=2 + n_seeds * (1 + 2 * n_shells))
	aggregated = aggregate_lorenz96_main_results(
		config,
		core_eval_root,
		shell_eval_root,
		seeds=seeds,
		smoke=bool(smoke),
		device=device,
		on_progress=lambda message: progress.update(message=message),
	)
	method_keys = {"alrnn_core", *baseline_keys}
	aggregated["per_seed_rows"] = [
		row for row in aggregated["per_seed_rows"] if str(row["method_key"]) in method_keys
	]
	aggregated["summary_rows"] = [
		row for row in aggregated["summary_rows"] if str(row["method_key"]) in method_keys
	]
	expected = {(int(seed), method) for seed in seeds for method in method_keys}
	actual = {
		(int(row["seed"]), str(row["method_key"]))
		for row in aggregated["per_seed_rows"]
	}
	if actual != expected:
		raise RuntimeError(
			"Incomplete Lorenz-96 result grid: "
			f"expected {len(expected)} seed-method rows, found {len(actual)}"
		)
	progress.end()

	write_csv(paths.per_seed_output, aggregated["per_seed_rows"])
	write_csv(
		paths.summary_csv,
		paper_summary_rows(
			aggregated["summary_rows"],
			required_seed_count=len(seeds),
		),
	)
	meta = {
		"mode": mode.value,
		"seed_list": seeds,
		"core_source_root": repo_relative_path(paths.repo_root, paths.core_source_root),
		"core_eval_root": repo_relative_path(paths.repo_root, core_eval_root),
		"shell_source_root": repo_relative_path(paths.repo_root, paths.shell_source_root),
		"shell_eval_root": repo_relative_path(paths.repo_root, shell_eval_root),
		"filter": str(dict(protocol.get("probabilistic_training", {})).get("filter", "ekf")),
		"filter_rank": dict(protocol.get("probabilistic_training", {})).get("filter_rank"),
		"metric_protocol": aggregated["metric_protocol"],
		"summary_rows": aggregated["summary_rows"],
		"summary_output": repo_relative_path(paths.repo_root, paths.summary_csv),
	}
	paths.meta_output.parent.mkdir(parents=True, exist_ok=True)
	write_json(paths.meta_output, meta)
	aggregated["summary_output"] = str(paths.summary_csv)
	return aggregated
