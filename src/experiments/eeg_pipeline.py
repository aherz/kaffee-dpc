"""EEG pipeline: deterministic core/stochastic shell training, evaluation and results export."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np

import torch

from data.eeg import SUBJECTS, load_eeg_recording
from experiments.eeg_eval import score_eeg_core, score_eeg_shell, write_core_metrics, write_shell_eval
from experiments.eeg_outputs import EEG_METHODS, aggregate_eeg_results, write_eeg_tables
from models.shplrnn import ClippedShallowPLRNN, LinearObservation
from models.shplrnn_shell import ShallowPLRNNShell
from training.train_shplrnn_prior import train_shplrnn_prior
from training.train_shplrnn_probabilistic import train_shplrnn_probabilistic


class RunMode(str, Enum):
	USE_CHECKPOINTS = "use-checkpoints"
	TRAIN_SHELLS = "train-shells"
	FROM_SCRATCH = "from-scratch"


# settings for quick (smoke) runs
SMOKE_HISTORY = 2
SMOKE_BURN = 4
SMOKE_SCORE = 32
SMOKE_MC = 32


@dataclass(frozen=True)
class EEGRunPaths:
	repo_root: Path
	config_path: Path
	output_root: Path
	core_source_root: Path
	shell_source_root: Path
	core_root: Path
	shell_root: Path


def load_config(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def resolve_paths(
	repo_root: Path,
	*,
	config: Path,
	output_root: Path,
	core_source_root: Path,
	shell_source_root: Path,
) -> EEGRunPaths:
	def _res(path: Path) -> Path:
		return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

	out = _res(output_root)
	return EEGRunPaths(
		repo_root=repo_root.resolve(),
		config_path=_res(config),
		output_root=out,
		core_source_root=_res(core_source_root),
		shell_source_root=_res(shell_source_root),
		core_root=out / "cores",
		shell_root=out / "shells",
	)


def _scoring_config(config: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
	scored = copy.deepcopy(config)
	if smoke:
		scored["evaluation"]["filter_history_steps"] = SMOKE_HISTORY
		scored["evaluation"]["rollout_burn_steps"] = SMOKE_BURN
		scored["evaluation"]["rollout_score_steps"] = SMOKE_SCORE
		scored["evaluation"]["mc_samples"] = SMOKE_MC
	return scored


def _training_config(config: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
	trained = copy.deepcopy(config)
	if not smoke:
		return trained
	shell = trained["shell_training"]
	shell["selector"]["rollout_steps"] = SMOKE_SCORE
	shell["window_total_steps"] = 8
	shell["covariance_burn_in_steps"] = 2
	shell["scored_steps"] = 6
	core = trained["core_training"]
	core["selector"]["score_steps"] = SMOKE_SCORE
	core["selector"]["initial_states"] = min(int(core["selector"]["initial_states"]), 2)
	return trained


def _prepare_run_config(config: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
	return _scoring_config(_training_config(config, smoke=smoke), smoke=smoke)


def _recording_for_scoring(subject: str, config: dict[str, Any]) -> np.ndarray:
	recording = load_eeg_recording(subject)
	score_steps = int(config["evaluation"]["rollout_score_steps"])
	if recording.shape[0] == score_steps:
		return recording
	if recording.shape[0] < score_steps:
		raise ValueError(f"EEG recording {subject} is shorter than the scoring window")
	return np.ascontiguousarray(recording[:score_steps])


def _set_cpu_threads(cpu_threads: int) -> None:
	torch.set_num_threads(max(1, int(cpu_threads)))
	try:
		torch.set_num_interop_threads(1)
	except RuntimeError:
		pass


def _core_dir(root: Path, subject: str, seed: int) -> Path:
	return root / subject / f"seed{seed:02d}"


def _shell_dir(root: Path, subject: str, seed: int, method: str) -> Path:
	return root / subject / f"seed{seed:02d}" / method


def _score_source_core(
	source_core: Path,
	output_core: Path,
	data: np.ndarray,
	*,
	config: dict[str, Any],
	device: str,
) -> None:
	metrics_path = output_core / "gtf_table_metrics_gpu.json"
	core = ClippedShallowPLRNN(int(config["model"]["latent_dim"]), int(config["model"]["hidden_dim"]))
	observation = LinearObservation(int(config["data"]["observed_dim"]), int(config["model"]["latent_dim"]))
	core.load_state_dict(torch.load(source_core / "core.pt", map_location="cpu", weights_only=True))
	observation.load_state_dict(torch.load(source_core / "observation.pt", map_location="cpu", weights_only=True))
	core = core.to(device).eval()
	observation = observation.to(device).eval()
	write_core_metrics(metrics_path, score_eeg_core(core, observation, data, config=config))


def _score_saved_core(core_dir: Path, data: np.ndarray, *, config: dict[str, Any], device: str) -> None:
	_score_source_core(core_dir, core_dir, data, config=config, device=device)


def _train_cores(
	*,
	paths: EEGRunPaths,
	config: dict[str, Any],
	subjects: list[str],
	seeds: list[int],
	device: str,
	smoke: bool,
) -> None:
	model_config = {**dict(config["model"]), "observed_dim": int(config["data"]["observed_dim"])}
	for subject in subjects:
		recording = load_eeg_recording(subject)
		scored = _recording_for_scoring(subject, config)
		for seed in seeds:
			core_dir = _core_dir(paths.core_root, subject, seed)
			if not (core_dir / "core.pt").is_file():
				print(f"train core {subject} seed{seed:02d}", flush=True)
				train_shplrnn_prior(
					recording,
					model_config=model_config,
					training_config=dict(config["core_training"]),
					seed=int(seed),
					output_dir=core_dir,
					device=device,
					smoke=smoke,
				)
			_score_saved_core(core_dir, scored, config=config, device=device)


def _train_and_score_shells(
	*,
	core_root: Path,
	shell_root: Path,
	config: dict[str, Any],
	subjects: list[str],
	seeds: list[int],
	methods: list[str],
	device: str,
	smoke: bool,
) -> None:
	for subject in subjects:
		recording = load_eeg_recording(subject)
		scored = _recording_for_scoring(subject, config)
		for seed in seeds:
			core_dir = _core_dir(core_root, subject, seed)
			if not (core_dir / "core.pt").is_file() or not (core_dir / "observation.pt").is_file():
				raise FileNotFoundError(f"EEG core is incomplete: {core_dir}")
			for method in methods:
				output = _shell_dir(shell_root, subject, seed, method)
				bundle = output / "shell_bundle.pt"
				eval_path = output / "eval.json"
				if not bundle.is_file():
					print(f"train shell {subject} seed{seed:02d} {method}", flush=True)
					train_shplrnn_probabilistic(
						recording,
						config=config,
						method=method,
						seed=int(seed),
						core_dir=core_dir,
						output_dir=output,
						device=device,
						smoke=smoke,
					)
				shell = ShallowPLRNNShell.load(bundle, method_key=method, device=device).eval()
				payload = score_eeg_shell(
					shell,
					scored,
					subject=subject,
					seed=seed,
					method=method,
					config=config,
				)
				write_shell_eval(eval_path, payload)
				print(f"scored {subject} seed{seed:02d} {method} NLL={payload['NLL']:.4f}", flush=True)


def _require_model_files(core_root: Path, shell_root: Path, subjects: list[str], seeds: list[int], methods: list[str]) -> None:
	missing: list[Path] = []
	for subject in subjects:
		for seed in seeds:
			core = _core_dir(core_root, subject, seed)
			for name in ("core.pt", "observation.pt"):
				path = core / name
				if not path.is_file():
					missing.append(path)
			for method in methods:
				bundle = _shell_dir(shell_root, subject, seed, method) / "shell_bundle.pt"
				if not bundle.is_file():
					missing.append(bundle)
	if missing:
		preview = "\n".join(str(path) for path in missing[:8])
		raise FileNotFoundError(f"Missing {len(missing)} EEG model files, including:\n{preview}")


def _rescore_checkpoints(
	*,
	paths: EEGRunPaths,
	config: dict[str, Any],
	subjects: list[str],
	seeds: list[int],
	methods: list[str],
	device: str,
) -> None:
	_require_model_files(paths.core_source_root, paths.shell_source_root, subjects, seeds, methods)
	for subject in subjects:
		scored = _recording_for_scoring(subject, config)
		for seed in seeds:
			_score_source_core(
				_core_dir(paths.core_source_root, subject, seed),
				_core_dir(paths.core_root, subject, seed),
				scored,
				config=config,
				device=device,
			)
			for method in methods:
				source = _shell_dir(paths.shell_source_root, subject, seed, method)
				eval_path = _shell_dir(paths.shell_root, subject, seed, method) / "eval.json"
				shell = ShallowPLRNNShell.load(source / "shell_bundle.pt", method_key=method, device=device).eval()
				payload = score_eeg_shell(
					shell,
					scored,
					subject=subject,
					seed=seed,
					method=method,
					config=config,
				)
				write_shell_eval(eval_path, payload)
				print(f"rescored {subject} seed{seed:02d} {method} NLL={payload['NLL']:.4f}", flush=True)


def run_eeg_main(
	*,
	paths: EEGRunPaths,
	config: dict[str, Any],
	mode: RunMode,
	device: str,
	cpu_threads: int,
	smoke: bool = False,
) -> dict[str, Any]:
	subjects = [str(subject) for subject in config["subjects"]]
	if subjects != list(SUBJECTS):
		raise ValueError(f"EEG subjects must be {SUBJECTS}")
	seeds = [int(seed) for seed in config["seeds"]]
	methods = [str(method) for method in config["methods"]]
	if list(methods) != list(EEG_METHODS):
		raise ValueError(f"EEG methods must be {EEG_METHODS}")
	if smoke:
		seeds = seeds[:1]
		subjects = subjects[:1]
	run_config = _prepare_run_config(config, smoke=smoke)
	rollout = (
		int(run_config["evaluation"]["rollout_burn_steps"]),
		int(run_config["evaluation"]["rollout_score_steps"]),
	)
	if mode is RunMode.USE_CHECKPOINTS:
		_set_cpu_threads(cpu_threads)
		_rescore_checkpoints(
			paths=paths,
			config=run_config,
			subjects=subjects,
			seeds=seeds,
			methods=methods,
			device=device,
		)
		core_root, shell_root = paths.core_root, paths.shell_root
	elif mode in (RunMode.TRAIN_SHELLS, RunMode.FROM_SCRATCH):
		_set_cpu_threads(cpu_threads)
		if mode is RunMode.FROM_SCRATCH:
			_train_cores(
				paths=paths,
				config=run_config,
				subjects=subjects,
				seeds=seeds,
				device=device,
				smoke=smoke,
			)
			core_root = paths.core_root
		else:
			for subject in subjects:
				scored = _recording_for_scoring(subject, run_config)
				for seed in seeds:
					_score_source_core(
						_core_dir(paths.core_source_root, subject, seed),
						_core_dir(paths.core_root, subject, seed),
						scored,
						config=run_config,
						device=device,
					)
			core_root = paths.core_source_root
		_train_and_score_shells(
			core_root=core_root,
			shell_root=paths.shell_root,
			config=run_config,
			subjects=subjects,
			seeds=seeds,
			methods=methods,
			device=device,
			smoke=smoke,
		)
		shell_root = paths.shell_root
		if mode is RunMode.TRAIN_SHELLS:
			core_root = paths.core_root
	else:
		raise ValueError(f"Unknown EEG mode: {mode}")
	aggregated = aggregate_eeg_results(
		core_root=core_root,
		shell_root=shell_root,
		subjects=subjects,
		seeds=seeds,
		methods=methods,
		expected_rollout=rollout,
		diagnostic=bool(smoke),
	)
	meta = {
		"name": "eeg_main",
		"mode": mode.value,
		"subjects": subjects,
		"seeds": seeds,
		"methods": methods,
		"rows": aggregated["summary_rows"],
	}
	write_eeg_tables(paths.output_root, aggregated, meta=meta)
	aggregated["summary_output"] = str(paths.output_root / "tables" / "eeg_by_recording.csv")
	return aggregated
