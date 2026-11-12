"""Helpers for the Lorenz-96 main ablation experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from data import (
	load_lorenz96_dataset,
	load_lorenz96_heldout_test,
	load_lorenz96_stochastic_evaluation,
)
from training import run_alrnn_probabilistic_ablation

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: str | Path | None) -> Path | None:
	if path is None:
		return None
	resolved = Path(path)
	if resolved.is_absolute():
		return resolved
	return _REPO_ROOT / resolved


def validate_lorenz96_main_config(config: dict[str, Any]) -> None:
	system = dict(config["system"])
	probabilistic = dict(config["probabilistic_training"])
	evaluation = dict(config["evaluation"])
	if str(system.get("type")) != "lorenz96":
		raise ValueError("lorenz96_main config must use system.type='lorenz96'")
	if int(system.get("state_dim")) != 20:
		raise ValueError("lorenz96_main config must use state_dim=20")
	if str(dict(config.get("training", {})).get("optimizer", "")).strip().lower() not in {"adam", "radam"}:
		raise ValueError("lorenz96_main training.optimizer must be 'Adam' or 'RAdam'")
	if int(probabilistic.get("iterations")) != 150:
		raise ValueError("lorenz96_main config must use 150 shell iterations")
	if str(probabilistic.get("scheduler", "")).lower() != "cosine":
		raise ValueError("lorenz96_main config must use cosine shell scheduler")
	filter_name = str(probabilistic.get("filter", "ekf")).strip().lower()
	if filter_name == "rrkf":
		if str(config.get("name")) == "lorenz96_main":
			raise ValueError("lorenz96_main cannot use rrkf; use configs/lorenz96_rrkf.json")
		if int(probabilistic.get("filter_rank", 0)) < 1:
			raise ValueError("probabilistic_training.filter_rank must be >= 1 for rrkf")
	elif filter_name != "ekf":
		raise ValueError(f"Unsupported probabilistic_training.filter: {filter_name!r}")
	if int(evaluation.get("state_space_rollout_steps", 50_000)) != 50_000:
		raise ValueError("lorenz96_main config must use stochastic state_space_rollout_steps=50000")
	if int(evaluation.get("lyapunov_burn_steps", 1000)) != 1000:
		raise ValueError("lorenz96_main config must use lyapunov_burn_steps=1000")
	if int(evaluation.get("lyapunov_rollout_steps", 30_000)) != 30_000:
		raise ValueError("lorenz96_main config must use lyapunov_rollout_steps=30000")
	horizon_steps = [int(h) for h in evaluation.get("spectrum_horizons_steps", [])]
	horizon_tau = [float(t) for t in evaluation.get("spectrum_horizons_tau_lambda", [])]
	if not horizon_steps or len(horizon_steps) != len(horizon_tau):
		raise ValueError("lorenz96_main evaluation.spectrum_horizons_steps/tau_lambda must be matched non-empty lists")
	if any(h < 1 for h in horizon_steps):
		raise ValueError("lorenz96_main evaluation.spectrum_horizons_steps must be positive")
	main_h = int(evaluation.get("main_forecast_horizon_steps", 0))
	if main_h not in horizon_steps:
		raise ValueError("lorenz96_main main_forecast_horizon_steps must appear in spectrum_horizons_steps")

	required_paths = [
		_resolve_repo_path(config["dataset"].get("path")),
		_resolve_repo_path(config["dataset"].get("info_path")),
		_resolve_repo_path(evaluation.get("reference_path")),
		_resolve_repo_path(evaluation.get("heldout_test_path")),
		_resolve_repo_path(evaluation.get("heldout_test_info_path")),
		_resolve_repo_path(evaluation.get("stochastic_evaluation_path")),
		_resolve_repo_path(evaluation.get("stochastic_evaluation_info_path")),
	]
	for path in required_paths:
		if path is None or not path.is_file():
			raise FileNotFoundError(f"Missing Lorenz-96 path: {path}")


def _seed_list(config: dict[str, Any]) -> list[int]:
	sweep = dict(config.get("seed_sweep", {}))
	start = int(sweep.get("start_seed", 0))
	count = int(sweep.get("count", 20))
	return list(range(start, start + count))


def _validate_evaluation_metadata(
	config: dict[str, Any],
	metadata: dict[str, Any],
	*,
	label: str,
	minimum_length: int,
) -> None:
	expected = {
		"dt": float(config["system"]["dt"]),
		"forcing": float(config["system"]["forcing"]),
		"state_dim": int(config["system"]["state_dim"]),
		"sigma_proc": float(config["system"]["sigma_proc"]),
		"sigma_proc_space": str(config["system"]["sigma_proc_space"]),
		"sigma_obs": float(config["system"]["sigma_obs"]),
	}
	for key, value in expected.items():
		if metadata.get(key) != value:
			raise ValueError(
				f"Lorenz-96 {label} metadata {key!r}={metadata.get(key)!r}, expected {value!r}"
			)
	if int(metadata.get("length", 0)) < int(minimum_length):
		raise ValueError(
			f"Lorenz-96 {label} has {metadata.get('length')} states, need at least {minimum_length}"
		)


def prepare_lorenz96_main_bundle(config: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
	validate_lorenz96_main_config(config)
	active = deepcopy(config)
	if bool(smoke):
		active["system"]["T_train"] = min(int(active["system"]["T_train"]), 2048)
		active["system"]["T_test"] = min(int(active["system"]["T_test"]), 1024)
		active["probabilistic_training"]["iterations"] = min(int(active["probabilistic_training"]["iterations"]), 5)

	dataset_cfg = dict(active.get("dataset", {}))
	dataset_path = _resolve_repo_path(dataset_cfg.get("path"))
	info_path = _resolve_repo_path(dataset_cfg.get("info_path"))
	if info_path is None:
		info_path = dataset_path.parent / "data_info.json"

	bundle = load_lorenz96_dataset(str(dataset_path), info_path=str(info_path))
	evaluation = dict(active["evaluation"])
	heldout_path = _resolve_repo_path(evaluation["heldout_test_path"])
	heldout_info_path = _resolve_repo_path(evaluation["heldout_test_info_path"])
	stochastic_path = _resolve_repo_path(evaluation["stochastic_evaluation_path"])
	stochastic_info_path = _resolve_repo_path(evaluation["stochastic_evaluation_info_path"])
	heldout_test = load_lorenz96_heldout_test(
		heldout_path,
		metadata_path=heldout_info_path,
		standardizer=dict(bundle["standardizer"]),
	)
	stochastic_evaluation = load_lorenz96_stochastic_evaluation(
		stochastic_path,
		metadata_path=stochastic_info_path,
		standardizer=dict(bundle["standardizer"]),
	)
	_validate_evaluation_metadata(
		config,
		dict(heldout_test["metadata"]),
		label="held-out test",
		minimum_length=int(config["system"]["T_test"]),
	)
	if int(heldout_test["metadata"]["length"]) != int(config["system"]["T_test"]):
		raise ValueError("Lorenz-96 held-out test length must match system.T_test")
	stochastic_minimum_length = max(
		int(evaluation["stochastic_start_index"]) + int(evaluation["state_space_rollout_steps"]),
		int(evaluation["stochastic_start_index"])
		+ int(evaluation["lyapunov_burn_steps"])
		+ int(evaluation["lyapunov_rollout_steps"])
		+ 1,
	)
	_validate_evaluation_metadata(
		config,
		dict(stochastic_evaluation["metadata"]),
		label="stochastic evaluation",
		minimum_length=stochastic_minimum_length,
	)

	protocol = {
		"name": active["name"],
		"system": dict(active["system"]),
		"model": dict(active["model"]),
		"training": dict(active["training"]),
		"seed_sweep": {
			**dict(active["seed_sweep"]),
			"seed_list": _seed_list(active),
		},
		"probabilistic_training": dict(active["probabilistic_training"]),
		"evaluation": dict(active["evaluation"]),
		"dataset_shapes": {
			"train_obs": list(bundle["train_obs"].shape),
			"test_obs": list(bundle["test_obs"].shape),
			"train_clean": list(bundle["train_clean"].shape),
			"test_clean": list(bundle["test_clean"].shape),
		},
		"dataset_source": {
			"path": str(dataset_path),
			"info_path": str(info_path),
			"heldout_test_path": str(heldout_path),
			"heldout_test_info_path": str(heldout_info_path),
			"stochastic_evaluation_path": str(stochastic_path),
			"stochastic_evaluation_info_path": str(stochastic_info_path),
		},
		"smoke": bool(smoke),
	}
	return {
		"dataset": bundle,
		"heldout_test": heldout_test,
		"stochastic_evaluation": stochastic_evaluation,
		"protocol": protocol,
	}


def train_lorenz96_core(*args: Any, **kwargs: Any):
	from training import train_alrnn_prior

	return train_alrnn_prior(*args, **kwargs)


__all__ = [
	"prepare_lorenz96_main_bundle",
	"run_alrnn_probabilistic_ablation",
	"train_lorenz96_core",
	"validate_lorenz96_main_config",
]
