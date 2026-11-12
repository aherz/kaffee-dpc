"""Helpers for the Lorenz-63 main ablation experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from data.lorenz63 import load_lorenz63_dataset, load_lorenz63_evaluation

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: str | Path | None) -> Path | None:
	if path is None:
		return None
	resolved = Path(path)
	if resolved.is_absolute():
		return resolved
	return _REPO_ROOT / resolved


def _seed_list(config: dict[str, Any]) -> list[int]:
	sweep = dict(config.get("seed_sweep", {}))
	start = int(sweep.get("start_seed", 0))
	count = int(sweep.get("count", 20))
	return list(range(start, start + count))


def dataset_path(config: dict[str, Any]) -> Path:
	path = _resolve_repo_path(dict(config.get("dataset", {})).get("path"))
	if path is None:
		raise ValueError("lorenz63_main dataset.path must be set")
	return path


def _evaluation_file(config: dict[str, Any], key: str) -> Path:
	path = _resolve_repo_path(dict(config.get("evaluation", {})).get(key))
	if path is None:
		raise ValueError(f"lorenz63_main evaluation.{key} must be set")
	return path


def validate_lorenz63_main_config(config: dict[str, Any]) -> None:
	system = dict(config["system"])
	model = dict(config["model"])
	training = dict(config.get("training", {}))
	probabilistic = dict(config["probabilistic_training"])
	evaluation = dict(config["evaluation"])
	if str(system.get("type")) != "lorenz63":
		raise ValueError("lorenz63_main config must use system.type='lorenz63'")
	if abs(float(system.get("dt", 0.0)) - 0.01) > 1e-12:
		raise ValueError("lorenz63_main config must use dt=0.01")
	if int(system.get("T_test", -1)) != 20_000:
		raise ValueError("lorenz63_main config must use T_test=20000")
	if int(model.get("latent_dim")) != 30:
		raise ValueError("lorenz63_main config must use latent_dim=30")
	if int(model.get("gated_dim")) != 10:
		raise ValueError("lorenz63_main config must use gated_dim=10")
	if int(model.get("observed_dim")) != 3:
		raise ValueError("lorenz63_main config must use observed_dim=3")
	if int(model.get("teacher_forcing_interval")) != 16:
		raise ValueError("lorenz63_main config must use teacher_forcing_interval=16")
	if str(training.get("optimizer", "")).strip().lower() not in {"adam", "radam"}:
		raise ValueError("lorenz63_main training.optimizer must be 'Adam' or 'RAdam'")
	if int(training.get("sequence_length")) != 200:
		raise ValueError("lorenz63_main config must use sequence_length=200")
	if int(training.get("batch_size")) != 16:
		raise ValueError("lorenz63_main config must use batch_size=16")
	if int(training.get("batches_per_epoch")) != 50:
		raise ValueError("lorenz63_main config must use batches_per_epoch=50")
	if int(training.get("epochs")) != 2000:
		raise ValueError("lorenz63_main config must use epochs=2000")
	if abs(float(training.get("lr_start")) - 0.001) > 1e-12:
		raise ValueError("lorenz63_main config must use lr_start=0.001")
	if abs(float(training.get("lr_end")) - 1e-5) > 1e-12:
		raise ValueError("lorenz63_main config must use lr_end=1e-5")
	if abs(float(training.get("alpha", 1.0)) - 1.0) > 1e-12:
		raise ValueError("lorenz63_main identity teacher forcing requires training.alpha=1.0")
	if int(probabilistic.get("iterations")) != 150:
		raise ValueError("lorenz63_main config must use 150 shell iterations")
	if str(probabilistic.get("scheduler", "")).lower() != "cosine":
		raise ValueError("lorenz63_main config must use cosine shell scheduler")
	if abs(float(probabilistic.get("lr", 0.0)) - 0.001) > 1e-12:
		raise ValueError("lorenz63_main config must use shell lr=0.001")
	if int(evaluation.get("main_forecast_horizon_steps", -1)) != 27:
		raise ValueError("lorenz63_main config must use main_forecast_horizon_steps=27")
	if int(evaluation.get("predictive_nll_starts", -1)) != 100:
		raise ValueError("lorenz63_main config must use predictive_nll_starts=100")
	if int(evaluation.get("predictive_forecast_starts", -1)) != 10:
		raise ValueError("lorenz63_main config must use predictive_forecast_starts=10")
	if int(evaluation.get("predictive_forecast_ensemble", -1)) != 200:
		raise ValueError("lorenz63_main config must use predictive_forecast_ensemble=200")
	if int(evaluation.get("state_space_rollout_steps", -1)) != 50_000:
		raise ValueError("lorenz63_main config must use state_space_rollout_steps=50000")
	if int(evaluation.get("lyapunov_burn_steps", -1)) != 1000:
		raise ValueError("lorenz63_main config must use lyapunov_burn_steps=1000")
	if int(evaluation.get("lyapunov_rollout_steps", -1)) != 30_000:
		raise ValueError("lorenz63_main config must use lyapunov_rollout_steps=30000")
	dataset = dataset_path(config)
	if not dataset.is_file():
		raise FileNotFoundError(f"Missing Lorenz-63 training dataset: {dataset}")
	for key in ("heldout_test_path", "stochastic_evaluation_path", "reference_path"):
		path = _evaluation_file(config, key)
		if not path.is_file():
			raise FileNotFoundError(f"Missing Lorenz-63 {key}: {path}")


def prepare_lorenz63_main_bundle(config: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
	validate_lorenz63_main_config(config)
	active = deepcopy(config)
	if bool(smoke):
		active["system"]["T_train"] = min(int(active["system"]["T_train"]), 2048)
		active["system"]["T_test"] = min(int(active["system"]["T_test"]), 1024)
		active["probabilistic_training"]["iterations"] = min(
			int(active["probabilistic_training"]["iterations"]), 5
		)

	path = dataset_path(active)
	bundle = load_lorenz63_dataset(path)
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
		"dataset_source": {"path": str(path)},
		"smoke": bool(smoke),
	}
	return {"dataset": bundle, "protocol": protocol}


def load_lorenz63_eval_bundle(config: dict[str, Any]) -> dict[str, Any]:
	validate_lorenz63_main_config(config)
	return load_lorenz63_evaluation(
		_evaluation_file(config, "heldout_test_path"),
		_evaluation_file(config, "stochastic_evaluation_path"),
		_evaluation_file(config, "reference_path"),
	)


def train_lorenz63_core(*args: Any, **kwargs: Any):
	from training import train_alrnn_prior

	return train_alrnn_prior(*args, **kwargs)


__all__ = [
	"dataset_path",
	"load_lorenz63_eval_bundle",
	"prepare_lorenz63_main_bundle",
	"train_lorenz63_core",
	"validate_lorenz63_main_config",
]
