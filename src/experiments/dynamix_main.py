"""Helpers for the DynaMix multisystem adaptation experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments.dynamix_outputs import DYNAMIX_BASELINE_KEYS

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: str | Path | None) -> Path | None:
	if path is None:
		return None
	resolved = Path(path)
	if resolved.is_absolute():
		return resolved
	return _REPO_ROOT / resolved


def validate_dynamix_config(config: dict[str, Any]) -> None:
	name = str(config.get("name", ""))
	if name != "dynamix_main":
		raise ValueError("config name must be 'dynamix_main'")
	systems = list(config.get("systems", []))
	if not systems:
		raise ValueError("dynamix_main config must list at least one system")
	data_noise = dict(config["data_noise"])
	if float(data_noise.get("observation_std_standardized", -1)) <= 0:
		raise ValueError("dynamix_main data_noise.observation_std_standardized must be positive")
	adapt = dict(config["adaptation"])
	if int(adapt.get("iterations", 0)) < 1:
		raise ValueError("dynamix_main adaptation.iterations must be >= 1")
	if float(adapt.get("lr", -1)) <= 0:
		raise ValueError("dynamix_main adaptation.lr must be positive")
	if str(adapt.get("optimizer", "")).strip().lower() not in {"adam", "radam"}:
		raise ValueError("dynamix_main adaptation.optimizer must be 'Adam' or 'RAdam'")
	baselines = DYNAMIX_BASELINE_KEYS
	schedules = dict(config.get("adaptation_schedules", {}))
	if tuple(schedules.keys()) != baselines:
		raise ValueError(
			"dynamix_main adaptation_schedules must have keys exactly like the method list"
		)
	for key in baselines:
		schedule = dict(schedules[key])
		train_total_t = int(schedule.get("train_total_t", -1))
		if key == "dynamix_base":
			if train_total_t != 0:
				raise ValueError("dynamix_base schedule must have train_total_t=0")
			continue
		if train_total_t <= 0:
			raise ValueError(f"{key} schedule train_total_t must be positive")
	trainable = [k for k in baselines if k != "dynamix_base"]
	ref_t = int(schedules[trainable[0]]["train_total_t"])
	for key in trainable:
		if int(schedules[key]["train_total_t"]) != ref_t:
			raise ValueError("dynamix_main requires identical train_total_t for all trainable methods")
	rmse_horizon = int(dict(config.get("evaluation", {})).get("rmse_forecast_horizon_steps", 0))
	if rmse_horizon < 1:
		raise ValueError("dynamix_main evaluation.rmse_forecast_horizon_steps must be >= 1")
	nll_horizon_k = float(dict(config.get("evaluation", {})).get("nll_horizon_tau_lambda", 0.0))
	if nll_horizon_k <= 0.0:
		raise ValueError("dynamix_main evaluation.nll_horizon_tau_lambda must be positive")
	nll_n_starts = int(dict(config.get("evaluation", {})).get("nll_n_starts", 0))
	if nll_n_starts < 1:
		raise ValueError("dynamix_main evaluation.nll_n_starts must be >= 1")
	if config.get("seed_exclusions"):
		raise ValueError(
			"dynamix_main does not use seed_exclusions; non-finite adapted seeds "
			"are dropped for every method at aggregation time"
		)
	checkpoint_path = _resolve_repo_path(dict(config.get("checkpoint", {})).get("base_model_path"))
	if checkpoint_path is None or not checkpoint_path.is_file():
		raise FileNotFoundError(
			f"Missing DynaMix base checkpoint at checkpoint.base_model_path: {checkpoint_path}"
		)


def matched_adaptation_train_total_t(config: dict[str, Any]) -> int:
	"""Shared trajectory length used to fit core-metric mean/std, including dynamix_base."""
	schedules = dict(config["adaptation_schedules"])
	trainable = [key for key in schedules if key != "dynamix_base"]
	if not trainable:
		raise ValueError("dynamix_main needs a trainable adaptation schedule")
	return int(schedules[trainable[0]]["train_total_t"])


def _seed_list(config: dict[str, Any]) -> list[int]:
	sweep = dict(config.get("seed_sweep", {}))
	start = int(sweep.get("start_seed", 0))
	count = int(sweep.get("count", 20))
	return list(range(start, start + count))


def validate_dynamix_main_config(config: dict[str, Any]) -> None:
	if str(config.get("name")) != "dynamix_main":
		raise ValueError("config name must be 'dynamix_main'")
	validate_dynamix_config(config)


def prepare_dynamix_bundle(config: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
	validate_dynamix_config(config)
	active = deepcopy(config)
	if bool(smoke):
		active["seed_sweep"]["count"] = min(int(active["seed_sweep"]["count"]), 2)
		active["systems"] = list(active["systems"][:2])
		active["adaptation"]["iterations"] = min(int(active["adaptation"]["iterations"]), 2)
		active["adaptation"]["context_length"] = min(int(active["adaptation"]["context_length"]), 256)
		for key, schedule in active["adaptation_schedules"].items():
			if key == "dynamix_base":
				continue
			schedule["train_total_t"] = min(int(schedule["train_total_t"]), 512)
		active["evaluation"]["qoi_rollout_steps"] = min(int(active["evaluation"]["qoi_rollout_steps"]), 512)
		active["evaluation"]["qoi_burn_in_steps"] = min(int(active["evaluation"]["qoi_burn_in_steps"]), 32)
		active["evaluation"]["qoi_eval_steps"] = min(int(active["evaluation"]["qoi_eval_steps"]), 256)

	checkpoint_path = _resolve_repo_path(dict(active["checkpoint"]).get("base_model_path"))
	seed_list = _seed_list(active)

	protocol = {
		"name": active["name"],
		"systems": list(active["systems"]),
		"data_noise": dict(active["data_noise"]),
		"seed_sweep": {
			**dict(active["seed_sweep"]),
			"seed_list": seed_list,
		},
		"checkpoint": {
			**dict(active["checkpoint"]),
			"base_model_path": str(checkpoint_path) if checkpoint_path is not None else None,
		},
		"adaptation": dict(active["adaptation"]),
		"adaptation_schedules": {
			key: dict(value) for key, value in dict(active["adaptation_schedules"]).items()
		},
		"baselines": list(active.get("baselines", DYNAMIX_BASELINE_KEYS)),
		"evaluation": dict(active["evaluation"]),
		"smoke": bool(smoke),
	}
	return {"protocol": protocol}


def prepare_dynamix_main_bundle(config: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
	return prepare_dynamix_bundle(config, smoke=smoke)


__all__ = [
	"matched_adaptation_train_total_t",
	"prepare_dynamix_bundle",
	"prepare_dynamix_main_bundle",
	"validate_dynamix_config",
	"validate_dynamix_main_config",
]
