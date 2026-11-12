"""DynaMix results export helpers."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _path_for_meta(path: Path, repo_root: Path = _REPO_ROOT) -> str:
	resolved = Path(path).resolve()
	try:
		return str(resolved.relative_to(Path(repo_root).resolve()))
	except ValueError:
		return resolved.name


DYNAMIX_SUMMARY_METRIC_FIELDS: tuple[str, ...] = (
	"NLL",
	"RMSE_10",
	"D_stsp_core_ratio",
	"D_H_core",
)
DYNAMIX_AUDIT_FIELDS: tuple[str, ...] = (
	"training_seconds",
	"cuda_peak_allocated_bytes",
)

SUMMARY_FIELD_MAP: tuple[tuple[str, str], ...] = (
	("NLL_median", "NLL (0.25 τ_λ)"),
	("RMSE_10_median", "RMSE_10"),
	("D_stsp_core_ratio_median", "D_stsp_core/D_stsp_base"),
	("D_H_core_median", "D_H^core"),
)

DYNAMIX_BASELINE_KEYS: tuple[str, ...] = (
	"dynamix_base",
	"frozen_core_shell",
	"open_loop_no_propagation",
	"open_loop_tied",
	"filtered_no_propagation",
	"kaffee_dynamix",
)

BASELINE_LABELS: dict[str, str] = {
	"dynamix_base": "DynaMix base (no adaptation)",
	"frozen_core_shell": "Frozen-Core Shell Calibration",
	"open_loop_no_propagation": "Open-loop, no propagation",
	"open_loop_tied": "Open-loop, tied propagation",
	"filtered_no_propagation": "Filtered, no propagation",
	"kaffee_dynamix": "Filtered, tied propagation (KAFFEE-DynaMix)",
}

# dynamix_base does not score NLL/RMSE
_MATCHED_EXCLUSION_SKIP_METHODS: frozenset[str] = frozenset({"dynamix_base"})


def _finite_or_none(value: Any) -> float | None:
	try:
		out = float(value)
	except (TypeError, ValueError):
		return None
	return out if np.isfinite(out) else None


def _adapted_paper_metrics_are_finite(row: dict[str, Any]) -> bool:
	if str(row.get("method_key")) in _MATCHED_EXCLUSION_SKIP_METHODS:
		return True
	return all(_finite_or_none(row.get(field)) is not None for field in DYNAMIX_SUMMARY_METRIC_FIELDS)


def matched_nonfinite_seed_exclusions(
	per_seed_rows: list[dict[str, Any]],
) -> dict[str, frozenset[int]]:
	"""Drop a (system, seed) for every method if any adapted method is non-finite there.

	Archive JSON files stay on disk. ``dynamix_base`` NLL/RMSE are not scored
	and do not trigger a drop. Do not nanmean Inf windows.
	"""
	tainted: dict[str, set[int]] = defaultdict(set)
	for row in per_seed_rows:
		if _adapted_paper_metrics_are_finite(row):
			continue
		tainted[str(row["system"])].add(int(row["seed"]))
	return {system: frozenset(seeds) for system, seeds in tainted.items() if seeds}


def seed_is_excluded(
	system: str,
	seed: int,
	exclusions: dict[str, frozenset[int]] | None,
) -> bool:
	if not exclusions:
		return False
	return int(seed) in exclusions.get(str(system), frozenset())


def expected_reporting_n_jobs(
	systems: list[str],
	seeds: list[int],
	exclusions: dict[str, frozenset[int]] | None,
) -> int:
	return sum(
		1
		for system in systems
		for seed in seeds
		if not seed_is_excluded(str(system), int(seed), exclusions)
	)


def exclusions_as_lists(exclusions: dict[str, frozenset[int]] | None) -> dict[str, list[int]]:
	if not exclusions:
		return {}
	return {str(system): sorted(int(s) for s in seeds) for system, seeds in exclusions.items()}


def _drop_excluded_seeds(
	aggregated: dict[str, Any],
	exclusions: dict[str, frozenset[int]] | None,
) -> dict[str, Any]:
	if not exclusions:
		return aggregated
	per_seed = [
		row
		for row in aggregated["per_seed_rows"]
		if not seed_is_excluded(str(row["system"]), int(row["seed"]), exclusions)
	]
	per_system: list[dict[str, Any]] = []
	summary_rows: list[dict[str, Any]] = []
	by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in per_seed:
		by_method[str(row["method_key"])].append(row)
	order = [str(row["method_key"]) for row in aggregated["summary_rows"]]
	seen: set[str] = set()
	for baseline_key in order:
		if baseline_key in seen:
			continue
		seen.add(baseline_key)
		method_rows = by_method.get(baseline_key, [])
		method_system_rows = _system_median_rows(method_rows, baseline_key)
		per_system.extend(method_system_rows)
		summary_rows.append(_summarize_rows(method_rows, method_system_rows, baseline_key))
	return {
		**aggregated,
		"per_seed_rows": per_seed,
		"per_system_rows": per_system,
		"summary_rows": summary_rows,
	}


def _rmse10(block: dict[str, Any]) -> float | None:
	mh = block.get("multihorizon") or {}
	entry = mh.get("10") or mh.get(10)
	if not isinstance(entry, dict):
		return None
	rmse = entry.get("rmse")
	if isinstance(rmse, dict):
		return _finite_or_none(rmse.get("mean"))
	return _finite_or_none(rmse)


def robust_se_median(values: Iterable[float]) -> float | None:
	arr = np.asarray(list(values), dtype=np.float64)
	finite = arr[np.isfinite(arr)]
	if finite.size < 2:
		return None
	q25, q75 = np.percentile(finite, [25.0, 75.0])
	return float(0.929 * (q75 - q25) / math.sqrt(finite.size))


def _system_median_rows(rows: list[dict[str, Any]], baseline_key: str) -> list[dict[str, Any]]:
	grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in rows:
		grouped[str(row["system"])].append(row)

	per_system: list[dict[str, Any]] = []
	for system, system_rows in sorted(grouped.items()):
		out: dict[str, Any] = {
			"method_key": baseline_key,
			"system": system,
			"n_jobs": len(system_rows),
			"n_seeds": len({int(row["seed"]) for row in system_rows}),
		}
		for field in (*DYNAMIX_SUMMARY_METRIC_FIELDS, *DYNAMIX_AUDIT_FIELDS):
			finite = [
				float(value)
				for value in (_finite_or_none(row.get(field)) for row in system_rows)
				if value is not None
			]
			out[f"{field}_n_finite_seeds"] = len(finite)
			is_complete = len(finite) == len(system_rows)
			out[field] = (
				float(np.median(finite))
				if finite and (field in DYNAMIX_AUDIT_FIELDS or is_complete)
				else None
			)
		per_system.append(out)
	return per_system


def _summarize_rows(
	rows: list[dict[str, Any]],
	per_system_rows: list[dict[str, Any]],
	baseline_key: str,
) -> dict[str, Any]:
	out: dict[str, Any] = {
		"method_key": baseline_key,
		"n_jobs": len(rows),
		"n_seeds": len({int(r["seed"]) for r in rows}) if rows else 0,
		"n_systems": len(per_system_rows),
	}
	for field in (*DYNAMIX_SUMMARY_METRIC_FIELDS, *DYNAMIX_AUDIT_FIELDS):
		vals = [_finite_or_none(row.get(field)) for row in per_system_rows]
		finite = [float(v) for v in vals if v is not None]
		out[f"{field}_median"] = float(np.median(finite)) if finite else None
		out[f"{field}_n_systems"] = len(finite)
		se = robust_se_median(finite)
		if se is not None:
			out[f"{field}_robust_se"] = se
	return out


def load_results_from_checkpoints(
	data_root: Path,
	*,
	systems: list[str],
	seed_filter: set[int] | None,
	baseline_keys: list[str],
) -> dict[str, Any]:
	per_seed_rows: list[dict[str, Any]] = []
	per_system_rows: list[dict[str, Any]] = []
	summary_rows: list[dict[str, Any]] = []
	result_root = Path(data_root) / "results"
	for baseline_key in baseline_keys:
		rows: list[dict[str, Any]] = []
		for fp in sorted((result_root / baseline_key).glob(f"result_{baseline_key}_*_seed*.json")):
			payload = json.loads(fp.read_text(encoding="utf-8"))
			system = str(payload.get("system"))
			if system not in systems:
				continue
			seed = int(payload.get("seed"))
			if seed_filter is not None and seed not in seed_filter:
				continue
			block = payload.get(baseline_key) or {}
			base = payload.get("baseline") or {}
			base_dstsp = _finite_or_none(base.get("D_stsp"))
			dstsp = _finite_or_none(block.get("D_stsp"))
			ratio = None
			if base_dstsp is not None and abs(base_dstsp) > 1.0e-12 and dstsp is not None:
				ratio = dstsp / abs(base_dstsp)
			row = {
				"method_key": baseline_key,
				"system": system,
				"seed": seed,
				"source_path": _path_for_meta(fp),
				"NLL": _finite_or_none(block.get("held_nll")),
				"RMSE_10": _rmse10(block),
				"D_stsp_core_ratio": ratio,
				"D_H_core": _finite_or_none(block.get("D_H")),
				"training_seconds": _finite_or_none((block.get("timing") or {}).get("wall_seconds")),
				"cuda_peak_allocated_bytes": _finite_or_none(
					(block.get("timing") or {}).get("cuda_peak_allocated_bytes")
				),
			}
			if baseline_key == "dynamix_base":
				row["D_stsp_core_ratio"] = 1.0
			rows.append(row)
			per_seed_rows.append(row)
		method_system_rows = _system_median_rows(rows, baseline_key)
		per_system_rows.extend(method_system_rows)
		summary_rows.append(_summarize_rows(rows, method_system_rows, baseline_key))
	return {
		"bundle_root": _path_for_meta(Path(data_root)),
		"per_seed_rows": per_seed_rows,
		"per_system_rows": per_system_rows,
		"summary_rows": summary_rows,
	}


def aggregate_dynamix_main_results(
	config: dict[str, Any],
	data_root: Path,
	*,
	systems: list[str],
	seeds: list[int] | None,
) -> dict[str, Any]:
	"""Aggregate seeds within systems, then summarize across systems."""
	seed_filter = set(int(s) for s in seeds) if seeds else None
	result_root = Path(data_root) / "results"
	if not result_root.is_dir():
		raise FileNotFoundError(
			f"Missing DynaMix result JSONs under {result_root}. "
			"Run scripts/run_dynamix_full.py --mode adapt first."
		)
	aggregated = load_results_from_checkpoints(
		Path(data_root),
		systems=systems,
		seed_filter=seed_filter,
		baseline_keys=[str(k) for k in config.get("baselines", DYNAMIX_BASELINE_KEYS)],
	)
	if seeds is not None:
		expected = {
			(str(method), str(system), int(seed))
			for method in config.get("baselines", DYNAMIX_BASELINE_KEYS)
			for system in systems
			for seed in seeds
		}
		actual_rows = [
			(str(row["method_key"]), str(row["system"]), int(row["seed"]))
			for row in aggregated["per_seed_rows"]
		]
		actual = set(actual_rows)
		if len(actual) != len(actual_rows) or actual != expected:
			raise RuntimeError(
				"Incomplete DynaMix result grid: "
				f"expected {len(expected)} unique jobs, found {len(actual)}"
			)
	exclusions = matched_nonfinite_seed_exclusions(aggregated["per_seed_rows"])
	dropped = _drop_excluded_seeds(aggregated, exclusions)
	dropped["seed_exclusions"] = exclusions_as_lists(exclusions)
	return dropped


__all__ = [
	"BASELINE_LABELS",
	"DYNAMIX_BASELINE_KEYS",
	"DYNAMIX_AUDIT_FIELDS",
	"DYNAMIX_SUMMARY_METRIC_FIELDS",
	"SUMMARY_FIELD_MAP",
	"aggregate_dynamix_main_results",
	"exclusions_as_lists",
	"expected_reporting_n_jobs",
	"load_results_from_checkpoints",
	"matched_nonfinite_seed_exclusions",
	"robust_se_median",
	"seed_is_excluded",
]
