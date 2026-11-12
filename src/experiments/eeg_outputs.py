"""Export EEG table results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from data.eeg import SUBJECTS
from metrics import robust_se_of_median

EEG_METHODS = (
	"frozen_core_shell",
	"open_loop_no_propagation",
	"open_loop_tied",
	"filtered_no_propagation",
	"kaffee",
)
EEG_LABELS = {
	"gtf_core": "shPLRNN+GTF (no shell)",
	"frozen_core_shell": "Frozen-core shell",
	"open_loop_no_propagation": "Open-loop, no propagation",
	"open_loop_tied": "Open-loop, tied propagation",
	"filtered_no_propagation": "Filtered, no propagation",
	"kaffee": "KAFFEE",
}
SHELL_FIELDS = ("NLL", "CRPS", "D_stsp_stoch", "D_H_stoch", "D_stsp_core", "D_H_core")


def _median_se(values: list[float], *, allow_single: bool = False) -> tuple[float, float | str]:
	array = np.asarray(values, dtype=np.float64)
	if array.size == 0 or not np.all(np.isfinite(array)):
		raise ValueError("EEG aggregation requires a complete finite seed grid")
	if allow_single and array.size == 1:
		return float(array[0]), ""
	se = robust_se_of_median(array.tolist())
	if se is None:
		raise ValueError("EEG aggregation needs at least two seeds")
	return float(np.median(array)), float(se)


def _check_record(
	payload: dict[str, Any],
	*,
	subject: str,
	seed: int,
	method: str,
	expected_rollout: tuple[int, int] = (2410, 9640),
) -> None:
	if payload.get("scoring_update") != "kalman" or int(payload.get("nll_horizon_steps", -1)) != 1:
		raise ValueError(f"EEG score is not the shared one-step filter: {subject} seed {seed} {method}")
	got = (int(payload.get("rollout_burn_steps", -1)), int(payload.get("rollout_score_steps", -1)))
	if got != tuple(expected_rollout):
		raise ValueError(f"EEG rollout window is not {expected_rollout[0]}/{expected_rollout[1]}: {subject} seed {seed} {method}")
	device = str(payload.get("core_rollout_device", ""))
	if device != "cpu" and not device.startswith("cuda"):
		raise ValueError(f"EEG noise-free rollout device is missing: {subject} seed {seed} {method}")
	if payload.get("subject") != subject or int(payload.get("seed", -1)) != int(seed) or payload.get("method") != method:
		raise ValueError(f"EEG eval identity mismatch: {subject} seed {seed} {method}")


def load_shell_eval(
	path: Path,
	*,
	subject: str,
	seed: int,
	method: str,
	expected_rollout: tuple[int, int] = (2410, 9640),
) -> dict[str, Any]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	_check_record(payload, subject=subject, seed=seed, method=method, expected_rollout=expected_rollout)
	return payload


def aggregate_eeg_results(
	*,
	core_root: Path,
	shell_root: Path,
	subjects: list[str] | None = None,
	seeds: list[int] | None = None,
	methods: list[str] | None = None,
	expected_rollout: tuple[int, int] = (2410, 9640),
	diagnostic: bool = False,
) -> dict[str, Any]:
	subjects = list(subjects or SUBJECTS)
	seeds = list(range(20) if seeds is None else seeds)
	methods = list(methods or EEG_METHODS)
	per_seed: list[dict[str, Any]] = []
	by_recording: list[dict[str, Any]] = []
	for subject in subjects:
		grouped: dict[str, dict[str, list[float]]] = {method: {field: [] for field in SHELL_FIELDS} for method in methods}
		core_dh: list[float] = []
		core_dstsp: list[float] = []
		for seed in seeds:
			core_path = core_root / subject / f"seed{seed:02d}" / "gtf_table_metrics_gpu.json"
			core = json.loads(core_path.read_text(encoding="utf-8"))
			device = str(core.get("device", ""))
			if device != "cpu" and not device.startswith("cuda"):
				raise ValueError(f"GTF cache has no device: {core_path}")
			core_dh.append(float(core["D_H"]))
			core_dstsp.append(float(core["D_stsp"]))
			per_seed.append({
				"subject": subject,
				"seed": int(seed),
				"method": "gtf_core",
				"NLL": "",
				"CRPS": "",
				"D_stsp_stoch": "",
				"D_H_stoch": "",
				"D_stsp_core": float(core["D_stsp"]),
				"D_H_core": float(core["D_H"]),
			})
			for method in methods:
				payload = load_shell_eval(
					shell_root / subject / f"seed{seed:02d}" / method / "eval.json",
					subject=subject,
					seed=seed,
					method=method,
					expected_rollout=expected_rollout,
				)
				row: dict[str, Any] = {"subject": subject, "seed": int(seed), "method": method}
				for field in SHELL_FIELDS:
					if field in ("D_stsp_stoch", "D_H_stoch"):
						repeats = payload[f"{field}_repeats"]
						if len(repeats) < 1:
							raise ValueError(f"Missing {field} rollout for {subject} seed {seed} {method}")
						value = float(repeats[0])
					else:
						value = float(payload[field])
					row[field] = value
					grouped[method][field].append(value)
				per_seed.append(row)
		dh_median, dh_se = _median_se(core_dh, allow_single=diagnostic)
		dstsp_median, dstsp_se = _median_se(core_dstsp, allow_single=diagnostic)
		by_recording.append({
			"subject": subject,
			"method": "gtf_core",
			"label": EEG_LABELS["gtf_core"],
			"n": len(seeds),
			"NLL_median": "",
			"NLL_se": "",
			"CRPS_median": "",
			"CRPS_se": "",
			"D_stsp_stoch_median": "",
			"D_stsp_stoch_se": "",
			"D_H_stoch_median": "",
			"D_H_stoch_se": "",
			"D_stsp_core_median": dstsp_median,
			"D_stsp_core_se": dstsp_se,
			"D_H_core_median": dh_median,
			"D_H_core_se": dh_se,
		})
		for method in methods:
			summary: dict[str, Any] = {
				"subject": subject,
				"method": method,
				"label": EEG_LABELS[method],
				"n": len(seeds),
			}
			for field in SHELL_FIELDS:
				median, se = _median_se(grouped[method][field], allow_single=diagnostic)
				summary[f"{field}_median"] = median
				summary[f"{field}_se"] = se
			by_recording.append(summary)
	return {"per_seed_rows": per_seed, "summary_rows": by_recording}


def write_eeg_tables(output_root: Path, aggregated: dict[str, Any], *, meta: dict[str, Any]) -> None:
	tables = output_root / "tables"
	tables.mkdir(parents=True, exist_ok=True)
	for name, rows in (("eeg_by_recording.csv", aggregated["summary_rows"]), ("eeg_per_seed.csv", aggregated["per_seed_rows"])):
		path = tables / name
		with path.open("w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
			writer.writeheader()
			writer.writerows(rows)
	(tables / "eeg_main.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
