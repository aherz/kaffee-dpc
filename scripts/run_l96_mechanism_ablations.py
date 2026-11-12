#!/usr/bin/env python3
"""Train and evaluate Lorenz-96 Update-SG / GTF-alpha ablations, see ../src/experiments/l96_mechanism_ablations.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from experiments.l96_mechanism_ablations import (
	SHELL_CHECKPOINT_NAME,
	canonicalize_method_key,
	evaluate_isolated_shell,
	run_isolated_shell,
	write_ablation_tables,
)
from experiments.lorenz96_main import prepare_lorenz96_main_bundle, validate_lorenz96_main_config

CORE_CHECKPOINT_NAME = "alrnn_core.pt"
DEFAULT_RUN_ROOT = Path("models/lorenz96_mechanism_ablations")
DEFAULT_OUTPUT_ROOT = Path("results/lorenz96_mechanism_ablations")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--mode", choices=["report", "train", "eval"], default="report")
	parser.add_argument("--config", type=Path, default=Path("configs/lorenz96_main.json"))
	parser.add_argument("--eval-root", type=Path, default=DEFAULT_RUN_ROOT)
	parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
	parser.add_argument("--core-root", type=Path, default=Path("models/lorenz96_main_cores"))
	parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--method", type=str, default=None)
	parser.add_argument("--device", type=str, default="cuda:0")
	parser.add_argument("--cpu-threads", type=int, default=1)
	parser.add_argument("--smoke", action="store_true")
	return parser.parse_args()


def _resolve(path: Path) -> Path:
	return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _set_cpu_threads(n_threads: int) -> None:
	n_threads = int(max(1, n_threads))
	torch.set_num_threads(n_threads)
	try:
		torch.set_num_interop_threads(1)
	except RuntimeError:
		pass


def _resolve_device(device_arg: str) -> torch.device:
	text = str(device_arg).strip().lower()
	if text == "cpu":
		return torch.device("cpu")
	if text.startswith("cuda"):
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA requested but unavailable")
		return torch.device(text)
	if text == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	raise ValueError(f"Unsupported device: {device_arg!r}")


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
	os.environ.setdefault("PYTHONUNBUFFERED", "1")
	args = parse_args()
	if args.mode == "report":
		written = write_ablation_tables(_resolve(args.eval_root), _resolve(args.output_root))
		for key, path in written.items():
			print(f"wrote {key} {path}", flush=True)
		return 0

	if args.seed is None or args.method is None:
		raise ValueError("--seed and --method are required for train/eval")
	method_key = canonicalize_method_key(str(args.method))
	_set_cpu_threads(int(args.cpu_threads))
	device = _resolve_device(args.device)
	config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
	validate_lorenz96_main_config(config)
	prepared = prepare_lorenz96_main_bundle(config, smoke=bool(args.smoke))
	seed = int(args.seed)
	run_dir = _resolve(args.run_root) / f"seed{seed:02d}" / method_key
	shell_path = run_dir / SHELL_CHECKPOINT_NAME
	metrics_path = run_dir / "eval.json"

	if args.mode == "train":
		core_path = _resolve(args.core_root) / f"seed{seed:02d}" / CORE_CHECKPOINT_NAME
		if not core_path.exists():
			raise FileNotFoundError(core_path)
		if shell_path.exists():
			raise FileExistsError(f"Refusing to overwrite existing shell checkpoint: {shell_path}")
		checkpoint = torch.load(core_path, map_location="cpu", weights_only=False)
		bundle = run_isolated_shell(
			checkpoint,
			prepared["dataset"],
			prepared["protocol"],
			method_key=method_key,
			seed=seed,
			device=device,
			max_iterations=4 if args.smoke else None,
			batch_size_override=2 if args.smoke else None,
			eval_every=1 if args.smoke else None,
		)
		model = bundle["model"]
		run_dir.mkdir(parents=True, exist_ok=True)
		torch.save(
			{
				"state_dict": model.state_dict(),
				"seed": seed,
				"baseline_key": method_key,
				"history": bundle["history"],
				"train_nll_history": bundle.get("train_nll_history"),
				"final_metrics": bundle["final_metrics"],
				"train_config": bundle["train_config"],
				"proc_std_exact": model._sampling_proc_std().detach().cpu().numpy(),
				"obs_std_exact": model._sampling_obs_std().detach().cpu().numpy(),
			},
			shell_path,
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
		_write_json(
			run_dir / "summary.json",
			{
				"seed": seed,
				"baseline_key": method_key,
				"baseline_label": bundle["method_label"],
				"selection_metric": bundle["selection_metric"],
				"selection_value": bundle["selection_value"],
				"selected_iteration": bundle.get("selected_iteration"),
				"train_config": bundle["train_config"],
				"smoke": bool(args.smoke),
			},
		)
		print(f"trained seed={seed:02d} {method_key} -> {run_dir}", flush=True)
		return 0

	if not shell_path.exists():
		raise FileNotFoundError(shell_path)
	checkpoint = torch.load(shell_path, map_location="cpu", weights_only=False)
	metrics = evaluate_isolated_shell(
		config,
		checkpoint,
		method_key=method_key,
		seed=seed,
		device=device,
		smoke=bool(args.smoke),
	)
	_write_json(metrics_path, metrics)
	print(
		f"eval seed={seed:02d} {method_key} "
		f"NLL={metrics.get('NLL')} G_tan={metrics.get('G_tan_transport')}",
		flush=True,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
