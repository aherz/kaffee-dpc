#!/usr/bin/env python3
"""Lorenz-96 mechanistic ablation: train deterministic cores and stochastic shells (optional; can also use pretrained cores and shells) and export metrics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from experiments.lorenz96_pipeline import (
	RunMode,
	load_config,
	parse_train_devices,
	resolve_paths,
	run_lorenz96_main,
	seed_list,
)
from experiments.lorenz96_main import validate_lorenz96_main_config  
from training import BASELINE_SPECS 


def _resolve_device(device_arg: str):
	import torch

	text = str(device_arg).strip().lower()
	if text == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if text == "cpu":
		return torch.device("cpu")
	if text.startswith("cuda"):
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
		return torch.device(text)
	raise ValueError(f"Unsupported device specification: {device_arg!r}")


def _set_cpu_threads(n_threads: int) -> None:
	import torch

	n_threads = int(max(1, n_threads))
	torch.set_num_threads(n_threads)
	try:
		torch.set_num_interop_threads(1)
	except RuntimeError:
		pass


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Modes:
  use-checkpoints   Default. Pretrained cores + shells; evaluation only.
  train-shells      Pretrained cores; train shells, then evaluate.
  from-scratch      Train cores, train shells, then evaluate.
""",
	)
	parser.add_argument("--config", type=Path, default=Path("configs/lorenz96_main.json"))
	parser.add_argument(
		"--mode",
		choices=[m.value for m in RunMode],
		default=RunMode.USE_CHECKPOINTS.value,
		help="Training mode (default: use-checkpoints).",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("results/lorenz96_main"),
		help="Output directory (tables under <output-root>/tables/).",
	)
	parser.add_argument("--core-source-root", type=Path, default=Path("models/lorenz96_main_cores"))
	parser.add_argument("--shell-source-root", type=Path, default=Path("models/lorenz96_main_shells"))
	parser.add_argument("--summary-csv", type=Path, default=None)
	parser.add_argument("--per-seed-output", type=Path, default=None)
	parser.add_argument("--meta-output", type=Path, default=None)
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument(
		"--gpus",
		type=str,
		default=None,
		help="Comma-separated CUDA device ids (e.g. 0,1,2) or 'all' for parallel training workers.",
	)
	parser.add_argument("--cpu-threads", type=int, default=2)
	parser.add_argument("--smoke", action="store_true")
	parser.add_argument("--max-seeds", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--baselines", nargs="+", default=None)
	parser.add_argument(
		"--filter-rank",
		type=int,
		default=None,
		help="Override probabilistic_training.filter_rank (rrkf).",
	)
	return parser.parse_args()


def main() -> int:
	os.environ.setdefault("PYTHONUNBUFFERED", "1")
	args = parse_args()
	mode = RunMode(str(args.mode))

	config = load_config((REPO_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config)
	if args.filter_rank is not None:
		config["probabilistic_training"]["filter_rank"] = int(args.filter_rank)
	validate_lorenz96_main_config(config)

	if str(config.get("probabilistic_training", {}).get("filter", "ekf")).strip().lower() == "rrkf":
		if args.output_root == Path("results/lorenz96_main"):
			args.output_root = Path("results/lorenz96_rrkf")
		if args.baselines is None:
			args.baselines = ["kaffee"]
		if mode is RunMode.USE_CHECKPOINTS and args.shell_source_root == Path("models/lorenz96_main_shells"):
			raise SystemExit(
				"rr-KAFFEE eval requires --shell-source-root pointing at rank-reduced shells"
			)

	device = _resolve_device(args.device)
	train_devices = parse_train_devices(args.gpus, fallback_device=device)
	_set_cpu_threads(int(args.cpu_threads))

	paths = resolve_paths(
		REPO_ROOT,
		config=args.config,
		output_root=args.output_root,
		core_source_root=args.core_source_root,
		shell_source_root=args.shell_source_root,
		summary_csv=args.summary_csv,
		per_seed_output=args.per_seed_output,
		meta_output=args.meta_output,
	)

	seeds = seed_list(config, smoke=bool(args.smoke), max_seeds=args.max_seeds, seed=args.seed)
	baseline_keys = [str(k) for k in args.baselines] if args.baselines else list(BASELINE_SPECS.keys())

	print(
		f"mode={mode.value} device={device} train_devices={train_devices} "
		f"seeds={seeds}",
		flush=True,
	)
	result = run_lorenz96_main(
		paths=paths,
		config=config,
		mode=mode,
		seeds=seeds,
		baseline_keys=baseline_keys,
		device=device,
		smoke=bool(args.smoke),
		train_devices=train_devices,
		cpu_threads=int(args.cpu_threads),
	)
	print(f"wrote {result['summary_output']}")
	print(f"wrote {paths.per_seed_output}")
	print(f"wrote {paths.meta_output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
