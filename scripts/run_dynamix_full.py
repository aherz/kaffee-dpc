#!/usr/bin/env python3
"""DynaMix multisystem adaptation: in-context probabilistic fine-tuning and results export."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from experiments.dynamix_main import validate_dynamix_config
from experiments.dynamix_outputs import DYNAMIX_BASELINE_KEYS
from experiments.dynamix_pipeline import (
	RunMode,
	load_config,
	parse_train_devices,
	resolve_paths,
	run_dynamix_main,
	seed_list,
)


def _display_path(path: Path) -> str:
	resolved = Path(path).resolve()
	try:
		return str(resolved.relative_to(REPO_ROOT))
	except ValueError:
		return resolved.name


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
  use-checkpoints   Score shells in models/dynamix_main_shells and write the tables.
  adapt             Train adaptations, then write the same table exports.
""",
	)
	parser.add_argument("--config", type=Path, default=Path("configs/dynamix_main.json"))
	parser.add_argument(
		"--mode",
		choices=[m.value for m in RunMode],
		default=RunMode.USE_CHECKPOINTS.value,
		help="Training mode (default: use-checkpoints).",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("results/dynamix_main"),
		help="Output directory (tables under <output-root>/tables/).",
	)
	parser.add_argument(
		"--adaptation-source-root",
		type=Path,
		default=Path("models/dynamix_main_shells"),
		help="Shell checkpoint directory for use-checkpoints.",
	)
	parser.add_argument("--summary-csv", type=Path, default=None)
	parser.add_argument("--per-seed-output", type=Path, default=None)
	parser.add_argument("--meta-output", type=Path, default=None)
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--gpus", type=str, default=None)
	parser.add_argument("--cpu-threads", type=int, default=2)
	parser.add_argument("--smoke", action="store_true")
	parser.add_argument("--max-seeds", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--systems", nargs="+", default=None)
	parser.add_argument("--baselines", nargs="+", default=None)
	return parser.parse_args()


def main() -> int:
	os.environ.setdefault("PYTHONUNBUFFERED", "1")
	args = parse_args()

	config_path = (REPO_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
	config = load_config(config_path)
	validate_dynamix_config(config)

	paths = resolve_paths(
		REPO_ROOT,
		config=config_path,
		output_root=args.output_root,
		adaptation_source_root=args.adaptation_source_root,
		summary_csv=args.summary_csv,
		per_seed_output=args.per_seed_output,
		meta_output=args.meta_output,
	)

	mode = RunMode(str(args.mode))
	device = _resolve_device(args.device)
	train_devices = parse_train_devices(args.gpus, fallback_device=device)
	_set_cpu_threads(int(args.cpu_threads))

	systems = [str(s) for s in args.systems] if args.systems else [str(s) for s in config["systems"]]
	seeds = seed_list(config, smoke=bool(args.smoke), max_seeds=args.max_seeds, seed=args.seed)
	configured_baselines = [str(k) for k in config.get("baselines", DYNAMIX_BASELINE_KEYS)]
	baseline_keys = [str(k) for k in args.baselines] if args.baselines else configured_baselines
	unknown_baselines = sorted(set(baseline_keys) - set(configured_baselines))
	if unknown_baselines:
		raise ValueError(f"Unknown DynaMix baselines: {unknown_baselines}")

	print(
		f"mode={mode.value} device={device} train_devices={train_devices} "
		f"systems={systems} seeds={seeds}",
		flush=True,
	)
	result = run_dynamix_main(
		paths=paths,
		config=config,
		mode=mode,
		systems=systems,
		seeds=seeds,
		baseline_keys=baseline_keys,
		device=device,
		smoke=bool(args.smoke),
		train_devices=train_devices,
		cpu_threads=int(args.cpu_threads),
	)
	print(f"wrote {_display_path(result['summary_output'])}")
	print(f"wrote {_display_path(paths.per_seed_output)}")
	print(f"wrote {_display_path(paths.meta_output)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
