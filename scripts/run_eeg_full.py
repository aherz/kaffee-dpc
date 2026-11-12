#!/usr/bin/env python3
"""Train and/or re-evaluate the EEG experiment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from experiments.eeg_pipeline import RunMode, load_config, resolve_paths, run_eeg_main


def resolve_device(device_arg: str) -> str:
	text = str(device_arg).strip().lower()
	if text == "auto":
		import torch
		return "cuda" if torch.cuda.is_available() else "cpu"
	if text == "cpu" or text.startswith("cuda"):
		return text
	raise ValueError(f"Unsupported device specification: {device_arg!r}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=Path("configs/eeg_main.json"))
	parser.add_argument("--mode", choices=[mode.value for mode in RunMode], default=RunMode.USE_CHECKPOINTS.value)
	parser.add_argument("--output-root", type=Path, default=Path("results/eeg_main"))
	parser.add_argument("--core-source-root", type=Path, default=Path("models/eeg_main_cores"))
	parser.add_argument("--shell-source-root", type=Path, default=Path("models/eeg_main_shells"))
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--cpu-threads", type=int, default=2)
	parser.add_argument("--smoke", action="store_true")
	return parser.parse_args()


def main() -> int:
	os.environ.setdefault("PYTHONUNBUFFERED", "1")
	args = parse_args()
	config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
	config = load_config(config_path)
	paths = resolve_paths(
		REPO_ROOT,
		config=config_path,
		output_root=args.output_root,
		core_source_root=args.core_source_root,
		shell_source_root=args.shell_source_root,
	)
	result = run_eeg_main(
		paths=paths,
		config=config,
		mode=RunMode(str(args.mode)),
		device=resolve_device(str(args.device)),
		cpu_threads=int(args.cpu_threads),
		smoke=bool(args.smoke),
	)
	print(f"wrote {result['summary_output']}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
