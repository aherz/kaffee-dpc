#!/usr/bin/env python3
"""Minimal example of the mechanistic ablation experiment on Lorenz-96 data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=Path("configs/lorenz96_main.json"))
	parser.add_argument("--output-root", type=Path, default=Path("scratch/lorenz96_minimal"))
	parser.add_argument("--mode", default="train-shells")
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--cpu-threads", type=int, default=2)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	repo_root = Path(__file__).resolve().parents[1]
	out = args.output_root
	cmd = [
		sys.executable,
		str(repo_root / "scripts" / "run_l96_full.py"),
		"--config",
		str(args.config),
		"--mode",
		str(args.mode),
		"--output-root",
		str(out),
		"--device",
		str(args.device),
		"--cpu-threads",
		str(args.cpu_threads),
		"--smoke",
	]
	print(f"running: {' '.join(cmd)}")
	subprocess.run(cmd, check=True, cwd=str(repo_root))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
