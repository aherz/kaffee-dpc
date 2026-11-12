#!/usr/bin/env python3
"""Lorenz-96 forecast-skill spectrum from trained shells."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from experiments.l96_forecast_skill import evaluate_forecast_skill_spectrum, plot_forecast_skill_spectrum
from experiments.lorenz96_main import validate_lorenz96_main_config
from experiments.lorenz96_pipeline import load_config, seed_list


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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=Path("configs/lorenz96_main.json"))
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("figures/l96_forecast_skill"),
		help="CSV/meta and default figure directory.",
	)
	parser.add_argument("--figure-dir", type=Path, default=None)
	parser.add_argument("--shell-source-root", type=Path, default=Path("models/lorenz96_main_shells"))
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--max-seeds", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--plot-only", action="store_true")
	return parser.parse_args()


def main() -> int:
	os.environ.setdefault("PYTHONUNBUFFERED", "1")
	args = parse_args()
	config_path = (REPO_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
	config = load_config(config_path)
	validate_lorenz96_main_config(config)
	output_root = (REPO_ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
	figure_dir = (
		(REPO_ROOT / args.figure_dir).resolve()
		if args.figure_dir is not None and not args.figure_dir.is_absolute()
		else (args.figure_dir.resolve() if args.figure_dir is not None else output_root)
	)
	csv_path = output_root / "forecast_skill_per_seed.csv"
	if not bool(args.plot_only):
		device = _resolve_device(args.device)
		shell_root = (
			REPO_ROOT / args.shell_source_root
			if not args.shell_source_root.is_absolute()
			else args.shell_source_root
		).resolve()
		seeds = seed_list(config, smoke=False, max_seeds=args.max_seeds, seed=args.seed)
		print(f"device={device} shells={_display_path(shell_root)} output={_display_path(output_root)}", flush=True)
		evaluate_forecast_skill_spectrum(
			config,
			shell_root=shell_root,
			device=device,
			seeds=seeds,
			skip_existing=bool(args.skip_existing),
			output_csv=csv_path,
		)
	if not csv_path.is_file():
		raise FileNotFoundError(f"Missing forecast-skill CSV: {csv_path}")
	out_pdf = plot_forecast_skill_spectrum(csv_path, figure_dir)
	print(f"wrote figure: {_display_path(out_pdf)}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
