#!/usr/bin/env python3
"""Generate DynaMix main-figure artifacts (see notebooks/dynamix_main_figure.ipynb to plot)."""

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
from experiments.dynamix_main_figure import (
	plot_dynamix_main_figure,
	resolve_paths,
	run_dynamix_main_figure_data,
)
from experiments.lorenz96_pipeline import load_config


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
	parser.add_argument("--config", type=Path, default=Path("configs/dynamix_main.json"))
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("figures/figures_inputs/dynamix_main_figure"),
		help="Directory for generated figure inputs.",
	)
	parser.add_argument(
		"--adaptation-source-root",
		type=Path,
		default=Path("results/dynamix_main/adaptations"),
		help="Adaptation results/checkpoints used for figure export.",
	)
	parser.add_argument("--figure-dir", type=Path, default=Path("figures/dynamix_main"))
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--panel-a-only", action="store_true", help="Export panel-(a) CSV only.")
	parser.add_argument("--plot", action="store_true", help="Also render the figure.")
	parser.add_argument("--plot-only", action="store_true", help="Plot from saved inputs (skip generation).")
	return parser.parse_args()


def main() -> int:
	os.environ.setdefault("PYTHONUNBUFFERED", "1")
	args = parse_args()

	config_path = (REPO_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
	config = load_config(config_path)
	validate_dynamix_config(config)

	device = _resolve_device(args.device)
	paths = resolve_paths(REPO_ROOT, output_root=args.output_root)
	adaptation_root = (
		REPO_ROOT / args.adaptation_source_root
		if not args.adaptation_source_root.is_absolute()
		else args.adaptation_source_root
	).resolve()
	figure_dir = (REPO_ROOT / args.figure_dir).resolve()

	if not bool(args.plot_only):
		print(
			f"device={device} output={_display_path(paths.output_root)} "
			f"adaptation_root={_display_path(adaptation_root)}",
			flush=True,
		)
		outputs = run_dynamix_main_figure_data(
			config,
			paths=paths,
			adaptation_root=adaptation_root,
			device=device,
			skip_existing=bool(args.skip_existing),
			skip_rollout=bool(args.panel_a_only),
		)
		for name, out_path in outputs.items():
			print(f"wrote {name}: {_display_path(out_path)}", flush=True)

	if bool(args.plot) or bool(args.plot_only):
		out_pdf = plot_dynamix_main_figure(paths, figure_dir=figure_dir)
		print(f"wrote figure: {_display_path(out_pdf)}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
