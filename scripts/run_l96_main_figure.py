#!/usr/bin/env python3
"""Generate Lorenz-96 figure artifacts (use notebooks/l96_main_figure.ipynb to plot)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from experiments.l96_main_figure import (
	plot_l96_main_figure,
	resolve_paths,
	run_l96_main_figure_data,
)
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
		default=Path("figures/figures_inputs/lorenz96_main_figure"),
		help="Directory for generated figure inputs.",
	)
	parser.add_argument("--figure-dir", type=Path, default=Path("figures/l96_main"))
	parser.add_argument("--core-source-root", type=Path, default=Path("models/lorenz96_main_cores"))
	parser.add_argument("--shell-source-root", type=Path, default=Path("models/lorenz96_main_shells"))
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--smoke", action="store_true")
	parser.add_argument("--max-seeds", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--skip-existing", action="store_true", help="Skip steps whose outputs already exist.")
	parser.add_argument("--plot", action="store_true", help="Also render the figure.")
	parser.add_argument("--plot-only", action="store_true", help="Plot from saved inputs (skip generation).")
	return parser.parse_args()


def main() -> int:
	os.environ.setdefault("PYTHONUNBUFFERED", "1")
	args = parse_args()

	config_path = (REPO_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
	config = load_config(config_path)
	validate_lorenz96_main_config(config)

	device = _resolve_device(args.device)
	paths = resolve_paths(REPO_ROOT, output_root=args.output_root)
	core_root = (REPO_ROOT / args.core_source_root).resolve()
	shell_root = (REPO_ROOT / args.shell_source_root).resolve()
	figure_dir = (REPO_ROOT / args.figure_dir).resolve()

	seeds = seed_list(config, smoke=bool(args.smoke), max_seeds=args.max_seeds, seed=args.seed)

	if not bool(args.plot_only):
		print(
			f"device={device} seeds={seeds} output={_display_path(paths.output_root)} "
			f"core_root={_display_path(core_root)} shell_root={_display_path(shell_root)}",
			flush=True,
		)
		run_l96_main_figure_data(
			config,
			paths=paths,
			core_root=core_root,
			shell_root=shell_root,
			seeds=seeds,
			device=device,
			smoke=bool(args.smoke),
			skip_existing=bool(args.skip_existing),
		)
		print(f"wrote artifacts under {_display_path(paths.output_root)}", flush=True)

	if bool(args.plot) or bool(args.plot_only):
		out_pdf = plot_l96_main_figure(paths, figure_dir=figure_dir)
		print(f"wrote figure {_display_path(out_pdf)}", flush=True)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
