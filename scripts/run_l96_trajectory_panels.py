#!/usr/bin/env python3
"""Generate Lorenz-96 optimization traces (a–d) without the full composite figure.

Each method and seed is trained once. Update 0 is the initial noise shell.
Later points are the weights after that many Adam steps, with a finer grid
through the first 25 updates.
"""

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
	plot_trajectory_panels_only,
	resolve_paths,
	run_trajectory_panel_data,
)
from experiments.lorenz96_main import validate_lorenz96_main_config
from experiments.lorenz96_pipeline import load_config, seed_list


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
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=Path("configs/lorenz96_main.json"))
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("figures/l96_trajectory_panels"),
	)
	parser.add_argument("--core-source-root", type=Path, default=Path("models/lorenz96_main_cores"))
	parser.add_argument("--shell-source-root", type=Path, default=Path("models/lorenz96_main_shells"))
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--cpu-threads", type=int, default=6)
	parser.add_argument("--smoke", action="store_true")
	parser.add_argument("--max-seeds", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--plot-only", action="store_true")
	return parser.parse_args()


def main() -> int:
	os.environ.setdefault("PYTHONUNBUFFERED", "1")
	args = parse_args()
	_set_cpu_threads(int(args.cpu_threads))

	config_path = (REPO_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
	config = load_config(config_path)
	validate_lorenz96_main_config(config)

	device = _resolve_device(args.device)
	paths = resolve_paths(REPO_ROOT, output_root=args.output_root)
	core_root = (REPO_ROOT / args.core_source_root).resolve()
	shell_root = (REPO_ROOT / args.shell_source_root).resolve()
	seeds = seed_list(config, smoke=bool(args.smoke), max_seeds=args.max_seeds, seed=args.seed)

	if not bool(args.plot_only):
		print(
			f"protocol=single_run_trajectory device={device} seeds={seeds} "
			f"output={paths.output_root}",
			flush=True,
		)
		result = run_trajectory_panel_data(
			config,
			paths=paths,
			core_root=core_root,
			shell_root=shell_root,
			seeds=seeds,
			device=device,
			smoke=bool(args.smoke),
			skip_existing=bool(args.skip_existing),
		)
		print(f"wrote {paths.trace_summary_csv}", flush=True)
		del result

	png_path, pdf_path = plot_trajectory_panels_only(paths, out_dir=paths.output_root)
	print(f"wrote panels {png_path} and {pdf_path}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
