#!/usr/bin/env python3
"""Write the rr-KAFFEE rank table and figure."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from experiments.l96_rrkf_performance import (
	RANKS,
	load_timing_cells,
	plot_timing_metrics,
	timing_exponent_rows,
	write_rank_report,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output-root", type=Path, default=Path("results/lorenz96_rrkf"))
	parser.add_argument("--figure-dir", type=Path, default=Path("figures/l96_rrkf"))
	parser.add_argument(
		"--dense-per-seed",
		type=Path,
		default=Path("results/lorenz96_main/tables/main_l96_ablation_per_seed.csv"),
	)
	parser.add_argument("--ranks", nargs="+", type=int, default=list(RANKS))
	parser.add_argument("--required-seeds", type=int, default=20)
	parser.add_argument("--no-dense", action="store_true")
	parser.add_argument(
		"--timing-csv",
		type=Path,
		default=Path("results/lorenz96_rrkf_timing/tables/rrkf_timing_per_cell.csv"),
	)
	return parser.parse_args()


def main() -> int:
	os.environ.setdefault("PYTHONUNBUFFERED", "1")
	args = parse_args()
	written = write_rank_report(
		args.output_root,
		ranks=[int(r) for r in args.ranks],
		figure_dir=args.figure_dir,
		dense_per_seed_csv=None if args.no_dense else args.dense_per_seed,
		required_seeds=int(args.required_seeds),
	)
	for key, path in written.items():
		print(f"wrote {key} {path}", flush=True)
	if args.timing_csv.is_file():
		timing = load_timing_cells(args.timing_csv)
		timing_fig = plot_timing_metrics(timing, args.figure_dir)
		if timing_fig is not None:
			print(f"wrote timing {timing_fig}", flush=True)
		exp_rows = timing_exponent_rows(timing)
		if exp_rows:
			exp_path = Path("results/lorenz96_rrkf_timing/tables/rrkf_timing_exponents.csv")
			if args.timing_csv.parent.name == "tables":
				exp_path = args.timing_csv.parent / "rrkf_timing_exponents.csv"
			pd.DataFrame(exp_rows).to_csv(exp_path, index=False)
			print(f"wrote exponents {exp_path}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
