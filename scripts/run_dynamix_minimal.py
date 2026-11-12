#!/usr/bin/env python3
"""Minimal DynaMix test, exporting the 13-system ODE specification table."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("scratch/dynamix_minimal"),
		help="Output directory (writes tables/dynamix_benchmark_systems.csv).",
	)
	return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
	return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def main() -> int:
	args = parse_args()
	repo_root = Path(__file__).resolve().parents[1]
	sys.path.insert(0, str(repo_root / "src"))

	from data import dynamix_benchmark_records  # noqa: PLC0415

	output_path = _resolve(repo_root, args.output_root / "tables" / "dynamix_benchmark_systems.csv")
	output_path.parent.mkdir(parents=True, exist_ok=True)

	rows = dynamix_benchmark_records()
	with output_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=["system", "parameters", "dt", "x0", "burn_in", "lambda1_ref"],
			lineterminator="\n",
		)
		writer.writeheader()
		writer.writerows(rows)

	print(f"wrote {len(rows)} rows to {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
