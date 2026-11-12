"""Smoke test: validate basic functionality."""

#!/usr/bin/env python3 

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> int:
	repo_root = Path(__file__).resolve().parents[1]
	sys.path.insert(0, str(repo_root / "src"))

	from data import make_lorenz96
	from experiments.lorenz96_main import validate_lorenz96_main_config
	from training import BASELINE_SPECS

	config_path = repo_root / "configs" / "lorenz96_main.json"
	config = __import__("json").loads(config_path.read_text(encoding="utf-8"))
	validate_lorenz96_main_config(config)

	l96 = make_lorenz96(8, seed=0, sigma_proc=0.1, sigma_proc_space="standardized")
	if l96.shape != (8, 20):
		raise SystemExit(f"unexpected Lorenz-96 smoke shape: {l96.shape}")
	if not np.isfinite(l96).all():
		raise SystemExit("Lorenz smoke generation produced non-finite values")

	print("smoke_test_ok")
	print(f"baseline_count={len(BASELINE_SPECS)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
