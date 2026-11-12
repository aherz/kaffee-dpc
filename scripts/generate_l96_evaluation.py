#!/usr/bin/env python3
"""Generate Lorenz-96 evaluation trajectories from metadata JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from data import load_lorenz96_dataset, make_lorenz96, standardize


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=Path("configs/lorenz96_main.json"))
	parser.add_argument(
		"--metadata",
		type=Path,
		default=Path("src/data/lorenz96_d20/heldout_test.json"),
	)
	parser.add_argument("--output", type=Path)
	return parser.parse_args()


def _resolve(path: Path) -> Path:
	return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> int:
	args = parse_args()
	config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
	metadata = json.loads(_resolve(args.metadata).read_text(encoding="utf-8"))
	output_path = _resolve(args.output or Path(metadata["path"]))
	if output_path.exists():
		raise FileExistsError(f"Refusing to overwrite existing evaluation data: {output_path}")

	dataset_cfg = dict(config["dataset"])
	dataset_path = _resolve(Path(dataset_cfg["path"]))
	info_path = _resolve(Path(dataset_cfg["info_path"]))
	dataset = load_lorenz96_dataset(str(dataset_path), info_path=str(info_path))
	system = dict(config["system"])

	clean_raw = make_lorenz96(
		int(metadata["length"]),
		dt=float(system["dt"]),
		N=int(system["state_dim"]),
		F=float(system["forcing"]),
		burn_in=int(system["burn_in"]),
		seed=int(metadata["process_seed"]),
		sigma_proc=float(system["sigma_proc"]),
		sigma_proc_space=str(system["sigma_proc_space"]),
		proc_scale_raw=np.asarray(metadata["sigma_proc_raw_per_dim"], dtype=np.float64),
	).astype(np.float32)
	obs_scale = np.asarray(dataset["data_info"]["obs_sigma_raw_per_dim"], dtype=np.float32)
	rng = np.random.default_rng(int(metadata["observation_noise_seed"]))
	obs_raw = clean_raw + rng.standard_normal(clean_raw.shape).astype(np.float32) * obs_scale
	stats = dict(dataset["standardizer"])

	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		output_path,
		clean_raw=clean_raw,
		obs_raw=obs_raw,
		clean_std=standardize(clean_raw, stats),
		obs_std=standardize(obs_raw, stats),
		process_seed=np.asarray(metadata["process_seed"], dtype=np.int64),
		observation_noise_seed=np.asarray(metadata["observation_noise_seed"], dtype=np.int64),
	)

	generated = {
		**metadata,
		"path": str(output_path.relative_to(REPO_ROOT)) if output_path.is_relative_to(REPO_ROOT) else str(output_path),
	}
	output_path.with_suffix(".json").write_text(
		json.dumps(generated, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)
	print(json.dumps(generated, indent=2, sort_keys=True))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
