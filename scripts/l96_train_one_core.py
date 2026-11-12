#!/usr/bin/env python3
"""Train one Lorenz-96 deterministic core for one seed (parallel worker)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from experiments.lorenz96_main import prepare_lorenz96_main_bundle, train_lorenz96_core, validate_lorenz96_main_config

CORE_CHECKPOINT_NAME = "alrnn_core.pt"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=Path("configs/lorenz96_main.json"))
	parser.add_argument("--seed", type=int, required=True)
	parser.add_argument("--out-root", type=Path, required=True)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--cpu-threads", type=int, default=1)
	parser.add_argument("--smoke", action="store_true")
	return parser.parse_args()


def _resolve(repo_root: Path, path: Path) -> Path:
	return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _set_cpu_threads(n_threads: int) -> None:
	n_threads = int(max(1, n_threads))
	torch.set_num_threads(n_threads)
	try:
		torch.set_num_interop_threads(1)
	except RuntimeError:
		pass


def _resolve_device(device_arg: str) -> torch.device:
	text = str(device_arg).strip().lower()
	if text == "cpu":
		return torch.device("cpu")
	if text.startswith("cuda"):
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA requested but unavailable")
		return torch.device(text)
	if text == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	raise ValueError(f"Unsupported device: {device_arg!r}")


def main() -> int:
	args = parse_args()
	repo_root = REPO_ROOT
	config_path = _resolve(repo_root, args.config)
	out_root = _resolve(repo_root, args.out_root)
	seed = int(args.seed)

	config = json.loads(config_path.read_text(encoding="utf-8"))
	validate_lorenz96_main_config(config)
	_set_cpu_threads(int(args.cpu_threads))
	device = _resolve_device(args.device)

	prepared = prepare_lorenz96_main_bundle(config, smoke=bool(args.smoke))
	dataset = prepared["dataset"]
	protocol = prepared["protocol"]

	run_dir = out_root / f"seed{seed:02d}"
	run_dir.mkdir(parents=True, exist_ok=True)
	model, train_info = train_lorenz96_core(
		dataset,
		protocol,
		seed=seed,
		device=device,
		max_epochs=4 if args.smoke else None,
		max_batches_per_epoch=2 if args.smoke else None,
		eval_every=1 if args.smoke else None,
		free_rollout_horizon=64 if args.smoke else 256,
	)
	checkpoint = {
		"state_dict": model.state_dict(),
		"M": int(protocol["model"]["latent_dim"]),
		"P": int(protocol["model"]["gated_dim"]),
		"N": int(protocol["model"]["observed_dim"]),
		"mean": dataset["standardizer"]["mean"].reshape(1, -1),
		"std": dataset["standardizer"]["std"].reshape(1, -1),
		"seed": seed,
		"data_seed": int(protocol["seed_sweep"].get("data_seed", 0)),
		"teacher_forcing_interval": int(protocol["model"]["teacher_forcing_interval"]),
		"selection_metric": train_info["selection_metric"],
		"selection_value": float(train_info["selection_value"]),
		"selection_epoch": int(train_info["selection_epoch"]),
		"config_json": json.dumps(config, sort_keys=True),
	}
	torch.save(checkpoint, run_dir / CORE_CHECKPOINT_NAME)
	print(f"done seed={seed:02d} -> {run_dir / CORE_CHECKPOINT_NAME}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
