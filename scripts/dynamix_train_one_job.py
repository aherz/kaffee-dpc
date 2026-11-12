#!/usr/bin/env python3
"""Train and evaluate one DynaMix adaptation job (parallel worker)."""

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

from experiments.dynamix_main import prepare_dynamix_bundle, validate_dynamix_config
from experiments.dynamix_pipeline import _result_path, _run_one_dynamix_job


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, required=True)
	parser.add_argument("--baseline", type=str, required=True)
	parser.add_argument("--system", type=str, required=True)
	parser.add_argument("--seed", type=int, required=True)
	parser.add_argument("--adaptation-root", type=Path, required=True)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--cpu-threads", type=int, default=2)
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


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
	args = parse_args()
	repo_root = REPO_ROOT
	config_path = _resolve(repo_root, args.config)
	adaptation_root = _resolve(repo_root, args.adaptation_root)
	_set_cpu_threads(int(args.cpu_threads))
	device = _resolve_device(args.device)

	config = json.loads(config_path.read_text(encoding="utf-8"))
	validate_dynamix_config(config)
	protocol = prepare_dynamix_bundle(config, smoke=bool(args.smoke))["protocol"]

	baseline_key = str(args.baseline)
	system = str(args.system)
	seed = int(args.seed)
	out = _result_path(adaptation_root, baseline_key, system, seed)
	if out.is_file():
		print(f"skip {baseline_key}:{system}:seed{seed}", flush=True)
		return 0

	result = _run_one_dynamix_job(
		repo_root=repo_root,
		protocol=protocol,
		adaptation_root=adaptation_root,
		baseline_key=baseline_key,
		system=system,
		seed=seed,
		device=device,
	)
	out.parent.mkdir(parents=True, exist_ok=True)
	_write_json(out, result)
	print(f"done {baseline_key}:{system}:seed{seed} -> {out}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
