#!/usr/bin/env python3
"""Train one Lorenz-63 probabilistic shell baseline for one seed (parallel worker)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from experiments.lorenz63_main import prepare_lorenz63_main_bundle, validate_lorenz63_main_config
from experiments.lorenz63_outputs import L63_METHODS
from training import run_alrnn_probabilistic_ablation

CORE_CHECKPOINT_NAME = "alrnn_core.pt"
SHELL_CHECKPOINT_NAME = "alrnn_shell.pt"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=Path("configs/lorenz63_main.json"))
	parser.add_argument("--seed", type=int, required=True)
	parser.add_argument("--baseline", type=str, required=True, choices=list(L63_METHODS))
	parser.add_argument(
		"--core-root",
		type=Path,
		default=Path("models/lorenz63_main_cores"),
		help="Directory with seedNN/alrnn_core.pt checkpoints.",
	)
	parser.add_argument("--out-root", type=Path, required=True)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--cpu-threads", type=int, default=1)
	parser.add_argument("--innovation-jitter", type=float, default=1e-6)
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


def _save_shell_checkpoint(path: Path, model: torch.nn.Module, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save({"state_dict": model.state_dict(), **payload}, path)


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
	args = parse_args()
	config_path = _resolve(REPO_ROOT, args.config)
	core_root = _resolve(REPO_ROOT, args.core_root)
	out_root = _resolve(REPO_ROOT, args.out_root)
	seed = int(args.seed)
	baseline_key = str(args.baseline)

	config = json.loads(config_path.read_text(encoding="utf-8"))
	validate_lorenz63_main_config(config)
	_set_cpu_threads(int(args.cpu_threads))
	device = _resolve_device(args.device)

	prepared = prepare_lorenz63_main_bundle(config, smoke=bool(args.smoke))
	checkpoint_path = core_root / f"seed{seed:02d}" / CORE_CHECKPOINT_NAME
	if not checkpoint_path.exists():
		raise FileNotFoundError(checkpoint_path)
	run_dir = out_root / f"seed{seed:02d}" / baseline_key
	shell_path = run_dir / SHELL_CHECKPOINT_NAME
	if shell_path.exists():
		raise FileExistsError(f"Refusing to overwrite existing shell checkpoint: {shell_path}")

	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	results = run_alrnn_probabilistic_ablation(
		checkpoint,
		prepared["dataset"],
		prepared["protocol"],
		baseline_keys=[baseline_key],
		seed=seed,
		device=device,
		innovation_jitter=float(args.innovation_jitter),
		use_jacobian_factor=True,
		max_iterations=4 if args.smoke else None,
		batch_size_override=2 if args.smoke else None,
		eval_every=1 if args.smoke else None,
	)
	bundle = results["baselines"][baseline_key]
	model = bundle["model"]
	run_dir.mkdir(parents=True, exist_ok=True)
	_save_shell_checkpoint(
		shell_path,
		model,
		{
			"seed": seed,
			"baseline_key": baseline_key,
			"history": bundle["history"],
			"train_nll_history": bundle.get("train_nll_history"),
			"final_metrics": bundle["final_metrics"],
			"train_config": bundle["train_config"],
			"proc_std_exact": model._sampling_proc_std().detach().cpu().numpy(),
			"obs_std_exact": model._sampling_obs_std().detach().cpu().numpy(),
		},
	)
	trace_rows = list(bundle.get("history", []))
	if trace_rows:
		pd.DataFrame(trace_rows).to_csv(run_dir / "trace.csv", index=False)
	train_nll_hist = list(bundle.get("train_nll_history", []))
	if train_nll_hist:
		pd.DataFrame(
			{
				"iter": np.arange(1, len(train_nll_hist) + 1, dtype=int),
				"train_nll": np.asarray(train_nll_hist, dtype=np.float64),
			}
		).to_csv(run_dir / "loss.csv", index=False)
	_write_json(
		run_dir / "summary.json",
		{
			"seed": seed,
			"baseline_key": baseline_key,
			"baseline_label": bundle["spec"].label,
			"selection_metric": bundle["selection_metric"],
			"selection_value": bundle["selection_value"],
			"selected_iteration": bundle.get("selected_iteration"),
			"smoke": bool(args.smoke),
		},
	)
	print(f"done seed={seed:02d} {baseline_key} -> {run_dir}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
