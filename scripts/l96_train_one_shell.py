#!/usr/bin/env python3
"""Train one Lorenz-96 probabilistic shell baseline for one seed (parallel worker)."""

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

from experiments.lorenz96_main import prepare_lorenz96_main_bundle, validate_lorenz96_main_config
from training import run_alrnn_probabilistic_ablation

CORE_CHECKPOINT_NAME = "alrnn_core.pt"
SHELL_CHECKPOINT_NAME = "alrnn_shell.pt"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=Path("configs/lorenz96_main.json"))
	parser.add_argument("--seed", type=int, required=True)
	parser.add_argument("--baseline", type=str, required=True)
	parser.add_argument(
		"--core-root",
		type=Path,
		default=Path("models/lorenz96_main_cores"),
		help="Directory with seedNN/alrnn_core.pt checkpoints.",
	)
	parser.add_argument("--out-root", type=Path, required=True)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--cpu-threads", type=int, default=1)
	parser.add_argument(
		"--filter-rank",
		type=int,
		default=None,
		help="Override probabilistic_training.filter_rank (rrkf configs only).",
	)
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
	repo_root = REPO_ROOT
	config_path = _resolve(repo_root, args.config)
	core_root = _resolve(repo_root, args.core_root)
	out_root = _resolve(repo_root, args.out_root)
	seed = int(args.seed)
	baseline_key = str(args.baseline)

	config = json.loads(config_path.read_text(encoding="utf-8"))
	if args.filter_rank is not None:
		config["probabilistic_training"]["filter_rank"] = int(args.filter_rank)
	validate_lorenz96_main_config(config)
	_set_cpu_threads(int(args.cpu_threads))
	device = _resolve_device(args.device)

	prepared = prepare_lorenz96_main_bundle(config, smoke=bool(args.smoke))
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
	summary = {
		"seed": seed,
		"baseline_key": baseline_key,
		"baseline_label": bundle["spec"].label,
		"selection_metric": bundle["selection_metric"],
		"selection_value": bundle["selection_value"],
		"selected_iteration": bundle.get("selected_iteration"),
		"filter": bundle["train_config"].get("filter"),
		"filter_rank": bundle["train_config"].get("filter_rank"),
		"smoke": bool(args.smoke),
	}
	_write_json(run_dir / "summary.json", summary)
	print(f"done seed={seed:02d} {baseline_key} -> {run_dir}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
