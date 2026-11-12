#!/usr/bin/env python3
"""Timed optimizer steps of dense KAFFEE vs rr-KAFFEE over a latent-dimension grid.

Set KAFFEE_RRKF_FLOAT64=0 (training otherwise runs RRKF SVD in float64, which is not the regime used for this experiment).

Timings for the paper were run on a single NVIDIA RTX PRO 6000 Blackwell Server Edition GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "lorenz96_rrkf_timing.json"
ENGINE = REPO_ROOT / "scripts" / "benchmark_rrkf_scaling.py"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument("--smoke", action="store_true")
	parser.add_argument("--output-root", type=Path, required=True)
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--cpu-threads", type=int, default=2)
	parser.add_argument(
		"--resume",
		action="store_true",
		help="Skip cells that already finished or hit OOM.",
	)
	parser.add_argument("--methods", nargs="+", default=None)
	parser.add_argument("--filter-rank", type=int, default=None)
	parser.add_argument("--latent-dims", nargs="+", type=int, default=None)
	return parser.parse_args()


def gpu_name() -> str | None:
	try:
		completed = subprocess.run(
			["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
			check=True,
			capture_output=True,
			text=True,
		)
	except (FileNotFoundError, subprocess.CalledProcessError):
		return None
	return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else None


def output_is_complete(output_dir: Path) -> bool:
	manifest_path = output_dir / "manifest.json"
	if not manifest_path.is_file():
		return False
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	return str(manifest.get("status")) == "complete"


def failure_is_oom(output_dir: Path) -> bool:
	texts: list[str] = []
	for path in sorted((output_dir / "workers").glob("*.result.json")) if (output_dir / "workers").is_dir() else []:
		texts.append(path.read_text(encoding="utf-8", errors="replace"))
	for path in sorted((output_dir / "logs").glob("*.log")) if (output_dir / "logs").is_dir() else []:
		texts.append(path.read_text(encoding="utf-8", errors="replace"))
	blob = "\n".join(texts).lower()
	return "out of memory" in blob or "cuda error: out of memory" in blob


def engine_cmd(
	*,
	cfg: dict[str, Any],
	output_dir: Path,
	device: str,
	cpu_threads: int,
	warmups: int,
	timed_updates: int,
	process_repeats: int,
	smoke: bool,
) -> list[str]:
	cmd = [
		sys.executable,
		str(ENGINE),
		"--output-dir",
		str(output_dir),
		"--observed-dims",
		str(int(cfg["observed_dim"])),
		"--batch-sizes",
		str(int(cfg["batch_size"])),
		"--window-steps",
		str(int(cfg["window_steps"])),
		"--gated-fraction",
		str(float(cfg["gated_fraction"])),
		"--burn-in-steps",
		str(int(cfg["burn_in_steps"])),
		"--warmups",
		str(int(warmups)),
		"--repeats",
		str(int(timed_updates)),
		"--process-repeats",
		str(int(process_repeats)),
		"--learning-rate",
		str(float(cfg["learning_rate"])),
		"--gradient-clip",
		str(float(cfg["gradient_clip"])),
		"--proc-std",
		str(float(cfg["proc_std"])),
		"--obs-std",
		str(float(cfg["obs_std"])),
		"--dtype",
		"float32",
		"--device",
		str(device),
		"--cpu-threads",
		str(int(cpu_threads)),
		"--reset-model-between-updates",
	]
	if smoke:
		cmd.append("--smoke")
	return cmd


def run_cell(
	*,
	cfg: dict[str, Any],
	output_dir: Path,
	cell_args: list[str],
	device: str,
	cpu_threads: int,
	warmups: int,
	timed_updates: int,
	process_repeats: int,
	smoke: bool,
	resume: bool,
) -> str:
	if output_dir.exists():
		if resume and output_is_complete(output_dir):
			print(f"skip complete {output_dir}", flush=True)
			return "complete"
		if resume and failure_is_oom(output_dir):
			print(f"skip oom {output_dir}", flush=True)
			return "oom"
		raise FileExistsError(f"output already exists: {output_dir}")

	env = os.environ.copy()
	env["KAFFEE_RRKF_FLOAT64"] = "0"
	command = [
		*engine_cmd(
			cfg=cfg,
			output_dir=output_dir,
			device=device,
			cpu_threads=cpu_threads,
			warmups=warmups,
			timed_updates=timed_updates,
			process_repeats=process_repeats,
			smoke=smoke,
		),
		*cell_args,
	]
	print(">>", " ".join(command), flush=True)
	completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=False)
	if completed.returncode == 0 and output_is_complete(output_dir):
		return "complete"
	if failure_is_oom(output_dir):
		return "oom"
	raise RuntimeError(f"timing cell failed: {output_dir}")


def run_smoke(args: argparse.Namespace, cfg: dict[str, Any], results_root: Path) -> None:
	methods = list(args.methods or cfg["methods"])
	rank = int(args.filter_rank if args.filter_rank is not None else cfg["filter_rank"])
		latent_dims = [int(x) for x in (args.latent_dims or [100])]
	status = run_cell(
		cfg=cfg,
		output_dir=results_root / "smoke",
		cell_args=[
			"--latent-dims",
			*[str(x) for x in latent_dims],
			"--methods",
			*methods,
			"--filter-ranks",
			str(rank),
		],
		device=args.device,
		cpu_threads=args.cpu_threads,
		warmups=2,
		timed_updates=3,
		process_repeats=1,
		smoke=True,
		resume=bool(args.resume),
	)
	if status != "complete":
		raise RuntimeError(f"smoke failed ({status})")


def run_dimensions(args: argparse.Namespace, cfg: dict[str, Any], results_root: Path) -> None:
	methods = list(args.methods or cfg["methods"])
	rank = int(args.filter_rank if args.filter_rank is not None else cfg["filter_rank"])
	latent_dims = [int(x) for x in (args.latent_dims or cfg["latent_dims"])]
	n_obs = int(cfg["observed_dim"])
	batch = int(cfg["batch_size"])
	window = int(cfg["window_steps"])
	stopped: set[str] = set()
	for latent_dim in latent_dims:
		for method in methods:
			if method in stopped:
				continue
			if method == "kaffee_rrkf" and rank > latent_dim:
				continue
			rank_tag = f"_r{rank:03d}" if method == "kaffee_rrkf" else ""
			# Folder tag: M = latent dim, N = observed dim (paper notation).
			output_dir = (
				results_root
				/ "dimension"
				/ f"{method}{rank_tag}_M{latent_dim}_N{n_obs}_B{batch}_T{window}"
			)
			cell_args = ["--latent-dims", str(latent_dim), "--methods", method, "--filter-ranks", str(rank)]
			status = run_cell(
				cfg=cfg,
				output_dir=output_dir,
				cell_args=cell_args,
				device=args.device,
				cpu_threads=args.cpu_threads,
				warmups=int(cfg["warmups"]),
				timed_updates=int(cfg["timed_updates"]),
				process_repeats=int(cfg["process_repeats"]),
				smoke=False,
				resume=bool(args.resume),
			)
			if status == "oom":
				stopped.add(method)
				print(f"oom at M={latent_dim} for {method}; skipping larger M", flush=True)


def main() -> int:
	args = parse_args()
	cfg = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
	results_root = Path(args.output_root).expanduser()
	results_root.mkdir(parents=True, exist_ok=True)
	os.environ["KAFFEE_RRKF_FLOAT64"] = "0"
	name = gpu_name()
	if name:
		print(f"gpu={name}", flush=True)
	if args.smoke:
		run_smoke(args, cfg, results_root)
	else:
		run_dimensions(args, cfg, results_root)
	print(f"finished {results_root}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
