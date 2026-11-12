#!/usr/bin/env python3
"""Train and evaluate rr-KAFFEE ranks (default 5, 10, 20, 40, 80)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_WORKER = REPO_ROOT / "scripts" / "l96_train_one_shell.py"
EVAL_ENGINE = REPO_ROOT / "scripts" / "run_l96_full.py"
DEFAULT_CONFIG = Path("configs/lorenz96_rrkf.json")
DEFAULT_OUTPUT = Path("results/lorenz96_rrkf")
RANKS = (5, 10, 20, 40, 80)
SHELL_NAME = "alrnn_shell.pt"
PER_SEED_CSV = "main_l96_ablation_per_seed.csv"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument(
		"--mode",
		choices=["use-checkpoints", "train-shells"],
		default="train-shells",
		help="train-shells trains from the committed cores (used in the paper) then evaluates; use-checkpoints evaluates existing shells.",
	)
	parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument("--core-source-root", type=Path, default=Path("models/lorenz96_main_cores"))
	parser.add_argument(
		"--ranks",
		nargs="+",
		type=int,
		default=list(RANKS),
		help="Rank grid (default: 5 10 20 40 80).",
	)
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--gpus", type=str, default=None)
	parser.add_argument(
		"--jobs-per-gpu",
		type=int,
		default=1,
		help="Concurrent train jobs per GPU.",
	)
	parser.add_argument(
		"--eval-jobs-per-gpu",
		type=int,
		default=1,
		help="Concurrent eval jobs per GPU.",
	)
	parser.add_argument("--cpu-threads", type=int, default=1)
	parser.add_argument("--smoke", action="store_true")
	parser.add_argument(
		"--skip-eval",
		action="store_true",
		help="Train only (skip the use-checkpoints evaluation pass).",
	)
	parser.add_argument("--max-seeds", type=int, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--baselines", nargs="+", default=["kaffee"])
	parser.add_argument("--dry-run", action="store_true")
	return parser.parse_args()


def _gpu_ids(gpus: str | None) -> list[str | None]:
	if not gpus:
		return [None]
	ids = [part.strip() for part in str(gpus).split(",") if part.strip()]
	if not ids:
		raise ValueError("--gpus was provided but no devices were parsed")
	return ids


def _config_dict(path: Path) -> dict:
	resolved = path if path.is_absolute() else REPO_ROOT / path
	return json.loads(resolved.read_text(encoding="utf-8"))


def _seeds(config: dict, *, smoke: bool, max_seeds: int | None, seed: int | None) -> list[int]:
	if seed is not None:
		return [int(seed)]
	sweep = dict(config.get("seed_sweep", {}))
	count = int(sweep.get("count", 20))
	if smoke:
		count = min(count, 2)
	if max_seeds is not None:
		count = min(count, int(max_seeds))
	start = int(sweep.get("start_seed", 0))
	return list(range(start, start + count))


def _rank_dir(output_root: Path, rank: int) -> Path:
	return output_root / f"r{int(rank):03d}"


def _shell_path(output_root: Path, rank: int, seed: int, baseline: str) -> Path:
	return _rank_dir(output_root, rank) / "shells" / f"seed{int(seed):02d}" / baseline / SHELL_NAME


def _eval_csv(output_root: Path, rank: int, seed: int) -> Path:
	return _rank_dir(output_root, rank) / "eval" / f"seed{int(seed):02d}" / "tables" / PER_SEED_CSV


def _train_cmd(args: argparse.Namespace, rank: int, seed: int, baseline: str) -> list[str]:
	cmd = [
		sys.executable,
		str(TRAIN_WORKER),
		"--config",
		str(args.config),
		"--seed",
		str(int(seed)),
		"--baseline",
		str(baseline),
		"--core-root",
		str(args.core_source_root),
		"--out-root",
		str(_rank_dir(args.output_root, rank) / "shells"),
		"--device",
		"cuda" if args.gpus else str(args.device),
		"--cpu-threads",
		str(int(args.cpu_threads)),
		"--filter-rank",
		str(int(rank)),
	]
	if args.smoke:
		cmd.append("--smoke")
	return cmd


def _eval_cmd(args: argparse.Namespace, rank: int, seed: int) -> list[str]:
	rank_out = _rank_dir(args.output_root, rank)
	cmd = [
		sys.executable,
		str(EVAL_ENGINE),
		"--config",
		str(args.config),
		"--mode",
		"use-checkpoints",
		"--output-root",
		str(rank_out / "eval" / f"seed{int(seed):02d}"),
		"--core-source-root",
		str(args.core_source_root),
		"--shell-source-root",
		str(rank_out / "shells"),
		"--filter-rank",
		str(int(rank)),
		"--device",
		"cuda" if args.gpus else str(args.device),
		"--cpu-threads",
		str(int(args.cpu_threads)),
		"--seed",
		str(int(seed)),
		"--baselines",
		*[str(k) for k in args.baselines],
	]
	if args.smoke:
		cmd.append("--smoke")
	return cmd


def _run_one(
	cmd: list[str],
	*,
	log_path: Path,
	slots: Queue,
) -> int:
	gpu = slots.get()
	try:
		log_path.parent.mkdir(parents=True, exist_ok=True)
		env = os.environ.copy()
		env["PYTHONUNBUFFERED"] = "1"
		if gpu is not None:
			env["CUDA_VISIBLE_DEVICES"] = str(gpu)
		print(f"start gpu={gpu} log={log_path}", flush=True)
		with log_path.open("w", encoding="utf-8") as handle:
			completed = subprocess.run(
				cmd,
				cwd=str(REPO_ROOT),
				env=env,
				stdout=handle,
				stderr=subprocess.STDOUT,
				check=False,
			)
		rc = int(completed.returncode)
		print(f"done rc={rc} log={log_path}", flush=True)
		return rc
	finally:
		slots.put(gpu)


def _run_pool(
	jobs: list[tuple[list[str], Path]],
	*,
	gpu_ids: list[str | None],
	jobs_per_gpu: int,
	dry_run: bool,
) -> int:
	if not jobs:
		print("no jobs", flush=True)
		return 0
	if dry_run:
		for cmd, log_path in jobs:
			print(" ".join(cmd) + f"  # {log_path}", flush=True)
		return 0
	slots: Queue = Queue()
	n_workers = max(1, int(jobs_per_gpu)) * max(1, len(gpu_ids))
	for _ in range(max(1, int(jobs_per_gpu))):
		for gpu in gpu_ids:
			slots.put(gpu)
	failures = 0
	with ThreadPoolExecutor(max_workers=n_workers) as pool:
		futures = [
			pool.submit(_run_one, cmd, log_path=log_path, slots=slots)
			for cmd, log_path in jobs
		]
		for future in as_completed(futures):
			if int(future.result()) != 0:
				failures += 1
	return failures


def _concat_rank_tables(output_root: Path, ranks: list[int]) -> None:
	sweep_frames = []
	for rank in ranks:
		rank_root = _rank_dir(output_root, rank)
		paths = sorted(rank_root.glob(f"eval/seed*/tables/{PER_SEED_CSV}"))
		if not paths:
			print(f"rank {rank}: no per-seed eval tables yet", flush=True)
			continue
		frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
		dest = rank_root / "tables" / PER_SEED_CSV
		dest.parent.mkdir(parents=True, exist_ok=True)
		frame.to_csv(dest, index=False)
		print(f"wrote {dest} rows={len(frame)} files={len(paths)}", flush=True)
		tagged = frame.copy()
		tagged.insert(0, "filter_rank", int(rank))
		sweep_frames.append(tagged)
	if not sweep_frames:
		return
	sweep = pd.concat(sweep_frames, ignore_index=True)
	sweep_dest = output_root / "tables" / PER_SEED_CSV
	sweep_dest.parent.mkdir(parents=True, exist_ok=True)
	sweep.to_csv(sweep_dest, index=False)
	print(f"wrote {sweep_dest} rows={len(sweep)}", flush=True)


def _try_rank_report(output_root: Path, ranks: list[int]) -> None:
	src = str(REPO_ROOT / "src")
	if src not in sys.path:
		sys.path.insert(0, src)
	from experiments.l96_rrkf_performance import write_rank_report

	written = write_rank_report(output_root, ranks=ranks)
	for key, path in written.items():
		print(f"wrote {key} {path}", flush=True)


def main() -> int:
	os.environ.setdefault("PYTHONUNBUFFERED", "1")
	args = parse_args()
	if args.output_root == Path("results/lorenz96_main"):
		raise SystemExit("use a separate --output-root; results/lorenz96_main is used for dense KAFFEE")
	ranks = [int(r) for r in args.ranks]
	if any(r < 1 for r in ranks):
		raise SystemExit("ranks must be >= 1")
	config = _config_dict(args.config)
	seeds = _seeds(
		config,
		smoke=bool(args.smoke),
		max_seeds=args.max_seeds,
		seed=args.seed,
	)
	baselines = [str(k) for k in args.baselines]
	gpu_ids = _gpu_ids(args.gpus)
	log_root = args.output_root / "logs"

	print(
		f"mode={args.mode} ranks={ranks} seeds={seeds} gpus={gpu_ids} "
		f"jobs_per_gpu={int(args.jobs_per_gpu)} eval_jobs_per_gpu={int(args.eval_jobs_per_gpu)}",
		flush=True,
	)

	train_failures = 0
	if args.mode == "train-shells":
		train_jobs: list[tuple[list[str], Path]] = []
		for rank in ranks:
			for seed in seeds:
				for baseline in baselines:
					if _shell_path(args.output_root, rank, seed, baseline).is_file():
						print(
							f"skip train r{rank:03d} seed{seed:02d} {baseline}",
							flush=True,
						)
						continue
					train_jobs.append(
						(
							_train_cmd(args, rank, seed, baseline),
							log_root / f"train_r{rank:03d}_seed{seed:02d}_{baseline}.log",
						)
					)
		train_failures = _run_pool(
			train_jobs,
			gpu_ids=gpu_ids,
			jobs_per_gpu=int(args.jobs_per_gpu),
			dry_run=bool(args.dry_run),
		)

	eval_failures = 0
	if args.skip_eval:
		print("skip eval (--skip-eval)", flush=True)
	else:
		eval_jobs: list[tuple[list[str], Path]] = []
		for rank in ranks:
			for seed in seeds:
				if _eval_csv(args.output_root, rank, seed).is_file():
					print(f"skip eval r{rank:03d} seed{seed:02d}", flush=True)
					continue
				missing = [
					baseline
					for baseline in baselines
					if not _shell_path(args.output_root, rank, seed, baseline).is_file()
				]
				if missing and not args.dry_run:
					print(
						f"skip eval r{rank:03d} seed{seed:02d}: missing shells {missing}",
						flush=True,
					)
					continue
				eval_jobs.append(
					(
						_eval_cmd(args, rank, seed),
						log_root / f"eval_r{rank:03d}_seed{seed:02d}.log",
					)
				)
		eval_failures = _run_pool(
			eval_jobs,
			gpu_ids=gpu_ids,
			jobs_per_gpu=int(args.eval_jobs_per_gpu),
			dry_run=bool(args.dry_run),
		)
	if not args.dry_run:
		_concat_rank_tables(args.output_root, ranks)
		_try_rank_report(args.output_root, ranks)

	failures = int(train_failures) + int(eval_failures)
	print(
		f"done failures={failures} train_jobs_failed={train_failures} eval_jobs_failed={eval_failures}",
		flush=True,
	)
	return 1 if failures else 0


if __name__ == "__main__":
	raise SystemExit(main())
