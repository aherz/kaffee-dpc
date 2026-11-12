#!/usr/bin/env python3
"""Time one Adam step of dense KAFFEE vs rr-KAFFEE.

Dense covariance uses transition_and_covariance; rr-KAFFEE uses the
Jacobian-factor product and a Joseph update when r > N (N = observed dim).
Set KAFFEE_RRKF_FLOAT64=0 so the RRKF SVD stays in float32.

Each (method, M) cell is a fresh process so peak GPU memory is not
contaminated by earlier cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from filtering.objectives import filtered_innovation_nll_batched
from models import ALRNNShell, ALRNNShellLR


METHODS = ("kaffee", "kaffee_rrkf")


def _device(text: str) -> torch.device:
	name = str(text).strip().lower()
	if name == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if name == "cpu":
		return torch.device("cpu")
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA requested but unavailable")
	return torch.device(name)




def _seed(seed: int) -> None:
	np.random.seed(int(seed))
	torch.manual_seed(int(seed))
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(int(seed))


def _sync(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.synchronize(device)


def _make_model(spec: dict[str, Any], device: torch.device, dtype: torch.dtype) -> ALRNNShell | ALRNNShellLR:
	m = int(spec["latent_dim"])
	n = int(spec["observed_dim"])
	gen = torch.Generator(device="cpu").manual_seed(int(spec["seed"]))
	embedding = torch.zeros((n, m), dtype=dtype)
	embedding[:, :n] = torch.eye(n, dtype=dtype)
	shared = dict(
		latent_dim=m,
		gated_dim=int(spec["gated_dim"]),
		observed_dim=n,
		A_init=torch.full((m,), 0.80, dtype=dtype),
		W_init=torch.randn((m, m), generator=gen, dtype=dtype) * (0.04 / math.sqrt(float(m))),
		h_init=torch.zeros((m,), dtype=dtype),
		embedding_init=embedding,
		proc_std_init=torch.full((m,), float(spec["proc_std"]), dtype=dtype),
		obs_std_init=torch.full((n,), float(spec["obs_std"]), dtype=dtype),
		train_core=True,
		covariance_transport="jacobian",
	)
	if str(spec.get("filter", "ekf")) == "rrkf":
		model: ALRNNShell | ALRNNShellLR = ALRNNShellLR(**shared, filter_rank=int(spec["filter_rank"]))
	else:
		model = ALRNNShell(**shared)
	return model.to(device=device, dtype=dtype)


def _windows(spec: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	b = int(spec["batch_size"])
	t = int(spec["window_steps"])
	n = int(spec["observed_dim"])
	gen = torch.Generator(device="cpu").manual_seed(int(spec["seed"]) + 17)
	x = torch.zeros((b, t + 1, n), dtype=dtype)
	x[:, 0, :] = 0.25 * torch.randn((b, n), generator=gen, dtype=dtype)
	for step in range(t):
		x[:, step + 1, :] = 0.80 * x[:, step, :] + 0.10 * torch.randn((b, n), generator=gen, dtype=dtype)
	return x.to(device=device)


def _step(
	model: ALRNNShell | ALRNNShellLR,
	windows: torch.Tensor,
	optimizer: torch.optim.Optimizer,
	spec: dict[str, Any],
) -> tuple[float, float]:
	optimizer.zero_grad(set_to_none=True)
	normalizer = float(spec["batch_size"]) * float(int(spec["window_steps"]) - int(spec["burn_in_steps"]))
	loss = filtered_innovation_nll_batched(model, windows, burn_in_steps=int(spec["burn_in_steps"])) / normalizer
	loss.backward()
	grad_norm = torch.nn.utils.clip_grad_norm_(
		[p for p in model.parameters() if p.requires_grad],
		float(spec["gradient_clip"]),
	)
	optimizer.step()
	return float(loss.detach().cpu()), float(grad_norm.detach().cpu())


def run_worker(spec: dict[str, Any]) -> dict[str, Any]:
	if str(spec.get("filter", "ekf")) == "rrkf":
		flag = os.environ.get("KAFFEE_RRKF_FLOAT64", "").strip().lower()
		if flag not in {"0", "false", "off", "no"}:
			raise RuntimeError("RRKF timing requires KAFFEE_RRKF_FLOAT64=0 (float32).")
	n_threads = max(1, int(spec["cpu_threads"]))
	torch.set_num_threads(n_threads)
	try:
		torch.set_num_interop_threads(1)
	except RuntimeError:
		pass
	torch.set_float32_matmul_precision("highest") # make sure to use highest precision for matmuls to mitigate numerical rounding errors
	_seed(int(spec["seed"]))
	device = _device(str(spec["device"]))
	dtype = torch.float32 if str(spec["dtype"]) == "float32" else torch.float64
	model = _make_model(spec, device, dtype)
	windows = _windows(spec, device, dtype)
	optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["learning_rate"]))
	state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

	def restore() -> None:
		with torch.no_grad():
			for name, tensor in model.state_dict().items():
				tensor.copy_(state[name])

	_step(model, windows, optimizer, spec)
	for _ in range(int(spec["warmups"])):
		restore()
		_step(model, windows, optimizer, spec)

	if device.type == "cuda":
		_sync(device)
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats(device)

	seconds: list[float] = []
	losses: list[float] = []
	grad_norms: list[float] = []
	for _ in range(int(spec["repeats"])):
		restore()
		_sync(device)
		t0 = time.perf_counter()
		loss, grad_norm = _step(model, windows, optimizer, spec)
		_sync(device)
		elapsed = time.perf_counter() - t0
		if not math.isfinite(elapsed) or elapsed <= 0.0:
			raise RuntimeError(f"invalid duration {elapsed}")
		seconds.append(elapsed)
		losses.append(loss)
		grad_norms.append(grad_norm)

	median = float(statistics.median(seconds))
	peak_cuda = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
	summary = {
		"status": "complete",
		"worker_id": str(spec["worker_id"]),
		"method": str(spec["method"]),
		"filter": str(spec.get("filter", "ekf")),
		"filter_rank": None if spec.get("filter_rank") is None else int(spec["filter_rank"]),
		"process_repeat": int(spec["process_repeat"]),
		"latent_dim": int(spec["latent_dim"]),
		"observed_dim": int(spec["observed_dim"]),
		"gated_dim": int(spec["gated_dim"]),
		"batch_size": int(spec["batch_size"]),
		"window_steps": int(spec["window_steps"]),
		"median_update_seconds": median,
		"mean_update_seconds": float(statistics.fmean(seconds)),
		"min_update_seconds": float(min(seconds)),
		"max_update_seconds": float(max(seconds)),
		"peak_cuda_allocated_bytes": peak_cuda,
		"all_losses_finite": all(math.isfinite(v) for v in losses),
		"all_grad_norms_finite": all(math.isfinite(v) for v in grad_norms),
	}
	return {"status": "complete", "summary": summary, "seconds": seconds}


def _child_main(argv: list[str]) -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--child-worker", action="store_true", required=True)
	parser.add_argument("--worker-spec", type=Path, required=True)
	parser.add_argument("--worker-output", type=Path, required=True)
	args = parser.parse_args(argv)
	spec = json.loads(args.worker_spec.read_text(encoding="utf-8"))
	try:
		result = run_worker(spec)
		args.worker_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
		print(f"{spec['worker_id']} {result['summary']['median_update_seconds']:.6f}s", flush=True)
		return 0
	except Exception as exc:
		args.worker_output.write_text(
			json.dumps({"status": "failed", "error": str(exc)}, indent=2) + "\n",
			encoding="utf-8",
		)
		raise


def _parse_args(argv: list[str]) -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--output-dir", type=Path, required=True)
	p.add_argument("--latent-dims", type=int, nargs="+", required=True)
	p.add_argument("--observed-dims", type=int, nargs="+", default=[20])
	p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
	p.add_argument("--window-steps", type=int, nargs="+", default=[10])
	p.add_argument("--methods", nargs="+", choices=METHODS, default=["kaffee", "kaffee_rrkf"])
	p.add_argument("--filter-ranks", type=int, nargs="+", default=[20])
	p.add_argument("--gated-fraction", type=float, default=0.5)
	p.add_argument("--burn-in-steps", type=int, default=2)
	p.add_argument("--warmups", type=int, default=20)
	p.add_argument("--repeats", type=int, default=20)
	p.add_argument("--process-repeats", type=int, default=5)
	p.add_argument("--learning-rate", type=float, default=0.005)
	p.add_argument("--gradient-clip", type=float, default=2.0)
	p.add_argument("--proc-std", type=float, default=0.02)
	p.add_argument("--obs-std", type=float, default=0.10)
	p.add_argument("--reset-model-between-updates", action="store_true")
	p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
	p.add_argument("--device", default="cuda")
	p.add_argument("--cpu-threads", type=int, default=2)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--smoke", action="store_true")
	return p.parse_args(argv)


def _specs(args: argparse.Namespace) -> list[dict[str, Any]]:
	out: list[dict[str, Any]] = []
	for m, n, b, t, method, proc in product(
		args.latent_dims,
		args.observed_dims,
		args.batch_sizes,
		args.window_steps,
		args.methods,
		range(int(args.process_repeats)),
	):
		if n > m:
			continue
		rrkf = method == "kaffee_rrkf"
		ranks = [r for r in args.filter_ranks if r <= m] if rrkf else [None]
		for rank in ranks:
			rank_tag = f"_r{int(rank):03d}" if rank is not None else ""
			out.append(
				{
					"worker_id": f"M{m}_N{n}_B{b}_T{t}_{method}{rank_tag}_p{proc:02d}",
					"method": method,
					"filter": "rrkf" if rrkf else "ekf",
					"filter_rank": rank,
					"process_repeat": proc,
					"seed": int(args.seed) + proc,
					"latent_dim": int(m),
					"observed_dim": int(n),
					"gated_dim": int(round(float(args.gated_fraction) * int(m))),
					"batch_size": int(b),
					"window_steps": int(t),
					"burn_in_steps": int(args.burn_in_steps),
					"warmups": int(args.warmups),
					"repeats": int(args.repeats),
					"learning_rate": float(args.learning_rate),
					"gradient_clip": float(args.gradient_clip),
					"device": str(args.device),
					"dtype": str(args.dtype),
					"cpu_threads": int(args.cpu_threads),
					"proc_std": float(args.proc_std),
					"obs_std": float(args.obs_std),
				}
			)
	return out


def _parent_main(argv: list[str]) -> int:
	args = _parse_args(argv)
	output_dir = args.output_dir.resolve()
	if output_dir.exists():
		raise FileExistsError(f"output directory already exists: {output_dir}")
	output_dir.mkdir(parents=True)
	(output_dir / "workers").mkdir()
	(output_dir / "logs").mkdir()
	specs = _specs(args)
	rows: list[dict[str, Any]] = []
	script = str(Path(__file__).resolve())
	for spec in specs:
		wid = spec["worker_id"]
		spec_path = output_dir / "workers" / f"{wid}.spec.json"
		result_path = output_dir / "workers" / f"{wid}.result.json"
		log_path = output_dir / "logs" / f"{wid}.log"
		spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
		env = dict(os.environ)
		env["KAFFEE_RRKF_FLOAT64"] = "0"
		for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
			env[name] = str(int(args.cpu_threads))
		with log_path.open("w", encoding="utf-8") as log:
			completed = subprocess.run(
				[sys.executable, script, "--child-worker", "--worker-spec", str(spec_path), "--worker-output", str(result_path)],
				cwd=str(Path(__file__).resolve().parents[1]),
				env=env,
				stdout=log,
				stderr=subprocess.STDOUT,
				check=False,
			)
		if completed.returncode != 0 or not result_path.is_file():
			(output_dir / "manifest.json").write_text(
				json.dumps({"status": "failed", "failed_worker_id": wid}, indent=2) + "\n",
				encoding="utf-8",
			)
			raise RuntimeError(f"worker failed: {wid} (see {log_path})")
		result = json.loads(result_path.read_text(encoding="utf-8"))
		if result.get("status") != "complete":
			(output_dir / "manifest.json").write_text(
				json.dumps({"status": "failed", "failed_worker_id": wid}, indent=2) + "\n",
				encoding="utf-8",
			)
			raise RuntimeError(f"worker failed: {wid} (see {log_path})")
		rows.append(result["summary"])
		print(f"{wid} {result['summary']['median_update_seconds']:.6f}s", flush=True)

	fields = list(rows[0].keys()) if rows else []
	with (output_dir / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fields)
		writer.writeheader()
		writer.writerows(rows)
	(output_dir / "manifest.json").write_text(
		json.dumps({"status": "complete", "n_workers": len(rows)}, indent=2) + "\n",
		encoding="utf-8",
	)
	print(f"wrote {output_dir}", flush=True)
	return 0


def main(argv: list[str] | None = None) -> int:
	active = list(sys.argv[1:] if argv is None else argv)
	if "--child-worker" in active:
		return _child_main(active)
	return _parent_main(active)


if __name__ == "__main__":
	raise SystemExit(main())
