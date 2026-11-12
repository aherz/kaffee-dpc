#!/usr/bin/env python3
"""Deterministic AL-RNN core pretraining on Lorenz-96."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


CORE_CHECKPOINT_NAME = "alrnn_core.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lorenz96_main.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/lorenz96_main_cores"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batches-per-epoch", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_device(device_arg: str) -> torch.device:
    text = str(device_arg).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cpu":
        return torch.device("cpu")
    if text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device(text)
    raise ValueError(f"Unsupported device specification: {device_arg!r}")


def _set_cpu_threads(n_threads: int) -> None:
    n_threads = int(max(1, n_threads))
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _seed_list(config: dict[str, Any], *, smoke: bool, max_seeds: int | None, seed: int | None) -> list[int]:
    if seed is not None:
        return [int(seed)]
    seed_cfg = dict(config["seed_sweep"])
    count = int(seed_cfg["count"])
    if smoke:
        count = min(count, 2)
    if max_seeds is not None:
        count = min(count, int(max_seeds))
    start_seed = int(seed_cfg.get("start_seed", 0))
    return list(range(start_seed, start_seed + count))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    output_root = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = _load_json(config_path)
    device = _resolve_device(args.device)
    _set_cpu_threads(int(args.cpu_threads))

    sys_path = str(repo_root / "src")
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from experiments import prepare_lorenz96_main_bundle, train_lorenz96_core 

    prepared = prepare_lorenz96_main_bundle(config, smoke=bool(args.smoke))
    dataset = prepared["dataset"]
    protocol = prepared["protocol"]
    seed_list = _seed_list(config, smoke=bool(args.smoke), max_seeds=args.max_seeds, seed=args.seed)

    runs: list[dict[str, Any]] = []
    for seed in seed_list:
        run_dir = output_root / f"seed{int(seed):02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model, train_info = train_lorenz96_core(
            dataset,
            protocol,
            seed=int(seed),
            device=device,
            max_epochs=int(args.epochs) if args.epochs is not None else (4 if args.smoke else None),
            max_batches_per_epoch=int(args.batches_per_epoch) if args.batches_per_epoch is not None else (2 if args.smoke else None),
            eval_every=int(args.eval_every) if args.eval_every is not None else (1 if args.smoke else None),
            free_rollout_horizon=64 if args.smoke else 256,
        )
        checkpoint = {
            "state_dict": model.state_dict(),
            "M": int(protocol["model"]["latent_dim"]),
            "P": int(protocol["model"]["gated_dim"]),
            "N": int(protocol["model"]["observed_dim"]),
            "mean": dataset["standardizer"]["mean"].reshape(1, -1),
            "std": dataset["standardizer"]["std"].reshape(1, -1),
            "seed": int(seed),
            "data_seed": int(protocol["seed_sweep"].get("data_seed", 0)),
            "teacher_forcing_interval": int(protocol["model"]["teacher_forcing_interval"]),
            "selection_metric": train_info["selection_metric"],
            "selection_value": float(train_info["selection_value"]),
            "selection_epoch": int(train_info["selection_epoch"]),
            "config_json": json.dumps(config, sort_keys=True),
        }
        torch.save(checkpoint, run_dir / CORE_CHECKPOINT_NAME)
        summary = {
            "seed": int(seed),
            "smoke": bool(args.smoke),
            "device": str(device),
            "config": config,
            "protocol": protocol,
            "selection_metric": train_info["selection_metric"],
            "selection_value": float(train_info["selection_value"]),
            "selection_epoch": int(train_info["selection_epoch"]),
            "final_metrics": train_info["final_metrics"],
            "epochs": int(train_info["epochs"]),
            "batches_per_epoch": int(train_info["batches_per_epoch"]),
            "eval_every": int(train_info["eval_every"]),
            "loss_tail": [float(x) for x in train_info["loss_tail"]],
        }
        _write_json(run_dir / "summary.json", summary)
        _write_history_csv(run_dir / "history.csv", list(train_info["history"]))
        runs.append(summary)

    _write_json(
        output_root / "manifest.json",
        {
            "config": str(config_path),
            "output_dir": str(output_root),
            "checkpoint_name": CORE_CHECKPOINT_NAME,
            "device": str(device),
            "smoke": bool(args.smoke),
            "seed_list": seed_list,
            "runs": runs,
        },
    )
    print(f"wrote {output_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())