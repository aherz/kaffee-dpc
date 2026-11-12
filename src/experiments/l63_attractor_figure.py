"""Lorenz-63 shell-off / shell-on attractor reconstruction figure, comparing methods.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_paper_style_spec = importlib.util.spec_from_file_location(
	"paper_figure_style",
	Path(__file__).resolve().parents[2] / "figures" / "paper_style.py",
)
_paper_style = importlib.util.module_from_spec(_paper_style_spec)
_paper_style_spec.loader.exec_module(_paper_style)
_L63_WIDTH = 11.2
_L63_POINTS = _paper_style.points(_L63_WIDTH)

from data.lorenz63 import standardize_raw_series, unstandardize_series
from experiments.l96_main_figure import TABLE1_PALETTE
from experiments.lorenz63_main import dataset_path, load_lorenz63_eval_bundle
from experiments.lorenz63_outputs import (
	_core_from_shell,
	_load_shell_checkpoint,
	_load_shell_model,
)
from models import rollout_free

DEFAULT_SEED = 6
REFERENCE_START = 2000
ROLLOUT_BURN = 30000
ROLLOUT_STEPS = 4000
FIGURE_SIZE = (_L63_WIDTH, 4.55)
FIGURE_DPI = 200

COLUMNS: tuple[tuple[str, str, str], ...] = (
	("truth", "True", "Truth"),
	("frozen_core_shell", "Frozen-core", "Prior"),
	("kaffee", "KAFFEE", "KAFFEE"),
	("filtered_no_propagation", "No-propagation", "No-transport"),
	("open_loop_tied", "OL-tied", "OL-tied"),
	("open_loop_no_propagation", "OL-no-prop.", "OL-no-transport"),
)

L63_ATTRACTOR_STYLE = _paper_style.rc_params(_L63_WIDTH)


@dataclass(frozen=True)
class L63AttractorPaths:
	repo_root: Path
	output_root: Path
	rollouts_npz: Path


def resolve_paths(repo_root: Path, *, output_root: Path) -> L63AttractorPaths:
	root = Path(output_root)
	if not root.is_absolute():
		root = Path(repo_root) / root
	root = root.resolve()
	return L63AttractorPaths(
		repo_root=Path(repo_root).resolve(),
		output_root=root,
		rollouts_npz=root / "rollouts.npz",
	)


def _window(series: np.ndarray, start: int, steps: int) -> np.ndarray:
	series = np.asarray(series, dtype=np.float64)
	stop = int(start) + int(steps)
	if stop > int(series.shape[0]):
		raise ValueError(f"Need {stop} steps, trajectory has {series.shape[0]}")
	return series[int(start) : stop]


def _model_rollouts(
	shell,
	*,
	x0_std: np.ndarray,
	standardizer: dict[str, Any],
	burn: int,
	steps: int,
	seed: int,
	device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
	core = _core_from_shell(shell, device)
	x0 = torch.as_tensor(x0_std.reshape(1, -1), dtype=torch.float32, device=device)
	core_std = (
		rollout_free(core, x0, steps=int(burn) + int(steps))[0, int(burn) :, : int(shell.observed_dim)]
		.detach()
		.cpu()
		.numpy()
	)
	shell_std = shell.rollout_observation_sequence(
		x0_std,
		steps=int(burn) + int(steps),
		seed=int(seed),
		sample_process_noise=True,
		sample_obs_noise=True,
	)[int(burn) :]
	return (
		unstandardize_series(core_std, standardizer),
		unstandardize_series(shell_std, standardizer),
	)


def export_rollouts(
	config: dict[str, Any],
	paths: L63AttractorPaths,
	*,
	core_root: Path,
	shell_root: Path,
	seed: int = DEFAULT_SEED,
	device: torch.device,
	skip_existing: bool = False,
) -> Path:
	"""Rollout the reference and the five shells for one core seed."""
	if bool(skip_existing) and paths.rollouts_npz.is_file():
		return paths.rollouts_npz
	evaluation = load_lorenz63_eval_bundle(config)
	dataset = np.load(dataset_path(config))
	standardizer = {
		"mean": np.asarray(dataset["mean"], dtype=np.float64),
		"std": np.asarray(dataset["std"], dtype=np.float64),
	}
	dataset.close()
	core_ref = _window(evaluation["test_core_raw"], REFERENCE_START, ROLLOUT_STEPS)
	shell_ref = _window(evaluation["stochastic_obs_raw"], REFERENCE_START, ROLLOUT_STEPS)
	x0_std = standardize_raw_series(core_ref[:1], standardizer)[0].astype(np.float32)

	core_rows = [core_ref]
	shell_rows = [shell_ref]
	for key, _title, _color_key in COLUMNS[1:]:
		checkpoint = _load_shell_checkpoint(shell_root, int(seed), key)
		shell = _load_shell_model(checkpoint, baseline=key, config=config, device=device)
		core_traj, shell_traj = _model_rollouts(
			shell,
			x0_std=x0_std,
			standardizer=standardizer,
			burn=ROLLOUT_BURN,
			steps=ROLLOUT_STEPS,
			seed=int(seed) + 17,
			device=device,
		)
		core_rows.append(core_traj)
		shell_rows.append(shell_traj)
		del shell

	paths.output_root.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		paths.rollouts_npz,
		core=np.stack(core_rows, axis=0),
		shell=np.stack(shell_rows, axis=0),
		columns=np.asarray([key for key, _title, _color in COLUMNS]),
	)
	return paths.rollouts_npz


def _shared_limits(core: np.ndarray, shell: np.ndarray) -> tuple[tuple[float, float], ...]:
	"""Axis limits from the reference data for a shared scale."""
	truth = np.concatenate([core[0], shell[0]], axis=0)
	limits = []
	for dim in range(3):
		lo = float(np.min(truth[:, dim]))
		hi = float(np.max(truth[:, dim]))
		pad = 0.08 * (hi - lo if hi > lo else 1.0)
		limits.append((lo - pad, hi + pad))
	return tuple(limits)


def _style_axis(ax, limits: tuple[tuple[float, float], ...]) -> None:
	ax.set_xlim(*limits[0])
	ax.set_ylim(*limits[1])
	ax.set_zlim(*limits[2])
	ax.view_init(elev=22, azim=-60)
	for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
		axis.set_ticklabels([])
		axis.line.set_linewidth(0.6)
		axis.pane.set_alpha(0.08)
	ax.grid(False)
	try:
		ax.set_box_aspect((1, 1, 1))
	except AttributeError:
		pass


def plot_l63_attractor_figure(
	paths: L63AttractorPaths,
	*,
	figure_dir: Path,
	dpi: int = FIGURE_DPI,
) -> Path:
	import matplotlib.pyplot as plt

	rollouts = np.load(paths.rollouts_npz)
	core = np.asarray(rollouts["core"], dtype=np.float64)
	shell = np.asarray(rollouts["shell"], dtype=np.float64)
	if core.shape[0] != len(COLUMNS) or shell.shape != core.shape:
		raise ValueError(f"Expected rollouts shaped ({len(COLUMNS)}, T, 3), got {core.shape} / {shell.shape}")
	limits = _shared_limits(core, shell)
	figure_dir = Path(figure_dir)
	figure_dir.mkdir(parents=True, exist_ok=True)

	with plt.rc_context(L63_ATTRACTOR_STYLE):
		fig = plt.figure(figsize=FIGURE_SIZE)
		grid = fig.add_gridspec(2, len(COLUMNS), left=0.055, right=0.992, bottom=0.03, top=0.86, wspace=0.12, hspace=0.28)
		for col, (_key, title, color_key) in enumerate(COLUMNS):
			color = TABLE1_PALETTE[color_key]
			for row, series in enumerate((core[col], shell[col])):
				ax = fig.add_subplot(grid[row, col], projection="3d")
				ax.plot(
					series[:, 0],
					series[:, 1],
					series[:, 2],
					color=color,
					lw=0.7 if row == 0 else 0.45,
					alpha=0.95 if row == 0 else 0.8,
					rasterized=True,
				)
				span = series.max(axis=0) - series.min(axis=0)
				if float(np.max(span)) < 1.0:
					ax.scatter(
						series[-1:, 0],
						series[-1:, 1],
						series[-1:, 2],
						color=color,
						s=18,
						depthshade=False,
					)
				_style_axis(ax, limits)
				if row == 0:
		
					shown = "No-\npropagation" if title == "No-propagation" else title
					ax.set_title(shown, pad=1)
					ax.text2D(
						0.0,
						1.16,
						f"({chr(ord('a') + col)})",
						transform=ax.transAxes,
						ha="left",
						va="bottom",
						fontsize=_L63_POINTS["panel"],
						fontweight="bold",
						clip_on=False,
					)
		fig.text(0.012, 0.66, "Core", rotation=90, va="center", ha="center", fontsize=_L63_POINTS["label"])
		fig.text(0.012, 0.28, "Shell-on", rotation=90, va="center", ha="center", fontsize=_L63_POINTS["label"])
		out_pdf = figure_dir / "l63_attractor_recon.pdf"
		fig.savefig(out_pdf, dpi=int(dpi), bbox_inches="tight", pad_inches=0.01)
		fig.savefig(figure_dir / "l63_attractor_recon.png", dpi=int(dpi), bbox_inches="tight", pad_inches=0.01)
		plt.close(fig)
	return out_pdf


__all__ = [
	"COLUMNS",
	"DEFAULT_SEED",
	"L63AttractorPaths",
	"export_rollouts",
	"plot_l63_attractor_figure",
	"resolve_paths",
]
