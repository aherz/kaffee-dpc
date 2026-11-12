"""Generate artifacts and plotting helpers for the main DynaMix figure.

Writes figure inputs under `figures/figures_inputs/dynamix_main_figure/`
and the rendered figure under `figures/dynamix_main/`.
`notebooks/dynamix_main_figure.ipynb` renders from those artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

_paper_style_spec = importlib.util.spec_from_file_location(
	"paper_figure_style",
	Path(__file__).resolve().parents[2] / "figures" / "paper_style.py",
)
_paper_style = importlib.util.module_from_spec(_paper_style_spec)
_paper_style_spec.loader.exec_module(_paper_style)
ONE_ROW_FIGSIZE = _paper_style.ONE_ROW_FIGSIZE
paper_points = _paper_style.points
paper_rc = _paper_style.rc_params

from data import (
	DYNAMIX_ODE_SPECS,
	add_dynamix_standardized_obs_noise,
	build_dynamix_window_dataset,
	generate_dynamix_ode_system,
)
from experiments.dynamix_eval import rollout_model_raw
from experiments.dynamix_outputs import (
	DYNAMIX_BASELINE_KEYS,
	load_results_from_checkpoints,
	matched_nonfinite_seed_exclusions,
	robust_se_median,
	seed_is_excluded,
)
from experiments.dynamix_pipeline import _checkpoint_path, _result_path
from filtering import build_identity_observation_matrix, kalman_predict, kalman_update
from models import DynaMixShell, compute_mixture_jacobian, load_dynamix_base, load_dynamix_shell_checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[2]

PANEL_A_METHODS: tuple[tuple[str, str, str, str], ...] = (
	("frozen_core_shell", "Frozen-core", "#9aa3af", "v"),
	("open_loop_tied", "Open-loop tied", "#D55E00", "^"),
	("kaffee_dynamix", "KAFFEE-DynaMix", "#009E73", "o"),
)

FILTERING_ROLLOUT_DEFAULTS = {
	"system": "lorenz63",
	"seed": 8,
	"horizon": 150,
	"obs_cadence": 10,
	"dim": 0,
	"context_length": 2000,
	"total_t": 5000,
	"sigma_noise": 0.1,
	"window_length": 56,
	"stride": 56,
	"covariance_burn_in_steps": 8,
	"core_baseline": "frozen",
	"shared_start": "ground_truth",
}

DEFAULT_FIGSIZE = ONE_ROW_FIGSIZE
DEFAULT_WIDTH_RATIOS = (0.52, 1.88)
PAPER_POINTS = paper_points(DEFAULT_FIGSIZE[0])

DYNAMIX_MAIN_STYLE = {
	**paper_rc(DEFAULT_FIGSIZE[0]),
	"grid.alpha": 0.22,
}

DYN_TRUE_COLOR = "#111827"
DYN_BASE_COLOR = "#9aa3af"
DYN_KAFFEE_COLOR = "#009E73"
DYN_NOISE_COLOR = "#9aa3af"


@dataclass(frozen=True)
class DynamixMainFigurePaths:
	repo_root: Path
	output_root: Path
	dpc_panel_csv: Path
	filtering_npz: Path


def resolve_paths(repo_root: Path | str | None = None, output_root: Path | str | None = None) -> DynamixMainFigurePaths:
	root = Path(repo_root or _REPO_ROOT).resolve()
	out = Path(output_root or root / "figures" / "figures_inputs" / "dynamix_main_figure").resolve()
	return DynamixMainFigurePaths(
		repo_root=root,
		output_root=out,
		dpc_panel_csv=out / "dpc_panel_rows.csv",
		filtering_npz=out / "filtering_rollout.npz",
	)


def _finite_float(value: Any) -> float | None:
	try:
		out = float(value)
	except (TypeError, ValueError):
		return None
	return out if math.isfinite(out) else None


def collect_dpc_panel_rows(
	adaptation_root: Path,
	*,
	systems: list[str],
	seed_exclusions: dict[str, frozenset[int]] | None = None,
) -> pd.DataFrame:
	"""Load panel-(a) rows from adaptation result JSONs."""
	rows: list[dict[str, Any]] = []
	root = Path(adaptation_root).resolve()
	for baseline_key, label, _color, _marker in PANEL_A_METHODS:
		result_dir = root / "results" / baseline_key
		if not result_dir.is_dir():
			continue
		for fp in sorted(result_dir.glob(f"result_{baseline_key}_*_seed*.json")):
			payload = json.loads(fp.read_text(encoding="utf-8"))
			system = str(payload.get("system"))
			if system not in systems:
				continue
			seed = int(payload.get("seed"))
			if seed_is_excluded(system, seed, seed_exclusions):
				continue
			block = payload.get(baseline_key) or {}
			base = payload.get("baseline") or {}
			base_dstsp = _finite_float(base.get("D_stsp"))
			dstsp = _finite_float(block.get("D_stsp"))
			nll = _finite_float(block.get("held_nll"))
			rel_d = (
				dstsp / abs(base_dstsp)
				if base_dstsp is not None and dstsp is not None and abs(base_dstsp) > 0
				else None
			)
			rows.append(
				{
					"method_key": baseline_key,
					"method": label,
					"system": system,
					"seed": seed,
					"nll": nll,
					"relD": rel_d,
				}
			)
	if not rows:
		raise FileNotFoundError(f"No DynaMix figure panel-(a) rows found under {root}")
	return pd.DataFrame(rows)


def system_median_dpc_panel(
	panel_df: pd.DataFrame,
	*,
	required_seed_count: int | None = None,
) -> pd.DataFrame:
	"""Median NLL and occupancy within each system. Drop systems with any non-finite seed.

	When ``required_seed_count`` is omitted, keep any system whose remaining
	seeds are all finite. That lets Dadras use 17 matched seeds while the
	other systems stay at 20.
	"""
	if panel_df.empty:
		return panel_df.copy()
	rows: list[dict[str, Any]] = []
	grouped = panel_df.groupby(["method_key", "method", "system"], sort=True)
	for (method_key, method, system), sub in grouped:
		nlls = sub["nll"].to_numpy(dtype=np.float64)
		rels = sub["relD"].to_numpy(dtype=np.float64)
		finite = np.isfinite(nlls) & np.isfinite(rels)
		n_rows = int(len(sub))
		if int(finite.sum()) != n_rows:
			continue
		if required_seed_count is not None and n_rows != int(required_seed_count):
			continue
		rows.append(
			{
				"method_key": method_key,
				"method": method,
				"system": system,
				"n_seeds": n_rows,
				"nll": float(np.median(nlls)),
				"relD": float(np.median(rels)),
			}
		)
	return pd.DataFrame(rows)


def export_dpc_panel_rows(
	config: dict[str, Any],
	paths: DynamixMainFigurePaths,
	*,
	adaptation_root: Path,
	skip_existing: bool = False,
) -> Path:
	paths.output_root.mkdir(parents=True, exist_ok=True)
	if bool(skip_existing) and paths.dpc_panel_csv.is_file():
		return paths.dpc_panel_csv
	adapted_keys = [
		str(k) for k in config.get("baselines", DYNAMIX_BASELINE_KEYS) if str(k) != "dynamix_base"
	]
	loaded = load_results_from_checkpoints(
		Path(adaptation_root),
		systems=list(config["systems"]),
		seed_filter=None,
		baseline_keys=adapted_keys,
	)
	df = collect_dpc_panel_rows(
		adaptation_root,
		systems=list(config["systems"]),
		seed_exclusions=matched_nonfinite_seed_exclusions(loaded["per_seed_rows"]),
	)
	df.to_csv(paths.dpc_panel_csv, index=False)
	return paths.dpc_panel_csv


@torch.no_grad()
def _sparse_ekf_rollout_raw(
	shell: DynaMixShell,
	*,
	z_start: torch.Tensor,
	context: torch.Tensor,
	obs_std: torch.Tensor,
	horizon: int,
	mean: np.ndarray,
	std: np.ndarray,
	cadence: int,
	include_initial: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	model = shell.model
	device = z_start.device
	dtype = z_start.dtype
	m_dim = int(model.M)
	n_dim = int(model.N)
	q_mat = shell.Q
	r_mat = shell.R
	b_mat = build_identity_observation_matrix(n_dim, m_dim, device=device, dtype=dtype)
	if z_start.ndim == 2:
		m_state = z_start[:, 0].clone()
	else:
		m_state = z_start.clone()
	p_mat = q_mat.clone()
	precomputed = model.precompute_cnn(context)
	means: list[np.ndarray] = []
	sigmas: list[np.ndarray] = []
	update_mask: list[bool] = []

	def _emit(do_update: bool) -> None:
		y_raw = m_state[:n_dim].detach().cpu().numpy().astype(np.float64) * std + mean
		s_raw = (
			p_mat[:n_dim, :n_dim].diagonal().clamp_min(1.0e-10).sqrt().detach().cpu().numpy().astype(np.float64) * std
		)
		means.append(y_raw)
		sigmas.append(s_raw)
		update_mask.append(bool(do_update))

	if bool(include_initial):
		_emit(False)
		t0 = 1
	else:
		t0 = 0
	for t in range(t0, int(horizon)):
		z_col = m_state.unsqueeze(1)
		m_pred_col = model.step(z_col, context, precomputed_cnn=precomputed)
		m_pred = m_pred_col[:, 0]
		f_t = compute_mixture_jacobian(model, z_col, context, precomputed)
		m_state, p_mat = kalman_predict(
			m_state,
			p_mat,
			z_mean_pred=m_pred,
			F_t=f_t,
			Gamma=q_mat,
			covariance_transport="jacobian",
		)
		p_mat = 0.5 * (p_mat + p_mat.T)
		do_update = (t % int(cadence) == 0)
		if do_update:
			m_state, p_mat, _ = kalman_update(m_state, p_mat, obs_std[t], B=b_mat, Lambda=r_mat)
			p_mat = 0.5 * (p_mat + p_mat.T)
		_emit(do_update)
	return (
		np.asarray(means, dtype=np.float64),
		np.asarray(sigmas, dtype=np.float64),
		np.asarray(update_mask, dtype=bool),
	)


def _filtering_checkpoint(repo_root: Path, adaptation_root: Path, system: str, seed: int) -> Path:
	root = (repo_root / adaptation_root).resolve()
	method = "kaffee_dynamix"
	result = _result_path(root, method, system, seed)
	if result.is_file():
		recorded = json.loads(result.read_text(encoding="utf-8")).get("checkpoint_path")
		if recorded:
			return (result.parent / recorded).resolve()

	checkpoint = _checkpoint_path(root, method, system, seed)
	if root == (repo_root / "results/dynamix_main/adaptations").resolve():
		# The original paper scores were archived separately from their checkpoints.
		if not (root / method).is_dir() and not (root / "checkpoints" / method).is_dir():
			return _checkpoint_path(repo_root / "models/dynamix_main_shells", method, system, seed)
	return checkpoint


def build_filtering_rollout(
	config: dict[str, Any],
	paths: DynamixMainFigurePaths,
	*,
	adaptation_root: Path,
	device: torch.device,
	rollout_options: dict[str, Any] | None = None,
	skip_existing: bool = False,
) -> Path:
	paths.output_root.mkdir(parents=True, exist_ok=True)
	if bool(skip_existing) and paths.filtering_npz.is_file():
		return paths.filtering_npz

	knobs = {**FILTERING_ROLLOUT_DEFAULTS, **(rollout_options or {})}
	system = str(knobs["system"])
	seed = int(knobs["seed"])
	context_length = int(knobs["context_length"])
	total_t = int(knobs["total_t"])
	horizon = int(knobs["horizon"])
	cadence = int(knobs["obs_cadence"])
	dim = int(knobs["dim"])
	sigma_noise = float(knobs["sigma_noise"])
	window_length = int(knobs["window_length"])
	stride = int(knobs["stride"])
	cov_burn = int(knobs["covariance_burn_in_steps"])

	x_clean, dt = generate_dynamix_ode_system(system, total_t, seed=seed)
	x_noisy = add_dynamix_standardized_obs_noise(
		x_clean,
		sigma=sigma_noise,
		seed=seed,
		context_length=context_length,
	)
	base_model = load_dynamix_base(device=device, repo_root=paths.repo_root)
	dataset = build_dynamix_window_dataset(
		x_noisy,
		base_model,
		context_length=context_length,
		window_length=window_length,
		stride=stride,
		device=device,
		covariance_burn_in_steps=cov_burn,
	)
	mean = dataset.mean.detach().cpu().numpy().astype(np.float64)
	std = dataset.std.detach().cpu().numpy().astype(np.float64)
	context_raw = dataset.raw_train[:context_length]

	obs_std = torch.as_tensor(
		((x_noisy[context_length : context_length + horizon] - mean) / std).astype(np.float32),
		device=device,
	)
	truth = x_clean[context_length : context_length + horizon]
	obs_raw = x_noisy[context_length : context_length + horizon]
	shared_start = str(knobs.get("shared_start", "ground_truth")).strip().lower()
	if shared_start not in {"ground_truth", "observation"}:
		raise ValueError(f"shared_start must be ground_truth or observation, got {shared_start!r}")

	ckpt = _filtering_checkpoint(paths.repo_root, adaptation_root, system, seed)
	if not ckpt.is_file():
		raise FileNotFoundError(f"Missing KAFFEE-DynaMix checkpoint for filtering panel: {ckpt}")
	adapted_model = load_dynamix_base(device=device, repo_root=paths.repo_root)
	shell = load_dynamix_shell_checkpoint(ckpt, model=adapted_model, trainable_shell=False).to(device)
	shell.eval()

	if shared_start == "ground_truth":
		x0_std = torch.as_tensor(((truth[0] - mean) / std).astype(np.float32), device=device)
		z_start = shell.latent_column_from_observation(x0_std)
		include_initial = True
	else:
		z_start = shell.latent_column_from_observation(obs_std[0])
		include_initial = False

	base_rollout = rollout_model_raw(
		base_model,
		context_raw=context_raw,
		mean=mean,
		std=std,
		horizon=horizon,
		device=device,
		z_start=z_start if include_initial else None,
		include_initial=include_initial,
	)
	base_pred = np.asarray(base_rollout["forecast"], dtype=np.float64)
	adapted_rollout = rollout_model_raw(
		shell.model,
		context_raw=context_raw,
		mean=mean,
		std=std,
		horizon=horizon,
		device=device,
		z_start=z_start if include_initial else None,
		include_initial=include_initial,
	)
	adapted_core_pred = np.asarray(adapted_rollout["forecast"], dtype=np.float64)
	filt_mean, filt_sig, update_mask = _sparse_ekf_rollout_raw(
		shell,
		z_start=z_start,
		context=dataset.context,
		obs_std=obs_std,
		horizon=horizon,
		mean=mean,
		std=std,
		cadence=cadence,
		include_initial=include_initial,
	)

	lambda1_ref = float(DYNAMIX_ODE_SPECS[str(system).strip().lower()].lambda1_ref)
	lyapunov_time = np.arange(horizon, dtype=np.float64) * float(dt) * max(lambda1_ref, 1.0e-8)

	np.savez(
		paths.filtering_npz,
		system=system,
		seed=seed,
		dim=dim,
		dt=float(dt),
		lambda1_ref=float(lambda1_ref),
		horizon=horizon,
		obs_cadence=cadence,
		core_baseline=str(knobs["core_baseline"]),
		truth=truth,
		obs_raw=obs_raw,
		base_pred=base_pred,
		adapted_core_pred=adapted_core_pred,
		filt_mean=filt_mean,
		filt_sig=filt_sig,
		update_mask=update_mask,
		lyapunov_time=lyapunov_time,
		shared_start=np.asarray(shared_start),
	)
	return paths.filtering_npz


def run_dynamix_main_figure_data(
	config: dict[str, Any],
	*,
	paths: DynamixMainFigurePaths,
	adaptation_root: Path,
	device: torch.device,
	rollout_options: dict[str, Any] | None = None,
	skip_existing: bool = False,
	skip_rollout: bool = False,
) -> dict[str, Path]:
	paths.output_root.mkdir(parents=True, exist_ok=True)
	dpc_csv = export_dpc_panel_rows(
		config,
		paths,
		adaptation_root=adaptation_root,
		skip_existing=bool(skip_existing),
	)
	rollout_npz = paths.filtering_npz
	if not bool(skip_rollout):
		rollout_npz = build_filtering_rollout(
			config,
			paths,
			adaptation_root=adaptation_root,
			device=device,
			rollout_options=rollout_options,
			skip_existing=bool(skip_existing),
		)
	return {"dpc_panel_csv": dpc_csv, "filtering_npz": rollout_npz}


@dataclass(frozen=True)
class DynamixMainFigureData:
	dpc_panel: pd.DataFrame
	rollout: dict[str, Any]


def load_dynamix_main_figure_data(paths: DynamixMainFigurePaths) -> DynamixMainFigureData:
	rollout_npz = np.load(paths.filtering_npz, allow_pickle=False)
	rollout = {key: rollout_npz[key] for key in rollout_npz.files}
	return DynamixMainFigureData(
		dpc_panel=pd.read_csv(paths.dpc_panel_csv),
		rollout=rollout,
	)


def plot_dpc_panel(ax, panel_df: pd.DataFrame, *, required_seed_count: int | None = None) -> None:
	span_x: list[float] = []
	span_y: list[float] = [1.0]
	system_df = panel_df
	if "seed" in panel_df.columns:
		system_df = system_median_dpc_panel(panel_df, required_seed_count=required_seed_count)
	for _baseline_key, label, color, marker in PANEL_A_METHODS:
		sub = system_df.loc[system_df["method"] == label]
		if sub.empty:
			sub = system_df.loc[system_df["method_key"] == _baseline_key]
		nlls = sub["nll"].to_numpy(dtype=np.float64)
		vals = sub["relD"].to_numpy(dtype=np.float64)
		finite = np.isfinite(nlls) & np.isfinite(vals)
		nlls = nlls[finite]
		vals = vals[finite]
		if nlls.size == 0:
			continue
		x_med = float(np.median(nlls))
		y_med = float(np.median(vals))
		x_se = robust_se_median(nlls)
		y_se = robust_se_median(vals)
		span_x.append(x_med)
		span_y.append(y_med)
		if x_se is not None:
			span_x.extend([x_med - float(x_se), x_med + float(x_se)])
		if y_se is not None:
			span_y.extend([y_med - float(y_se), y_med + float(y_se)])
		ax.errorbar(
			x_med,
			y_med,
			xerr=None if x_se is None else float(x_se),
			yerr=None if y_se is None else float(y_se),
			fmt="none",
			ecolor=color,
			elinewidth=1.05,
			capsize=2.4,
			capthick=0.9,
			alpha=0.9,
			zorder=4,
		)
		ax.plot(
			x_med,
			y_med,
			marker,
			ms=7.0,
			color=color,
			mec="black",
			mew=0.45,
			alpha=0.96,
			zorder=5,
			label=label,
		)
	if span_x:
		span = max(span_x) - min(span_x)
		pad_x = max(0.04, 0.25 * span if span > 0 else 0.04)
		ax.set_xlim(min(span_x) - pad_x, max(span_x) + pad_x)
	ylo, yhi = min(span_y), max(span_y)
	pad_y = max(0.04, 0.12 * (yhi - ylo) if yhi > ylo else 0.04)
	ylo_ax = ylo - pad_y
	legend_clearance = 0.58
	yhi_ax = ylo_ax + (yhi - ylo_ax) / legend_clearance
	ax.set_ylim(ylo_ax, yhi_ax)
	ax.axhline(1.0, color=DYN_NOISE_COLOR, ls="--", lw=1.0, alpha=0.5, zorder=0)
	ax.set_box_aspect(1.0)
	ax.set_xlabel("NLL")
	ax.set_ylabel(r"$D_{\mathrm{stsp}}^{\mathrm{core}}$ (rel. to baseline)")
	ax.text(
		0.0,
		1.02,
		"(a)",
		transform=ax.transAxes,
		ha="left",
		va="bottom",
		fontsize=PAPER_POINTS["panel"],
		fontweight="bold",
	)
	ax.legend(
		loc="upper right",
		bbox_to_anchor=(0.985, 0.985),
		ncol=1,
		frameon=False,
		fontsize=PAPER_POINTS["legend"],
		handlelength=0.95,
		handletextpad=0.32,
		labelspacing=0.45,
		borderaxespad=0.0,
	)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)


def plot_filtering_panel(ax, rollout: dict[str, Any]) -> None:
	dim = int(rollout["dim"])
	t = np.asarray(rollout["lyapunov_time"], dtype=np.float64)
	truth = np.asarray(rollout["truth"], dtype=np.float64)
	obs_raw = np.asarray(rollout["obs_raw"], dtype=np.float64)
	base_pred = np.asarray(rollout["base_pred"], dtype=np.float64)
	adapted_core_pred = np.asarray(rollout["adapted_core_pred"], dtype=np.float64)
	filt_mean = np.asarray(rollout["filt_mean"], dtype=np.float64)
	filt_sig = np.asarray(rollout["filt_sig"], dtype=np.float64)
	update_mask = np.asarray(rollout["update_mask"], dtype=bool)
	core_baseline = str(rollout.get("core_baseline", "frozen"))

	baseline_series = [truth[:, [dim]], filt_mean[:, [dim]]]
	if core_baseline in {"frozen", "both"}:
		baseline_series.append(base_pred[:, [dim]])
	if core_baseline in {"adapted", "both"}:
		baseline_series.append(adapted_core_pred[:, [dim]])
	y_stack = np.concatenate(baseline_series, axis=0).ravel()
	ylo, yhi = np.nanpercentile(y_stack, [1, 99])
	ypad = 0.12 * (yhi - ylo)

	ax.fill_between(
		t,
		filt_mean[:, dim] - 2.0 * filt_sig[:, dim],
		filt_mean[:, dim] + 2.0 * filt_sig[:, dim],
		color=DYN_KAFFEE_COLOR,
		alpha=0.20,
		lw=0,
		zorder=2,
	)
	ax.plot(t, truth[:, dim], color=DYN_TRUE_COLOR, lw=1.85, ls="--", label="truth", zorder=5)
	if core_baseline in {"frozen", "both"}:
		ax.plot(t, base_pred[:, dim], color=DYN_BASE_COLOR, lw=1.6, alpha=0.95, label="DynaMix", zorder=3)
	if core_baseline in {"adapted", "both"}:
		ax.plot(
			t,
			adapted_core_pred[:, dim],
			color="#D55E00",
			lw=1.55,
			alpha=0.82,
			ls=":",
			label="KAFFEE-DynaMix",
			zorder=4,
		)
	ax.plot(t, filt_mean[:, dim], color=DYN_KAFFEE_COLOR, lw=1.95, label="KAFFEE-DynaMix", zorder=6)
	upd = np.where(update_mask)[0]
	ax.scatter(
		t[upd],
		obs_raw[upd, dim],
		s=10,
		facecolors="white",
		edgecolors="black",
		linewidths=0.55,
		alpha=0.9,
		zorder=7,
		label="noisy obs.",
	)
	ax.set_xlim(0.0, float(t[-1]))
	ax.set_ylim(ylo - ypad, yhi + ypad)
	ax.set_xlabel(r"horizon ($\tau_\lambda$)")
	ax.set_ylabel(r"$x$")
	ax.text(
		0.0,
		1.02,
		"(b)",
		transform=ax.transAxes,
		ha="left",
		va="bottom",
		fontsize=PAPER_POINTS["panel"],
		fontweight="bold",
	)
	ax.legend(loc="upper right", frameon=False, fontsize=PAPER_POINTS["legend"], handlelength=1.25, borderpad=0.12)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)


def make_dynamix_main_figure_layout(
	*,
	figsize: tuple[float, float] = DEFAULT_FIGSIZE,
	width_ratios: tuple[float, float] = DEFAULT_WIDTH_RATIOS,
):
	import matplotlib.pyplot as plt

	fig, (ax_a, ax_b) = plt.subplots(
		1,
		2,
		figsize=figsize,
		gridspec_kw={"width_ratios": list(width_ratios)},
	)
	fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.88, wspace=0.22)
	return fig, {"dpc": ax_a, "filtering": ax_b}


def draw_dynamix_main_figure(fig, axes: dict[str, Any], data: DynamixMainFigureData) -> None:
	import matplotlib.pyplot as plt

	with plt.rc_context(DYNAMIX_MAIN_STYLE):
		plot_dpc_panel(axes["dpc"], data.dpc_panel)
		plot_filtering_panel(axes["filtering"], data.rollout)


def save_dynamix_main_figure(fig, figure_dir: Path, *, dpi: int = 360) -> Path:
	figure_dir = Path(figure_dir).resolve()
	figure_dir.mkdir(parents=True, exist_ok=True)
	out_pdf = figure_dir / "dynamix_context_adaptation.pdf"
	out_png = figure_dir / "dynamix_context_adaptation.png"
	fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
	fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", pad_inches=0.01)
	return out_pdf


def plot_dynamix_main_figure(
	paths: DynamixMainFigurePaths,
	*,
	figure_dir: Path,
	figsize: tuple[float, float] = DEFAULT_FIGSIZE,
	dpi: int = 360,
) -> Path:
	data = load_dynamix_main_figure_data(paths)
	fig, axes = make_dynamix_main_figure_layout(figsize=figsize)
	draw_dynamix_main_figure(fig, axes, data)
	return save_dynamix_main_figure(fig, figure_dir, dpi=dpi)


__all__ = [
	"DYNAMIX_MAIN_STYLE",
	"DynamixMainFigureData",
	"DynamixMainFigurePaths",
	"build_filtering_rollout",
	"collect_dpc_panel_rows",
	"draw_dynamix_main_figure",
	"export_dpc_panel_rows",
	"load_dynamix_main_figure_data",
	"make_dynamix_main_figure_layout",
	"plot_dynamix_main_figure",
	"resolve_paths",
	"run_dynamix_main_figure_data",
	"save_dynamix_main_figure",
	"system_median_dpc_panel",
]
