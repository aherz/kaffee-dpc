"""Aggregate the rr-KAFFEE rank sweep into a table and figure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_paper_style_spec = importlib.util.spec_from_file_location(
	"paper_figure_style",
	Path(__file__).resolve().parents[2] / "figures" / "paper_style.py",
)
_paper_style = importlib.util.module_from_spec(_paper_style_spec)
_paper_style_spec.loader.exec_module(_paper_style)
_RRKF_WIDTH = 5.5
_RRKF_POINTS = _paper_style.points(_RRKF_WIDTH)

from metrics import robust_se_of_median


RANKS = (5, 10, 20, 40, 80)
PER_SEED_CSV = "main_l96_ablation_per_seed.csv"
METHOD_KEY = "kaffee"
REQUIRED_SEEDS = 20
DENSE_RANK_LABEL = 100

# Paper table uses the propagation G_tan.
METRIC_SPEC: tuple[tuple[str, str, str], ...] = (
	("NLL", "NLL", "NLL"),
	("CRPS", "CRPS", "CRPS"),
	("G_tan_transport", "G_tan", r"$G_{\mathrm{tan}}$"),
	("D_stsp_stoch", "D_stsp_stoch", r"$D_{\mathrm{stsp}}^{\mathrm{stoch}}$"),
	("abs_signed_h_lambda_error_stoch", "|Delta h_lambda^stoch|", r"$|\Delta h_\lambda^{\mathrm{stoch}}|$"),
	("D_stsp_core", "D_stsp_core", r"$D_{\mathrm{stsp}}^{\mathrm{core}}$"),
)

PLOT_STYLE = _paper_style.rc_params(_RRKF_WIDTH)


LINE_WIDTH = 1.5
FIG_DPI = 200
KAFFEE_COLOR = "#009E73"
DENSE_COLOR = "0.35"
SLOPE_PTS = 4  # Number of last complete latent sizes in the log-log fit 


def load_rank_per_seed(output_root: Path, ranks: tuple[int, ...] | list[int] = RANKS) -> pd.DataFrame:
	"""Load kaffee per-seed rows, tagged with "filter_rank"."""
	output_root = Path(output_root)
	frames: list[pd.DataFrame] = []
	for rank in ranks:
		rank_csv = output_root / f"r{int(rank):03d}" / "tables" / PER_SEED_CSV
		if rank_csv.is_file():
			part = pd.read_csv(rank_csv)
			if "filter_rank" not in part.columns:
				part.insert(0, "filter_rank", int(rank))
			frames.append(part)
			continue
		seed_paths = sorted(
			(output_root / f"r{int(rank):03d}").glob(f"eval/seed*/tables/{PER_SEED_CSV}")
		)
		if not seed_paths:
			continue
		part = pd.concat([pd.read_csv(path) for path in seed_paths], ignore_index=True)
		part.insert(0, "filter_rank", int(rank))
		frames.append(part)
	if not frames:
		for name in ("rrkf_rank_per_seed.csv", PER_SEED_CSV):
			sweep_csv = output_root / "tables" / name
			if sweep_csv.is_file():
				frame = pd.read_csv(sweep_csv)
				if "filter_rank" in frame.columns:
					frames.append(frame)
					break
	if not frames:
		return pd.DataFrame()
	frame = pd.concat(frames, ignore_index=True)
	if "method_key" in frame.columns:
		frame = frame.loc[frame["method_key"].astype(str) == METHOD_KEY].copy()
	return frame.reset_index(drop=True)


def load_dense_kaffee_per_seed(per_seed_csv: Path) -> pd.DataFrame:
	path = Path(per_seed_csv)
	if not path.is_file():
		return pd.DataFrame()
	frame = pd.read_csv(path)
	if "method_key" in frame.columns:
		frame = frame.loc[frame["method_key"].astype(str) == METHOD_KEY].copy()
	frame = frame.copy()
	frame["filter_rank"] = int(DENSE_RANK_LABEL)
	return frame.reset_index(drop=True)


def _summarize_group(rows: pd.DataFrame, *, required_seeds: int) -> dict[str, Any]:
	seeds = sorted({int(s) for s in rows["seed"].tolist()}) if "seed" in rows.columns else []
	summary: dict[str, Any] = {
		"seed_count": int(len(seeds)),
		"n_required": int(required_seeds),
		"complete": int(len(seeds)) == int(required_seeds),
	}
	for field, paper_column, _ylabel in METRIC_SPEC:
		if field not in rows.columns:
			summary[f"{paper_column}_median"] = float("nan")
			summary[f"{paper_column}_robust_se"] = float("nan")
			summary[f"{paper_column}_n_finite"] = 0
			continue
		vals = np.asarray(rows[field], dtype=np.float64)
		finite = vals[np.isfinite(vals)]
		summary[f"{paper_column}_n_finite"] = int(finite.size)
		summary[f"{paper_column}_median"] = float(np.median(finite)) if finite.size else float("nan")
		se = robust_se_of_median(finite)
		summary[f"{paper_column}_robust_se"] = float(se) if se is not None else float("nan")
	return summary


def summarize_ranks(
	per_seed: pd.DataFrame,
	*,
	ranks: tuple[int, ...] | list[int] = RANKS,
	required_seeds: int = REQUIRED_SEEDS,
) -> pd.DataFrame:
	rows: list[dict[str, Any]] = []
	for rank in ranks:
		part = per_seed.loc[per_seed["filter_rank"].astype(int) == int(rank)] if not per_seed.empty else per_seed
		summary = _summarize_group(part, required_seeds=required_seeds)
		summary["filter_rank"] = int(rank)
		summary["source"] = "rrkf"
		rows.append(summary)
	return pd.DataFrame(rows)


def summarize_dense(per_seed: pd.DataFrame, *, required_seeds: int = REQUIRED_SEEDS) -> dict[str, Any] | None:
	if per_seed.empty:
		return None
	summary = _summarize_group(per_seed, required_seeds=required_seeds)
	summary["filter_rank"] = int(DENSE_RANK_LABEL)
	summary["source"] = "dense_table1"
	return summary


def _format_cell(median: float, se: float, *, complete: bool) -> str:
	if not complete or not np.isfinite(median):
		return ""
	if np.isfinite(se):
		return f"{median:.3f} ± {se:.3f}"
	return f"{median:.3f}"


def paper_table_rows(rank_summary: pd.DataFrame, dense: dict[str, Any] | None = None) -> list[dict[str, str]]:
	rows: list[dict[str, str]] = []
	for _, summary in rank_summary.iterrows():
		complete = bool(summary["complete"]) and int(summary["seed_count"]) == int(summary["n_required"])
		row: dict[str, str] = {
			"rank": str(int(summary["filter_rank"])),
			"n": str(int(summary["seed_count"])),
		}
		for _field, paper_column, _ylabel in METRIC_SPEC:
			row[paper_column] = _format_cell(
				float(summary[f"{paper_column}_median"]),
				float(summary[f"{paper_column}_robust_se"]),
				complete=complete and int(summary[f"{paper_column}_n_finite"]) == int(summary["n_required"]),
			)
		rows.append(row)
	if dense is not None:
		complete = bool(dense["complete"]) and int(dense["seed_count"]) == int(dense["n_required"])
		row = {"rank": "dense (Table 1)", "n": str(int(dense["seed_count"]))}
		for _field, paper_column, _ylabel in METRIC_SPEC:
			row[paper_column] = _format_cell(
				float(dense[f"{paper_column}_median"]),
				float(dense[f"{paper_column}_robust_se"]),
				complete=complete and int(dense[f"{paper_column}_n_finite"]) == int(dense["n_required"]),
			)
		rows.append(row)
	return rows


def plot_rank_metrics(
	per_seed: pd.DataFrame,
	rank_summary: pd.DataFrame,
	figure_dir: Path,
	*,
	dense: dict[str, Any] | None = None,
	dpi: int = FIG_DPI,
) -> Path | None:
	if rank_summary.empty or per_seed.empty:
		return None
	figure_dir = Path(figure_dir)
	figure_dir.mkdir(parents=True, exist_ok=True)
	panels = METRIC_SPEC
	with plt.rc_context(PLOT_STYLE):
		fig, axes = plt.subplots(2, 3, figsize=(5.5, 2.30))
		fig.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.97, wspace=0.48, hspace=0.52)
		ranks = [int(r) for r in rank_summary["filter_rank"].tolist()]
		xpos = np.arange(len(ranks), dtype=np.float64)
		for ax, (field, paper_column, ylabel) in zip(axes.ravel(), panels, strict=True):
			ys = [float(v) for v in rank_summary[f"{paper_column}_median"].tolist()]
			err = [float(v) if np.isfinite(v) else 0.0 for v in rank_summary[f"{paper_column}_robust_se"].tolist()]
			finite = np.asarray([np.isfinite(y) for y in ys], dtype=bool)
			ax.errorbar(
				xpos[finite],
				np.asarray(ys, dtype=np.float64)[finite],
				yerr=np.asarray(err, dtype=np.float64)[finite],
				color=KAFFEE_COLOR,
				marker="o",
				markersize=3.2,
				linewidth=LINE_WIDTH,
				elinewidth=0.8,
				capsize=1.6,
				label="rr-KAFFEE",
			)
			if dense is not None and np.isfinite(float(dense[f"{paper_column}_median"])):
				y = float(dense[f"{paper_column}_median"])
				se = float(dense[f"{paper_column}_robust_se"])
				ax.axhline(y, color=DENSE_COLOR, linestyle="--", linewidth=1.0, zorder=1)
				if np.isfinite(se):
					ax.axhspan(y - se, y + se, color=DENSE_COLOR, alpha=0.08, zorder=0)
			if ax in (axes[0, 1], axes[1, 1]):
				ax.set_xlabel("rank")
			ax.set_ylabel(ylabel)
			ax.set_xticks(xpos)
			ax.set_xticklabels([str(r) for r in ranks])
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
		handles, labels = axes[0, 0].get_legend_handles_labels()
		if dense is not None:
			handles.append(plt.Line2D([0], [0], color=DENSE_COLOR, linestyle="--", linewidth=1.0))
			labels.append("dense KAFFEE")
		axes[0, 0].legend(
			handles,
			labels,
			loc="upper right",
			frameon=False,
			fontsize=_RRKF_POINTS["legend"],
			handlelength=1.6,
			borderpad=0.15,
			labelspacing=0.25,
		)
		out_pdf = figure_dir / "rrkf_rank_metrics.pdf"
		fig.savefig(out_pdf, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
		fig.savefig(figure_dir / "rrkf_rank_metrics.png", dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
		plt.close(fig)
	return out_pdf


def write_rank_report(
	output_root: Path,
	*,
	ranks: tuple[int, ...] | list[int] = RANKS,
	figure_dir: Path | None = None,
	dense_per_seed_csv: Path | None = Path("results/lorenz96_main/tables/main_l96_ablation_per_seed.csv"),
	required_seeds: int = REQUIRED_SEEDS,
) -> dict[str, Path]:
	output_root = Path(output_root)
	tables = output_root / "tables"
	tables.mkdir(parents=True, exist_ok=True)
	figure_dir = Path(figure_dir) if figure_dir is not None else Path("figures/l96_rrkf")

	per_seed = load_rank_per_seed(output_root, ranks=ranks)
	if not per_seed.empty:
		per_seed_path = tables / "rrkf_rank_per_seed.csv"
		per_seed.to_csv(per_seed_path, index=False)
	else:
		per_seed_path = tables / "rrkf_rank_per_seed.csv"

	rank_summary = summarize_ranks(per_seed, ranks=ranks, required_seeds=required_seeds)
	rank_summary_path = tables / "rrkf_rank_summary.csv"
	rank_summary.to_csv(rank_summary_path, index=False)

	dense_summary = None
	if dense_per_seed_csv is not None:
		dense_rows = load_dense_kaffee_per_seed(Path(dense_per_seed_csv))
		dense_summary = summarize_dense(dense_rows, required_seeds=required_seeds)

	paper_rows = paper_table_rows(rank_summary, dense=dense_summary)
	paper_path = tables / "rrkf_rank_paper_table.csv"
	pd.DataFrame(paper_rows).to_csv(paper_path, index=False)

	figure_path = plot_rank_metrics(
		per_seed,
		rank_summary,
		figure_dir,
		dense=dense_summary,
	)

	meta = {
		"ranks": [int(r) for r in ranks],
		"required_seeds": int(required_seeds),
		"shells": "models/lorenz96_rrkf_shells",
		"n_kaffee_rows": int(len(per_seed)),
		"seeds_by_rank": {
			str(int(rank)): int((per_seed["filter_rank"] == int(rank)).sum()) if not per_seed.empty else 0
			for rank in ranks
		},
	}
	meta_path = tables / "rrkf_rank_report.json"
	meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	written = {
		"per_seed": per_seed_path,
		"summary": rank_summary_path,
		"paper_table": paper_path,
		"meta": meta_path,
	}
	if figure_path is not None:
		written["figure"] = figure_path
	return written


def load_timing_cells(per_cell_csv: Path) -> pd.DataFrame:
	frame = pd.read_csv(per_cell_csv)
	if frame.empty:
		return frame
	frame = frame.loc[frame["status"].astype(str) == "complete"].copy()
	return frame.reset_index(drop=True)


def _loglog_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
	slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
	return float(slope), float(intercept)


def upper_range_power_law(
	part: pd.DataFrame,
	y_column: str,
	*,
	n_fit: int = SLOPE_PTS,
) -> dict[str, Any] | None:
	complete = part.sort_values("latent_dim")
	complete = complete.loc[np.isfinite(complete[y_column].to_numpy(dtype=np.float64))]
	fit = complete.tail(int(n_fit))
	if len(fit) < 2:
		return None
	x = fit["latent_dim"].to_numpy(dtype=np.float64)
	y = fit[y_column].to_numpy(dtype=np.float64)
	if np.any(x <= 0.0) or np.any(y <= 0.0) or not np.all(np.isfinite(y)):
		return None
	slope, intercept = _loglog_slope(x, y)
	if not np.isfinite(slope) or not np.isfinite(intercept):
		return None
	return {
		"slope": slope,
		"intercept": intercept,
		"M_min": float(x.min()),
		"M_max": float(x.max()),
		"n_fit": int(len(fit)),
	}


def _asymmetric_err(
	median: np.ndarray,
	low: np.ndarray,
	high: np.ndarray,
) -> np.ndarray:
	lower = np.clip(median - low, a_min=0.0, a_max=None)
	upper = np.clip(high - median, a_min=0.0, a_max=None)
	return np.vstack([lower, upper])


FIT_LINE_WIDTH = 1.7


def _draw_power_law(ax, fit: dict[str, Any], color: str) -> None:
	"""Draw the upper-range fit past the last point."""
	xs = np.array([fit["M_min"], fit["M_max"] * 1.85], dtype=np.float64)
	ys = np.exp(fit["intercept"] + fit["slope"] * np.log(xs))
	ax.plot(xs, ys, color=color, linestyle="--", linewidth=FIT_LINE_WIDTH, zorder=5, solid_capstyle="butt")


def plot_timing_metrics(per_cell: pd.DataFrame, figure_dir: Path, *, dpi: int = FIG_DPI) -> Path | None:
	if per_cell.empty:
		return None
	figure_dir = Path(figure_dir)
	figure_dir.mkdir(parents=True, exist_ok=True)
	method_style = {
		"kaffee": {"color": DENSE_COLOR, "fit_color": "#0072B2", "label": "dense KAFFEE", "marker": "s"},
		"kaffee_rrkf": {"color": KAFFEE_COLOR, "fit_color": "#56B4E9", "label": "rr-KAFFEE (r=20)", "marker": "o"},
	}
	has_time_err = {"q25_update_seconds", "q75_update_seconds"}.issubset(per_cell.columns)
	has_mem_err = {"q25_cuda_gib", "q75_cuda_gib"}.issubset(per_cell.columns)

	scale_points = _paper_style.points(5.35)
	with plt.rc_context(_paper_style.rc_params(5.35)):
		fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.55))
		fig.subplots_adjust(left=0.10, right=0.985, bottom=0.22, top=0.96, wspace=0.40)
		fit_entries: dict[int, list[tuple[str, str]]] = {0: [], 1: []}
		short_name = {"kaffee": "dense", "kaffee_rrkf": "rr-KAFFEE"}
		for method in ("kaffee_rrkf", "kaffee"):
			style = method_style[method]
			part = per_cell.loc[per_cell["method"].astype(str) == method].sort_values("latent_dim")
			if part.empty:
				continue
			m = part["latent_dim"].to_numpy(dtype=np.float64)
			time_y = part["median_update_seconds"].to_numpy(dtype=np.float64)
			mem_y = part["peak_cuda_gib"].to_numpy(dtype=np.float64)
			time_kw: dict[str, Any] = dict(
				color=style["color"],
				marker=style["marker"],
				markersize=3.0,
				markeredgewidth=0.4,
				linewidth=LINE_WIDTH,
				linestyle="-",
				label=style["label"],
				zorder=3,
			)
			if has_time_err:
				axes[0].errorbar(
					m,
					time_y,
					yerr=_asymmetric_err(
						time_y,
						part["q25_update_seconds"].to_numpy(dtype=np.float64),
						part["q75_update_seconds"].to_numpy(dtype=np.float64),
					),
					capsize=1.6,
					elinewidth=0.7,
					**time_kw,
				)
			else:
				axes[0].plot(m, time_y, **time_kw)
			mem_kw = dict(time_kw)
			mem_kw.pop("label")
			if has_mem_err:
				axes[1].errorbar(
					m,
					mem_y,
					yerr=_asymmetric_err(
						mem_y,
						part["q25_cuda_gib"].to_numpy(dtype=np.float64),
						part["q75_cuda_gib"].to_numpy(dtype=np.float64),
					),
					capsize=1.6,
					elinewidth=0.7,
					**mem_kw,
				)
			else:
				axes[1].plot(m, mem_y, **mem_kw)
			time_fit = upper_range_power_law(part, "median_update_seconds")
			mem_fit = upper_range_power_law(part, "peak_cuda_gib")
			name = short_name[method]
			if time_fit is not None and np.isfinite(time_fit["slope"]):
				_draw_power_law(axes[0], time_fit, style["fit_color"])
				fit_entries[0].append((rf"{name} $\propto M^{{{time_fit['slope']:.2f}}}$", style["fit_color"]))
			if mem_fit is not None and np.isfinite(mem_fit["slope"]):
				_draw_power_law(axes[1], mem_fit, style["fit_color"])
				fit_entries[1].append((rf"{name} $\propto M^{{{mem_fit['slope']:.2f}}}$", style["fit_color"]))
		axes[0].set_ylabel("training update (s)")
		axes[1].set_ylabel("peak memory (GiB)")
		for idx, ax in enumerate(axes):
			ax.set_xscale("log")
			ax.set_yscale("log")
			ax.minorticks_off()
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
			if fit_entries[idx] and idx != 0:
				fit_handles = [
					plt.Line2D([0], [0], color=color, linestyle="--", linewidth=FIT_LINE_WIDTH)
					for _label, color in fit_entries[idx]
				]
				fit_labels = [label for label, _color in fit_entries[idx]]
				ax.legend(
					fit_handles,
					fit_labels,
					loc="upper left",
					frameon=False,
					fontsize=scale_points["legend"],
					handlelength=1.8,
					borderpad=0.15,
					labelspacing=0.25,
				)
		method_handles, method_labels = axes[0].get_legend_handles_labels()
		fit_handles = [
			plt.Line2D([0], [0], color=color, linestyle="--", linewidth=FIT_LINE_WIDTH)
			for _label, color in fit_entries[0]
		]
		fit_labels = [label for label, _color in fit_entries[0]]
		axes[0].legend(
			[*method_handles, *fit_handles],
			[*method_labels, *fit_labels],
			loc="upper left",
			frameon=False,
			fontsize=scale_points["legend"],
			handlelength=1.8,
			borderpad=0.15,
			labelspacing=0.25,
		)
		fig.canvas.draw()
		span_left = axes[0].get_position()
		span_right = axes[1].get_position()
		fig.text(
			0.5 * (span_left.x0 + span_right.x1),
			span_left.y0 - 0.14,
			r"latent dimension $M$",
			ha="center",
			va="top",
			fontsize=scale_points["label"],
		)
		out_pdf = figure_dir / "rrkf_scaling.pdf"
		fig.savefig(out_pdf, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
		fig.savefig(figure_dir / "rrkf_scaling.png", dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
		plt.close(fig)
	return out_pdf

def timing_exponent_rows(per_cell: pd.DataFrame) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for method in ("kaffee", "kaffee_rrkf"):
		part = per_cell.loc[per_cell["method"].astype(str) == method]
		if part.empty:
			continue
		for quantity, column in (
			("time", "median_update_seconds"),
			("memory", "peak_cuda_gib"),
		):
			fit = upper_range_power_law(part, column)
			if fit is None:
				continue
			rows.append(
				{
					"method": method,
					"quantity": quantity,
					"n_fit": fit["n_fit"],
					"M_min": fit["M_min"],
					"M_max": fit["M_max"],
					"exponent": fit["slope"],
				}
			)
	return rows
