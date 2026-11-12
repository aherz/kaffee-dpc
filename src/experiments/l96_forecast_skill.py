"""Lorenz-96 appendix forecast-skill spectrum (NLL, CRPS, G_tan^(H) at multiple assimilation intervals H).

Writes under "figures/l96_forecast_skill/":

- "forecast_skill_per_seed.csv" — scores at each horizon/assimilation interval H
- "l96_forecast_skill_spectrum.{pdf,png}" — Appx. figure

"notebooks/l96_forecast_skill_appendix.ipynb" plots the figure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

_paper_style_spec = importlib.util.spec_from_file_location(
	"paper_figure_style",
	Path(__file__).resolve().parents[2] / "figures" / "paper_style.py",
)
_paper_style = importlib.util.module_from_spec(_paper_style_spec)
_paper_style_spec.loader.exec_module(_paper_style)
_FORECAST_WIDTH = 6.6
_FORECAST_POINTS = _paper_style.points(_FORECAST_WIDTH)

from experiments.l96_main_figure import (
	METHOD_DISPLAY_LABELS,
	RAW_TO_PALETTE_KEY,
	TABLE1_PALETTE,
)
from experiments.lorenz96_main import prepare_lorenz96_main_bundle
from experiments.lorenz96_outputs import (
	PROBABILISTIC_METHODS,
	_evaluate_grounding,
	_evaluate_probabilistic_predictive,
	_load_probabilistic_checkpoint,
	_load_probabilistic_model,
	_predictive_eval_config,
)
from metrics import PredictiveEvalConfig, robust_se_of_median

SPECTRUM_PLOT_METHOD_KEYS = (
	"open_loop_no_propagation",
	"open_loop_tied",
	"kaffee",
	"filtered_no_propagation",
)
TABLE1_HORIZON_TAU_LAMBDA = 0.25
TABLE1_HORIZON_LEGEND = r"Table 1 horizon = $0.25\,\tau_\lambda$"

SPECTRUM_STYLE = _paper_style.rc_params(_FORECAST_WIDTH)
LINE_WIDTH = 1.5  # tent_nll_decomposition curve width
FIG_DPI = 200


def spectrum_horizons(config: dict[str, Any]) -> list[tuple[int, float]]:
	evaluation = dict(config["evaluation"])
	steps = [int(h) for h in evaluation["spectrum_horizons_steps"]]
	taus = [float(t) for t in evaluation["spectrum_horizons_tau_lambda"]]
	if len(steps) != len(taus):
		raise ValueError("spectrum_horizons_steps and spectrum_horizons_tau_lambda length mismatch")
	return list(zip(steps, taus, strict=True))


def evaluate_forecast_skill_spectrum(
	config: dict[str, Any],
	*,
	shell_root: Path,
	device: torch.device,
	seeds: list[int] | None = None,
	skip_existing: bool = True,
	output_csv: Path,
) -> Path:
	"""Score shells at each configured forecast horizon."""
	output_csv = Path(output_csv)
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	bundle = prepare_lorenz96_main_bundle(config, smoke=False)
	heldout = bundle["heldout_test"]
	standardizer = dict(bundle["dataset"]["standardizer"])
	predictive_base = _predictive_eval_config(config)
	dynamical = bundle["protocol"]["evaluation"]
	horizons = spectrum_horizons(config)
	seed_list = list(seeds) if seeds is not None else list(bundle["protocol"]["seed_sweep"]["seed_list"])
	test_obs_std = np.asarray(heldout["obs_std"], dtype=np.float32)
	test_obs_raw = np.asarray(heldout["obs_raw"], dtype=np.float64)
	test_clean_std = np.asarray(heldout["clean_std"], dtype=np.float32)
	dt = float(config["system"]["dt"])
	forcing = float(config["system"]["forcing"])

	rows: list[dict[str, Any]] = []
	if output_csv.is_file() and bool(skip_existing):
		existing = pd.read_csv(output_csv)
		rows = existing.to_dict(orient="records")
		done = {
			(str(r["method_key"]), int(r["seed"]), int(r["horizon_steps"]))
			for r in rows
		}
	else:
		done = set()

	n_jobs = len(PROBABILISTIC_METHODS) * len(seed_list) * len(horizons)
	print(f"forecast-skill jobs {n_jobs} done={len(done)} device={device}", flush=True)
	for spec in PROBABILISTIC_METHODS:
		for seed in seed_list:
			pending = [(h, tau) for h, tau in horizons if (spec.key, int(seed), int(h)) not in done]
			if not pending:
				continue
			checkpoint = _load_probabilistic_checkpoint(Path(shell_root), int(seed), spec.key)
			shell = _load_probabilistic_model(
				checkpoint,
				method_key=spec.key,
				config=config,
				device=device,
			)
			for horizon, tau in pending:
				predictive = PredictiveEvalConfig(
					forecast_horizon=int(horizon),
					forecast_starts=int(predictive_base.forecast_starts),
					nll_starts=int(predictive_base.nll_starts),
					forecast_ensemble=int(predictive_base.forecast_ensemble),
					obs_floor=float(predictive_base.obs_floor),
				)
				pred = _evaluate_probabilistic_predictive(
					shell,
					test_obs_std,
					test_obs_raw,
					standardizer,
					predictive=predictive,
					seed=int(seed),
				)
				gtan = _evaluate_grounding(
					shell,
					test_obs_std,
					test_clean_std,
					dt=dt,
					forcing=forcing,
					standardizer=standardizer,
					window_start=int(dynamical.get("gtan_window_start", 200)),
					window_length=int(dynamical.get("gtan_window_steps", 4000)),
					filter_warmup=int(dynamical.get("gtan_warmup_steps", 200)),
					assim_every=int(horizon),
				)
				row = {
					"seed": int(seed),
					"method_key": spec.key,
					"method_label": spec.label,
					"horizon_steps": int(horizon),
					"horizon_tau_lambda": float(tau),
					"NLL": float(pred["NLL"]),
					"CRPS": float(pred["CRPS"]),
					"G_tan": float(gtan["G_tan"]),
					"G_tan_transport": float(gtan["G_tan_transport"]),
				}
				rows.append(row)
				done.add((spec.key, int(seed), int(horizon)))
				pd.DataFrame(rows).to_csv(output_csv, index=False)
				print(
					f"  {spec.key} seed{seed:02d} H={horizon} "
					f"NLL={row['NLL']:.4f} CRPS={row['CRPS']:.4f} "
					f"G_tan_H={row['G_tan_transport']:.4f}",
					flush=True,
				)
			del shell
			if device.type == "cuda":
				torch.cuda.empty_cache()

	frame = pd.DataFrame(rows)
	frame.to_csv(output_csv, index=False)
	return output_csv


def plot_forecast_skill_spectrum(
	per_seed_csv: Path,
	figure_dir: Path,
	*,
	dpi: int = FIG_DPI,
) -> Path:
	"""Three-panel median +- robust SE vs horizon in Lyapunov times."""
	frame = pd.read_csv(per_seed_csv)
	figure_dir = Path(figure_dir)
	figure_dir.mkdir(parents=True, exist_ok=True)
	metrics = (
		("NLL", "NLL"),
		("CRPS", "CRPS"),
		("G_tan_transport", r"$G_{\mathrm{tan}}^{(H)}$"),
	)
	plot_specs = [spec for spec in PROBABILISTIC_METHODS if spec.key in SPECTRUM_PLOT_METHOD_KEYS]
	plot_specs = sorted(plot_specs, key=lambda spec: SPECTRUM_PLOT_METHOD_KEYS.index(spec.key))
	with plt.rc_context(SPECTRUM_STYLE):
		fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.02))
		fig.subplots_adjust(left=0.07, right=0.995, bottom=0.24, top=0.78, wspace=0.42)
		panel_letters = ("(a)", "(b)", "(c)")
		for ax, (field, ylabel), letter in zip(axes, metrics, panel_letters, strict=True):
			for spec in plot_specs:
				sub = frame.loc[frame["method_key"] == spec.key]
				if sub.empty:
					sub = frame.loc[frame["method_label"] == spec.label]
				if sub.empty:
					continue
				palette_key = RAW_TO_PALETTE_KEY.get(spec.label)
				if palette_key is None or palette_key == "Prior":
					raise ValueError(f"spectrum plot refused palette key {palette_key!r} for {spec.key}")
				color = TABLE1_PALETTE[palette_key]
				display = METHOD_DISPLAY_LABELS.get(palette_key, spec.label)
				xs: list[float] = []
				ys: list[float] = []
				yerr: list[float] = []
				for tau, grp in sub.groupby("horizon_tau_lambda", sort=True):
					values = grp[field].to_numpy(dtype=np.float64)
					finite = values[np.isfinite(values)]
					se = robust_se_of_median(finite)
					xs.append(float(tau))
					ys.append(float(np.median(finite)) if finite.size else float("nan"))
					yerr.append(0.0 if se is None else float(se))
				ax.errorbar(
					xs,
					ys,
					yerr=yerr,
					color=color,
					marker="o",
					markersize=3.2,
					linewidth=LINE_WIDTH,
					elinewidth=0.8,
					capsize=1.6,
					capthick=0.8,
					label=display,
				)
			ax.axvline(
				TABLE1_HORIZON_TAU_LAMBDA,
				color="0.45",
				linestyle=":",
				linewidth=1.0,
				zorder=1,
			)
			ax.set_ylabel(ylabel)
			ax.set_xlim(0.0, 1.05)
			ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
			ax.text(
				0.0,
				1.01,
				letter,
				transform=ax.transAxes,
				ha="left",
				va="bottom",
				fontsize=_FORECAST_POINTS["panel"],
				fontweight="bold",
				clip_on=False,
			)
		fig.canvas.draw()
		span_left = axes[0].get_position()
		span_right = axes[-1].get_position()
		fig.text(
			0.5 * (span_left.x0 + span_right.x1),
			span_left.y0 - 0.13,
			r"horizon ($\tau_\lambda$)",
			ha="center",
			va="top",
			fontsize=_FORECAST_POINTS["label"],
		)
		handles, labels = axes[0].get_legend_handles_labels()
		handles.append(
			Line2D([0], [0], color="0.45", linestyle=":", linewidth=1.0, label=TABLE1_HORIZON_LEGEND)
		)
		labels.append(TABLE1_HORIZON_LEGEND)
		fig.legend(
			handles,
			labels,
			loc="lower center",
			ncol=5,
			frameon=False,
			bbox_to_anchor=(0.5, 0.86),
			borderaxespad=0.0,
		)
		out_pdf = figure_dir / "l96_forecast_skill_spectrum.pdf"
		fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
		fig.savefig(figure_dir / "l96_forecast_skill_spectrum.png", dpi=int(dpi), bbox_inches="tight", pad_inches=0.01)
		plt.close(fig)
	return out_pdf


__all__ = [
	"evaluate_forecast_skill_spectrum",
	"plot_forecast_skill_spectrum",
	"spectrum_horizons",
]
