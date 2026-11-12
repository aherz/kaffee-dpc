"""
Size conventions for figure in the paper.
"""

from __future__ import annotations

REFERENCE_WIDTH_IN = 6.85
LABEL_PT = 8.0
TICK_PT = 7.5
LEGEND_PT = 7.0
PANEL_PT = 9.0
TITLE_PT = 9.0
AXES_LINEWIDTH = 0.8

ONE_ROW_FIGSIZE = (REFERENCE_WIDTH_IN, 2.10)
TWO_ROW_FIGSIZE = (10.95, 4.05)


def scale_for(canvas_width_in: float, *, include_fraction: float = 1.0) -> float:
	return float(canvas_width_in) / REFERENCE_WIDTH_IN / float(include_fraction)


def points(canvas_width_in: float, *, include_fraction: float = 1.0) -> dict[str, float]:
	scale = scale_for(canvas_width_in, include_fraction=include_fraction)
	return {
		"label": LABEL_PT * scale,
		"tick": TICK_PT * scale,
		"legend": LEGEND_PT * scale,
		"panel": PANEL_PT * scale,
		"title": TITLE_PT * scale,
	}


def rc_params(canvas_width_in: float, *, include_fraction: float = 1.0) -> dict[str, float | str | bool]:
	sized = points(canvas_width_in, include_fraction=include_fraction)
	return {
		"font.family": "DejaVu Sans",
		"axes.labelsize": sized["label"],
		"axes.titlesize": sized["title"],
		"xtick.labelsize": sized["tick"],
		"ytick.labelsize": sized["tick"],
		"legend.fontsize": sized["legend"],
		"axes.linewidth": AXES_LINEWIDTH,
		"axes.spines.top": False,
		"axes.spines.right": False,
	}
