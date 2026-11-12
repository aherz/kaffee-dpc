"""Evaluation metrics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .predictive import (
	PredictiveEvalConfig,
	ensemble_path_nll,
	forecast_starts,
	gaussian_nll_diag,
	marginal_crps_ensemble,
	rollout_probabilistic_ensemble,
)
from .power_spectrum_error import compute_smoothed_power_spectrum, hellinger_distance, power_spectrum_error
from .state_space_divergence import state_space_divergence, state_space_divergence_binning
from .lyapunov import (
	le_metrics_from_jacobians,
	lorenz63_lyapunov_spectrum_from_samples,
	lorenz63_rds_lyapunov_spectrum,
	lorenz96_lyapunov_spectrum_from_samples,
	lorenz96_rds_lyapunov_spectrum,
	lyap_error_dict,
	lyapunov_spectrum_from_jacobians,
	positive_lyap_sum,
)
from .gtan import (
	GroundingWindowResult,
	gtan_from_log_variances,
	gtan_from_variances,
	gtan_transport_from_log_variances,
	gtan_transport_from_variances,
	l63_ekf_grounding_window_from_shell,
	l96_ekf_grounding_window_from_shell,
	lorenz63_rk4_step_jacobian_raw,
	lorenz63_rk4_step_jacobian_standardized,
	lorenz96_rk4_step_jacobian_raw,
	lorenz96_rk4_step_jacobian_standardized,
	spearman_rank_correlation,
)


def robust_se_of_median(values: Iterable[float]) -> float | None:
	"""Robust SE of the median: 0.929 * IQR / sqrt(n)."""
	arr = np.asarray(list(values), dtype=np.float64)
	finite = arr[np.isfinite(arr)]
	if finite.size < 2:
		return None
	q25, q75 = np.percentile(finite, [25.0, 75.0])
	return float(0.929 * (q75 - q25) / math.sqrt(finite.size))


__all__ = [
	"PredictiveEvalConfig",
	"ensemble_path_nll",
	"forecast_starts",
	"gaussian_nll_diag",
	"marginal_crps_ensemble",
	"rollout_probabilistic_ensemble",
	"compute_smoothed_power_spectrum",
	"hellinger_distance",
	"le_metrics_from_jacobians",
	"lorenz63_lyapunov_spectrum_from_samples",
	"lorenz63_rds_lyapunov_spectrum",
	"lorenz96_lyapunov_spectrum_from_samples",
	"lorenz96_rds_lyapunov_spectrum",
	"lyap_error_dict",
	"lyapunov_spectrum_from_jacobians",
	"positive_lyap_sum",
	"state_space_divergence",
	"power_spectrum_error",
	"state_space_divergence_binning",
	"GroundingWindowResult",
	"gtan_from_log_variances",
	"gtan_from_variances",
	"gtan_transport_from_log_variances",
	"gtan_transport_from_variances",
	"l63_ekf_grounding_window_from_shell",
	"l96_ekf_grounding_window_from_shell",
	"lorenz63_rk4_step_jacobian_raw",
	"lorenz63_rk4_step_jacobian_standardized",
	"lorenz96_rk4_step_jacobian_raw",
	"lorenz96_rk4_step_jacobian_standardized",
	"spearman_rank_correlation",
	"robust_se_of_median",
]
