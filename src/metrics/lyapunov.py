"""Implementation of Lyapunov-spectrum and unstable-volume metrics.

We use the variational RK4 method with QR-based re-orthonormalization for computing the full Lyapunov spectrum from a sequence of Jacobian matrices along a trajectory (after discarding the first few transient steps). The leading Lyapunov exponent captures the asymptotic exponential divergence rate of nearby trajectories (necessarily >0 for continuous chaotic systems), while the sum of positive Lyapunov exponents (h_lambda) captures the average exponential volume growth rate in state space, which relates to the "uncertainty production" of the dynamics, also known as the Kolmogorov-Sinai (KS) entropy for deterministic systems. To evaluate the stochastic Lyapunov spectrum (of the respective Random Dynamical System (RDS)), we use the RK4 Jacobian at a fixed process-noise realization (while excluding observation noise)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


def _variational_rk4_step_from_samples(jacobian_fn, x_t: np.ndarray, x_t1: np.ndarray, Y: np.ndarray, dt: float) -> np.ndarray:
	"""RK4 step of the variational flow Y' = J(x) Y using successive trajectory samples."""
	x0 = np.asarray(x_t, dtype=np.float64).reshape(-1)
	x1 = np.asarray(x_t1, dtype=np.float64).reshape(-1)
	xm = 0.5 * (x0 + x1)
	J0 = jacobian_fn(x0)
	Jm = jacobian_fn(xm)
	J1 = jacobian_fn(x1)
	Y = np.asarray(Y, dtype=np.float64)
	k1 = J0 @ Y
	k2 = Jm @ (Y + 0.5 * float(dt) * k1)
	k3 = Jm @ (Y + 0.5 * float(dt) * k2)
	k4 = J1 @ (Y + float(dt) * k3)
	return Y + (float(dt) / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def lorenz96_lyapunov_spectrum_from_samples(
	X_raw: np.ndarray,
	*,
	dt: float,
	burn_in: int = 200,
	max_T: int = 80_000,
	qr_every: int = 1,
) -> np.ndarray:
	"""Continuous-time Lorenz-96 Lyapunov spectrum (per unit time) from raw trajectory samples.

	Propagates the tangent flow with the analytic Lorenz-96 Jacobian via variational RK4 between
	successive samples, with QR re-orthonormalization. This produces the deterministic reference spectrum.
	"""
	from data import lorenz96_jacobian

	X_raw = np.asarray(X_raw, dtype=np.float64)
	if X_raw.ndim != 2:
		raise ValueError(f"X_raw must have shape (T, N), got {X_raw.shape}")
	N = int(X_raw.shape[1])
	dt = float(dt)
	burn_in = int(burn_in)
	qr_every = int(max(1, qr_every))
	T = int(min(int(max_T), int(X_raw.shape[0] - 1)))
	if T <= burn_in + 10:
		return np.full(N, np.nan, dtype=np.float64)

	Q = np.eye(N, dtype=np.float64)
	logs = np.zeros(N, dtype=np.float64)
	scored_steps = 0
	for t in range(T):
		Q = _variational_rk4_step_from_samples(lorenz96_jacobian, X_raw[t], X_raw[t + 1], Q, dt)
		if ((t + 1) % qr_every) != 0:
			continue
		Q, R = np.linalg.qr(Q, mode="reduced")
		if t < burn_in:
			continue
		diag_r = np.maximum(np.abs(np.diag(R)), 1e-300)
		logs += np.log(diag_r)
		scored_steps += qr_every
	if scored_steps <= 0:
		return np.full(N, np.nan, dtype=np.float64)
	return (logs / (float(scored_steps) * dt)).astype(np.float64)


def lorenz96_rds_lyapunov_spectrum(
	X_raw: np.ndarray,
	*,
	dt: float,
	burn_in: int = 200,
	max_T: int | None = None,
	forcing: float = 8.0,
) -> np.ndarray:
	"""Lyapunov spectrum of the stochastic Lorenz-96 random dynamical system (RDS; with additive process noise).

	The derivative is taken at fixed process-noise realization. Additive noise changes the trajectory at which the RK4 map Jacobian is evaluated,
	but contributes no additional Jacobian term.
	"""
	from .gtan import lorenz96_rk4_step_jacobian_raw

	X_raw = np.asarray(X_raw, dtype=np.float64)
	if X_raw.ndim != 2:
		raise ValueError(f"X_raw must have shape (T, N), got {X_raw.shape}")
	if not np.isfinite(dt) or float(dt) <= 0.0:
		raise ValueError(f"dt must be positive, got {dt!r}")
	burn = int(burn_in)
	if burn < 0:
		raise ValueError(f"burn_in must be non-negative, got {burn_in}")
	stop = int(X_raw.shape[0] - 1) if max_T is None else min(int(max_T), int(X_raw.shape[0] - 1))
	if stop <= burn + 10:
		return np.full(int(X_raw.shape[1]), np.nan, dtype=np.float64)

	dim = int(X_raw.shape[1])
	q = np.eye(dim, dtype=np.float64)
	logs = np.zeros(dim, dtype=np.float64)
	for t in range(stop):
		jacobian = lorenz96_rk4_step_jacobian_raw(
			X_raw[t],
			dt=float(dt),
			forcing=float(forcing),
		)
		q, r = np.linalg.qr(jacobian @ q, mode="reduced")
		if t >= burn:
			logs += np.log(np.maximum(np.abs(np.diag(r)), 1e-300))
	return logs / (float(stop - burn) * float(dt))


def lorenz63_lyapunov_spectrum_from_samples(
	X_raw: np.ndarray,
	*,
	dt: float,
	burn_in: int = 1000,
	max_T: int = 31_000,
	sigma: float = 10.0,
	rho: float = 28.0,
	beta: float = 8.0 / 3.0,
) -> np.ndarray:
	"""Reference Lorenz-63 spectrum (per unit time) from raw trajectory samples.

	Same RK4 estimator as "lorenz96_lyapunov_spectrum_from_samples".
	"""
	from data.lorenz63 import lorenz63_jacobian

	X_raw = np.asarray(X_raw, dtype=np.float64)
	if X_raw.ndim != 2 or int(X_raw.shape[1]) != 3:
		raise ValueError(f"X_raw must have shape (T, 3), got {X_raw.shape}")
	dt = float(dt)
	burn_in = int(burn_in)
	T = int(min(int(max_T), int(X_raw.shape[0] - 1)))
	if T <= burn_in + 10:
		return np.full(3, np.nan, dtype=np.float64)

	def jacobian_fn(state: np.ndarray) -> np.ndarray:
		return lorenz63_jacobian(state, sigma=float(sigma), rho=float(rho), beta=float(beta))

	Q = np.eye(3, dtype=np.float64)
	logs = np.zeros(3, dtype=np.float64)
	for t in range(T):
		Q = _variational_rk4_step_from_samples(jacobian_fn, X_raw[t], X_raw[t + 1], Q, dt)
		Q, R = np.linalg.qr(Q, mode="reduced")
		if t < burn_in:
			continue
		logs += np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
	scored_steps = T - burn_in
	if scored_steps <= 0:
		return np.full(3, np.nan, dtype=np.float64)
	return (logs / (float(scored_steps) * dt)).astype(np.float64)


def lorenz63_rds_lyapunov_spectrum(
	X_raw: np.ndarray,
	*,
	dt: float,
	burn_in: int = 1000,
	max_T: int | None = None,
	sigma: float = 10.0,
	rho: float = 28.0,
	beta: float = 8.0 / 3.0,
) -> np.ndarray:
	"""Lyapunov spectrum of the stochastic Lorenz-63 random dynamical system (RDS; with additive process noise).

	Matches "lorenz96_rds_lyapunov_spectrum".
	"""
	from .gtan import lorenz63_rk4_step_jacobian_raw

	X_raw = np.asarray(X_raw, dtype=np.float64)
	if X_raw.ndim != 2 or int(X_raw.shape[1]) != 3:
		raise ValueError(f"X_raw must have shape (T, 3), got {X_raw.shape}")
	if not np.isfinite(dt) or float(dt) <= 0.0:
		raise ValueError(f"dt must be positive, got {dt!r}")
	burn = int(burn_in)
	if burn < 0:
		raise ValueError(f"burn_in must be non-negative, got {burn_in}")
	stop = int(X_raw.shape[0] - 1) if max_T is None else min(int(max_T), int(X_raw.shape[0] - 1))
	if stop <= burn + 10:
		return np.full(3, np.nan, dtype=np.float64)

	q = np.eye(3, dtype=np.float64)
	logs = np.zeros(3, dtype=np.float64)
	for t in range(stop):
		jacobian = lorenz63_rk4_step_jacobian_raw(
			X_raw[t],
			dt=float(dt),
			sigma=float(sigma),
			rho=float(rho),
			beta=float(beta),
		)
		q, r = np.linalg.qr(jacobian @ q, mode="reduced")
		if t >= burn:
			logs += np.log(np.maximum(np.abs(np.diag(r)), 1e-300))
	return logs / (float(stop - burn) * float(dt))


def _to_numpy_jacobian_seq(jacobians: Iterable[Any]) -> list[np.ndarray]:
	matrices: list[np.ndarray] = []
	for matrix in jacobians:
		if isinstance(matrix, torch.Tensor):
			array = matrix.detach().cpu().numpy()
		else:
			array = np.asarray(matrix)
		if array.ndim != 2 or array.shape[0] != array.shape[1]:
			raise ValueError(f"Each Jacobian must have shape (D, D), got {array.shape}")
		matrices.append(np.asarray(array, dtype=np.float64))
	return matrices


def lyapunov_spectrum_from_jacobians(jacobians: Iterable[Any], *, burn: int = 0) -> np.ndarray:
	matrices = _to_numpy_jacobian_seq(jacobians)
	if int(burn) < 0:
		raise ValueError(f"burn must be non-negative, got {burn}")
	if int(burn) >= len(matrices):
		raise ValueError(f"burn={burn} >= n_jacobians={len(matrices)}; need a longer rollout.")
	dim = int(matrices[0].shape[-1])
	q = np.eye(dim, dtype=np.float64)
	logs = np.zeros(dim, dtype=np.float64)
	for index, matrix in enumerate(matrices):
		q, r = np.linalg.qr(matrix @ q, mode="reduced")
		if index >= int(burn):
			diag_r = np.maximum(np.abs(np.diag(r)), 1e-300)
			logs += np.log(diag_r)
	return logs / float(len(matrices) - int(burn))


def positive_lyap_sum(spectrum: np.ndarray) -> float:
	values = np.asarray(spectrum, dtype=np.float64)
	if values.size == 0 or not np.all(np.isfinite(values)):
		return float("nan")
	return float(np.maximum(values, 0.0).sum())


def lyap_error_dict(
	spectrum_per_step: np.ndarray,
	ref_spectrum_per_unit_time: np.ndarray | float,
	*,
	dt: float,
	suffix: str = "",
) -> dict[str, float]:
	if not np.isfinite(dt) or float(dt) <= 0.0:
		raise ValueError(f"dt must be a positive float, got {dt!r}")
	spectrum = np.asarray(spectrum_per_step, dtype=np.float64) / float(dt)
	nan_dict = {
		f"signed_LE_error{suffix}": float("nan"),
		f"abs_signed_LE_error{suffix}": float("nan"),
		f"h_lambda{suffix}": float("nan"),
		f"signed_h_lambda_error{suffix}": float("nan"),
		f"abs_signed_h_lambda_error{suffix}": float("nan"),
	}
	if spectrum.size == 0 or not np.isfinite(spectrum[0]):
		return nan_dict

	h_pred = positive_lyap_sum(spectrum)
	if np.isscalar(ref_spectrum_per_unit_time):
		lam1_ref = float(ref_spectrum_per_unit_time)
		h_ref = float("nan")
	else:
		reference = np.asarray(ref_spectrum_per_unit_time, dtype=np.float64)
		if reference.size == 0 or not np.isfinite(reference[0]):
			out = dict(nan_dict)
			out[f"h_lambda{suffix}"] = float(h_pred)
			return out
		lam1_ref = float(reference[0])
		h_ref = positive_lyap_sum(reference)

	signed_le = float(spectrum[0] - lam1_ref)
	if np.isfinite(h_ref):
		signed_h = float(h_pred - h_ref)
		abs_h = abs(signed_h)
	else:
		signed_h = float("nan")
		abs_h = float("nan")
	return {
		f"signed_LE_error{suffix}": signed_le,
		f"abs_signed_LE_error{suffix}": abs(signed_le),
		f"h_lambda{suffix}": float(h_pred),
		f"signed_h_lambda_error{suffix}": signed_h,
		f"abs_signed_h_lambda_error{suffix}": abs_h,
	}


def le_metrics_from_jacobians(
	jacobians: Iterable[Any] | None,
	*,
	burn: int,
	dt: float,
	ref_spectrum_per_unit_time: np.ndarray | float,
	suffix: str = "",
) -> dict[str, float]:
	nan_dict = {
		f"signed_LE_error{suffix}": float("nan"),
		f"abs_signed_LE_error{suffix}": float("nan"),
		f"h_lambda{suffix}": float("nan"),
		f"signed_h_lambda_error{suffix}": float("nan"),
		f"abs_signed_h_lambda_error{suffix}": float("nan"),
	}
	if jacobians is None:
		return nan_dict
	matrices = _to_numpy_jacobian_seq(jacobians)
	if len(matrices) <= int(burn) + 10:
		return nan_dict
	try:
		spectrum = lyapunov_spectrum_from_jacobians(matrices, burn=int(burn))
	except Exception:
		return nan_dict
	return lyap_error_dict(
		spectrum,
		ref_spectrum_per_unit_time,
		dt=float(dt),
		suffix=suffix,
	)


__all__ = [
	"le_metrics_from_jacobians",
	"lorenz96_lyapunov_spectrum_from_samples",
	"lorenz96_rds_lyapunov_spectrum",
	"lyap_error_dict",
	"lyapunov_spectrum_from_jacobians",
	"positive_lyap_sum",
]