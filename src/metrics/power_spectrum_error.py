"""Power-spectrum error implementation for (stochastic and deterministic) dynamical fidelity. We use the Hellinger distance between smoothed power spectra of reference and generated trajectories, averaged across coordinates. The smoothing is done with a Gaussian kernel, and the power spectra are normalized to sum to 1 (i.e., they form a probability distribution) before computing the Hellinger distance. This metric captures differences in the distribution of "energy"/power across frequencies, which is important for assessing the fidelity of generated trajectories in terms of their temporal dynamics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def compute_smoothed_power_spectrum(x: np.ndarray, *, smoothing: int) -> np.ndarray:
	x = np.asarray(x, dtype=np.float64).reshape(-1)
	n = int(x.size)
	if n <= 1:
		return np.ones(1, dtype=np.float64)
	mean = float(np.mean(x))
	std = float(np.std(x))
	if (not np.isfinite(std)) or std <= 1e-12:
		k = np.fft.rfft(np.zeros(n, dtype=np.float64)).size
		return np.full(k, 1.0 / k, dtype=np.float64)
	standardized = np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
	power = (np.abs(np.fft.rfft(standardized)) ** 2) * (2.0 / float(n))
	power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
	if int(smoothing) > 0:
		power = gaussian_filter1d(power, int(smoothing))
	power = np.maximum(power, 0.0)
	total = float(np.sum(power))
	if (not np.isfinite(total)) or total <= 0.0:
		return np.full(power.size, 1.0 / power.size, dtype=np.float64)
	return (power / total).astype(np.float64)


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
	p = np.maximum(np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
	q = np.maximum(np.nan_to_num(np.asarray(q, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
	sum_p = float(np.sum(p))
	sum_q = float(np.sum(q))
	p = p / sum_p if sum_p > 0.0 else np.full_like(p, 1.0 / p.size)
	q = q / sum_q if sum_q > 0.0 else np.full_like(q, 1.0 / q.size)
	bc = float(np.clip(np.sum(np.sqrt(p * q)), 0.0, 1.0))
	return float(np.sqrt(max(0.0, 1.0 - bc)))


def power_spectrum_error(reference: np.ndarray, generated: np.ndarray, *, smoothing: int = 20) -> float:
	reference = np.asarray(reference, dtype=np.float64)
	generated = np.asarray(generated, dtype=np.float64)
	if reference.ndim != 2 or generated.ndim != 2:
		raise ValueError(f"reference and generated must have shape (T, D), got {reference.shape} and {generated.shape}")
	if reference.shape[1] != generated.shape[1]:
		raise ValueError(f"reference and generated must share coordinate dimension, got {reference.shape} and {generated.shape}")
	distances: list[float] = []
	for dim in range(int(reference.shape[1])):
		ref_ps = compute_smoothed_power_spectrum(reference[:, dim], smoothing=int(smoothing))
		gen_ps = compute_smoothed_power_spectrum(generated[:, dim], smoothing=int(smoothing))
		distances.append(hellinger_distance(ref_ps, gen_ps))
	return float(np.mean(distances)) if distances else float("nan")


__all__ = ["compute_smoothed_power_spectrum", "hellinger_distance", "power_spectrum_error"]