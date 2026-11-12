"""State-space divergence metric implementation for invariant-measure fidelity. For low-dimensional systems we use histograms to estimate the KL divergence between long reference and surrogate-generated trajectories (i.e. state distributions). For high-dimensional systems (N>5), we approximate both state distributions by isotropic GMMs centered on the trajectory samples, then estimate their KL divergence with MC samples from the reference mixture."""

from __future__ import annotations

import numpy as np


HIGHDIMENSIONAL_BINNING_THRESHOLD = 5
DEFAULT_DSTSP_GMM_SIGMA2 = 1.0
DEFAULT_DSTSP_GMM_MAX_T = 100_000
DEFAULT_DSTSP_GMM_MC_SAMPLES = 1_000
DEFAULT_DSTSP_GMM_BATCH_SIZE = 512


def _validate_state_space_arrays(generated: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	generated = np.asarray(generated, dtype=np.float64)
	reference = np.asarray(reference, dtype=np.float64)
	if generated.ndim != 2 or reference.ndim != 2:
		raise ValueError(f"generated and reference must have shape (T, D), got {generated.shape} and {reference.shape}")
	if generated.shape[1] != reference.shape[1]:
		raise ValueError(f"generated and reference must share coordinate dimension, got {generated.shape} and {reference.shape}")
	return generated, reference


def _calc_histogram(x: np.ndarray, n_bins: int, min_: np.ndarray, max_: np.ndarray) -> np.ndarray:
	x = np.asarray(x, dtype=np.float64)
	finite_rows = np.isfinite(x).all(axis=1)
	if not np.any(finite_rows):
		return np.zeros((int(n_bins),) * int(x.shape[1]), dtype=np.float64)
	x = x[finite_rows]
	min_ = np.asarray(min_, dtype=np.float64)
	max_ = np.asarray(max_, dtype=np.float64)
	denom = np.maximum(max_ - min_, 1e-12)
	coordinates = np.floor(float(n_bins) * (x - min_) / denom).astype(np.int64)
	coordinates = np.clip(coordinates, 0, int(n_bins) - 1)
	histogram = np.zeros((int(n_bins),) * int(x.shape[1]), dtype=np.float64)
	np.add.at(histogram, tuple(coordinates[:, dim] for dim in range(coordinates.shape[1])), 1.0)
	return histogram


def _normalize_histogram(histogram: np.ndarray, n_bins: int, smoothing_alpha: float = 1e-5) -> np.ndarray | None:
	total = float(np.sum(histogram))
	if total <= 0.0:
		return None
	dim = int(histogram.ndim)
	return (histogram + smoothing_alpha) / (total + smoothing_alpha * (int(n_bins) ** dim))


def _logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
	max_values = np.max(values, axis=axis, keepdims=True)
	max_values = np.where(np.isfinite(max_values), max_values, 0.0)
	return np.log(np.sum(np.exp(values - max_values), axis=axis)) + max_values.squeeze(axis)


def _log_mixture_density_gmm_diag(
	samples: np.ndarray,
	centers: np.ndarray,
	*,
	sigma: float,
	batch_size: int,
) -> np.ndarray:
	samples = np.asarray(samples, dtype=np.float64)
	centers = np.asarray(centers, dtype=np.float64)
	n_samples, dim = samples.shape
	n_centers = int(centers.shape[0])
	inv_sigma2 = 1.0 / (float(sigma) * float(sigma))
	const = -float(dim) * float(np.log(float(sigma)))
	ss = np.sum(samples * samples, axis=1, dtype=np.float64)
	logsum = np.full((n_samples,), -np.inf, dtype=np.float64)

	batch_size = int(max(16, int(batch_size)))
	for start in range(0, n_centers, batch_size):
		mu = centers[start : start + batch_size]
		cc = np.sum(mu * mu, axis=1, dtype=np.float64)
		cross = samples @ mu.T
		distance2 = np.maximum(ss[:, None] + cc[None, :] - 2.0 * cross, 0.0)
		log_likelihood = (-0.5 * inv_sigma2) * distance2 + const
		logsum = np.logaddexp(logsum, _logsumexp(log_likelihood, axis=1))

	return logsum - float(np.log(float(n_centers)))


def _state_space_divergence_gmm(
	generated: np.ndarray,
	reference: np.ndarray,
	*,
	sigma2: float = DEFAULT_DSTSP_GMM_SIGMA2,
	max_T: int | None = DEFAULT_DSTSP_GMM_MAX_T,
	mc_samples: int = DEFAULT_DSTSP_GMM_MC_SAMPLES,
	seed: int = 0,
	batch_size: int = DEFAULT_DSTSP_GMM_BATCH_SIZE,
) -> float:
	generated, reference = _validate_state_space_arrays(generated, reference)
	generated = generated[np.isfinite(generated).all(axis=1)]
	reference = reference[np.isfinite(reference).all(axis=1)]
	if generated.shape[0] < 10 or reference.shape[0] < 10:
		return float("nan")

	if max_T is None:
		effective_T = int(min(reference.shape[0], generated.shape[0]))
	else:
		effective_T = int(min(int(max_T), reference.shape[0], generated.shape[0]))
	if effective_T < 10:
		return float("nan")

	ref_centers = reference[:effective_T]
	gen_centers = generated[:effective_T]
	sigma = float(np.sqrt(max(float(sigma2), 0.0)))
	if sigma <= 0.0:
		return float("nan")

	rng = np.random.default_rng(int(seed))
	n_mc = int(max(1, int(mc_samples)))
	indices = rng.integers(low=0, high=effective_T, size=n_mc, endpoint=False)
	samples = ref_centers[indices] + sigma * rng.standard_normal(size=(n_mc, ref_centers.shape[1]))

	log_p = _log_mixture_density_gmm_diag(samples, ref_centers, sigma=sigma, batch_size=int(batch_size))
	log_q = _log_mixture_density_gmm_diag(samples, gen_centers, sigma=sigma, batch_size=int(batch_size))
	mask = np.isfinite(log_p) & np.isfinite(log_q)
	if not bool(np.any(mask)):
		return float("nan")
	return float(np.mean(log_p[mask] - log_q[mask]))


def state_space_divergence_binning(
	generated: np.ndarray,
	reference: np.ndarray,
	*,
	n_bins: int = 30,
	sigma2: float = DEFAULT_DSTSP_GMM_SIGMA2,
	max_T: int | None = DEFAULT_DSTSP_GMM_MAX_T,
	mc_samples: int = DEFAULT_DSTSP_GMM_MC_SAMPLES,
	seed: int = 0,
	batch_size: int = DEFAULT_DSTSP_GMM_BATCH_SIZE,
) -> float:
	generated, reference = _validate_state_space_arrays(generated, reference)
	if reference.shape[1] > int(HIGHDIMENSIONAL_BINNING_THRESHOLD):
		return _state_space_divergence_gmm(
			generated,
			reference,
			sigma2=float(sigma2),
			max_T=max_T,
			mc_samples=int(mc_samples),
			seed=int(seed),
			batch_size=int(batch_size),
		)
	min_ = np.nanmin(reference, axis=0)
	max_ = np.nanmax(reference, axis=0)
	hist_gen = _calc_histogram(generated, n_bins=int(n_bins), min_=min_, max_=max_)
	hist_ref = _calc_histogram(reference, n_bins=int(n_bins), min_=min_, max_=max_)
	p_gen = _normalize_histogram(hist_gen, n_bins=int(n_bins))
	p_ref = _normalize_histogram(hist_ref, n_bins=int(n_bins))
	if p_gen is None or p_ref is None:
		return float("nan")
	return float(np.sum(p_ref * np.log(p_ref / p_gen)))


state_space_divergence = state_space_divergence_binning


__all__ = [
	"DEFAULT_DSTSP_GMM_BATCH_SIZE",
	"DEFAULT_DSTSP_GMM_MAX_T",
	"DEFAULT_DSTSP_GMM_MC_SAMPLES",
	"DEFAULT_DSTSP_GMM_SIGMA2",
	"HIGHDIMENSIONAL_BINNING_THRESHOLD",
	"state_space_divergence",
	"state_space_divergence_binning",
]