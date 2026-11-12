"""Filtering, covariance propagation, and probabilistic scoring. Modules follow the following notation: latent states are z_t, observations are x_t, predictive/filtered state covariances are Sigma, process/observation covariances are Gamma and Lambda, and the innovation covariance is Omega."""

from .covariance import add_covariance_jitter, joseph_update, symmetrize_covariance, transport_covariance
from .ekf import build_identity_observation_matrix, innovation_log_likelihood, innovation_statistics, kalman_predict, kalman_update
from .objectives import filtered_innovation_nll_batched, open_loop_gaussian_nll_batched

__all__ = [
	"add_covariance_jitter",
	"build_identity_observation_matrix",
	"filtered_innovation_nll_batched",
	"innovation_log_likelihood",
	"innovation_statistics",
	"joseph_update",
	"kalman_predict",
	"kalman_update",
	"open_loop_gaussian_nll_batched",
	"symmetrize_covariance",
	"transport_covariance",
]