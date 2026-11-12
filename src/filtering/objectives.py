"""Probabilistic training objectives for core-plus-shell experiments.

These helpers evaluate either filtered innovation NLL or open-loop Gaussian NLL on observation windows x_{1:T}, while the shell model supplies the latent transition z_t -> z_{t+1} and the shell covariances Gamma and Lambda.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .covariance import add_covariance_jitter, jacobian_factor_predict, symmetrize_covariance
from .ekf import (
	build_identity_observation_matrix,
	innovation_log_likelihood,
	innovation_statistics,
	innovation_statistics_identity_observation,
	kalman_predict,
	kalman_update,
	kalman_update_identity_observation,
)
from .ekf_lr import RRKF_FLOAT64, rrkf_innovation_log_likelihood, rrkf_predict, rrkf_update


class ShellObjectiveModel(Protocol):
	latent_dim: int
	observed_dim: int
	covariance_transport: str
	Gamma: torch.Tensor
	Lambda: torch.Tensor

	def initialize_from_observation(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

	def transition_and_jacobian(self, z_mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class ShellLRObjectiveModel(ShellObjectiveModel, Protocol):
	filter_rank: int
	Gamma_sqrt: torch.Tensor
	Lambda_sqrt: torch.Tensor

	def transition_mean(self, z_mean: torch.Tensor) -> torch.Tensor: ...

	def transition_and_jacobian_factor(
		self,
		z_mean: torch.Tensor,
		factor: torch.Tensor,
		*,
		transport_dtype: torch.dtype | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]: ...


def _structured_covariance_methods(model: ShellObjectiveModel):
	mean_transition = getattr(model, "transition_mean", None)
	covariance_transition = getattr(model, "transition_and_covariance", None)
	if callable(mean_transition) and callable(covariance_transition):
		return mean_transition, covariance_transition
	return None


def _validate_window_inputs(x_windows: torch.Tensor, *, burn_in_steps: int) -> None:
	if x_windows.ndim != 3:
		raise ValueError("x_windows must have shape (batch, time, observed_dim)")
	total_steps = x_windows.shape[1] - 1
	if total_steps < 1:
		raise ValueError("x_windows must contain at least two time points")
	if int(burn_in_steps) < 0 or int(burn_in_steps) > total_steps:
		raise ValueError("burn_in_steps must lie between 0 and window length minus one")


def _run_gaussian_window_objective_ekf(
	model: ShellObjectiveModel,
	x_windows: torch.Tensor,
	*,
	burn_in_steps: int,
	filtered: bool,
	innovation_jitter: float,
	initial_state: torch.Tensor | None,
	targets_include_initial: bool,
	use_jacobian_factor: bool = False,
) -> torch.Tensor:
	"""Filtered or open-loop Gaussian innovation objective.

	Jacobian-tied propagation uses J Sigma J^T + Gamma. 
	use_jacobian_factor implements the same while using a covariance square-root factor with a float-64 QR decomposition to help with stability/positive semidefiniteness of the predicted covariance (currently only used for Lorenz-63 shells).
	"""
	if initial_state is None:
		total_steps = x_windows.shape[1] - 1
		target_offset = 1
		z_mean, Sigma = model.initialize_from_observation(x_windows[:, 0, :])
	else:
		total_steps = x_windows.shape[1] if bool(targets_include_initial) else x_windows.shape[1] - 1
		target_offset = 0 if bool(targets_include_initial) else 1
		z_mean = initial_state
		Gamma0 = model.Gamma.to(device=z_mean.device, dtype=z_mean.dtype)
		Sigma = Gamma0.unsqueeze(0).expand(z_mean.shape[0], model.latent_dim, model.latent_dim).clone()
	structured = _structured_covariance_methods(model)
	B = None
	if structured is None:
		B = build_identity_observation_matrix(
			model.observed_dim,
			model.latent_dim,
			device=z_mean.device,
			dtype=z_mean.dtype,
		)
	Gamma = model.Gamma.to(device=z_mean.device, dtype=z_mean.dtype)
	Lambda = model.Lambda.to(device=z_mean.device, dtype=z_mean.dtype)
	total_nll = torch.zeros((), device=z_mean.device, dtype=z_mean.dtype)
	transport_mode = str(model.covariance_transport).strip().lower()
	use_factor = bool(use_jacobian_factor) and transport_mode == "jacobian"
	process_std = None
	factor = None
	if use_factor:
		process_std = model.proc_std.to(device=z_mean.device, dtype=z_mean.dtype).reshape(-1)
		factor = torch.diag(process_std).expand(z_mean.shape[0], -1, -1).contiguous()

	for step in range(total_steps):
		if use_factor:
			z_mean_pred, J_t = model.transition_and_jacobian(z_mean)
			factor = jacobian_factor_predict(factor, J_t, process_std)
			Sigma_pred = (factor @ factor.transpose(-1, -2)).to(dtype=z_mean.dtype)
		elif structured is None:
			z_mean_pred, J_t = model.transition_and_jacobian(z_mean)
			z_mean_pred, Sigma_pred = kalman_predict(
				z_mean,
				Sigma,
				Gamma=Gamma,
				z_mean_pred=z_mean_pred,
				J_t=J_t,
				covariance_transport=transport_mode,
			)
		elif transport_mode == "identity":
			mean_transition, _ = structured
			z_mean_pred = mean_transition(z_mean)
			z_mean_pred, Sigma_pred = kalman_predict(
				z_mean,
				Sigma,
				Gamma=Gamma,
				z_mean_pred=z_mean_pred,
				covariance_transport="identity",
			)
		elif transport_mode == "jacobian":
			_, covariance_transition = structured
			transported_covariance, z_mean_pred = covariance_transition(z_mean, Sigma)
			z_mean_pred, Sigma_pred = kalman_predict(
				z_mean,
				Sigma,
				Gamma=Gamma,
				z_mean_pred=z_mean_pred,
				transported_covariance=transported_covariance,
				covariance_transport="jacobian",
			)
		else:
			raise ValueError(f"Unsupported covariance propagation: {model.covariance_transport!r}")
		x_t = x_windows[:, step + target_offset, :]

		if filtered:
			if structured is not None:
				z_mean, Sigma, log_likelihood = kalman_update_identity_observation(
					z_mean_pred,
					Sigma_pred,
					x_t,
					Lambda=Lambda,
					Omega_jitter=innovation_jitter,
				)
			else:
				assert B is not None
				z_mean, Sigma, log_likelihood = kalman_update(
					z_mean_pred,
					Sigma_pred,
					x_t,
					B=B,
					Lambda=Lambda,
					Omega_jitter=innovation_jitter,
				)
			if use_factor:
				factor = torch.linalg.cholesky(symmetrize_covariance(Sigma))
		else:
			if use_factor and structured is not None:
				observed_dim = int(x_t.shape[-1])
				rows = factor[..., :observed_dim, :]
				residual = x_t - z_mean_pred[..., :observed_dim]
				Omega_t = add_covariance_jitter(
					rows @ rows.transpose(-1, -2) + Lambda,
					innovation_jitter,
				)
				residual = residual.to(dtype=Omega_t.dtype)
			elif structured is not None:
				residual, Omega_t, _ = innovation_statistics_identity_observation(
					z_mean_pred,
					Sigma_pred,
					x_t,
					Lambda=Lambda,
					Omega_jitter=innovation_jitter,
				)
			else:
				assert B is not None
				residual, Omega_t = innovation_statistics(
					z_mean_pred,
					Sigma_pred,
					x_t,
					B=B,
					Lambda=Lambda,
					Omega_jitter=innovation_jitter,
				)
			log_likelihood = innovation_log_likelihood(residual, Omega_t)
			z_mean, Sigma = z_mean_pred, Sigma_pred

		if step >= int(burn_in_steps):
			total_nll = total_nll - log_likelihood.sum()

	return total_nll


def _run_gaussian_window_objective_rrkf(
	model: ShellLRObjectiveModel,
	x_windows: torch.Tensor,
	*,
	burn_in_steps: int,
	filtered: bool,
	initial_state: torch.Tensor | None,
	targets_include_initial: bool,
) -> torch.Tensor:
	rank = int(model.filter_rank)
	if rank < 1:
		raise ValueError("filter_rank must be >= 1")

	if initial_state is None:
		total_steps = x_windows.shape[1] - 1
		target_offset = 1
		z_mean, Sigma_sqrt = model.initialize_from_observation(x_windows[:, 0, :])
	else:
		total_steps = x_windows.shape[1] if bool(targets_include_initial) else x_windows.shape[1] - 1
		target_offset = 0 if bool(targets_include_initial) else 1
		z_mean = initial_state
		Sigma_sqrt = model.Gamma_sqrt.unsqueeze(0).expand(z_mean.shape[0], model.latent_dim, rank).clone()

	if Sigma_sqrt.ndim != 3 or int(Sigma_sqrt.shape[-1]) != rank:
		raise ValueError(
			"ALRNNShellLR.initialize_from_observation must return Sigma_sqrt with shape (batch, latent_dim, r)"
		)

	device = z_mean.device
	dtype = z_mean.dtype
	B = build_identity_observation_matrix(
		model.observed_dim,
		model.latent_dim,
		device=device,
		dtype=dtype,
	)
	total_nll = torch.zeros((), device=device, dtype=dtype)
	transport_mode = str(model.covariance_transport).strip().lower()
	transport_dtype = torch.float64 if RRKF_FLOAT64 else dtype

	for step in range(total_steps):
		if transport_mode == "identity":
			z_mean_pred = model.transition_mean(z_mean)
			transported_sqrt = Sigma_sqrt
		elif transport_mode == "jacobian":
			transported_sqrt, z_mean_pred = model.transition_and_jacobian_factor(
				z_mean,
				Sigma_sqrt,
				transport_dtype=transport_dtype,
			)
		else:
			raise ValueError(f"Unsupported covariance propagation: {model.covariance_transport!r}")
		z_mean_pred, Sigma_sqrt_pred = rrkf_predict(
			z_mean,
			Sigma_sqrt,
			Gamma_sqrt=model.Gamma_sqrt,
			r=rank,
			z_mean_pred=z_mean_pred,
			transported_sqrt=transported_sqrt,
		)
		x_t = x_windows[:, step + target_offset, :]
		if filtered:
			z_mean, Sigma_sqrt, log_likelihood = rrkf_update(
				z_mean_pred,
				Sigma_sqrt_pred,
				x_t,
				B=B,
				Lambda_sqrt=model.Lambda_sqrt,
				r=rank,
				return_log_likelihood=True,
			)
		else:
			log_likelihood = rrkf_innovation_log_likelihood(
				z_mean_pred,
				Sigma_sqrt_pred,
				x_t,
				B=B,
				Lambda_sqrt=model.Lambda_sqrt,
			)
			z_mean, Sigma_sqrt = z_mean_pred, Sigma_sqrt_pred

		if step >= int(burn_in_steps):
			total_nll = total_nll - log_likelihood.sum()

	return total_nll


def _run_gaussian_window_objective(
	model: ShellObjectiveModel,
	x_windows: torch.Tensor,
	*,
	burn_in_steps: int = 0,
	filtered: bool,
	innovation_jitter: float = 1e-6,
	initial_state: torch.Tensor | None = None,
	targets_include_initial: bool = False,
	use_jacobian_factor: bool = False,
) -> torch.Tensor:
	_validate_window_inputs(x_windows, burn_in_steps=burn_in_steps)
	if getattr(model, "filter_rank", None) is not None:
		return _run_gaussian_window_objective_rrkf(
			model,
			x_windows,
			burn_in_steps=burn_in_steps,
			filtered=filtered,
			initial_state=initial_state,
			targets_include_initial=targets_include_initial,
		)
	return _run_gaussian_window_objective_ekf(
		model,
		x_windows,
		burn_in_steps=burn_in_steps,
		filtered=filtered,
		innovation_jitter=innovation_jitter,
		initial_state=initial_state,
		targets_include_initial=targets_include_initial,
		use_jacobian_factor=use_jacobian_factor,
	)


def filtered_innovation_nll_batched(
	model: ShellObjectiveModel,
	x_windows: torch.Tensor,
	*,
	burn_in_steps: int = 0,
	innovation_jitter: float = 1e-6,
	initial_state: torch.Tensor | None = None,
	targets_include_initial: bool = False,
	use_jacobian_factor: bool = False,
) -> torch.Tensor:
	"""Return the total filtered innovation NLL over a scored batch window."""

	return _run_gaussian_window_objective(
		model,
		x_windows,
		burn_in_steps=burn_in_steps,
		filtered=True,
		innovation_jitter=innovation_jitter,
		initial_state=initial_state,
		targets_include_initial=targets_include_initial,
		use_jacobian_factor=use_jacobian_factor,
	)


def open_loop_gaussian_nll_batched(
	model: ShellObjectiveModel,
	x_windows: torch.Tensor,
	*,
	burn_in_steps: int = 0,
	innovation_jitter: float = 1e-6,
	initial_state: torch.Tensor | None = None,
	targets_include_initial: bool = False,
	use_jacobian_factor: bool = False,
) -> torch.Tensor:
	"""Return the total open-loop Gaussian NLL over a scored batch window."""

	return _run_gaussian_window_objective(
		model,
		x_windows,
		burn_in_steps=burn_in_steps,
		filtered=False,
		innovation_jitter=innovation_jitter,
		initial_state=initial_state,
		targets_include_initial=targets_include_initial,
		use_jacobian_factor=use_jacobian_factor,
	)


__all__ = [
	"ShellObjectiveModel",
	"filtered_innovation_nll_batched",
	"open_loop_gaussian_nll_batched",
]
