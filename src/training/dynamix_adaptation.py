"""DynaMix probabilistic adaptation/training."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from data import DynaMixWindowDataset
from filtering import filtered_innovation_nll_batched, open_loop_gaussian_nll_batched
from models import DynaMixShell

CovarianceTransport = Literal["jacobian", "identity"]
Objective = Literal["shell", "filtered", "open_loop"]


def _make_optimizer(params: list[torch.nn.Parameter], *, name: str, lr: float) -> torch.optim.Optimizer:
	key = str(name).strip().lower()
	if key == "adam":
		return torch.optim.Adam(params, lr=float(lr))
	if key == "radam":
		return torch.optim.RAdam(params, lr=float(lr))
	raise ValueError(f"Unsupported optimizer: {name!r}")


def _stack_z_inits(z_inits: list[torch.Tensor]) -> torch.Tensor:
	if not z_inits:
		raise ValueError("z_inits must be non-empty")
	return torch.stack([z[:, 0] if z.ndim == 2 else z for z in z_inits], dim=0)


def _window_z_inits(shell: DynaMixShell, windows: list[torch.Tensor]) -> list[torch.Tensor]:
	return [shell.latent_column_from_observation(obs[0]).detach() for obs in windows]


def _stack_windows(windows: list[torch.Tensor]) -> torch.Tensor:
	if not windows:
		raise ValueError("windows must be non-empty")
	return torch.stack(windows, dim=0)


def _objective_total_nll(
	shell: DynaMixShell,
	*,
	objective: Objective,
	context: torch.Tensor,
	z_inits: list[torch.Tensor],
	windows: list[torch.Tensor],
	burn_in_steps: int,
	n_ensemble_train: int,
) -> torch.Tensor:
	if objective == "shell":
		n_scored = max(int(windows[0].shape[0]) - 1 - int(burn_in_steps), 1)
		return shell.batched_windowed_nll(
			z_inits,
			[context] * len(z_inits),
			windows,
			n_ensemble=int(n_ensemble_train),
			burn_in_steps=int(burn_in_steps),
		) * float(len(windows) * n_scored * int(shell.observed_dim))

	x_windows = _stack_windows(windows)
	initial_state = _stack_z_inits(z_inits)
	if objective == "filtered":
		return filtered_innovation_nll_batched(
			shell,
			x_windows,
			burn_in_steps=int(burn_in_steps),
			initial_state=initial_state,
			targets_include_initial=False,
		)
	if objective == "open_loop":
		return open_loop_gaussian_nll_batched(
			shell,
			x_windows,
			burn_in_steps=int(burn_in_steps),
			initial_state=initial_state,
			targets_include_initial=False,
		)
	raise ValueError(f"Unsupported DynaMix objective: {objective!r}")


def train_dynamix_shell(
	shell: DynaMixShell,
	dataset: DynaMixWindowDataset,
	*,
	objective: Objective,
	transport: CovarianceTransport = "jacobian",
	train_core: bool,
	n_iters: int,
	batch_size: int,
	lr: float,
	seed: int,
	covariance_burn_in_steps: int,
	n_ensemble_train: int = 64,
	gradient_clip: float = 1.0,
	optimizer: str = "adam",
) -> dict[str, list[float]]:
	shell.covariance_transport = str(transport)
	shell.set_objective_context(dataset.context)
	for param in shell.model.parameters():
		param.requires_grad_(bool(train_core))
	shell.log_proc_std.requires_grad_(True)
	shell.log_obs_std.requires_grad_(True)
	trainable = [p for p in shell.parameters() if p.requires_grad]
	opt = _make_optimizer(trainable, name=optimizer, lr=float(lr))
	rng = np.random.default_rng(int(seed))
	history: dict[str, list[float]] = {"train_nll": [], "sigma_proc": [], "sigma_obs": []}

	for _it in range(int(n_iters)):
		shell.train()
		opt.zero_grad(set_to_none=True)
		idx = rng.choice(len(dataset.windows_train), size=int(batch_size), replace=True)
		window_batch = [dataset.windows_train[int(i)] for i in idx]
		z_batch = _window_z_inits(shell, window_batch)
		total_nll = _objective_total_nll(
			shell,
			objective=objective,
			context=dataset.context,
			z_inits=z_batch,
			windows=window_batch,
			burn_in_steps=int(covariance_burn_in_steps),
			n_ensemble_train=int(n_ensemble_train),
		)
		n_scored = max(int(window_batch[0].shape[0]) - 1 - int(covariance_burn_in_steps), 1)
		normalizer = float(len(window_batch) * n_scored * int(shell.observed_dim))
		loss = total_nll / normalizer
		loss.backward()
		torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(gradient_clip))
		opt.step()
		history["train_nll"].append(float(loss.detach().cpu().item()))
		history["sigma_proc"].append(float(shell.sigma_proc.detach().cpu().item()))
		history["sigma_obs"].append(float(shell.sigma_obs.detach().cpu().item()))
	return history


__all__ = [
	"CovarianceTransport",
	"Objective",
	"train_dynamix_shell",
]
