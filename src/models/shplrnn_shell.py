"""Clipped shallow PLRNN stochastic shell implementation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from filtering.covariance import transport_covariance
from .shplrnn import LinearObservation, ClippedShallowPLRNN


@dataclass(frozen=True)
class ShellMethod:
	key: str
	objective: str
	transport: str
	train_core: bool
	detach_transport_jacobian: bool = False


METHOD_SPECS = {
	"kaffee": ShellMethod("kaffee", "filtered", "jacobian", True),
	"frozen_core_shell": ShellMethod("frozen_core_shell", "filtered", "jacobian", False),
	"filtered_no_propagation": ShellMethod("filtered_no_propagation", "filtered", "identity", True),
	"open_loop_tied": ShellMethod("open_loop_tied", "open_loop", "jacobian", True),
	"open_loop_no_propagation": ShellMethod("open_loop_no_propagation", "open_loop", "identity", True),
}


class ShallowPLRNNShell(torch.nn.Module):
	def __init__(
		self,
		core: ClippedShallowPLRNN,
		observation: LinearObservation,
		*,
		proc_std: torch.Tensor,
		obs_std: torch.Tensor,
		method: ShellMethod,
		initial_covariance: str = "projected_observation",
	) -> None:
		super().__init__()
		self.core = core
		self.observation = observation
		self.method_key = method.key
		self.objective = method.objective
		self.covariance_transport = method.transport
		self.detach_transport_jacobian = bool(method.detach_transport_jacobian)
		self.initial_covariance = str(initial_covariance)
		for parameter in self.core.parameters():
			parameter.requires_grad_(method.train_core)
		for parameter in self.observation.parameters():
			parameter.requires_grad_(False)
		self.log_proc_std = torch.nn.Parameter(torch.log(proc_std.detach().clone().clamp_min(1e-6)))
		self.log_obs_std = torch.nn.Parameter(torch.log(obs_std.detach().clone().clamp_min(1e-6)))

	@property
	def proc_std(self) -> torch.Tensor:
		return torch.exp(self.log_proc_std)

	@property
	def obs_std(self) -> torch.Tensor:
		return torch.exp(self.log_obs_std)

	@property
	def process_covariance(self) -> torch.Tensor:
		return torch.diag(self.proc_std.square())

	@property
	def observation_covariance(self) -> torch.Tensor:
		return torch.diag(self.obs_std.square())

	def initialize(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Latent belief from the (pseudo-)inverse readout."""
		z = self.observation.inverse(x0).detach()
		if z.ndim == 1:
			z = z.unsqueeze(0)
		if self.initial_covariance == "projected_observation":
			pinv = torch.linalg.pinv(self.observation.B)
			sigma = pinv @ self.observation_covariance @ pinv.T
			sigma = 0.5 * (sigma + sigma.T)
			sigma = sigma + 1e-6 * torch.eye(sigma.shape[0], device=sigma.device, dtype=sigma.dtype)
		elif self.initial_covariance == "process":
			sigma = self.process_covariance
		else:
			raise ValueError(f"Unknown initial covariance: {self.initial_covariance}")
		return z, sigma.unsqueeze(0).expand(z.shape[0], -1, -1).clone()

	def transition_and_jacobian(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		predicted = self.core(z)
		jacobians = torch.stack([self.core.transition_jacobian(row) for row in z])
		if self.detach_transport_jacobian:
			jacobians = jacobians.detach()
		return predicted, jacobians

	def predict_covariance(self, sigma: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
		return transport_covariance(
			sigma,
			self.process_covariance.unsqueeze(0),
			jacobian=jacobian,
			mode=self.covariance_transport,
		)

	def bundle(self) -> dict[str, Any]:
		return {
			"method_key": self.method_key,
			"objective": self.objective,
			"core": self.core.state_dict(),
			"observation": self.observation.state_dict(),
			"log_proc_std": self.log_proc_std.detach().cpu(),
			"log_obs_std": self.log_obs_std.detach().cpu(),
			"latent_dim": self.core.latent_dim,
			"hidden_dim": self.core.hidden_dim,
			"observed_dim": self.observation.observed_dim,
			"covariance_transport": self.covariance_transport,
			"detach_jacobian_in_transport": self.detach_transport_jacobian,
		}

	@classmethod
	def from_states(
		cls,
		core_state: dict[str, torch.Tensor],
		observation_state: dict[str, torch.Tensor],
		*,
		latent_dim: int,
		hidden_dim: int,
		observed_dim: int,
		proc_std: torch.Tensor,
		obs_std: torch.Tensor,
		method: ShellMethod,
		initial_covariance: str = "projected_observation",
	) -> "ShallowPLRNNShell":
		core = ClippedShallowPLRNN(latent_dim, hidden_dim)
		observation = LinearObservation(observed_dim, latent_dim)
		core.load_state_dict(core_state)
		observation.load_state_dict(observation_state)
		return cls(
			core,
			observation,
			proc_std=proc_std,
			obs_std=obs_std,
			method=method,
			initial_covariance=initial_covariance,
		)

	@classmethod
	def load(cls, path: str | Path, *, method_key: str, device: torch.device | str = "cpu") -> "ShallowPLRNNShell":
		bundle = torch.load(path, map_location="cpu", weights_only=True)
		if method_key not in METHOD_SPECS:
			raise ValueError(f"Unknown shell method: {method_key}")
		expected = METHOD_SPECS[method_key]
		stored_method = bundle.get("method_key")
		if stored_method is not None and str(stored_method) != method_key:
			raise ValueError(f"Checkpoint method is {stored_method}, expected {method_key}: {path}")
		if str(bundle.get("covariance_transport", "")) != expected.transport:
			raise ValueError(f"Checkpoint covariance propagation does not match {method_key}: {path}")
		if bool(bundle.get("detach_jacobian_in_transport", False)) != expected.detach_transport_jacobian:
			raise ValueError(f"Checkpoint Jacobian handling does not match {method_key}: {path}")
		for name in ("log_proc_std", "log_obs_std"):
			value = bundle.get(name)
			if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
				raise ValueError(f"Invalid {name} in shell bundle: {path}")
		for group in ("core", "observation"):
			state = bundle.get(group)
			if not isinstance(state, dict) or not state or any(
				not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
				for value in state.values()
			):
				raise ValueError(f"Invalid {group} state in shell bundle: {path}")
		if bundle["log_proc_std"].numel() != int(bundle["latent_dim"]) or bundle["log_obs_std"].numel() != int(bundle["observed_dim"]):
			raise ValueError(f"Noise dimensions disagree with the shell: {path}")
		model = cls.from_states(
			bundle["core"],
			bundle["observation"],
			latent_dim=int(bundle["latent_dim"]),
			hidden_dim=int(bundle["hidden_dim"]),
			observed_dim=int(bundle["observed_dim"]),
			proc_std=torch.exp(bundle["log_proc_std"]),
			obs_std=torch.exp(bundle["log_obs_std"]),
			method=expected,
		)
		with torch.no_grad():
			model.log_proc_std.copy_(bundle["log_proc_std"])
			model.log_obs_std.copy_(bundle["log_obs_std"])
		if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
			raise ValueError(f"Non-finite reconstructed shell state: {path}")
		return model.to(device).eval()

	def clone_bundle(self) -> dict[str, Any]:
		return copy.deepcopy(self.bundle())


@torch.no_grad()
def core_rollout(
	core: ClippedShallowPLRNN,
	observation: LinearObservation,
	x0: np.ndarray | torch.Tensor,
	*,
	burn_steps: int,
	score_steps: int,
) -> np.ndarray:
	"""Noise-free observation rollout.."""
	device, dtype = core.A.device, core.A.dtype
	z = observation.inverse(torch.as_tensor(x0, device=device, dtype=dtype).reshape(-1))
	outputs = []
	for step in range(int(burn_steps) + int(score_steps)):
		if step > 0:
			z = core(z.unsqueeze(0)).squeeze(0)
		outputs.append(observation(z.unsqueeze(0)).squeeze(0))
	value = torch.stack(outputs).cpu().numpy().astype(np.float64)
	return np.ascontiguousarray(value[int(burn_steps) :])


@torch.no_grad()
def stochastic_rollout(
	shell: ShallowPLRNNShell,
	x0: np.ndarray | torch.Tensor,
	*,
	burn_steps: int,
	score_steps: int,
	seed: int,
) -> np.ndarray:
	"""Noisy observation rollout."""
	device, dtype = shell.core.A.device, shell.core.A.dtype
	z = shell.observation.inverse(torch.as_tensor(x0, device=device, dtype=dtype).reshape(-1))
	generator = torch.Generator(device=device).manual_seed(int(seed))
	outputs = []
	for step in range(int(burn_steps) + int(score_steps)):
		if step > 0:
			z = shell.core(z.unsqueeze(0)).squeeze(0)
			z = z + shell.proc_std * torch.randn(z.shape, device=device, dtype=dtype, generator=generator)
		x = shell.observation(z.unsqueeze(0)).squeeze(0)
		x = x + shell.obs_std * torch.randn(x.shape, device=device, dtype=dtype, generator=generator)
		outputs.append(x)
	value = torch.stack(outputs).cpu().numpy().astype(np.float64)
	return np.ascontiguousarray(value[int(burn_steps) :])
