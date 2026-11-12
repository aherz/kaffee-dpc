"""Training for shPLRNN cores ("deterministic priors") via generalized teacher forcing (GTF)."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metrics import state_space_divergence
from models.shplrnn import LinearObservation, ClippedShallowPLRNN
from models.shplrnn_shell import core_rollout


class WindowSampler:
	def __init__(self, data: np.ndarray, *, length: int, batch_size: int, generator: torch.Generator) -> None:
		self.data = torch.as_tensor(np.asarray(data, dtype=np.float32))
		self.length = int(length)
		self.batch_size = int(batch_size)
		self.generator = generator

	def sample(self) -> torch.Tensor:
		last_exclusive = self.data.shape[0] - self.length - 1
		starts = torch.randint(0, last_exclusive, (self.batch_size,), generator=self.generator)
		return torch.stack([self.data[int(start) : int(start) + self.length + 1] for start in starts])


def _gtf_loss(core: ClippedShallowPLRNN, observation: LinearObservation, x: torch.Tensor, alpha: float) -> torch.Tensor:
	z = observation.inverse(x[:, 0])
	forcing = observation.inverse(x).detach()
	outputs = []
	for step in range(1, x.shape[1]):
		z_dynamic = core(z)
		z = (1.0 - float(alpha)) * z_dynamic + float(alpha) * forcing[:, step]
		outputs.append(z_dynamic)
	predicted = observation(torch.stack(outputs, dim=1))
	return torch.mean((predicted - x[:, 1:]) ** 2)


def _regularization(core: ClippedShallowPLRNN, observation: LinearObservation, *, latent_weight: float, observation_weight: float) -> torch.Tensor:
	latent = (
		(1.0 - core.A).square().sum() + core.W1.square().sum() + core.W2.square().sum()
		+ core.h1.square().sum() + core.h2.square().sum()
	)
	singular = torch.linalg.svdvals(observation.B)
	condition = (1.0 - singular[0] / (singular[-1] + 1e-8)).square()
	return float(latent_weight) * latent + float(observation_weight) * condition


def _learning_rate(config: dict[str, Any], epoch: int) -> float:
	start, end, epochs = float(config["lr_start"]), float(config["lr_end"]), int(config["epochs"])
	gamma = math.exp(math.log(end / start) / float(epochs))
	return max(end, start * gamma ** float(int(epoch) - 1))


def _full_span_indices(length: int, count: int) -> np.ndarray:
	if not 1 <= int(count) <= int(length):
		raise ValueError("full-span index count must lie in [1, length]")
	if int(count) == int(length):
		return np.arange(int(length), dtype=np.int64)
	return np.linspace(0, int(length) - 1, num=int(count), dtype=np.int64)


def _balanced_equal_start_pool(trajectories: Sequence[np.ndarray], *, target_length: int) -> np.ndarray:
	"""Pool several rollouts so each start contributes the same number of centers."""
	paths = [np.asarray(value, dtype=np.float64) for value in trajectories]
	if not paths or any(value.shape != paths[0].shape for value in paths):
		raise ValueError("balanced pooling requires equal-shaped trajectories")
	starts = len(paths)
	target = int(target_length)
	if target < starts or target > starts * paths[0].shape[0]:
		raise ValueError("target length cannot represent every start")
	base, remainder = divmod(target, starts)
	counts = [base + (index < remainder) for index in range(starts)]
	selected = [path[_full_span_indices(path.shape[0], count)] for path, count in zip(paths, counts, strict=True)]
	rows = [selected[start][time] for time in range(max(counts)) for start in range(starts) if time < counts[start]]
	return np.ascontiguousarray(np.stack(rows), dtype=np.float64)


@dataclass(frozen=True)
class _CoreSelector:
	reference: np.ndarray
	sigma2: float
	mc_samples: int
	seed: int


def _selector(train: np.ndarray, config: dict[str, Any], seed: int) -> tuple[np.ndarray, int, int, _CoreSelector]:
	selector = dict(config["selector"])
	steps = min(int(selector["score_steps"]), train.shape[0])
	reference = np.asarray(train, dtype=np.float64)[_full_span_indices(train.shape[0], steps)]
	starts = np.asarray(train, dtype=np.float64)[_full_span_indices(train.shape[0], int(selector["initial_states"]))]
	context = _CoreSelector(
		reference=reference,
		sigma2=float(selector["sigma2"]),
		mc_samples=int(selector["mc_samples"]),
		seed=int(selector["metric_seed_base"]) + int(seed),
	)
	return starts, int(np.floor(float(selector["burn_fraction"]) * steps)), steps, context


@torch.no_grad()
def _score_core(core, observation, starts, burn_steps, score_steps, context: _CoreSelector) -> float:
	paths = [core_rollout(core, observation, x0, burn_steps=burn_steps, score_steps=score_steps) for x0 in starts]
	if not all(np.all(np.isfinite(path)) and np.max(np.abs(path)) < 1e6 for path in paths):
		return float("inf")
	pooled = _balanced_equal_start_pool(paths, target_length=context.reference.shape[0])
	return state_space_divergence(
		pooled,
		context.reference,
		sigma2=context.sigma2,
		max_T=context.reference.shape[0],
		mc_samples=context.mc_samples,
		seed=context.seed,
	)


def train_shplrnn_prior(
	train: np.ndarray,
	*,
	model_config: dict[str, Any],
	training_config: dict[str, Any],
	seed: int,
	output_dir: str | Path,
	device: torch.device | str = "cpu",
	smoke: bool = False,
) -> dict[str, Any]:
	np.random.seed(int(seed))
	torch.manual_seed(int(seed))
	generator = torch.Generator().manual_seed(int(seed))
	device = torch.device(device)
	core = ClippedShallowPLRNN(int(model_config["latent_dim"]), int(model_config["hidden_dim"]), generator=generator).to(device)
	observation = LinearObservation(int(model_config.get("observed_dim", train.shape[1])), int(model_config["latent_dim"]), generator=generator).to(device)
	sampler = WindowSampler(
		train,
		length=int(training_config["sequence_length"]),
		batch_size=int(training_config["batch_size"]),
		generator=generator,
	)
	parameters = list(core.parameters()) + list(observation.parameters())
	learning_rate = float(training_config["lr_start"])
	optimizer = torch.optim.RAdam(parameters, lr=learning_rate)
	starts, burn_steps, score_steps, selector_context = _selector(train, training_config, seed)
	epochs = 1 if smoke else int(training_config["epochs"])
	batches = 1 if smoke else int(training_config["batches_per_epoch"])
	checkpoint_every = 1 if smoke else int(training_config["checkpoint_every_epochs"])
	best_score, best_epoch, best_core, best_observation = float("inf"), -1, None, None
	candidates = []
	for epoch in range(1, epochs + 1):
		learning_rate = _learning_rate({**training_config, "epochs": epochs}, epoch) if not smoke else float(training_config["lr_start"])
		for group in optimizer.param_groups:
			group["lr"] = learning_rate
		for _ in range(batches):
			x = sampler.sample().to(device)
			optimizer.zero_grad(set_to_none=True)
			loss = _gtf_loss(core, observation, x, float(training_config["gtf_alpha"]))
			loss = loss + _regularization(
				core,
				observation,
				latent_weight=float(training_config["latent_regularization"]),
				observation_weight=float(training_config["observation_regularization"]),
			)
			loss.backward()
			optimizer.step()
		if epoch % checkpoint_every == 0 or epoch == epochs:
			core.eval(); observation.eval()
			score = _score_core(core, observation, starts, burn_steps, score_steps, selector_context)
			candidates.append({"epoch": epoch, "training_reference_d_stsp": score})
			if np.isfinite(score) and score <= best_score:
				best_score, best_epoch = float(score), int(epoch)
				best_core, best_observation = copy.deepcopy(core.state_dict()), copy.deepcopy(observation.state_dict())
			core.train(); observation.train()
	if best_core is None or best_observation is None:
		raise RuntimeError("No finite core checkpoint candidate")
	output = Path(output_dir)
	output.mkdir(parents=True, exist_ok=True)
	torch.save(best_core, output / "core.pt")
	torch.save(best_observation, output / "observation.pt")
	summary = {
		"seed": int(seed),
		"epochs": int(epochs),
		"batches_per_epoch": int(batches),
		"selected_epoch": best_epoch,
		"selected_training_d_stsp": best_score,
		"smoke": bool(smoke),
		"candidates": candidates,
	}
	(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
	return summary
