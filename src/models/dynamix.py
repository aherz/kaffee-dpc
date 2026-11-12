"""DynaMix core. Uses DynaMix-python (https://github.com/DurstewitzLab/DynaMix-python, https://arxiv.org/abs/2505.13192) with a batched expert `step()` and `_use_batched` mixture-Jacobian propagation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from dynamix.model.dynamix import AlmostLinearRNN, GatingNetwork

BASE_MODEL_NAME = "dynamix-3d-alrnn-v1.0"
BASE_CHECKPOINT_FILE = f"{BASE_MODEL_NAME}.safetensors"
BASE_CONFIG_FILE = f"config_{BASE_MODEL_NAME.removeprefix('dynamix-')}.json"


def _base_model_dir(repo_root: Path | None = None) -> Path:
	return (repo_root or Path(__file__).resolve().parents[2]) / "models" / "dynamix_base"


class DynaMix(nn.Module):
	def __init__(
		self,
		M: int,
		N: int,
		Experts: int,
		P: int = 2,
		*,
		expert_type: str = "almost_linear_rnn",
		probabilistic_expert: bool = False,
		dtype: torch.dtype = torch.float32,
		temperature: float = 1.0,
	) -> None:
		if str(expert_type) != "almost_linear_rnn":
			raise ValueError(f"Only almost_linear_rnn experts are supported; got {expert_type!r}")
		if bool(probabilistic_expert):
			raise ValueError("probabilistic_expert is not supported")

		super().__init__()
		self.expert_type = "almost_linear_rnn"
		self.probabilistic_expert = False
		self.dtype = dtype
		self.M = int(M)
		self.N = int(N)
		self.Experts = int(Experts)
		self.P = int(P)
		self.temperature = float(temperature)

		self.experts = nn.ModuleList(
			AlmostLinearRNN(self.M, self.P, probabilistic=False, dtype=dtype) for _ in range(self.Experts)
		)
		self.gating_network = GatingNetwork(self.N, self.M, self.Experts, dtype=dtype)
		self.B = nn.Parameter(self._uniform_init((self.N, self.M), dtype=dtype))
		self._use_batched = self.P > 0

	def step(
		self,
		z: torch.Tensor,
		context: torch.Tensor,
		precomputed_cnn: torch.Tensor | None = None,
	) -> torch.Tensor:
		w_exp = self.gating_network(context, z, precomputed_cnn=precomputed_cnn)
		if self._use_batched:
			expert_next = self._batched_alrnn_forward(z)
		else:
			expert_next = torch.stack([expert(z) for expert in self.experts], dim=0)
		return torch.sum(expert_next * w_exp.unsqueeze(1), dim=0)

	def forward(
		self,
		z: torch.Tensor,
		context: torch.Tensor,
		precomputed_cnn: torch.Tensor | None = None,
	) -> torch.Tensor:
		return self.step(z, context, precomputed_cnn=precomputed_cnn)

	def precompute_cnn(self, context: torch.Tensor) -> torch.Tensor:
		context_for_conv = context.permute(1, 2, 0)
		encoded = self.gating_network.conv(context_for_conv)
		return encoded.permute(2, 0, 1)

	def _batched_alrnn_forward(self, z: torch.Tensor) -> torch.Tensor:
		e = self.Experts
		p = self.P
		a_all = torch.stack([expert.A for expert in self.experts])
		w_all = torch.stack([expert.W for expert in self.experts])
		h_all = torch.stack([expert.h for expert in self.experts])
		z1 = z[:-p, :]
		zcat = torch.cat([z1, F.relu(z[-p:, :])], dim=0)
		return (
			a_all.unsqueeze(-1) * z.unsqueeze(0)
			+ torch.bmm(w_all, zcat.unsqueeze(0).expand(e, -1, -1))
			+ h_all.unsqueeze(-1)
		)

	@staticmethod
	def _uniform_init(shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
		din = shape[-1]
		radius = 1.0 / float(np.sqrt(din))
		return (torch.rand(shape, dtype=dtype) * 2.0 - 1.0) * radius


def load_dynamix_base(
	*,
	device: torch.device | str = "cpu",
	repo_root: Path | None = None,
) -> DynaMix:
	"""Load base weights from `models/dynamix_base` in this repo."""
	model_dir = _base_model_dir(repo_root)
	config_path = model_dir / BASE_CONFIG_FILE
	checkpoint_path = model_dir / BASE_CHECKPOINT_FILE
	if not config_path.is_file():
		raise FileNotFoundError(f"Missing DynaMix config: {config_path}")
	if not checkpoint_path.is_file():
		raise FileNotFoundError(f"Missing DynaMix checkpoint: {checkpoint_path}")

	arch = json.loads(config_path.read_text(encoding="utf-8"))["architecture"]
	state_dict = load_file(str(checkpoint_path))
	model = DynaMix(
		M=int(arch["M"]),
		N=int(arch["N"]),
		Experts=int(arch["Experts"]),
		P=int(arch["P"]),
		expert_type=str(arch["expert_type"]),
		probabilistic_expert=bool(arch["probabilistic_expert"]),
		temperature=float(arch.get("temperature", 1.0)),
	)
	model.load_state_dict(state_dict, strict=True)
	return model.to(device).eval()


__all__ = [
	"AlmostLinearRNN",
	"BASE_CHECKPOINT_FILE",
	"BASE_CONFIG_FILE",
	"BASE_MODEL_NAME",
	"DynaMix",
	"GatingNetwork",
	"load_dynamix_base",
]
