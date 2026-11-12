"""Model implementations."""

from __future__ import annotations

from typing import Any

from .alrnn import ALRNN, WindowedTimeSeriesDataset, rollout_free, rollout_itf, teacher_force
from .alrnn_shell import ALRNNShell
from .alrnn_shell_lr import ALRNNShellLR
from .shplrnn import LinearObservation, ClippedShallowPLRNN
from .shplrnn_shell import ShallowPLRNNShell

__all__ = [
	"ALRNN",
	"ALRNNShell",
	"ALRNNShellLR",
	"LinearObservation",
	"ClippedShallowPLRNN",
	"ShallowPLRNNShell",
	"DynaMix",
	"DynaMixShell",
	"WindowedTimeSeriesDataset",
	"compute_mixture_jacobian",
	"load_dynamix_base",
	"load_dynamix_shell_checkpoint",
	"rollout_free",
	"rollout_itf",
	"teacher_force",
]

_DYNAMIX_EXPORTS: dict[str, tuple[str, str]] = {
	"DynaMix": (".dynamix", "DynaMix"),
	"load_dynamix_base": (".dynamix", "load_dynamix_base"),
	"DynaMixShell": (".dynamix_shell", "DynaMixShell"),
	"compute_mixture_jacobian": (".dynamix_shell", "compute_mixture_jacobian"),
	"load_dynamix_shell_checkpoint": (".dynamix_shell", "load_dynamix_shell_checkpoint"),
}


def __getattr__(name: str) -> Any:
	if name not in _DYNAMIX_EXPORTS:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	import importlib

	module_name, attr_name = _DYNAMIX_EXPORTS[name]
	value = getattr(importlib.import_module(module_name, __name__), attr_name)
	globals()[name] = value
	return value
