"""Data-generation for ODE-systems for DynaMix experiments."""

from __future__ import annotations

from .dynamix_ode import DYNAMIX_ODE_SPECS, DynaMixOdeSpec


DYNAMIX_BENCHMARK_ORDER: tuple[str, ...] = (
    "aizawa",
    "burke_shaw",
    "chen",
    "chua",
    "dadras",
    "genesio_tesi",
    "hadley",
    "halvorsen",
    "lorenz63",
    "nose_hoover",
    "qi_chen",
    "rabinovich_fabrikant",
    "sprott_b",
)


def _format_vector(values: tuple[float, float, float]) -> str:
    return f"({values[0]:g}, {values[1]:g}, {values[2]:g})"


def dynamix_benchmark_systems() -> tuple[DynaMixOdeSpec, ...]:
    return tuple(DYNAMIX_ODE_SPECS[system_key] for system_key in DYNAMIX_BENCHMARK_ORDER)


def dynamix_benchmark_records() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in dynamix_benchmark_systems():
        rows.append(
            {
                "system": spec.name,
                "parameters": spec.parameters,
                "dt": f"{spec.dt:g}",
                "x0": _format_vector(spec.x0),
                "burn_in": str(spec.burn_in),
                "lambda1_ref": f"{spec.lambda1_ref:.4f}",
            }
        )
    return rows


__all__ = [
    "DYNAMIX_BENCHMARK_ORDER",
    "dynamix_benchmark_records",
    "dynamix_benchmark_systems",
]
