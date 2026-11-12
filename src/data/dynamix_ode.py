"""Trajectory generation for the held-out ODE systems in the DynaMix experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class DynaMixOdeSpec:
	name: str
	parameters: str
	dt: float
	x0: tuple[float, float, float]
	burn_in: int
	lambda1_ref: float  # dysts/Gilpin 2021; scored NLL window via dynamix_nll_horizon_steps


DYNAMIX_ODE_SPECS: dict[str, DynaMixOdeSpec] = {
	"lorenz63": DynaMixOdeSpec(
		"Lorenz-63", "sigma=10, rho=28, beta=8/3", 0.02, (1.0, 1.0, 1.0), 500, 0.9018,
	),
	"halvorsen": DynaMixOdeSpec(
		"Halvorsen",
		"a=1.4, b=4",
		0.01,
		(-8.6807408, -2.4741399, 0.070775762),
		1000,
		0.6849,
	),
	"sprott_b": DynaMixOdeSpec(
		"Sprott B", "---", 0.05, (0.1, 0.5, 0.1), 500, 0.2067,
	),
	"chen": DynaMixOdeSpec(
		"Chen", "a=35, b=3, c=28", 0.005, (-10.0, 0.0, 37.0), 2000, 2.0418,
	),
	"aizawa": DynaMixOdeSpec(
		"Aizawa",
		"a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1",
		0.02,
		(0.1, 0.0, 0.0),
		1000,
		0.0878,
	),
	"dadras": DynaMixOdeSpec(
		"Dadras", "p=3, q=2.7, r=1.7, s=2, e=9", 0.02, (1.0, 1.0, 1.0), 500, 0.3667
	),
	"chua": DynaMixOdeSpec(
		"Chua", "alpha=15.6, beta=28, m0=-1.143, m1=-0.714", 0.02, (0.7, 0.0, 0.0), 500, 0.4069
	),
	"genesio_tesi": DynaMixOdeSpec(
		"Genesio-Tesi", "a=0.44, b=1.1, c=1", 0.02, (0.1, 0.1, 0.1), 500, 0.1079
	),
	"burke_shaw": DynaMixOdeSpec(
		"Burke-Shaw",
		"n=10, e=13",
		0.005,
		(-0.79153188, 0.65244479, 0.22517831),
		1000,
		2.3139,
	),
	"nose_hoover": DynaMixOdeSpec(
		"Nose-Hoover", "a=1.5", 0.02, (0.1, 0.0, 0.0), 500, 0.0052
	),
	"hadley": DynaMixOdeSpec(
		"Hadley",
		"a=0.2, b=4, f=9, g=1",
		0.02,
		(1.3664423, 0.9309146, 0.64283417),
		500,
		0.2152,
	),
	"rabinovich_fabrikant": DynaMixOdeSpec(
		"Rabinovich-Fabrikant",
		"gamma=0.87, alpha=1.1",
		0.01,
		(-1.0, 0.0, 0.5),
		1000,
		0.2147,
	),
	"qi_chen": DynaMixOdeSpec(
		"Qi-Chen", "a=38, b=8/3, c=80", 0.005, (1.0, 1.0, 5.0), 2000, 3.6770
	),
}


def dynamix_required_reference_steps(context_length: int) -> int:
	return int(context_length) + 55_000

def dynamix_scored_window_length(
	lambda1: float | None,
	dt: float | None,
	*,
	k: float = 1.0,
	lo: int = 24,
	hi: int = 128,
	fallback: int = 64,
) -> int:
	"""Compute scored window length according to the Lyapunov time of a system."""
	if lambda1 is None or dt is None:
		return int(fallback)
	if not np.isfinite(float(lambda1)) or not np.isfinite(float(dt)) or float(lambda1) <= 0.0:
		return int(hi)
	window = float(k) * (1.0 / float(lambda1)) / float(dt)
	if not np.isfinite(window):
		return int(hi)
	return int(max(int(lo), min(int(hi), round(window))))


def dynamix_nll_horizon_steps(
	lambda1: float | None,
	dt: float | None,
	*,
	k: float = 0.25,
	lo: int = 24,
	hi: int = 128,
) -> int:
	"""Local open-loop NLL horizon: round(k / λ₁ / dt), clipped to [lo, hi].
	"""
	if float(k) <= 0.0:
		raise ValueError(f"nll horizon Lyapunov multiple must be positive, got {k}")
	return dynamix_scored_window_length(lambda1, dt, k=float(k), lo=int(lo), hi=int(hi), fallback=int(lo))


def dynamix_ode_drift(system_name: str, x: np.ndarray) -> np.ndarray:
	"""Continuous-time drift for one held-out DynaMix ODE system."""
	system = str(system_name).strip().lower()
	x0, y0, z0 = [float(v) for v in np.asarray(x, dtype=np.float64).reshape(-1)]

	if system == "lorenz63":
		sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
		return np.array(
			[
				sigma * (y0 - x0),
				x0 * (rho - z0) - y0,
				x0 * y0 - beta * z0,
			],
			dtype=np.float64,
		)
	if system == "halvorsen":
		a, b = 1.4, 4.0
		return np.array(
			[
				-a * x0 - b * y0 - b * z0 - y0 * y0,
				-a * y0 - b * z0 - b * x0 - z0 * z0,
				-a * z0 - b * x0 - b * y0 - x0 * x0,
			],
			dtype=np.float64,
		)
	if system == "sprott_b":
		return np.array(
			[
				y0 * z0,
				x0 - y0,
				1.0 - x0 * y0,
			],
			dtype=np.float64,
		)
	if system == "chen":
		a, b, c = 35.0, 3.0, 28.0
		return np.array(
			[
				a * (y0 - x0),
				(c - a) * x0 - x0 * z0 + c * y0,
				x0 * y0 - b * z0,
			],
			dtype=np.float64,
		)
	if system == "aizawa":
		a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
		return np.array(
			[
				(z0 - b) * x0 - d * y0,
				d * x0 + (z0 - b) * y0,
				c + a * z0 - z0**3 / 3.0 - (x0 * x0 + y0 * y0) * (1.0 + e * z0) + f * z0 * x0**3,
			],
			dtype=np.float64,
		)
	if system == "dadras":
		p, q, r, s, e = 3.0, 2.7, 1.7, 2.0, 9.0
		return np.array(
			[
				y0 - p * x0 + q * y0 * z0,
				r * y0 - x0 * z0 + z0,
				s * x0 * y0 - e * z0,
			],
			dtype=np.float64,
		)
	if system == "chua":
		alpha, beta, m0, m1 = 15.6, 28.0, -1.143, -0.714
		h = m1 * x0 + 0.5 * (m0 - m1) * (abs(x0 + 1.0) - abs(x0 - 1.0))
		return np.array(
			[
				alpha * (y0 - x0 - h),
				x0 - y0 + z0,
				-beta * y0,
			],
			dtype=np.float64,
		)
	if system == "genesio_tesi":
		a, b, c = 0.44, 1.1, 1.0
		return np.array(
			[
				y0,
				z0,
				-c * x0 - b * y0 - a * z0 + x0 * x0,
			],
			dtype=np.float64,
		)
	if system == "burke_shaw":
		n, e = 10.0, 13.0
		return np.array(
			[
				-n * (x0 + y0),
				y0 - n * x0 * z0,
				n * x0 * y0 + e,
			],
			dtype=np.float64,
		)
	if system == "nose_hoover":
		a = 1.5
		return np.array(
			[
				y0,
				-x0 + y0 * z0,
				a - y0 * y0,
			],
			dtype=np.float64,
		)
	if system == "hadley":
		a, b, f, g = 0.2, 4.0, 9.0, 1.0
		return np.array(
			[
				-y0 * y0 - z0 * z0 - a * x0 + a * f,
				x0 * y0 - b * x0 * z0 - y0 + g,
				b * x0 * y0 + x0 * z0 - z0,
			],
			dtype=np.float64,
		)
	if system == "rabinovich_fabrikant":
		gamma, alpha = 0.87, 1.1
		return np.array(
			[
				y0 * (z0 - 1.0 + x0 * x0) + gamma * x0,
				x0 * (3.0 * z0 + 1.0 - x0 * x0) + gamma * y0,
				-2.0 * z0 * (alpha + x0 * y0),
			],
			dtype=np.float64,
		)
	if system == "qi_chen":
		a, b, c = 38.0, 8.0 / 3.0, 80.0
		return np.array(
			[
				a * (y0 - x0) + y0 * z0,
				c * x0 + y0 - x0 * z0,
				x0 * y0 - b * z0,
			],
			dtype=np.float64,
		)

	raise KeyError(f"No ODE drift implemented for {system_name!r}")


def generate_dynamix_ode_system(system_name: str, n_steps: int, *, seed: int = 0) -> tuple[np.ndarray, float]:
	"""Generate one ODE trajectory."""
	key = str(system_name).strip().lower()
	if key not in DYNAMIX_ODE_SPECS:
		raise KeyError(f"Unknown DynaMix ODE system {system_name!r}")
	spec = DYNAMIX_ODE_SPECS[key]
	rng = np.random.default_rng(int(seed))
	x0 = np.asarray(spec.x0, dtype=np.float64) + 0.01 * rng.standard_normal(3)
	total_t = int(n_steps) + int(spec.burn_in)
	t_eval = np.arange(total_t, dtype=np.float64) * float(spec.dt)

	def rhs(_t: float, y: np.ndarray) -> np.ndarray:
		return dynamix_ode_drift(key, y)

	sol = solve_ivp(
		rhs,
		t_span=(float(t_eval[0]), float(t_eval[-1])),
		y0=x0,
		t_eval=t_eval,
		method="RK45",
		rtol=1.0e-8,
		atol=1.0e-10,
		max_step=float(spec.dt),
	)
	y = sol.y.T
	if not sol.success or y.shape != (total_t, 3):
		raise RuntimeError(f"ODE solve failed for {key}: {sol.message}")
	x = y[int(spec.burn_in) : int(spec.burn_in) + int(n_steps)]
	if not np.all(np.isfinite(x)):
		raise RuntimeError(f"ODE solve produced non-finite states for {key}")
	return x.astype(np.float32), float(spec.dt)


def add_dynamix_standardized_obs_noise(x_raw: np.ndarray, *, sigma: float, seed: int, context_length: int) -> np.ndarray:
	x = np.asarray(x_raw, dtype=np.float32)
	if float(sigma) <= 0.0:
		return x.copy()
	rng = np.random.default_rng(int(seed))
	moment_end = min(max(int(context_length), 2), int(x.shape[0]))
	std = x[:moment_end].std(axis=0)
	std = np.where(std < 1.0e-6, 1.0, std)
	noise = rng.standard_normal(x.shape).astype(np.float32) * (float(sigma) * std.astype(np.float32))
	return (x + noise).astype(np.float32)


__all__ = [
	"DYNAMIX_ODE_SPECS",
	"DynaMixOdeSpec",
	"add_dynamix_standardized_obs_noise",
	"dynamix_nll_horizon_steps",
	"dynamix_ode_drift",
	"dynamix_required_reference_steps",
	"dynamix_scored_window_length",
	"generate_dynamix_ode_system",
]
