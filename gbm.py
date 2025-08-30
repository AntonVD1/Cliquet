from __future__ import annotations

"""Geometric Brownian motion simulation with time-dependent drift.

This module provides a discretised GBM simulator where the drift at each
step is derived from discount factors. A callable ``discount_factor`` is
expected, which returns the discount factor between two ``datetime.date``
objects. The continuous forward rate used for the drift is computed as
``-ln(DF)/dt`` for each step, making the risk-free rate dynamic over time.
"""

from datetime import date
import math
from typing import Callable, Sequence, Optional

import numpy as np

from helpers import year_fraction_365


def simulate_gbm(
    s0: float,
    vol: float,
    dates: Sequence[date],
    discount_factor: Callable[[date, date], float],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate a geometric Brownian motion path.

    Parameters
    ----------
    s0 : float
        Starting asset price.
    vol : float
        Constant volatility of the process.
    dates : Sequence[date]
        Strictly increasing sequence of dates including the start date.
    discount_factor : Callable[[date, date], float]
        Function returning the discount factor between two dates.
    rng : np.random.Generator, optional
        Optional NumPy random generator. ``np.random.default_rng()`` is used
        if not provided.

    Returns
    -------
    np.ndarray
        Simulated asset prices for each date.
    """
    if len(dates) < 2:
        raise ValueError("dates must include at least start and end")

    rng = rng or np.random.default_rng()
    path = np.empty(len(dates), dtype=float)
    path[0] = s0

    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i + 1]
        dt = year_fraction_365(d0, d1)
        if dt <= 0:
            raise ValueError("dates must be strictly increasing")
        df = discount_factor(d0, d1)
        if df <= 0:
            raise ValueError("discount factor must be positive")
        r = -math.log(df) / dt  # continuous forward rate
        drift = (r - 0.5 * vol * vol) * dt
        diffusion = vol * math.sqrt(dt) * rng.normal()
        path[i + 1] = path[i] * math.exp(drift + diffusion)

    return path


def plot_gbm_simulations(
    s0: float,
    vol: float,
    dates: Sequence[date],
    discount_factor: Callable[[date, date], float],
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate and plot multiple GBM paths.

    Parameters
    ----------
    s0 : float
        Starting asset price.
    vol : float
        Constant volatility of the process.
    dates : Sequence[date]
        Strictly increasing sequence of dates including the start date.
    discount_factor : Callable[[date, date], float]
        Function returning the discount factor between two dates.
    n : int
        Number of simulated paths.
    rng : np.random.Generator, optional
        Optional NumPy random generator. ``np.random.default_rng()`` is used
        if not provided.

    Returns
    -------
    np.ndarray
        Simulated asset prices with shape ``(n, len(dates))``.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = rng or np.random.default_rng()
    paths = np.empty((n, len(dates)), dtype=float)
    for i in range(n):
        paths[i] = simulate_gbm(s0, vol, dates, discount_factor, rng)

    import matplotlib.pyplot as plt  # Import here to keep optional

    for path in paths:
        plt.plot(dates, path)

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{n} GBM simulations")
    plt.grid(True)
    plt.show()

    return paths


__all__ = ["simulate_gbm", "plot_gbm_simulations"]

