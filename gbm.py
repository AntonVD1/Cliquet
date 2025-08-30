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
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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


def plot_gbm_paths(
    s0: float,
    vol: float,
    dates: Sequence[date],
    discount_factor: Callable[[date, date], float],
    n: int,
    rng: Optional[np.random.Generator] = None,
    *,
    show: bool = True,
    ax: Optional[Axes] = None,
) -> np.ndarray:
    """Simulate and plot ``n`` geometric Brownian motion paths.

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
    show : bool, default True
        Whether to display the plot using :func:`matplotlib.pyplot.show`.
    ax : matplotlib.axes.Axes, optional
        Existing axes on which to draw the paths.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, len(dates))`` containing the simulated paths.
    """
    rng = rng or np.random.default_rng()
    paths = np.vstack(
        [simulate_gbm(s0, vol, dates, discount_factor, rng) for _ in range(n)]
    )

    ax = ax or plt.gca()
    for path in paths:
        ax.plot(dates, path, alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"GBM simulations ({n})")
    if show:
        plt.show()

    return paths


__all__ = ["simulate_gbm", "plot_gbm_paths"]

