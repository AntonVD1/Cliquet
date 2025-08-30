from __future__ import annotations

"""Simple Monte Carlo pricer for a Cliquet option.

This pricer uses the :func:`simulate_gbm` function to generate asset paths
under a geometric Brownian motion with time‑dependent drift derived from
discount factors.  The payoff is the sum of capped and floored *absolute*
price differences over each reset period.

The implementation purposely avoids heavy external dependencies such as
QuantLib and relies only on the standard library and NumPy.
"""

from dataclasses import dataclass
from datetime import date, timedelta
import math
from typing import Callable

import numpy as np

from gbm import simulate_gbm
from helpers import year_fraction_365


@dataclass
class CliquetOptionPricer:
    """Monte Carlo pricer for a discrete‑reset Cliquet option.

    Parameters
    ----------
    S : float
        Initial underlying spot price.
    r : float
        Continuously compounded risk‑free rate.
    dividend : float
        Continuously compounded dividend yield.  Only ``r - dividend`` is used
        for the drift of the GBM simulation.
    sigma : float
        Constant volatility.
    start_date : ``datetime.date``
        Simulation start and option valuation date.
    periods : int
        Number of reset periods.
    tenor_years : float
        Length of each reset period in years (e.g. ``1.0`` for annual
        resets).
    cap, floor : float
        Local cap and floor applied to the *absolute* price change per period.
    steps_per_period : int, optional
        Number of simulation time steps per period.
    samples : int, optional
        Number of Monte Carlo paths.
    seed : int, optional
        Random seed for reproducibility.
    """

    S: float
    r: float
    dividend: float
    sigma: float
    start_date: date
    periods: int
    tenor_years: float
    cap: float
    floor: float
    steps_per_period: int = 10
    samples: int = 50_000
    seed: int = 42

    def __post_init__(self) -> None:
        # total number of simulation steps
        self.total_steps = self.periods * self.steps_per_period

        # generate the list of dates used for simulation
        dt_years = (self.tenor_years * self.periods) / self.total_steps
        self._dates = [self.start_date]
        for i in range(1, self.total_steps + 1):
            self._dates.append(
                self.start_date + timedelta(days=int(round(i * dt_years * 365)))
            )

        self.maturity_date = self._dates[-1]
        self.total_years = year_fraction_365(self.start_date, self.maturity_date)

        # Discount factor for the GBM drift (risk‑free minus dividend)
        def _df(d0: date, d1: date) -> float:
            dt = year_fraction_365(d0, d1)
            return math.exp(-(self.r - self.dividend) * dt)

        self._discount_factor: Callable[[date, date], float] = _df

    # ------------------------------------------------------------------
    def _path_payoff(self, path: np.ndarray) -> float:
        """Cliquet payoff using absolute price differences per period."""
        payoff = 0.0
        prev_price = path[0]

        for i in range(self.periods):
            idx = (i + 1) * self.steps_per_period
            st = path[idx]
            diff = st - prev_price
            payoff += min(max(diff, self.floor), self.cap)
            prev_price = st

        return payoff

    # ------------------------------------------------------------------
    def price(self) -> float:
        """Estimate the option price via Monte Carlo simulation."""
        rng = np.random.default_rng(self.seed)
        sum_payoffs = 0.0

        for _ in range(self.samples):
            path = simulate_gbm(
                self.S, self.sigma, self._dates, self._discount_factor, rng
            )
            sum_payoffs += self._path_payoff(path)

        mean_payoff = sum_payoffs / float(self.samples)
        discount = math.exp(-self.r * self.total_years)
        return discount * mean_payoff


__all__ = ["CliquetOptionPricer"]

