from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Optional, List
import numpy as np
import math


def year_fraction_act365(d0: date, d1: date) -> float:
    return (d1 - d0).days / 365.0


@dataclass(frozen=True)
class GBM:
    """
    Geometric Brownian Motion (GBM) simulator with user-specified drift.

    Model dynamics:
        dS_t = mu * S_t dt + sigma * S_t dW_t

    Exact solution (from start_date to t):
        S_t = S_0 * exp( (mu - 0.5*sigma^2) * T + sigma * sqrt(T) * Z ),  Z ~ N(0,1)

    Parameters
    ----------
    s0 : float
        Initial spot price.
    sigma : float
        Volatility of the process (annualized).
    mu : float
        Continuous drift rate (annualized).
    start_date : date
        Start/valuation date of the simulation.
    """
    s0: float
    sigma: float
    mu: float
    start_date: date

    def simulate_at(
        self,
        to_date: date,
        n: int,
        seed: Optional[int] = None,
        antithetic: bool = False,
    ) -> np.ndarray:
        """
        Simulate marginal S(to_date) drawn directly from S(start_date) (no path continuity across dates).
        """
        t = year_fraction_act365(self.start_date, to_date)
        if t == 0.0:
            return np.full(n, self.s0, dtype=float)

        rng = np.random.default_rng(seed)
        if antithetic:
            if n % 2 != 0:
                raise ValueError("antithetic=True requires even n")
            half = n // 2
            z_half = rng.standard_normal(half)
            z = np.concatenate([z_half, -z_half])
        else:
            z = rng.standard_normal(n)

        exponent = (self.mu - 0.5 * self.sigma**2) * t + self.sigma * math.sqrt(t) * z
        return self.s0 * np.exp(exponent)

    def simulate_path_matrix(
        self,
        reset_dates: List[date],
        n: int,
        seed: Optional[int] = None,
        antithetic: bool = False,
        spot0: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simulate continuous GBM paths sequentially across the provided reset_dates.
        Returns an array of shape (n, len(reset_dates)) where each row is one continuous path.

        Step update over Δt between consecutive dates:
            S_{t+Δt} = S_t * exp( (mu - 0.5*sigma^2) * Δt + sigma * sqrt(Δt) * Z ),  Z ~ N(0,1)
        """
        dates = list(reset_dates)  # preserve input order
        m = len(dates)
        out = np.empty((n, m), dtype=float)

        # initial spots
        S = np.full(n, float(self.s0 if (spot0 is None or spot0 <= 0) else spot0), dtype=float)

        rng = np.random.default_rng(seed)
        if antithetic and (n % 2 != 0):
            raise ValueError("antithetic=True requires even n")

        prev_date = self.start_date
        mu = float(self.mu)
        sig = float(self.sigma)

        for j, d in enumerate(dates):
            dt = year_fraction_act365(prev_date, d)

            if dt > 0.0:
                if antithetic:
                    half = n // 2
                    z_half = rng.standard_normal(half)
                    z = np.concatenate([z_half, -z_half])
                else:
                    z = rng.standard_normal(n)

                drift = (mu - 0.5 * sig * sig) * dt
                diff = sig * math.sqrt(dt) * z
                S = S * np.exp(drift + diff)

            # if dt == 0, S remains unchanged
            out[:, j] = S
            prev_date = d

        return out
