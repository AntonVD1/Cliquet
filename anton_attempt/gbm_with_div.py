from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Optional, List, Sequence, Tuple, Dict
import numpy as np
import math

def year_fraction_act365(d0: date, d1: date) -> float:
    return (d1 - d0).days / 365.0

@dataclass(frozen=True)
class GBM:
    """
    Geometric Brownian Motion (GBM) with optional *discrete cash dividends*.

    Dynamics between dates (no q):
        S_{t+Δ} = S_t * exp( (mu - 0.5*sigma^2)*Δ + sigma*sqrt(Δ)*Z ),  Z~N(0,1)
    Jump at an ex-date τ with cash dividend D (in rands per share):
        S_{τ^+} = max(S_{τ^-} - D, 0)   (if clip_at_zero=True)

    Parameters
    ----------
    s0 : float
        Initial spot price (cum-div price at start_date).
    sigma : float
        Volatility (annualized).
    mu : float
        Continuous drift (annualized). Under Q, typically set to r (NOT r - q here).
    start_date : date
        Simulation valuation/start date.
    dividends : Optional[List[Tuple[date, float]]]
        Optional list of (ex_date, cash_div_in_rands). Single-stock assumption.
        If multiple entries share the exact same date, the last one wins.
    clip_at_zero : bool
        Clamp S >= 0 after a dividend jump.
    """
    s0: float
    sigma: float
    mu: float
    start_date: date
    dividends: Optional[List[Tuple[date, float]]] = None
    clip_at_zero: bool = True

    # ---- internal: map ex-dates to cash amounts (no aggregation across stocks) ----
    def _div_map(self, start: date, end: date) -> Dict[date, float]:
        m: Dict[date, float] = {}
        if self.dividends:
            for d_ex, cash in self.dividends:
                if start <= d_ex <= end:
                    m[d_ex] = float(cash)  # last one wins if same date repeats
        return m

    def simulate_at(
        self,
        to_date: date,
        n: int,
        seed: Optional[int] = None,
        antithetic: bool = False,
    ) -> np.ndarray:
        """
        Simulate marginal S(to_date) with discrete cash-div jumps applied at any ex-dates ≤ to_date.
        """
        if to_date < self.start_date:
            raise ValueError("to_date must be on/after start_date")

        # Build mini timeline: ex-dates up to to_date (inclusive), then to_date.
        div_map = self._div_map(self.start_date, to_date)
        timeline = sorted(set(div_map.keys()) | {to_date})

        S = np.full(n, float(self.s0), dtype=float)
        rng = np.random.default_rng(seed)
        if antithetic and (n % 2 != 0):
            raise ValueError("antithetic=True requires even n")

        prev = self.start_date
        mu, sig = float(self.mu), float(self.sigma)

        for t in timeline:
            dt = year_fraction_act365(prev, t)
            if dt > 0.0:
                if antithetic:
                    half = n // 2
                    z_half = rng.standard_normal(half)
                    z = np.concatenate([z_half, -z_half])
                else:
                    z = rng.standard_normal(n)
                S *= np.exp((mu - 0.5 * sig * sig) * dt + sig * math.sqrt(dt) * z)

            # Apply cash dividend jump at t (if any)
            D = div_map.get(t)
            if D is not None:
                S -= D
                if self.clip_at_zero:
                    S = np.maximum(S, 0.0)

            prev = t

        return S

    def simulate_path_matrix(
        self,
        reset_dates: Sequence[date],
        n: int,
        seed: Optional[int] = None,
        antithetic: bool = False,
        spot0: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simulate continuous GBM paths across the provided reset_dates with cash-div jumps.
        Returns shape (n, len(reset_dates)), each row = one path.
        """
        dates = list(reset_dates)
        if not dates:
            raise ValueError("reset_dates must be non-empty")
        if any(d < self.start_date for d in dates):
            raise ValueError("all reset_dates must be on/after start_date")
        if any(dates[i] >= dates[i+1] for i in range(len(dates)-1)):
            raise ValueError("reset_dates must be strictly increasing")

        out = np.empty((n, len(dates)), dtype=float)

        # Initial spot (cum-div)
        S = np.full(n, float(self.s0 if (spot0 is None or spot0 <= 0) else spot0), dtype=float)

        rng = np.random.default_rng(seed)
        if antithetic and (n % 2 != 0):
            raise ValueError("antithetic=True requires even n")

        # Union timeline = reset dates ∪ ex-dates
        div_map = self._div_map(self.start_date, dates[-1])
        timeline = sorted(set(dates) | set(div_map.keys()))

        prev = self.start_date
        mu, sig = float(self.mu), float(self.sigma)
        j_out = 0
        for t in timeline:
            dt = year_fraction_act365(prev, t)
            if dt > 0.0:
                if antithetic:
                    half = n // 2
                    z_half = rng.standard_normal(half)
                    z = np.concatenate([z_half, -z_half])
                else:
                    z = rng.standard_normal(n)

                S *= np.exp((mu - 0.5 * sig * sig) * dt + sig * math.sqrt(dt) * z)

            # Dividend jump at t, if any
            D = div_map.get(t)
            if D is not None:
                S -= D
                if self.clip_at_zero:
                    S = np.maximum(S, 0.0)

            # Record if t is a reset date
            if j_out < len(dates) and t == dates[j_out]:
                out[:, j_out] = S
                j_out += 1

            prev = t

        if j_out != len(dates):
            raise RuntimeError("Internal timeline mismatch when writing outputs.")

        return out
