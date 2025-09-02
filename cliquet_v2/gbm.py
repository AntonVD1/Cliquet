from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Optional, Protocol
import numpy as np
import math


class DiscountEngine(Protocol):
    def discount_factor(self, valuation_date: date, start_date: date, end_date: date) -> float: ...


def year_fraction_act365(d0: date, d1: date) -> float:
    return (d1 - d0).days / 365.0


@dataclass(frozen=True)
class GBM:
    """
    GBM with drift tied to the discount engine:
      exponent uses  ln(DF(start→to)) - 0.5*sigma^2*t + sigma*sqrt(t)*Z

    Where DF(start→to) is obtained via:
      discount_engine.discount_factor(valuation_date=start_date,
                                      start_date=start_date,
                                      end_date=to_date)
    """
    s0: float
    sigma: float
    start_date: date
    discount_engine: DiscountEngine

    def simulate_at(
        self,
        to_date: date,
        n: int,
        seed: Optional[int] = None,
        antithetic: bool = False,
    ) -> np.ndarray:
        t = year_fraction_act365(self.start_date, to_date)
        if t == 0.0:
            return np.full(n, self.s0, dtype=float)

        df = self.discount_engine.discount_factor(
            valuation_date=self.start_date,
            start_date=self.start_date,
            end_date=to_date,
        )
        drift_factor = math.log(df)

        rng = np.random.default_rng(seed)
        if antithetic:
            if n % 2 != 0:
                raise ValueError("antithetic=True requires even n")
            half = n // 2
            z_half = rng.standard_normal(half)
            z = np.concatenate([z_half, -z_half])
        else:
            z = rng.standard_normal(n)

        diffusion = self.sigma * math.sqrt(abs(t)) * z * (1.0 if t >= 0 else -1.0)
        exponent = drift_factor - 0.5 * (self.sigma ** 2) * t + diffusion
        return self.s0 * np.exp(exponent)
