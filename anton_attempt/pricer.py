from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional
import numpy as np

@dataclass
class CliquetPricer:
    flag: str = "c"  # "c" => S_i/S_{i-1}-1 ; "p" => S_{i-1}/S_i-1

    # Caps / floors
    local_cap: float = 1.0e308
    local_floor: float = -1.0e308
    global_cap: float = 10_000_000.0
    global_floor: float = -0.4

    # Notional & underlying
    nominal: float = 1.0
    stock_price: float = 0.0  # if <= 0, defaults to gbm.s0

    # Dates
    reset_dates: List[date] = field(default_factory=list)
    valdate: date | None = None
    expiry: date | None = None

    # Simulation config
    gbm: "GBM" | None = None
    n_sims: int = 0
    antithetic: bool = False
    seed: Optional[int] = None

    # Discounting
    discount_engine: object | None = None  # any object with .discount_factor(valuation_date, start_date, end_date)

    # Outputs
    sim_prices: Optional[np.ndarray] = field(init=False, default=None)   # (n_sims, n_dates)
    sim_payoffs: Optional[np.ndarray] = field(init=False, default=None)  # (n_sims, n_dates)

    def __post_init__(self):
        spot0 = self.stock_price if self.stock_price > 0 else float(self.gbm.s0)  # type: ignore
        self.reset_dates = sorted(self.reset_dates)

        # 1) Simulate continuous (sequential) GBM paths across reset dates
        self.sim_prices = self.gbm.simulate_path_matrix(              # type: ignore
            reset_dates=self.reset_dates,
            n=self.n_sims,
            seed=self.seed,
            antithetic=self.antithetic,
            spot0=spot0,
        ).astype(float)  # shape: (n_sims, n_dates)

        # 2) Previous-price matrix (path-wise)
        n_sims, n_dates = self.sim_prices.shape
        prev = np.empty_like(self.sim_prices, dtype=float)
        prev[:, 0] = spot0
        if n_dates > 1:
            prev[:, 1:] = self.sim_prices[:, :-1]

        # 3) Local payoffs (elementwise, safe denom), then local cap/floor
        tiny = 1e-12
        if self.flag == "c":
            denom = np.maximum(prev, tiny)
            local = self.sim_prices / denom - 1.0
        else:
            denom = np.maximum(self.sim_prices, tiny)
            local = prev / denom - 1.0

        self.sim_payoffs = np.clip(local, self.local_floor, self.local_cap)

    def price(self) -> float:
        avg_per_reset = self.sim_payoffs.mean(axis=0)  # type: ignore
        end_payoff = float(avg_per_reset.sum())
        end_payoff = max(self.global_floor, min(self.global_cap, end_payoff))
        df = self.discount_engine.discount_factor(  # type: ignore
            valuation_date=self.valdate, start_date=self.valdate, end_date=self.expiry
        )
        return self.nominal * end_payoff * df
