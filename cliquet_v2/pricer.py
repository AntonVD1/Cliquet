from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import List, Literal, Optional, Protocol
import numpy as np

# ---- Protocol for discounting ----
class DiscountEngine(Protocol):
    def discount_factor(self, valuation_date: date, start_date: date, end_date: date) -> float: ...

FlagType = Literal["c", "p"]
SettlementType = Literal["Business", "Calendar"]


@dataclass
class CliquetPricer:
    # Flags / types
    flag: FlagType = "c"  # 'c' = call-style (S/K - 1), 'p' = put-style (K/S - 1)
    pay_type: int = 1
    cap_type: int = 1
    floor_type: int = 1

    # Caps / floors
    local_cap: float = 1.0e308
    global_cap: float = 10_000_000.0
    local_floor: float = -1.0e308
    global_floor: float = -0.4

    # Notional / Nominal
    nominal: float = 1.0

    # Underlying & strike
    stock_price: float = 0.0
    K: float = 0.0

    # Dates
    reset_dates: List[date] = field(default_factory=list)
    sigma: float = 0.0
    startdate: date | None = None
    valdate: date | None = None
    expiry: date | None = None

    # Settlement
    spot_days: int = 3
    settlement_type: SettlementType = "Business"

    # Simulation config
    gbm: "GBM" | None = None
    n_sims: int = 0
    antithetic: bool = False
    seed: Optional[int] = None

    # Discounting
    discount_engine: DiscountEngine | None = None

    # Outputs
    sim_prices: Optional[np.ndarray] = field(init=False, default=None)   # (n_sims, n_dates)
    sim_payoffs: Optional[np.ndarray] = field(init=False, default=None)  # (n_sims, n_dates)

    def __post_init__(self):
        if self.gbm is None:
            raise ValueError("CliquetPricer requires a GBM instance.")
        if self.discount_engine is None:
            raise ValueError("CliquetPricer requires a discount engine instance.")
        if self.n_sims <= 0:
            raise ValueError("n_sims must be positive.")
        if self.antithetic and (self.n_sims % 2 != 0):
            raise ValueError("antithetic=True requires n_sims to be even.")
        if not self.reset_dates:
            raise ValueError("reset_dates must be non-empty.")
        if self.valdate is None or self.expiry is None:
            raise ValueError("valdate and expiry must be provided.")

        # 1) Simulate terminal prices at each reset date
        cols = []
        for i, d in enumerate(self.reset_dates):
            col = self.gbm.simulate_at(
                to_date=d,
                n=self.n_sims,
                seed=None if self.seed is None else self.seed + i,
                antithetic=self.antithetic,
            )
            cols.append(col)
        self.sim_prices = np.column_stack(cols)  # shape (n_sims, n_dates)

        # 2) Compute local payoffs per reset per simulation with local cap/floor
        tiny = 1e-12
        if self.flag == "c":
            # call-style: S/K - 1
            denom = max(float(self.K), tiny)
            base = self.sim_prices / denom - 1.0
        else:
            # put-style: K/S - 1
            denom = np.maximum(self.sim_prices, tiny)
            base = float(self.K) / denom - 1.0

        self.sim_payoffs = np.clip(base, self.local_floor, self.local_cap)  # (n_sims, n_dates)

    def price(self) -> float:
        """
        Price = nominal * clip_global( sum_over_resets( average_over_sims(local_payoff) ) )
                          * DF(valdate -> expiry)
        """
        if self.sim_payoffs is None:
            raise RuntimeError("sim_payoffs not available.")

        avg_per_reset = self.sim_payoffs.mean(axis=0)   # (n_dates,)
        end_payoff = float(avg_per_reset.sum())
        end_payoff = max(self.global_floor, min(self.global_cap, end_payoff))  # global floor/cap

        df = self.discount_engine.discount_factor(
            valuation_date=self.valdate, start_date=self.valdate, end_date=self.expiry
        )
        return self.nominal * end_payoff * df
