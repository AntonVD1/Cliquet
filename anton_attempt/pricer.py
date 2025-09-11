from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional
import numpy as np
import QuantLib as ql

@dataclass
class CliquetPricer:
    # Payoff type: "c" => S_i/S_{i-1}-1 ; "p" => S_{i-1}/S_i-1
    flag: str = "c"

    # Notional & underlying
    nominal: float = 1.0
    stock_price: float = 0.0  # if <= 0, defaults to gbm.s0 (interpreted as T+lag spot)

    # Dates (UNSHIFTED)
    reset_dates: List[date] = field(default_factory=list)
    valdate: date | None = None
    expiry: date | None = None

    # Simulation config
    gbm: "GBM" | None = None
    n_sims: int = 0
    antithetic: bool = False
    seed: Optional[int] = None

    # Discounting
    discount_engine: object | None = None  # needs .discount_factor(valuation_date, start_date, end_date) -> float

    # Local/global caps
    local_cap: float = 1.0e308
    local_floor: float = -1.0e308
    global_cap: float = 10_000_000.0
    global_floor: float = -0.4

    # Underlying spot-day settings (business-day shift)
    underlying_spot_days: int = 3
    calendar: ql.Calendar = field(default_factory=lambda: ql.SouthAfrica())
    bday_convention: ql.BusinessDayConvention = ql.Following

    # Outputs (optional)
    sim_prices: Optional[np.ndarray] = field(init=False, default=None)
    sim_payoffs: Optional[np.ndarray] = field(init=False, default=None)

    # Internal (shifted) dates
    _reset_shifted: List[date] = field(init=False, default_factory=list)
    _val_shifted: Optional[date] = field(init=False, default=None)
    _exp_shifted: Optional[date] = field(init=False, default=None)

    def _shift(self, d: date, n: int) -> date:
        qd = ql.Date(d.day, d.month, d.year)
        sd = self.calendar.advance(qd, ql.Period(n, ql.Days), self.bday_convention)
        return date(sd.year(), sd.month(), sd.dayOfMonth())

    def __post_init__(self):
        if self.gbm is None or self.discount_engine is None:
            raise ValueError("gbm and discount_engine must be provided")
        if self.valdate is None or self.expiry is None:
            raise ValueError("valdate and expiry must be set")

        self.reset_dates = sorted(self.reset_dates)

        lag = int(self.underlying_spot_days)
        self._reset_shifted = [self._shift(d, lag) for d in self.reset_dates]
        self._val_shifted = self._shift(self.valdate, lag)
        self._exp_shifted = self._shift(self.expiry, lag)

        spot0 = self.stock_price if self.stock_price > 0 else float(self.gbm.s0)  # type: ignore

        # Simulate on shifted fixings; GBM stays unaware of spot days
        self.sim_prices = self.gbm.simulate_path_matrix(              # type: ignore
            reset_dates=self._reset_shifted,
            n=self.n_sims,
            seed=self.seed,
            antithetic=self.antithetic,
            spot0=spot0,
        ).astype(float)

        # Local payoffs path-wise
        n_sims, n_dates = self.sim_prices.shape
        prev = np.empty_like(self.sim_prices)
        prev[:, 0] = spot0
        if n_dates > 1:
            prev[:, 1:] = self.sim_prices[:, :-1]

        tiny = 1e-12
        if self.flag == "c":
            local = self.sim_prices / np.maximum(prev, tiny) - 1.0
        else:
            local = prev / np.maximum(self.sim_prices, tiny) - 1.0

        self.sim_payoffs = np.clip(local, self.local_floor, self.local_cap)

    def price(self) -> float:
        # Sum of expected locals with global cap/floor
        avg_per_reset = self.sim_payoffs.mean(axis=0)  # type: ignore
        end_payoff = float(avg_per_reset.sum())
        end_payoff = max(self.global_floor, min(self.global_cap, end_payoff))

        # Front/back DF ratio: DF(0, expiry+lag) / DF(0, lag)
        df_front = self.discount_engine.discount_factor(  # type: ignore
            valuation_date=self.valdate, start_date=self.valdate, end_date=self._val_shifted
        )
        df_back = self.discount_engine.discount_factor(  # type: ignore
            valuation_date=self.valdate, start_date=self.valdate, end_date=self._exp_shifted
        )
        return self.nominal * end_payoff * (df_back / df_front)
