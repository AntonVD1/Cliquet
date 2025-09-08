from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, List, Tuple
import numpy as np

CENTS_TO_RANDS = 0.01  # CSV dividends are in cents → convert to rands

# ---------- CSV + PV helpers ----------
def _parse_date(s: str) -> date:
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date format in dividends CSV: {s!r}")

def _to_float(x: str) -> float:
    return float(str(x).replace(",", "").strip())

def _read_dividends_csv(csv_path: str) -> List[Tuple[date, float]]:
    """
    Wide CSV repeating triplets:
      stock_name, pay_date, amount_in_cents, <blank?>, stock_name, pay_date, amount_in_cents, <blank?>, ...
    Returns a flat list: [(pay_date, amount_in_rands), ...]
    """
    import csv
    out: List[Tuple[date, float]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f, skipinitialspace=True)
        for row in rdr:
            if not row:
                continue
            i, n = 0, len(row)

            def nxt(k: int) -> int:
                while k < n and (row[k] is None or str(row[k]).strip() == ""):
                    k += 1
                return k

            while True:
                i = nxt(i)
                if i >= n: break
                # stock = row[i]  # not needed for aggregate PV
                i = nxt(i + 1);  0
                if i >= n: break
                date_cell = str(row[i]).strip()
                i = nxt(i + 1)
                if i >= n: break
                amt_cell = str(row[i]).strip()
                i += 1

                if date_cell and amt_cell:
                    try:
                        pay_dt = _parse_date(date_cell)
                        amt = _to_float(amt_cell) * CENTS_TO_RANDS
                        if amt != 0.0:
                            out.append((pay_dt, amt))
                    except ValueError:
                        continue
    return out

def _pv_future_divs_from_list(
    divs: List[Tuple[date, float]],
    disc_engine,          # your Discounter
    valuation_date: date, # first arg used by Discounter
    at_date: date,        # discount-to date and cutoff for "future"
) -> float:
    """
    PV at 'at_date' of all dividends with pay_date >= at_date.
    Uses DF(at_date -> pay_date) = disc_engine.discount_factor(valuation_date, at_date, pay_date).
    Dividends before valuation_date are ignored.
    """
    total = 0.0
    cutoff = max(at_date, valuation_date)
    for pd, amt in divs:
        if pd >= cutoff:
            df = disc_engine.discount_factor(valuation_date, at_date, pd)
            total += amt * df
    return total


# ---------- The adapter ----------
@dataclass(frozen=True)
class GBMDividendAdjusted:
    """
    Wraps an existing GBM and adjusts outputs for dividends:
      - Subtract PV(all future dividends at start_date) from the initial level (ex-div scaling)
      - Add PV(future dividends) at each reported date

    Pass an instance of this adapter anywhere your code expects a GBM:
        pricer = CliquetPricer(..., gbm=GBMDividendAdjusted(base_gbm, dividends_csv, disc_engine), ...)
    """
    base: object            # the original GBM instance
    dividends_csv: str
    disc_engine: object     # your Discounter

    # Expose common GBM attributes in case callers access them
    @property
    def s0(self) -> float: return float(self.base.s0)
    @property
    def sigma(self) -> float: return float(self.base.sigma)
    @property
    def mu(self) -> float: return float(self.base.mu)
    @property
    def start_date(self) -> date: return self.base.start_date

    # Preload dividends once
    def _divs(self) -> List[Tuple[date, float]]:
        return _read_dividends_csv(self.dividends_csv)

    # Convenience: PV of future dividends at an arbitrary date
    def pv_div_at(self, at_date: date) -> float:
        return _pv_future_divs_from_list(self._divs(), self.disc_engine, self.start_date, at_date)

    # Marginal simulation, adjusted
    def simulate_at(
        self,
        to_date: date,
        n: int,
        seed: Optional[int] = None,
        antithetic: bool = False,
    ) -> np.ndarray:
        # Raw marginal GBM (cum-like)
        raw = self.base.simulate_at(to_date, n, seed=seed, antithetic=antithetic)

        # Ex-div scaling at t0: s0_ex = s0 - PV_all(start)
        pv_start = self.pv_div_at(self.start_date)
        if self.s0 <= 0:
            raise ValueError("GBM s0 must be positive.")
        scale = (self.s0 - pv_start) / self.s0
        if scale < 0:
            raise ValueError(f"PV of dividends ({pv_start:.6g}) exceeds s0 ({self.s0:.6g}).")

        # Add PV of future dividends at to_date
        pv_future_at_to = self.pv_div_at(to_date)
        return raw * scale + pv_future_at_to

    # Path simulation, adjusted
    def simulate_path_matrix(
        self,
        reset_dates: List[date],
        n: int,
        seed: Optional[int] = None,
        antithetic: bool = False,
        spot0: Optional[float] = None,
    ) -> np.ndarray:
        # 1) Raw GBM paths
        raw = self.base.simulate_path_matrix(
            reset_dates=reset_dates,
            n=n,
            seed=seed,
            antithetic=antithetic,
            spot0=spot0,
        )

        # 2) Ex-div scaling based on the actual starting spot used
        s0_used = float(self.s0 if (spot0 is None or spot0 <= 0) else spot0)
        if s0_used <= 0:
            raise ValueError("spot0/s0 must be positive.")
        pv_start = self.pv_div_at(self.start_date)
        scale = (s0_used - pv_start) / s0_used
        if scale < 0:
            raise ValueError(f"PV of dividends ({pv_start:.6g}) exceeds starting spot ({s0_used:.6g}).")
        ex_div = raw * scale  # shape (n, m)

        # 3) Deterministic PV-of-future-divs add-back at each date
        divs = self._divs()  # avoid re-reading CSV multiple times
        pv_vec = np.array(
            [_pv_future_divs_from_list(divs, self.disc_engine, self.start_date, d) for d in reset_dates],
            dtype=float,
        )  # shape (m,)

        return ex_div + pv_vec[np.newaxis, :]
