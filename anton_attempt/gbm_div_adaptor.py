from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Tuple, Optional
import numpy as np

CENTS_TO_RANDS = 0.01  # CSV dividends are in cents

# ---------- CSV + PV helpers ----------
def _parse_date(s: str) -> date:
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date format: {s!r}")

def _to_float(x: str) -> float:
    return float(str(x).replace(",", "").strip())

def _read_dividends_csv(csv_path: str) -> List[Tuple[date, float]]:
    """
    Wide CSV repeating triplets per row:
      stock_name, pay_date, amount_in_cents, <blank?>, stock_name, pay_date, amount_in_cents, <blank?>, ...
    Returns [(pay_date, amount_in_rands), ...]
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
                # stock = row[i]  # not needed for aggregate PV add-back
                i = nxt(i + 1); 
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
    discounter,            # your Discounter
    valuation_date: date,  # first arg used by discount_factor
    at_date: date,         # discount-to date AND "future" cutoff
) -> float:
    """
    PV at 'at_date' of all dividends with pay_date >= at_date.
    Each discounted from pay_date back to at_date:
       DF(at_date -> pay_date) = discounter.discount_factor(valuation_date, at_date, pay_date)
    Dividends before valuation_date are ignored.
    """
    total = 0.0
    cutoff = max(at_date, valuation_date)
    for pd, amt in divs:
        if pd >= cutoff:
            df = discounter.discount_factor(valuation_date, at_date, pd)
            total += amt * df
    return total

# ---------- GBM adapter (add-back only) ----------
@dataclass(frozen=True)
class GBMDividendAddbackAdapter:
    """
    Wrap a GBM that was run EX-DIV (i.e., you already subtracted PV at start from s0),
    and add back the PV of FUTURE dividends at each requested date.

    Use it anywhere a GBM is expected (same methods exposed).
    """
    base: object                 # the original GBM instance
    dividends_csv: str
    discounter: object           # your Discounter
    valuation_date: Optional[date] = None  # defaults to base.start_date

    def __post_init__(self):
        # Cache dividends once
        object.__setattr__(self, "_divs", _read_dividends_csv(self.dividends_csv))
        # Choose valuation date
        val = self.valuation_date if self.valuation_date is not None else self.base.start_date
        object.__setattr__(self, "_valdate", val)

    # passthrough handy attrs (optional)
    @property
    def start_date(self) -> date: return self.base.start_date
    @property
    def s0(self) -> float: return self.base.s0
    @property
    def sigma(self) -> float: return self.base.sigma
    @property
    def mu(self) -> float: return self.base.mu

    # Marginal simulate with add-back (if you ever use simulate_at)
    def simulate_at(
        self,
        to_date: date,
        n: int,
        seed: Optional[int] = None,
        antithetic: bool = False,
    ) -> np.ndarray:
        raw = self.base.simulate_at(to_date, n, seed=seed, antithetic=antithetic)  # ex-div path
        pv_future = _pv_future_divs_from_list(self._divs, self.discounter, self._valdate, to_date)
        return raw + pv_future

    # Path matrix with add-back (what your pricer uses)
    def simulate_path_matrix(
        self,
        reset_dates: List[date],
        n: int,
        seed: Optional[int] = None,
        antithetic: bool = False,
        spot0: Optional[float] = None,
    ) -> np.ndarray:
        # 1) Run the base GBM normally (it should already be ex-div via your adjusted s0)
        raw = self.base.simulate_path_matrix(
            reset_dates=reset_dates,
            n=n,
            seed=seed,
            antithetic=antithetic,
            spot0=spot0,
        )  # shape (n, m)

        # 2) Deterministic add-back per date: PV of FUTURE dividends valued AT that date
        pv_vec = np.array(
            [_pv_future_divs_from_list(self._divs, self.discounter, self._valdate, d) for d in reset_dates],
            dtype=float,
        )  # shape (m,)

        return raw + pv_vec[np.newaxis, :]
