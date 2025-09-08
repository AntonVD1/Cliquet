from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, List, Tuple
import numpy as np
import math

# ---- helpers (local to this module) ----
def year_fraction_act365(d0: date, d1: date) -> float:
    return (d1 - d0).days / 365.0

def _parse_date(s: str) -> date:
    s = str(s).strip()
    fmts = ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date format in dividends CSV: {s!r}")

def _to_float(x: str) -> float:
    return float(str(x).replace(",", "").strip())

CENTS_TO_RANDS = 0.01  # CSV dividend amounts are in CENTS → convert to RANDS

def _read_dividends_csv(csv_path: str) -> List[Tuple[date, float]]:
    """
    Reads a 'wide' CSV laid out as repeating triplets:
        stock_name, pay_date, amount_in_cents, <blank?>, stock_name, pay_date, amount_in_cents, <blank?>, ...
    Returns a flat list of (pay_date, amount_in_rands), ignoring blank/malformed cells and zeros.
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
                i = nxt(i + 1)
                if i >= n: break
                date_cell = str(row[i]).strip()
                i = nxt(i + 1)
                if i >= n: break
                amt_cell = str(row[i]).strip()
                i += 1  # advance

                if date_cell and amt_cell:
                    try:
                        pay_dt = _parse_date(date_cell)
                        amt_cents = _to_float(amt_cell)
                        amt_rands = amt_cents * CENTS_TO_RANDS
                        if amt_rands != 0.0:
                            out.append((pay_dt, amt_rands))
                    except ValueError:
                        continue
    return out

# ---- NEW: PV of future dividends at a specified date ----
def pv_div(
    dividends_csv: str,
    disc_engine: object,   # your Discounter
    valuation_date: date,  # first arg to discount_factor (the curve's "val" date)
    at_date: date,         # the date to which we discount and from which we consider "future"
) -> float:
    """
    PV at 'at_date' of all dividends with pay_date >= at_date.
    Each dividend is discounted individually from pay_date back to at_date using:
        DF(at_date -> pay_date) = disc_engine.discount_factor(valuation_date, at_date, pay_date)
    """
    divs = _read_dividends_csv(dividends_csv)  # [(pay_date, amount_in_rands)]
    total = 0.0
    # Guard: if at_date is before valuation_date, the ratio is still well-defined; we also
    # enforce that dividends earlier than valuation_date are ignored.
    cutoff = max(at_date, valuation_date)
    for pd, a in divs:
        if pd >= cutoff:
            df = disc_engine.discount_factor(valuation_date, at_date, pd)
            total += a * df
    return total


# ---------------- GBM ----------------
@dataclass(frozen=True)
class GBM_with_div:
    """
    Geometric Brownian Motion (GBM) with optional dividend adjustments.

    Optional dividends behavior when both 'dividends_csv' and 'disc_engine' are provided:
      1) Initial spot is reduced by PV of all future dividends at start_date:
           S0_ex = s0 - pv_div(..., at_date=start_date)
      2) At each reset date 'd', the reported value adds back the PV at 'd' of future dividends:
           S_cum(d) = S_ex(d) + pv_div(..., at_date=d)
    """
    s0: float
    sigma: float
    mu: float
    start_date: date

    dividends_csv: Optional[str] = None
    disc_engine: Optional[object] = None  # expects Discounter

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
        Simulate continuous GBM paths sequentially across reset_dates.
        Returns (n, len(reset_dates)) where each row is a path.

        If dividends_csv and disc_engine are provided:
          - initial S is reduced by PV(all future dividends) at start_date
          - at each reset date d, output adds PV(all future dividends at d)
        """
        dates = list(reset_dates)
        m = len(dates)
        out = np.empty((n, m), dtype=float)

        base_s0 = float(self.s0 if (spot0 is None or spot0 <= 0) else spot0)

        # ---- dividends (optional) via pv_div ----
        use_divs = (self.dividends_csv is not None) and (self.disc_engine is not None)
        if use_divs:
            pv_all = pv_div(self.dividends_csv, self.disc_engine, self.start_date, self.start_date)
            pv_at_dates = [
                pv_div(self.dividends_csv, self.disc_engine, self.start_date, d) for d in dates
            ]
        else:
            pv_all = 0.0
            pv_at_dates = [0.0] * m

        # initial ex-div spots
        S = np.full(n, base_s0 - pv_all, dtype=float)
        if np.any(S < 0.0):
            raise ValueError(f"Ex-div initial spot negative (s0={base_s0} - pv_all={pv_all}).")

        # ---- stochastic evolution ----
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

            # ex-div path + PV of FUTURE dividends from this date (valued AT this date)
            out[:, j] = S + pv_at_dates[j]
            prev_date = d

        return out
