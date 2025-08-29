# curves_from_wide_csv.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Tuple
from datetime import date, datetime
import math
import pandas as pd
import numpy as np

from discount.helpers import year_fraction_365, cont_from_naca, naca_from_cont


# -------------------------
# Yield curve object
# -------------------------
@dataclass
class YieldCurveNACA365:
    """
    Yield curve built from NACA/365 spot pillars (Date, NACA).
    Internally stores a continuous-compounded zero curve Z(t) with
    linear interpolation in time t (years) from a chosen valuation_date.

        r_cont_i = ln(1 + r_naca_i)
        Z(t)     = linear_interp( t_i, r_cont_i )
        D(t)     = exp( - Z(t) * t )

    Discount factor between two dates dA -> dB:
        DF(dA->dB) = D(t_B) / D(t_A) = exp( -[Z(t_B)*t_B - Z(t_A)*t_A] ).
    """
    name: str
    pillars: pd.DataFrame  # columns: ["date", "naca"], sorted ascending by date
    valuation_date: date   # anchor for t=0 (not required to be "today")

    def __post_init__(self):
        if self.pillars.empty:
            raise ValueError(f"{self.name}: no pillars provided")
        # Unique, sorted
        self.pillars = (
            self.pillars.dropna(subset=["date", "naca"])
                        .drop_duplicates(subset=["date"])
                        .sort_values("date")
                        .reset_index(drop=True)
        )
        # Precompute times and cont rates
        self._t = np.array([year_fraction_365(self.valuation_date, d) for d in self.pillars["date"]], dtype=float)
        if np.any(self._t < -1e-12):
            raise ValueError(f"{self.name}: pillar date before valuation_date; adjust valuation_date.")
        self._r_cont = np.array([cont_from_naca(x) for x in self.pillars["naca"]], dtype=float)

        # Sanity: times must be strictly increasing (duplicates were dropped)
        if not np.all(np.diff(self._t) > 0):
            raise ValueError(f"{self.name}: pillar times must be strictly increasing.")

    # ---- core interpolation ----
    def _Z_cont(self, t: float) -> float:
        """Piecewise-linear interpolation of continuous zero Z(t). No extrapolation by default."""
        if t < self._t[0] - 1e-12 or t > self._t[-1] + 1e-12:
            raise ValueError(f"{self.name}: requested time {t:.6f} y is outside pillar range [{self._t[0]:.6f}, {self._t[-1]:.6f}]")
        # Exact match?
        j = np.searchsorted(self._t, t)
        if j < len(self._t) and abs(self._t[j] - t) < 1e-12:
            return self._r_cont[j]
        if j == 0:
            j = 1
        if j == len(self._t):
            j = len(self._t) - 1
        t0, t1 = self._t[j - 1], self._t[j]
        z0, z1 = self._r_cont[j - 1], self._r_cont[j]
        w = (t - t0) / (t1 - t0)
        return z0 * (1.0 - w) + z1 * w

    def _t_of(self, d: date) -> float:
        return year_fraction_365(self.valuation_date, d)

    # ---- public API ----
    def naca_rate(self, d: date) -> float:
        """Interpolated NACA/365 spot at date d."""
        t = self._t_of(d)
        zc = self._Z_cont(t)
        return naca_from_cont(zc)

    def discount_factor(self, d_from: date, d_to: date) -> float:
        """
        Discount factor from d_from -> d_to under this zero curve (Act/365F).
        Raises if either date is outside pillar span.
        """
        if d_from == d_to:
            return 1.0
        tA = self._t_of(d_from)
        tB = self._t_of(d_to)
        zA = self._Z_cont(tA)
        zB = self._Z_cont(tB)
        # D(t) = exp(-Z(t) * t), so DF(A->B) = D(tB)/D(tA)
        return math.exp(-(zB * tB - zA * tA))

# -------------------------
# CSV loader for "wide" format
# -------------------------
def load_wide_yield_curves(csv_path: str) -> Dict[str, YieldCurveNACA365]:
    """
    Load a single CSV that contains repeated groups of columns:
        YieldCurve | StartDate | Date | NACA | [blank] | YieldCurve | StartDate | Date | NACA | [blank] | ...

    Returns a dict: {curve_name: YieldCurveNACA365}
    - Uses each curve's **earliest pillar date** as the valuation_date (anchor).
    - Ignores the 'StartDate' column (as per your note).
    """
    raw = pd.read_csv(csv_path, dtype=str)  # read all as strings first; we'll parse manually
    cols = list(raw.columns)

    def _is_blank(colname: str) -> bool:
        if colname is None:
            return True
        c = str(colname).strip().lower()
        return (c == "") or ("unnamed" in c)

    curves: Dict[str, YieldCurveNACA365] = {}
    i = 0
    while i <= len(cols) - 4:
        c0, c1, c2, c3 = cols[i:i+4]
        if str(c0).strip().lower() == "yieldcurve" and str(c1).strip().lower() == "startdate" \
           and str(c2).strip().lower() == "date" and str(c3).strip().lower() == "naca":
            # Extract this group
            name_series = raw[c0].dropna().astype(str).str.strip()
            curve_name = name_series[name_series != ""].iloc[0] if not name_series.empty else f"Curve_{len(curves)+1}"

            # Dates & NACA
            dates = raw[c2].apply(_to_date)
            naca = pd.to_numeric(raw[c3], errors="coerce")

            pillars = pd.DataFrame({"date": dates, "naca": naca}).dropna()
            if not pillars.empty:
                valn = min(pillars["date"])
                curve = YieldCurveNACA365(name=curve_name, pillars=pillars, valuation_date=valn)
                curves[curve_name] = curve

            # Advance past this group and an optional blank separator
            jmp = 4
            if i + 4 < len(cols) and _is_blank(cols[i+4]):
                jmp = 5
            i += jmp
        else:
            i += 1

    if not curves:
        raise ValueError("No curve groups (YieldCurve/StartDate/Date/NACA) found in the CSV headers.")

    return curves

# -------------------------
# Convenience facade
# -------------------------
class CurveBook:
    """
    Holds multiple YieldCurveNACA365 objects loaded from a single wide CSV.
    """
    def __init__(self, csv_path: str):
        self.curves: Dict[str, YieldCurveNACA365] = load_wide_yield_curves(csv_path)

    def list_curves(self) -> Iterable[str]:
        return self.curves.keys()

    def naca_rate(self, curve_name: str, on_date: date) -> float:
        return self.curves[curve_name].naca_rate(on_date)

    def discount_factor(self, curve_name: str, d_from: date, d_to: date) -> float:
        return self.curves[curve_name].discount_factor(d_from, d_to)

