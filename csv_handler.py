from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Tuple
from datetime import date, datetime
import math
import pandas as pd
import numpy as np
from discount_engine import YieldCurveNACA365
from helpers import _to_date



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
