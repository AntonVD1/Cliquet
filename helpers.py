from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Tuple
from datetime import date, datetime
import math
import pandas as pd
import numpy as np


def _to_date(x) -> Optional[date]:
    """Robust parse to datetime.date (returns None on failure)."""
    if pd.isna(x):
        return None
    try:
        # Allow day-first (common outside US); pandas handles most formats
        dt = pd.to_datetime(x, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date()
    except Exception:
        return None

def year_fraction_365(d0: date, d1: date) -> float:
    """Act/365F style: simple days/365."""
    return (d1 - d0).days / 365.0

def cont_from_naca(naca_365: float) -> float:
    """Continuous zero from NACA/365 spot: r_cont = ln(1 + r_naca)."""
    return math.log1p(naca_365)

def naca_from_cont(r_cont: float) -> float:
    """NACA/365 from continuous: r_naca = exp(r_cont) - 1."""
    return math.exp(r_cont) - 1.0