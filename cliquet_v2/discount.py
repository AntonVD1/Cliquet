import csv
from datetime import datetime, date
from typing import Dict, Optional


def parse_date(s: str) -> date:
    """Parse date string into datetime.date (supports dash and slash)."""
    s = s.strip()
    formats = [
        "%Y-%m-%d",  # 2025-07-28
        "%d-%m-%Y",  # 28-07-2025
        "%m-%d-%Y",  # 07-28-2025
        "%Y/%m/%d",  # 2025/07/28
        "%d/%m/%Y",  # 28/07/2025
        "%m/%d/%Y",  # 07/28/2025
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {s}")


def year_fraction(d0: date, d1: date) -> float:
    """ACT/365F year fraction."""
    return (d1 - d0).days / 365.0


class Discounter:
    def __init__(self, csv_path: str, curve_name: Optional[str] = None):
        """
        Load curve data from CSV.

        CSV columns (by name or order):
        - Column 1: Curve name
        - Column 2: StartDate (ignored)
        - Column 3: Date (pillar date)
        - Column 4: NACA rate
        """
        self.rates: Dict[date, float] = {}
        self.curve_name = curve_name

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if curve_name and row[reader.fieldnames[0]] != curve_name:
                    continue
                pillar_date = parse_date(row["Date"])
                naca = float(row["NACA"])
                self.rates[pillar_date] = naca

        if not self.rates:
            raise ValueError(f"No data found for curve '{curve_name}' in {csv_path}")

    def _df_from_val(self, valuation_date: date, target_date: date) -> float:
        """Return DF(val→target) using exact stored NACA for that date."""
        if target_date not in self.rates:
            raise KeyError(f"No rate found for {target_date}")
        r_naca = self.rates[target_date]
        t = year_fraction(valuation_date, target_date)
        return (1 + r_naca) ** (-t)

    def discount_factor(self, valuation_date: date, start_date: date, end_date: date) -> float:
        """
        Compute DF(start→end) = DF(val→end) / DF(val→start).
        """
        df_end = self._df_from_val(valuation_date, end_date)
        df_start = self._df_from_val(valuation_date, start_date)
        return df_end / df_start
