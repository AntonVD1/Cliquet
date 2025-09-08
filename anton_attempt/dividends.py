import numpy as np
from datetime import date, datetime
from typing import List, Tuple

CENTS_TO_RANDS = 0.01  # CSV amounts are in cents

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
    Wide CSV repeating triplets: stock_name, pay_date, amount_in_cents, <blank?>, ...
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
                        amt_rands = _to_float(amt_cell) * CENTS_TO_RANDS
                        if amt_rands != 0.0:
                            out.append((pay_dt, amt_rands))
                    except ValueError:
                        continue
    return out

def _pv_future_divs_from_list(divs: List[Tuple[date, float]], disc_engine, valuation_date: date, at_date: date) -> float:
    """
    PV at 'at_date' of all dividends with pay_date >= at_date.
    Each discounted individually from pay_date back to 'at_date':
        DF(at_date -> pay) = discount_factor(valuation_date, at_date, pay)
    Dividends before valuation_date are ignored.
    """
    total = 0.0
    cutoff = max(at_date, valuation_date)
    for pd, amt in divs:
        if pd >= cutoff:
            df = disc_engine.discount_factor(valuation_date, at_date, pd)
            total += amt * df
    return total

def adjust_paths_with_dividends(
    raw_paths: np.ndarray,            # shape (n_sims, m_dates) from GBM.simulate_path_matrix
    reset_dates: List[date],
    start_date: date,                 # valuation date for discount engine
    s0_used: float,                   # initial spot used by the GBM run (s0 or spot0)
    dividends_csv: str,
    disc_engine,                      # your Discounter
) -> np.ndarray:
    """
    Produce a cum-div-style series from raw GBM paths:
      1) Scale raw paths so they start ex-div: multiply by (s0 - PV_all(start)) / s0
      2) At each date d, add PV of all FUTURE dividends (pay >= d), valued at d.

    Returns an array with the same shape as raw_paths.
    """
    if s0_used <= 0:
        raise ValueError("s0_used must be positive.")

    # Read dividends once
    divs = _read_dividends_csv(dividends_csv)

    # 1) PV of all future divs at start_date (to go ex-div at t0)
    pv_start = _pv_future_divs_from_list(divs, disc_engine, start_date, start_date)
    scale = (s0_used - pv_start) / s0_used
    if scale < 0:
        raise ValueError(f"PV of dividends ({pv_start:.6g}) exceeds starting spot ({s0_used:.6g}).")

    ex_div_paths = raw_paths * scale  # same multiplicative factors, lower starting level

    # 2) PV of future divs at each reset date (deterministic add-back per column)
    pv_future_vec = np.array(
        [_pv_future_divs_from_list(divs, disc_engine, start_date, d) for d in reset_dates],
        dtype=float
    )  # shape (m_dates,)

    adjusted = ex_div_paths + pv_future_vec[np.newaxis, :]
    return adjusted
