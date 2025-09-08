from datetime import date, datetime
from typing import List, Tuple
import csv

CENTS_TO_RANDS = 0.01  # input amounts are in cents

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

def pv_dividends_at_start(
    dividends_csv: str,
    discounter,          # your Discounter
    start_date: date,    # valuation date AND discount start
) -> float:
    """
    Present value at 'start_date' of all dividends with pay_date >= start_date.

    CSV layout (repeating triplets per row):
        stock_name, pay_date, amount_in_cents, <blank?>, stock_name, pay_date, amount_in_cents, <blank?>, ...

    DF used per dividend:
        DF(start_date -> pay_date) = discounter.discount_factor(start_date, start_date, pay_date)
    """
    total = 0.0
    with open(dividends_csv, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f, skipinitialspace=True)
        for row in rdr:
            if not row:
                continue
            i, n = 0, len(row)

            def next_nonempty(k: int) -> int:
                while k < n and (row[k] is None or str(row[k]).strip() == ""):
                    k += 1
                return k

            while True:
                # stock_name (ignored for aggregate PV)
                i = next_nonempty(i)
                if i >= n: break

                # pay_date
                i = next_nonempty(i + 1)
                if i >= n: break
                date_cell = str(row[i]).strip()

                # amount_in_cents
                i = next_nonempty(i + 1)
                if i >= n: break
                amt_cell = str(row[i]).strip()

                # move past separator (blank) or next triplet
                i += 1

                if not date_cell or not amt_cell:
                    continue

                try:
                    pay_dt = _parse_date(date_cell)
                    amt_rands = _to_float(amt_cell) * CENTS_TO_RANDS
                except ValueError:
                    continue  # skip malformed triplets

                if amt_rands == 0.0 or pay_dt < start_date:
                    continue  # exclude non-future or zero

                # discount each dividend from its pay date back to start_date
                df = discounter.discount_factor(start_date, start_date, pay_dt)
                total += amt_rands * df

    return total
