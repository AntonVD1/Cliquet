import csv
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Dict
from discount import Discounter


# --------- utils ---------
def parse_date(s: str) -> date:
    """Parse date string into datetime.date (supports dash and slash)."""
    s = str(s).strip()
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
    raise ValueError(f"Unrecognized date format: {s!r}")


def _to_float(x: str) -> float:
    """Parse numeric cells that may include commas or spaces."""
    return float(str(x).replace(",", "").strip())


# --------- data model ---------
@dataclass(frozen=True)
class Dividend:
    """Single dividend cash flow."""
    stock: str
    pay_date: date
    amount: float


# --------- CSV reader ---------
def read_dividends_csv(csv_path: str) -> List[Dividend]:
    """
    Parse a 'wide' CSV where columns repeat as triplets:
        stock_name, pay_date, amount, <blank?>, stock_name, pay_date, amount, <blank?>, ...
    The <blank> separator may be present between triplets but is often absent after the last one.
    """
    out: List[Dividend] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, skipinitialspace=True)

        for row in reader:
            if not row:
                continue

            i, n = 0, len(row)

            def next_nonempty(k: int) -> int:
                while k < n and (row[k] is None or str(row[k]).strip() == ""):
                    k += 1
                return k

            while True:
                # stock
                i = next_nonempty(i)
                if i >= n:
                    break
                stock_cell = str(row[i]).strip()

                # pay_date
                i = next_nonempty(i + 1)
                if i >= n:
                    break
                date_cell = str(row[i]).strip()

                # amount
                i = next_nonempty(i + 1)
                if i >= n:
                    break
                amt_cell = str(row[i]).strip()

                # advance at least one position for next loop
                i += 1

                # validate/append
                if stock_cell and date_cell and amt_cell:
                    try:
                        pay_dt = parse_date(date_cell)
                        amount = _to_float(amt_cell)
                    except ValueError:
                        # skip malformed triplets (header cells, bad dates/numbers)
                        continue

                    if amount != 0.0:
                        out.append(Dividend(stock_cell, pay_dt, amount))

    return out


# --------- PV handler ---------
class DividendHandler:
    """
    Computes present value of dividends read from the wide CSV layout.
    """

    def __init__(self, dividends: List[Dividend]):
        self.dividends = dividends

    def _eligible(self, valuation_date: date) -> List[Dividend]:
        """Keep dividends with pay_date >= valuation_date."""
        return [d for d in self.dividends if d.pay_date >= valuation_date]

    def pv_total(
        self,
        valuation_date: date,
        discount_start_date: date,
        discounter: "Discounter",
    ) -> float:
        """
        Aggregate PV of all eligible dividends discounted to valuation_date.

        DF used:
            DF(start -> pay) = discounter.discount_factor(valuation_date, discount_start_date, pay_date)
        """
        total = 0.0
        for d in self._eligible(valuation_date):
            df = discounter.discount_factor(valuation_date, discount_start_date, d.pay_date)
            total += d.amount * df
        return total

    def pv_by_stock(
        self,
        valuation_date: date,
        discount_start_date: date,
        discounter: "Discounter",
    ) -> Dict[str, float]:
        """Per-stock PV breakdown at valuation_date using the given discount_start_date."""
        acc: Dict[str, float] = {}
        for d in self._eligible(valuation_date):
            df = discounter.discount_factor(valuation_date, discount_start_date, d.pay_date)
            acc[d.stock] = acc.get(d.stock, 0.0) + d.amount * df
        return acc

    def pv_total_up_to(
    self,
    valuation_date: date,
    at_date: date,
    discounter: "Discounter",
    ) -> float:
        """
        PV (at valuation_date) of all eligible dividends with pay_date <= at_date,
        discounted from 'at_date' to each pay_date via the engine.
        """
        total = 0.0
        for d in self._eligible(valuation_date):
            if d.pay_date <= at_date:
                df = discounter.discount_factor(valuation_date, at_date, d.pay_date)
                total += d.amount * df
        return total

# --------- example run ---------
if __name__ == "__main__":
    # Build the discounter from your curve CSV (columns: CurveName, StartDate, Date, NACA)
    discounter = Discounter(
        csv_path=r"C:\Coding\Cliquet\cliquet_v2\dummy_curve_data.csv",
        curve_name="ZAR-SWAP"
    )

    # Read dividends from the wide-layout CSV
    dividends = read_dividends_csv(r"C:\coding\Cliquet\anton_attempt\dividend_test_data.csv")

    # Compute PVs at chosen dates
    valdate = date(2027, 1, 28)
    df_start = date(2027, 1, 28) 

    handler = DividendHandler(dividends)

    total_pv = handler.pv_total(valdate, df_start, discounter)
    by_stock = handler.pv_by_stock(valdate, df_start, discounter)

    print(f"Total dividend PV @ {valdate}: {total_pv:,.2f}")
    for s, pv in sorted(by_stock.items()):
        print(f"  {s}: {pv:,.2f}")
