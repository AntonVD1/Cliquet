from datetime import date
import tempfile
import csv
import numpy as np
from discount import Discounter
from gbm import GBM
from pricer import CliquetPricer


disc_engine = Discounter(r"C:\\Coding\\Cliquet\\cliquet_v2\\dummy_curve_data.csv", curve_name="ZAR-SWAP")

# ---- Step 4. Build GBM with discount engine ----
gbm = GBM(
    s0=9375,
    sigma=0.415884,
    start_date=date(2025, 7, 28),
    discount_engine=disc_engine
)

# ---- Step 5. Build the CliquetPricer ----
pricer = CliquetPricer(
    flag="c",                        # call-style payoff
    nominal=1_000_000,              # <-- new: notional/nominal
    local_cap=100_000,              # local cap
    local_floor=0,                  # local floor
    global_cap=1_000_000,           # global cap
    global_floor=0.0,               # global floor
    stock_price=9_375,
    K=9_200,
    reset_dates=[
        date(2025, 10, 28),
        date(2026, 1, 28),
        date(2026, 4, 28),
        date(2026, 7, 28),
        date(2026, 10, 28),
        date(2027, 1, 28),
    ],
    sigma=0.415884,
    startdate=date(2025, 7, 28),
    valdate=date(2025, 7, 28),
    expiry=date(2027, 1, 28),
    gbm=gbm,
    n_sims=10_000,
    antithetic=True,
    seed=42,
    discount_engine=disc_engine,
)

# ---- Step 6. Request the price ----
price = pricer.price()
print("Cliquet price =", price)
