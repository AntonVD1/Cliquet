from datetime import date
import math
from discount import Discounter, year_fraction
from gbm import GBM
from pricer import CliquetPricer

# --- Dates & discounting ---
start_date = date(2025, 7, 28)
end_date   = date(2027, 1, 28)
T = year_fraction(start_date, end_date)

disc_engine = Discounter(
    r"C:\\Coding\\Cliquet\\cliquet_v2\\dummy_curve_data.csv",
    curve_name="ZAR-SWAP"
)

# Risk-neutral drift implied by start->end discount factor
drift = -math.log(disc_engine.discount_factor(start_date, start_date, end_date)) / T

# --- GBM with explicit drift ---
gbm = GBM(
    s0=9375,
    sigma=0.509079,
    mu=drift,
    start_date=start_date
)

# --- Cliquet pricer ---
cliquet = CliquetPricer(
    flag="c",                 # call-style local return
    local_cap=1_000_000_000,  # very high local cap (effectively none)
    local_floor=0.0,          # 0 local floor (ratchet)
    global_cap=100_000_000,   # very high global cap (effectively none)
    global_floor=0.0,
    nominal=10_000,
    stock_price=9375,
    reset_dates=[
        date(2025, 10, 28),
        date(2026, 1, 28),
        date(2026, 4, 28),
        date(2026, 7, 28),
        date(2026, 10, 28),
        date(2027, 1, 28),
    ],
    valdate=start_date,
    expiry=end_date,
    gbm=gbm,
    n_sims=50000,
    antithetic=True,
    seed=41,
    discount_engine=disc_engine,
)

# --- Price ---
price = cliquet.price()
print("Cliquet option price:", price)
print((price-6011.76)/6011.76)