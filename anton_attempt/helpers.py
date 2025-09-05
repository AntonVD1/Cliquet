from datetime import date
import numpy as np

def simulate_prices_at_resets(
    gbm,
    reset_dates: list[date],
    n_sims: int = 5000,
    seed: int | None = 42,
    antithetic: bool = True,
) -> np.ndarray:
    """
    Returns an array of shape (n_sims, n_dates) with simulated prices at each reset date.
    """
    if antithetic and (n_sims % 2 != 0):
        raise ValueError("antithetic=True requires an even n_sims")

    reset_dates = sorted(reset_dates)
    cols = []
    for i, d in enumerate(reset_dates):
        col = gbm.simulate_at(
            to_date=d,
            n=n_sims,
            seed=None if seed is None else seed + i,
            antithetic=antithetic,
        )
        cols.append(col.astype(float))
    return np.column_stack(cols)