import QuantLib as ql
import math
 
class CliquetOptionPricer:
    """
    Monte Carlo pricer for a discrete-reset Cliquet (ratchet) option under the Black-Scholes-Merton model.
 
    Attributes:
        S (float): initial underlying spot price
        r (float): continuously compounded risk-free rate
        dividend (float): continuously compounded dividend yield
        sigma (float): constant volatility
        start_date (ql.Date): evaluation and simulation start date
        periods (int): number of reset periods
        tenor (ql.Period): tenor of each reset
        cap (float): local cap per period (e.g., 0.05 for 5%)
        floor (float): local floor per period (e.g., -0.02 for -2%)
        steps_per_period (int): time steps per period for the Monte Carlo
        samples (int): number of Monte Carlo samples
        seed (int): random seed for reproducibility
    """
        #Begin with the constructor method
    def __init__(self, S, r, dividend, sigma, start_date: ql.Date, periods, tenor: ql.Period, cap, floor, steps_per_period: int = 10, samples: int = 50000, seed: int = 42):
        # Store inputs
        self.S = S
        self.r = r
        self.dividend = dividend
        self.sigma = sigma
        self.start_date = start_date
        self.periods = periods
        self.tenor = tenor
        self.cap = cap
        self.floor = floor
        self.steps_per_period = steps_per_period
        self.samples = samples
        self.seed = seed
 
        # Derived parameters
        self.total_steps = self.periods * self.steps_per_period
        # Calendar and day-count convention for date calculations
        self.calendar = ql.TARGET()
        self.day_count = ql.Actual365Fixed()
        # Set global evaluation date
        ql.Settings.instance().evaluationDate = self.start_date
 
        # Build QuantLib components
        self._build_term_structures()
        self._build_process()            
        # Compute total time in years
        self._compute_maturity()
 
    def _build_term_structures(self):
        #Build flat term structures for interest rates, dividends, and volatility. At least for now
        # Quote for the spot price
        self.spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.S))
        # Flat forward curve for risk-free rate
        self.yield_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.start_date, self.r, self.day_count)
        )
        # Flat forward curve for dividend yield
        self.dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.start_date, self.dividend, self.day_count)
        )
        # Black volatility surface (flat volatility)
        self.vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(self.start_date, self.calendar, self.sigma, self.day_count)
        )
 
    def _build_process(self):
        """Initialize the Black-Scholes-Merton stochastic process for the underlying.""" #(Geometric Brownian Motion )
        self.process = ql.BlackScholesMertonProcess(
            self.spot_handle,
            self.dividend_ts,
            self.yield_ts,
            self.vol_ts
        )
 
    def _compute_maturity(self):
        """Compute option maturity date and total time in years."""
        # Maturity date = start date + tenor * periods
        self.maturity_date = self.calendar.advance(
            self.start_date, self.tenor * self.periods
        )
        # Total time in years for simulation
        self.total_years = self.day_count.yearFraction(
            self.start_date, self.maturity_date
        )
        
    def _path_payoff(self, path):
        """
        Cliquet payoff using absolute price differences per period.
        cap/floor are ABSOLUTE amounts (same units as S).
        """
        payoff = 0.0
        prev_price = path[0]

        for i in range(self.periods):
            idx = (i + 1) * self.steps_per_period
            st = path[idx]
            # absolute difference (not a return)
            diff = st - prev_price
            # local floor/cap in absolute units
            payoff += min(max(diff, self.floor), self.cap)
            prev_price = st

        return payoff
 
    def price(self) -> float:
        """
        Price the Cliquet option via Monte Carlo simulation.
 
        Returns:
            float: present value (NPV) of the option
        """
        # Build time grid for simulation
        time_grid = ql.TimeGrid(self.total_years, self.total_steps)
 
        # Random number generators: uniform -> Gaussian
        uniform_generator = ql.UniformRandomSequenceGenerator(
            self.total_steps,
            ql.UniformRandomGenerator(self.seed)
        )
        gaussian_generator = ql.GaussianRandomSequenceGenerator(uniform_generator)
 
        # Path generator under the BSM process
        path_generator = ql.GaussianPathGenerator(
            self.process,
            self.total_years,
            self.total_steps,
            gaussian_generator,
            False  # no antithetic variates
        )
 
        # Monte Carlo loop
        sum_payoffs = 0.0
        for _ in range(self.samples):
            seq = path_generator.next()
            path = seq.value()
            sum_payoffs += self._path_payoff(path)
 
        # average payoff
        mean_payoff = sum_payoffs / float(self.samples)
        # discount back to present value
        discount = math.exp(-self.r * self.total_years)
        npv = discount * mean_payoff
        return npv