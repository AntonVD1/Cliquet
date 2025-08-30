# Cliquet Option Pricer

This repository implements a simple Monte Carlo pricer for a discrete-reset
Cliquet option.

## Risk-neutral dynamics

Under the risk-neutral measure the underlying spot price $S_t$ follows a
geometric Brownian motion with constant volatility $\sigma$ and dividend
yield $q$:

$$
\mathrm{d}S_t = S_t[(r - q)\,\mathrm{d}t + \sigma\,\mathrm{d}W_t].
$$

## Payoff

Let the option reset at dates $t_0 < t_1 < \dots < t_n$. For each period the
absolute price change is capped above and floored below. The payoff of one path
is

$$
P = \sum_{i=1}^{n} \min\bigl(\max(S_{t_i} - S_{t_{i-1}},\, \text{floor}),\, \text{cap}\bigr).
$$

## Pricing

The value at $t_0$ discounts the expected payoff under the risk-neutral
measure:

$$
V_0 = e^{-rT} \mathbb{E}[P], \qquad T = t_n - t_0.
$$

Monte Carlo simulation approximates this expectation by averaging the payoff over
many simulated paths.

