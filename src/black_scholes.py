"""
Normalized Black-Scholes pricing and synthetic grid generation.

All formulas operate in terms of total volatility v = sigma * sqrt(T)
and log-moneyness k = log(F/K). The normalized call price is:

  c_BS(k, v) = Phi(-k/v + v/2) - exp(k) * Phi(-k/v - v/2)

where d1 = -k/v + v/2 and d2 = d1 - v.
"""
import numpy as np
from scipy.stats import norm


def bs_call(k: float, v: float) -> float:
    """Normalized Black-Scholes call price c(k, v).

    Parameters
    ----------
    k : float
        Log-moneyness k = log(F/K).
    v : float
        Total volatility v = sigma * sqrt(T), must be > 0.

    Returns
    -------
    float
        Normalized call price in [max(1-exp(k), 0), 1).
    """
    if v <= 0.0:
        return max(1.0 - np.exp(k), 0.0)
    d1 = -k / v + v / 2.0
    d2 = d1 - v
    return float(norm.cdf(d1) - np.exp(k) * norm.cdf(d2))


def bs_call_vec(k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized normalized BS call price.

    Parameters
    ----------
    k : np.ndarray
        Log-moneyness values.
    v : np.ndarray
        Total volatility values (same shape as k).

    Returns
    -------
    np.ndarray
        Normalized call prices.
    """
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    d1 = -k / v + v / 2.0
    d2 = d1 - v
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)


def bs_call_intrinsic(k: float) -> float:
    """Intrinsic value of normalized call: max(1 - exp(k), 0)."""
    return max(1.0 - np.exp(k), 0.0)


def bs_call_time_value(k: float, c: float) -> float:
    """Time value c_tv = c - intrinsic(k)."""
    return c - bs_call_intrinsic(k)


def generate_grid(v_values: np.ndarray, delta_values: np.ndarray) -> dict:
    """Generate synthetic benchmark grid from total vols and call deltas.

    For each (v, Delta) pair:
      d1 = Phi^{-1}(Delta)
      k  = v * (v/2 - d1)
      d2 = d1 - v
      c  = Phi(d1) - exp(k) * Phi(d2)

    Parameters
    ----------
    v_values : np.ndarray
        Array of true total volatilities.
    delta_values : np.ndarray
        Array of Black-Scholes call deltas in (0, 1).

    Returns
    -------
    dict with keys: k, c, v_true, delta, d1, d2
        Each is a 1-D array of length len(v_values) * len(delta_values).
    """
    V, D = np.meshgrid(v_values, delta_values, indexing='ij')
    V = V.ravel()
    D = D.ravel()
    d1 = norm.ppf(D)
    k = V * (V / 2.0 - d1)
    d2 = d1 - V
    c = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    return {'k': k, 'c': c, 'v_true': V, 'delta': D, 'd1': d1, 'd2': d2}


def make_grid1() -> dict:
    """Standard benchmark: 41 x 8 = 328 options.

    v in {0.01, 0.05, 0.10, 0.15, ..., 2.00}
    Delta in {0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95}
    """
    v_vals = np.concatenate([[0.01, 0.05], np.arange(0.10, 2.01, 0.05)])
    delta_vals = np.array([0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95])
    return generate_grid(v_vals, delta_vals)


def make_grid2() -> dict:
    """Granular benchmark: 197 x 10 = 1970 options.

    v in {0.01, 0.05, 0.06, 0.07, ..., 2.00}
    Delta in {0.01, 0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95, 0.99}
    """
    v_vals = np.concatenate([[0.01, 0.05], np.arange(0.06, 2.01, 0.01)])
    delta_vals = np.array([0.01, 0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95, 0.99])
    return generate_grid(v_vals, delta_vals)
