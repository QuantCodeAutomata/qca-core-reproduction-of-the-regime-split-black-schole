"""
Fourth-order Householder (HH-4) refinement for implied total volatility.

At current iterate v with f = c_BS(k,v) - c and f' = phi(d1):
  r     = f / f'
  alpha = d1 * d2 / v
  beta  = ((d1*d2)^2 - (d1^2 + d2^2) - d1*d2) / v^2
  denom = -6 + 6*r*alpha - r^2*beta

Update:
  v <- v + 3*r*(2 - r*alpha) / denom   if |denom| >= 1e-20
  v <- v - r                            (Newton fallback otherwise)

The loop is repeat-until: at least one update is performed before
checking convergence criterion |f| < tol.
"""
import numpy as np
from scipy.stats import norm


def hh4_update(k: float, v: float, c: float) -> tuple:
    """Single HH-4 update step.

    Parameters
    ----------
    k : float
        Log-moneyness.
    v : float
        Current total-volatility iterate (must be > 0).
    c : float
        Target normalized call price.

    Returns
    -------
    (v_new, f, 1) : tuple
        Updated iterate, residual f = c_BS(k,v) - c, and update count 1.
    """
    d1 = -k / v + v / 2.0
    d2 = d1 - v
    f = norm.cdf(d1) - np.exp(k) * norm.cdf(d2) - c
    fp = norm.pdf(d1)   # phi(d1) = partial c_BS / partial v

    if abs(fp) < 1e-300:
        return v, f, 1  # degenerate: vega vanishes

    r = f / fp
    alpha = d1 * d2 / v
    beta = ((d1 * d2)**2 - (d1**2 + d2**2) - d1 * d2) / (v * v)
    denom = -6.0 + 6.0 * r * alpha - r**2 * beta

    if abs(denom) < 1e-20:
        v_new = v - r   # Newton fallback
    else:
        v_new = v + 3.0 * r * (2.0 - r * alpha) / denom

    return v_new, f, 1


def hh4_refine(k: float, v0: float, c: float,
               tol: float = 1e-14,
               max_iter: int = 100) -> tuple:
    """Iterate HH-4 updates until |f| < tol.

    Implements the repeat-until structure: at least one update is always
    performed before the convergence check.  Convergence is tested at the
    CURRENT iterate (before the would-be next update) once at least one
    update has been applied.  This matches the paper's loop semantics:

        do:
            compute f at v, apply HH-4 step, v <- v_new, n_updates += 1
        while |f(v)| >= tol      <-- f at the UPDATED v, checked next cycle

    Equivalently, each cycle first checks the residual at v (which was
    produced by the previous update), then optionally updates again.

    Parameters
    ----------
    k : float
        Log-moneyness.
    v0 : float
        Initial seed for total volatility.
    c : float
        Target normalized call price.
    tol : float
        Convergence tolerance on |f| = |c_BS(k,v) - c|.
    max_iter : int
        Safety cap on iterations.

    Returns
    -------
    (v_hat, n_updates, converged) : tuple
        Recovered total volatility, number of HH-4 updates applied,
        and convergence flag (True iff |f(v_hat)| < tol).
    """
    v = v0
    n_updates = 0
    while True:
        # Compute residual at current v (the iterate produced by the last update,
        # or the seed on the first pass).
        d1 = -k / v + v / 2.0
        d2 = d1 - v
        f = norm.cdf(d1) - np.exp(k) * norm.cdf(d2) - c

        # After at least one update: check if we already have |f| < tol
        # before committing to another update.
        if n_updates >= 1 and abs(f) < tol:
            return v, n_updates, True

        if n_updates >= max_iter:
            return v, n_updates, abs(f) < tol

        # Apply one HH-4 update (fp, r, alpha, beta, denom).
        fp = norm.pdf(d1)
        if abs(fp) < 1e-300:
            # Degenerate: vega vanishes; just return current v.
            return v, n_updates, abs(f) < tol
        r = f / fp
        alpha = d1 * d2 / v
        beta = ((d1 * d2) ** 2 - (d1 ** 2 + d2 ** 2) - d1 * d2) / (v * v)
        denom = -6.0 + 6.0 * r * alpha - r ** 2 * beta
        if abs(denom) < 1e-20:
            v = v - r   # Newton fallback
        else:
            v = v + 3.0 * r * (2.0 - r * alpha) / denom
        n_updates += 1
