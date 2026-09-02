"""
Regime-split implied total-volatility solver (Algorithm 1).

Solves c_BS(k, v) = c for total volatility v = sigma * sqrt(T) using
analytically derived regime-split initial seeds followed by HH-4 refinement.

Regime cutoffs (paper Table 1 / Algorithm 1):
    kappa_1   = 0.001   ATM boundary
    kappa_2   = 0.81    lower polynomial boundary
    kappa_3   = 1.155   transition
    kappa_4   = 1.347   deep-OTM boundary
    c_star    = 0.02128 tail-guard price threshold
    kappa_star = 0.5    tail-guard moneyness threshold
"""
import numpy as np
from scipy.stats import norm
from .seeds import (
    atm_seed, vP3_seed, vP7_seed, deep_otm_seed, otm_equivalent_price,
)
from .refinement import hh4_refine
from .inverse_normal import acklam_ppf

# ---------------------------------------------------------------------------
# Regime boundary constants
# ---------------------------------------------------------------------------
KAPPA_1 = 0.001
KAPPA_2 = 0.81
KAPPA_3 = 1.155
KAPPA_4 = 1.347
C_STAR = 0.02128
KAPPA_STAR = 0.5


def _compute_seed(k: float, c: float, use_tail_guard: bool = True) -> tuple:
    """Compute initial seed and regime label per Algorithm 1.

    Branch order:
      A: kappa < KAPPA_1  → v_ATM
      B: c_seed < C_STAR and kappa > KAPPA_STAR (tail guard, only if enabled) → v_q,1
      C: KAPPA_1 <= kappa <= KAPPA_4 (polynomial region)
         C1: kappa <= KAPPA_2     → max(v_P7, v_ATM)
         C2: kappa <= KAPPA_3     → max((v_P3+v_P7)/2, v_ATM)
         C3: kappa <= KAPPA_4     → max(v_P3, v_ATM)
      D: kappa > KAPPA_4  → v_q,1

    Parameters
    ----------
    k : float
        Log-moneyness.
    c : float
        Normalized call price.
    use_tail_guard : bool
        Whether to apply Branch B (near-OTM tail guard).

    Returns
    -------
    (v0, regime, v_ATM) : tuple
        Seed, regime label ('A','B','C','D'), and ATM seed value.
    """
    intrinsic = max(1.0 - np.exp(k), 0.0)
    c_tv = max(c - intrinsic, 0.0)
    v_ATM = atm_seed(c_tv)
    kappa = abs(k)
    c_seed = otm_equivalent_price(k, c)

    # Branch A: ATM
    if kappa < KAPPA_1:
        return v_ATM, 'A', v_ATM

    # Branch B: tail guard (near-OTM override)
    if use_tail_guard and c_seed < C_STAR and kappa > KAPPA_STAR:
        _, v_q1 = deep_otm_seed(c_seed, kappa, ppf_func=acklam_ppf)
        return v_q1, 'B', v_ATM

    # Branch D: deep OTM
    if kappa > KAPPA_4:
        _, v_q1 = deep_otm_seed(c_seed, kappa, ppf_func=acklam_ppf)
        return v_q1, 'D', v_ATM

    # Branch C: polynomial region.
    # Pass signed k so vP7 can use a Newton fallback when discriminant N≤0.
    vp7 = vP7_seed(c_seed, kappa, k=k)
    vp3 = vP3_seed(c_seed, kappa)

    if kappa <= KAPPA_2:
        v0 = max(vp7, v_ATM)
    elif kappa <= KAPPA_3:
        v0 = max((vp3 + vp7) / 2.0, v_ATM)
    else:
        v0 = max(vp3, v_ATM)

    return v0, 'C', v_ATM


def solve_iv(k: float, c: float, tol: float = 1e-14,
             use_tail_guard: bool = True) -> dict:
    """Solve for implied total volatility v from (k, c).

    Parameters
    ----------
    k : float
        Log-moneyness k = log(F/K).
    c : float
        Normalized call price (must satisfy c >= max(1-exp(k),0) and c < 1).
    tol : float
        HH-4 convergence tolerance on |c_BS(k,v)-c|.
    use_tail_guard : bool
        Whether to apply the near-OTM tail-guard branch.

    Returns
    -------
    dict with keys:
        v_hat     : recovered total volatility
        v0        : initial seed
        v_ATM     : ATM seed (reference)
        regime    : regime label ('A','B','C','D')
        n_updates : number of HH-4 updates
        f_final   : final residual c_BS(k,v_hat)-c
        converged : bool, whether |f_final| < tol
    """
    intrinsic = max(1.0 - np.exp(k), 0.0)
    if c < intrinsic - 1e-13:
        raise ValueError(f"Price below intrinsic: c={c:.6e}, intrinsic={intrinsic:.6e}")
    if c >= 1.0:
        raise ValueError(f"Price at or above 1: c={c:.6e}")

    v0, regime, v_ATM = _compute_seed(k, c, use_tail_guard=use_tail_guard)

    if v0 <= 0.0 or not np.isfinite(v0):
        v0 = v_ATM if (v_ATM > 0.0 and np.isfinite(v_ATM)) else 1e-4

    v_hat, n_updates, converged = hh4_refine(k, v0, c, tol=tol)

    d1 = -k / v_hat + v_hat / 2.0
    d2 = d1 - v_hat
    f_final = float(norm.cdf(d1) - np.exp(k) * norm.cdf(d2) - c)

    return {
        'v_hat': v_hat,
        'v0': v0,
        'v_ATM': v_ATM,
        'regime': regime,
        'n_updates': n_updates,
        'f_final': f_final,
        'converged': bool(converged),  # ensure Python bool for `is True` checks
    }


def solve_grid(grid: dict, tol: float = 1e-14,
               use_tail_guard: bool = True) -> dict:
    """Solve implied total volatility for all options in a grid.

    Parameters
    ----------
    grid : dict
        Output of make_grid1() / make_grid2() with keys k, c, v_true.
    tol : float
        HH-4 convergence tolerance.
    use_tail_guard : bool
        Whether to apply the near-OTM tail guard (Branch B).

    Returns
    -------
    dict with per-option arrays:
        v_hat, v0, regime, n_updates, abs_error, seed_ratio, eta_v,
        v_true, k, c
    """
    k_arr = grid['k']
    c_arr = grid['c']
    v_true_arr = grid['v_true']
    n = len(k_arr)

    v_hat_arr = np.empty(n)
    v0_arr = np.empty(n)
    n_updates_arr = np.empty(n, dtype=int)
    regime_arr = []

    for i in range(n):
        k = float(k_arr[i])
        c = float(c_arr[i])

        intrinsic = max(1.0 - np.exp(k), 0.0)
        c_tv = max(c - intrinsic, 0.0)
        v_ATM_ = atm_seed(c_tv)
        kappa = abs(k)
        c_seed = otm_equivalent_price(k, c)

        # Regime selection (inline for speed)
        if kappa < KAPPA_1:
            v0 = v_ATM_
            regime = 'A'
        elif use_tail_guard and c_seed < C_STAR and kappa > KAPPA_STAR:
            _, v0 = deep_otm_seed(c_seed, kappa, ppf_func=acklam_ppf)
            regime = 'B'
        elif kappa > KAPPA_4:
            _, v0 = deep_otm_seed(c_seed, kappa, ppf_func=acklam_ppf)
            regime = 'D'
        else:
            # Pass signed k for Newton fallback when discriminant N≤0
            vp7 = vP7_seed(c_seed, kappa, k=k)
            vp3 = vP3_seed(c_seed, kappa)
            if kappa <= KAPPA_2:
                v0 = max(vp7, v_ATM_)
            elif kappa <= KAPPA_3:
                v0 = max((vp3 + vp7) / 2.0, v_ATM_)
            else:
                v0 = max(vp3, v_ATM_)
            regime = 'C'

        if v0 <= 0.0 or not np.isfinite(v0):
            v0 = v_ATM_ if (v_ATM_ > 0.0 and np.isfinite(v_ATM_)) else 1e-4

        v_hat_arr[i], n_updates_arr[i], _ = hh4_refine(k, v0, c, tol=tol)
        v0_arr[i] = v0
        regime_arr.append(regime)

    abs_error = np.abs(v_hat_arr - v_true_arr)
    seed_ratio = np.where(v_true_arr > 0, v0_arr / v_true_arr, np.nan)
    eta_v = np.abs(seed_ratio - 1.0)

    return {
        'v_hat': v_hat_arr,
        'v0': v0_arr,
        'regime': np.array(regime_arr),
        'n_updates': n_updates_arr,
        'abs_error': abs_error,
        'seed_ratio': seed_ratio,
        'eta_v': eta_v,
        'v_true': v_true_arr,
        'k': k_arr,
        'c': c_arr,
    }


def aggregate_metrics(results: dict, filter_near_otm: bool = False) -> dict:
    """Compute aggregate convergence metrics from solve_grid output.

    Parameters
    ----------
    results : dict
        Output of solve_grid().
    filter_near_otm : bool
        If True, exclude near-OTM tail-guard cases (regime 'B') from stats.

    Returns
    -------
    dict with keys:
        n_options, max_abs_error, mean_updates,
        n_1_update, n_2_updates, n_gt2_updates, frac_le2
    """
    nu = results['n_updates']
    abs_err = results['abs_error']

    if filter_near_otm:
        mask = results['regime'] != 'B'
        nu = nu[mask]
        abs_err = abs_err[mask]

    n = len(nu)
    if n == 0:
        return {'n_options': 0, 'max_abs_error': np.nan, 'mean_updates': np.nan,
                'n_1_update': 0, 'n_2_updates': 0, 'n_gt2_updates': 0, 'frac_le2': np.nan}

    return {
        'n_options': n,
        'max_abs_error': float(np.max(abs_err)),
        'mean_updates': float(np.mean(nu)),
        'n_1_update': int(np.sum(nu == 1)),
        'n_2_updates': int(np.sum(nu == 2)),
        'n_gt2_updates': int(np.sum(nu > 2)),
        'frac_le2': float(np.sum(nu <= 2) / n),
    }
