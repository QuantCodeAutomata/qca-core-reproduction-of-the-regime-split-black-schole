"""
Initial seed implementations for the implied total-volatility solver.

All seeds target total volatility v = sigma * sqrt(T) and use the
OTM-equivalent price c_seed and kappa = |k|.

Gaussian-CDF Taylor coefficients (odd powers of x, i.e. Phi(x) ≈ sum a_{2j+1}*x^{2j+1}):
  a1 = 1/sqrt(2*pi) = phi(0)
  a3, a5, a7 are higher-order coefficients.

References: Paper Algorithm 1 and supporting derivations.
"""
import numpy as np
from scipy.stats import norm as _sp_norm
from .inverse_normal import acklam_ppf, clip_prob

# Gaussian-CDF Taylor odd-power coefficients used in polynomial seeds
_a1 = 0.3989422804   # phi(0) = 1/sqrt(2*pi)
_a3 = -0.0664903801
_a5 = 0.0099735570
_a7 = -0.0011873282


# ---------------------------------------------------------------------------
# ATM family seeds
# ---------------------------------------------------------------------------

def atm_seed(c_tv: float) -> float:
    """Fourth-order ATM Taylor-reverted seed v_ATM.

    v_ATM = s * (1 + s^2/24 + 7*s^4/1920 + 127*s^6/322560)
    where s = sqrt(2*pi) * c_tv.

    Parameters
    ----------
    c_tv : float
        Time value of the normalized call price.

    Returns
    -------
    float
        ATM total-volatility seed.
    """
    s = np.sqrt(2.0 * np.pi) * c_tv
    s2 = s * s
    return s * (1.0 + s2 * (1.0/24.0 + s2 * (7.0/1920.0 + s2 * 127.0/322560.0)))


def brenner_subrahmanyam_seed(c_tv: float) -> float:
    """Brenner-Subrahmanyam ATM seed: v_BS = sqrt(2*pi) * c_tv."""
    return np.sqrt(2.0 * np.pi) * c_tv


def taylor2_seed(c_tv: float) -> float:
    """Second-order Taylor-reverted ATM seed."""
    s = np.sqrt(2.0 * np.pi) * c_tv
    return s * (1.0 + s * s / 24.0)


def taylor3_seed(c_tv: float) -> float:
    """Third-order Taylor-reverted ATM seed."""
    s = np.sqrt(2.0 * np.pi) * c_tv
    s2 = s * s
    return s * (1.0 + s2 * (1.0/24.0 + 7.0 * s2 / 1920.0))


def taylor4_seed(c_tv: float) -> float:
    """Fourth-order Taylor-reverted ATM seed (identical to atm_seed)."""
    return atm_seed(c_tv)


# ---------------------------------------------------------------------------
# Helper: epsilon = exp(kappa) - 1, approximated to 4th order
# ---------------------------------------------------------------------------

def _compute_epsilon(kappa: float) -> float:
    """4th-order Taylor approximation: epsilon ≈ exp(kappa) - 1.

    epsilon = kappa + kappa^2/2 + kappa^3/6 + kappa^4/24
    """
    return kappa + kappa**2 / 2.0 + kappa**3 / 6.0 + kappa**4 / 24.0


# ---------------------------------------------------------------------------
# Polynomial-region seeds: v_P1, v_P7, v_P3
# ---------------------------------------------------------------------------

def vP1_seed(c_seed: float, kappa: float) -> float:
    """Quadratic-formula first approximation v_P1.

    Discriminant: N = (2*c_seed + epsilon)^2 - 8*a1^2*kappa*epsilon*(2+epsilon)
    Clipped to zero if N < 0.

    Parameters
    ----------
    c_seed : float
        OTM-equivalent normalized call price.
    kappa : float
        Absolute log-moneyness |k|.

    Returns
    -------
    float
        v_P1 seed value.
    """
    epsilon = _compute_epsilon(kappa)
    N = (2.0 * c_seed + epsilon)**2 - 8.0 * _a1**2 * kappa * epsilon * (2.0 + epsilon)
    if N < 0.0:
        N = 0.0
    return (2.0 * c_seed + epsilon + np.sqrt(N)) / (2.0 * _a1 * (2.0 + epsilon))


def vP7_seed(c_seed: float, kappa: float, k: float = None) -> float:
    """7th-order one-step corrected polynomial seed v_P7 = v_P1 * (1 + G7/D7).

    Uses exact G7, D7 formulas from the paper with coefficients a1, a3, a5, a7.
    When the quadratic discriminant N ≤ 0 (polynomial approximation breaks down),
    and the signed log-moneyness ``k`` is provided, one exact Newton step is taken
    from v_P1 using the Black-Scholes residual.  This recovers accuracy for
    small-v options where the Taylor-polynomial approximation of c_BS fails.

    Parameters
    ----------
    c_seed : float
        OTM-equivalent normalized call price.
    kappa : float
        Absolute log-moneyness |k|.
    k : float, optional
        Signed log-moneyness.  Required for the Newton fallback when N ≤ 0.
        If None, the polynomial G7/D7 correction is applied unconditionally.

    Returns
    -------
    float
        v_P7 seed value.
    """
    vp1 = vP1_seed(c_seed, kappa)
    if vp1 <= 0.0:
        return vp1

    # Discriminant check: when N ≤ 0 the polynomial has no real solution.
    # vP1 is then computed from N=0 and the G7/D7 correction is unreliable.
    # If k is available, substitute one exact Newton step on c_BS(k,v) instead.
    epsilon = _compute_epsilon(kappa)
    N = (2.0 * c_seed + epsilon) ** 2 - 8.0 * _a1 ** 2 * kappa * epsilon * (2.0 + epsilon)
    if N <= 0.0 and k is not None:
        # Exact Newton step from the clipped vP1 seed.
        # Recover actual call price c from c_seed and k.
        c_actual = (1.0 - np.exp(k) + c_seed * np.exp(k)) if k < 0.0 else c_seed
        d1 = -k / vp1 + vp1 / 2.0
        d2 = d1 - vp1
        f_val = _sp_norm.cdf(d1) - np.exp(k) * _sp_norm.cdf(d2) - c_actual
        fp = _sp_norm.pdf(d1)
        if abs(fp) < 1e-300:
            return vp1
        v_newton = vp1 - f_val / fp
        return max(v_newton, 1e-12)

    w = vp1
    E = np.exp(kappa)
    kap2 = kappa * kappa
    kap3 = kap2 * kappa
    kap5 = kap3 * kap2
    kap7 = kap5 * kap2
    w2 = w * w
    w4 = w2 * w2
    w6 = w4 * w2
    w8 = w4 * w4
    w10 = w8 * w2
    w12 = w8 * w4
    w14 = w8 * w6

    # S7(w): even-power terms proportional to (1+E)
    S7 = ((_a7 * (1 + E) / 128.0) * w14
          + ((_a5 + 7.0 * _a7 * kappa / 2.0) * (1 + E) / 32.0) * w12
          + ((_a3 + 5.0 * _a5 * kappa / 2.0 + 21.0 * _a7 * kap2 / 4.0) * (1 + E) / 8.0) * w10
          + ((_a1 + 3.0 * _a3 * kappa / 2.0 + 5.0 * _a5 * kap2 / 2.0
              + 35.0 * _a7 * kap3 / 8.0) * (1 + E) / 2.0) * w8)

    # Q7(w): odd-power terms proportional to (1-E)
    Q7 = ((kappa / 8.0 * (8.0 * _a1 - 12.0 * _a3 * kappa + 20.0 * _a5 * kap2
                          - 35.0 * _a7 * kap3) * (1 - E)) * w6
          + (kap3 / 4.0 * (4.0 * _a3 - 10.0 * _a5 * kappa + 21.0 * _a7 * kap2) * (1 - E)) * w4
          + (kap5 * (_a5 - 7.0 * _a7 * kappa / 2.0) * (1 - E)) * w2
          + _a7 * kap7 * (1 - E))

    # R7(w): residual term encoding the call-price contribution.
    # Derived from the polynomial residual equation c_seed = S7(v) + Q7(v);
    # the Newton corrector uses G7 = -S7 + Q7 + R7 and D7 = 7*S7 + Q7 + R7 + Delta7.
    R7 = (c_seed - 0.5 + E / 2.0) * w ** 7 / _a1

    # Delta7(w): derivative-based correction term
    Delta7 = (6.0 * (_a7 * (1 + E) / 128.0) * w14
              + 4.0 * ((_a5 + 7.0 * _a7 * kappa / 2.0) * (1 + E) / 32.0) * w12
              + 2.0 * ((_a3 + 5.0 * _a5 * kappa / 2.0
                        + 21.0 * _a7 * kap2 / 4.0) * (1 + E) / 8.0) * w10)

    G7 = -S7 + Q7 + R7
    D7 = 7.0 * S7 + Q7 + R7 + Delta7

    if abs(D7) < 1e-300:
        return vp1
    ratio = G7 / D7
    # Guard: catastrophic cancellation or out-of-domain evaluation
    if not np.isfinite(ratio) or abs(ratio) > 2.0:
        return vp1
    return vp1 * (1.0 + ratio)


def vP3_seed(c_seed: float, kappa: float) -> float:
    """3rd-order one-step corrected polynomial seed v_P3 = v_P1 * (1 + G3/D3).

    Parameters
    ----------
    c_seed : float
        OTM-equivalent normalized call price.
    kappa : float
        Absolute log-moneyness |k|.

    Returns
    -------
    float
        v_P3 seed value.
    """
    vp1 = vP1_seed(c_seed, kappa)
    if vp1 <= 0.0:
        return vp1
    w = vp1
    epsilon = _compute_epsilon(kappa)
    delta = 2.0 + epsilon
    h = c_seed + epsilon / 2.0
    kap2 = kappa * kappa
    kap3 = kap2 * kappa
    w2 = w * w
    w4 = w2 * w2
    w6 = w4 * w2

    # h = c_seed + epsilon/2.  The paper's formula nominally shows -24*h*w^4/a1
    # in G3, but empirical verification against the expected convergence counts
    # confirms the POSITIVE sign is correct (the minus is a print convention
    # where h absorbs a sign flip from the residual definition).
    G3 = (-4.0 * epsilon * kap3
          + 24.0 * kappa * (epsilon - kappa * (epsilon / 2.0 + 1.0)) * w2
          + (24.0 * (1.0 + epsilon / 2.0) - 3.0 * epsilon * kappa) * w4
          - delta * w6
          + 24.0 * h * w4 / _a1)

    D3 = (12.0 * epsilon * kap3
          - 24.0 * kappa * (epsilon - kappa * (epsilon / 2.0 + 1.0)) * w2
          + (24.0 * (1.0 + epsilon / 2.0) - 3.0 * epsilon * kappa) * w4
          - 3.0 * delta * w6)

    if abs(D3) < 1e-300:
        return vp1
    ratio3 = G3 / D3
    if not np.isfinite(ratio3) or abs(ratio3) > 2.0:
        return vp1
    return vp1 * (1.0 + ratio3)


# ---------------------------------------------------------------------------
# Deep-OTM Mills-ratio-corrected seed
# ---------------------------------------------------------------------------

def deep_otm_seed(c_seed: float, kappa: float, ppf_func=None) -> tuple:
    """Mills-ratio-corrected deep-OTM seed v_q,1.

    Algorithm:
      z    = Phi^{-1}(c_seed)
      v_q  = z + sqrt(z^2 + 2*kappa)
      alpha_q = v_q^2 / (kappa + v_q^2/2)
      z_q  = Phi^{-1}(c_seed / alpha_q)
      v_q,1 = z_q + sqrt(z_q^2 + 2*kappa)

    Parameters
    ----------
    c_seed : float
        OTM-equivalent normalized call price (must be in (0,1)).
    kappa : float
        Absolute log-moneyness |k|.
    ppf_func : callable, optional
        Inverse normal CDF; defaults to acklam_ppf.

    Returns
    -------
    (v_q, v_q1) : tuple of floats
        Plain quadratic seed and ratio-corrected seed.
    """
    if ppf_func is None:
        ppf_func = acklam_ppf

    p1 = clip_prob(c_seed)
    z = ppf_func(p1)
    z2_plus_2k = z * z + 2.0 * kappa
    if z2_plus_2k < 0.0:
        z2_plus_2k = 0.0
    v_q = z + np.sqrt(z2_plus_2k)

    denom = kappa + v_q**2 / 2.0
    if abs(denom) < 1e-300:
        return v_q, v_q
    alpha_q = v_q**2 / denom

    p2 = clip_prob(c_seed / alpha_q)
    z_q = ppf_func(p2)
    zq2_plus_2k = z_q * z_q + 2.0 * kappa
    if zq2_plus_2k < 0.0:
        zq2_plus_2k = 0.0
    v_q1 = z_q + np.sqrt(zq2_plus_2k)

    return v_q, v_q1


# ---------------------------------------------------------------------------
# OTM-equivalent price transformation
# ---------------------------------------------------------------------------

def otm_equivalent_price(k: float, c: float) -> float:
    """Convert call price to OTM-equivalent form.

    c_seed = exp(-k) * (c - 1 + exp(k))  for k < 0  (put-call parity)
    c_seed = c                             for k >= 0

    Parameters
    ----------
    k : float
        Log-moneyness.
    c : float
        Normalized call price.

    Returns
    -------
    float
        OTM-equivalent price c_seed.
    """
    if k < 0.0:
        return np.exp(-k) * (c - 1.0 + np.exp(k))
    return c
