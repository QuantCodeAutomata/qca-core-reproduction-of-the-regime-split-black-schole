"""
Inverse normal CDF approximations.

Acklam's piecewise rational approximation and the Soranzo-Epure
fully algebraic approximation, as required by the paper's deep-OTM
seed formulas.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Acklam's piecewise rational inverse normal constants
# ---------------------------------------------------------------------------
_A = np.array([-3.969683028665376e+01, 2.209460984245205e+02,
               -2.759285104469687e+02, 1.383577518672690e+02,
               -3.066479806614716e+01, 2.506628277459239e+00])
_B = np.array([-5.447609879822406e+01, 1.615858368580409e+02,
               -1.556989798598866e+02, 6.680131188771972e+01,
               -1.328068155288572e+01])
_C = np.array([-7.784894002430293e-03, -3.223964580411365e-01,
               -2.400758277161838e+00, -2.549732539343734e+00,
                4.374664141464968e+00, 2.938163982698783e+00])
_D = np.array([7.784695709041462e-03, 3.224671290700398e-01,
               2.445134137142996e+00, 3.754408661907416e+00])
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def acklam_ppf(p: float) -> float:
    """Acklam's piecewise rational inverse normal CDF for scalar p in (0,1).

    Parameters
    ----------
    p : float
        Probability in (0, 1).

    Returns
    -------
    float
        Approximate Phi^{-1}(p).
    """
    p = float(p)
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    if p < _P_LOW:
        q = np.sqrt(-2.0 * np.log(p))
        return ((((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) /
                (((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1.0)))
    elif p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        return ((((((_A[0]*r+_A[1])*r+_A[2])*r+_A[3])*r+_A[4])*r+_A[5])*q /
                ((((((_B[0]*r+_B[1])*r+_B[2])*r+_B[3])*r+_B[4])*r+1.0)))
    else:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -((((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) /
                 (((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1.0)))


def acklam_ppf_vec(p: np.ndarray) -> np.ndarray:
    """Vectorized Acklam inverse normal for array p."""
    p = np.asarray(p, dtype=np.float64)
    out = np.empty_like(p)
    for i in range(p.size):
        out.flat[i] = acklam_ppf(p.flat[i])
    return out


# ---------------------------------------------------------------------------
# Soranzo-Epure algebraic inverse normal constants
# ---------------------------------------------------------------------------
_SE_ALPHA1 = 0.1487936
_SE_ALPHA2 = 0.0005160
_SE_BETA1 = 1.2735457
_SE_BETA2 = 0.1480931


def soranzo_epure_ppf(p: float) -> float:
    """Soranzo-Epure fully algebraic inverse normal for scalar p in (0,1).

    Solves A(L)*u^2 + B(L)*u + C(L) = 0 for the positive root u, then
    returns sign(p-1/2)*sqrt(u).  The correct root is selected via the
    numerically stable form u = 2*C / (-B - copysign(sqrt_disc, B)),
    which avoids catastrophic cancellation and always picks the root
    that is positive when A<0 and C>0 (the typical regime here).

    Parameters
    ----------
    p : float
        Probability in (0, 1).

    Returns
    -------
    float
        Approximate Phi^{-1}(p) via algebraic formula.
    """
    p = float(p)
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    sign = 1.0 if p >= 0.5 else -1.0
    q = 2.0 * max(p, 1.0 - p) - 1.0   # q in [0, 1)
    # L = log(1 - q^2) = log(4*p*(1-p)); always <= 0
    L = np.log(max(1.0 - q * q, 1e-300))
    A_L = -_SE_ALPHA1 / 2.0 - _SE_ALPHA2 * L / 2.0
    B_L = -_SE_BETA1 - _SE_BETA2 * L
    C_L = -2.0 * L
    disc = B_L * B_L - 4.0 * A_L * C_L
    disc = max(disc, 0.0)
    sqrt_disc = np.sqrt(disc)
    # Numerically stable form for the root with larger magnitude denominator:
    # u = 2*C / (-B - sign(B)*sqrt_disc).  With B<0 this becomes
    # u = 2*C / (-B + sqrt_disc), which is the positive root when C>0.
    denom = -B_L - np.copysign(sqrt_disc, B_L)
    if abs(denom) < 1e-300:
        return sign * np.sqrt(max(C_L / A_L, 0.0))  # degenerate fallback
    u = 2.0 * C_L / denom
    return sign * np.sqrt(max(u, 0.0))


def soranzo_epure_ppf_vec(p: np.ndarray) -> np.ndarray:
    """Vectorized Soranzo-Epure inverse normal."""
    p = np.asarray(p, dtype=np.float64)
    out = np.empty_like(p)
    for i in range(p.size):
        out.flat[i] = soranzo_epure_ppf(p.flat[i])
    return out


# Largest float64 strictly less than 1.0; used as the safe upper bound for
# clip_prob so that clip_prob(1.0) < 1.0 even when `tiny` underflows to 0
# in 1.0 - tiny (which happens for tiny < ~2.2e-16 in float64).
_ONE_BELOW = np.nextafter(1.0, 0.0)


def clip_prob(p: float, tiny: float = 1e-300) -> float:
    """Clip probability to (tiny, 1-tiny) for safe inverse-CDF calls.

    The upper bound is always capped at the largest float64 strictly below 1
    so that clip_prob(1.0) < 1.0 regardless of `tiny`.

    Parameters
    ----------
    p : float
        Raw probability value.
    tiny : float
        Minimum distance from 0; also determines 1-tiny when representable.

    Returns
    -------
    float
        Clipped probability strictly inside (0, 1).
    """
    hi = min(_ONE_BELOW, 1.0 - tiny) if (1.0 - tiny < 1.0) else _ONE_BELOW
    return max(tiny, min(hi, float(p)))
