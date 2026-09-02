"""
Theoretical derivation of regime boundaries for the solver.

Reproduces the paper's analytical justification of:
  - kappa_2, kappa_3, kappa_4 from Gaussian-CDF Taylor truncation tolerances
  - HH-4 two-step admissibility threshold eta_v*
  - Tail-guard price threshold c* and moneyness threshold kappa*
"""
import numpy as np

# Gaussian CDF Taylor odd-power coefficients: Phi(x) ≈ sum_{j} a_{2j+1} * x^{2j+1}
a1 = 0.3989422804   # phi(0) = 1/sqrt(2*pi)
a3 = -0.0664903801
a5 = 0.0099735570
a7 = -0.0011873282
a9 = 0.0001154347   # first neglected term = 1/(3456*sqrt(2*pi))


def admissible_argument(epsilon_phi: float, a_next: float = None) -> float:
    """Maximum admissible CDF argument from P7 Taylor truncation tolerance.

    From the first neglected term |a_next| * |x|^9 <= epsilon_phi:
        a_eta = (epsilon_phi / |a_next|)^(1/9)

    Parameters
    ----------
    epsilon_phi : float
        Target CDF truncation tolerance.
    a_next : float, optional
        Magnitude of first neglected coefficient; defaults to |a9|.

    Returns
    -------
    float
        a_eta: maximum admissible argument of the truncated Taylor series.
    """
    if a_next is None:
        a_next = abs(a9)
    return (epsilon_phi / a_next) ** (1.0 / 9.0)


def kmax_peak(a_eta: float) -> float:
    """Peak moneyness supported by the P7 polynomial approximation region.

    From k_max(v; a_eta) = a_eta*v - v^2/2 (using |d2| <= a_eta as constraint),
    differentiate with respect to v to get peak at v* = a_eta:
        k_max^peak = a_eta^2 / 2

    Parameters
    ----------
    a_eta : float
        Admissible CDF argument bound.

    Returns
    -------
    float
        k_max^peak = a_eta^2 / 2.
    """
    return a_eta**2 / 2.0


def hh4_admissibility(epsilon_v: float = 1e-14) -> float:
    """HH-4 two-step admissibility threshold for the initial relative seed error.

    For two successive quartic contractions (each reduces error to 4th power),
    the initial relative seed error eta_v must satisfy:
        eta_v* = epsilon_v^(1/16)
    so that after two HH-4 updates the residual is below epsilon_v.

    Parameters
    ----------
    epsilon_v : float
        Target total-volatility accuracy tolerance.

    Returns
    -------
    float
        eta_v*: maximum allowable initial relative seed error.
    """
    return epsilon_v ** (1.0 / 16.0)


def tail_guard_price(phi0: float = a1, eta_v: float = None,
                     v_rep: float = 0.4) -> float:
    """Derive the tail-guard price threshold c*.

    Approximation: c*(v) ≈ phi(0) * eta_v* * v
    Evaluated at representative total volatility v_rep = 0.4.

    Parameters
    ----------
    phi0 : float
        phi(0) = 1/sqrt(2*pi) ≈ 0.3989422804.
    eta_v : float, optional
        HH-4 admissibility threshold; computed from epsilon_v=1e-14 if None.
    v_rep : float
        Representative total volatility.

    Returns
    -------
    float
        c_star: tail-guard price threshold.
    """
    if eta_v is None:
        eta_v = hh4_admissibility(1e-14)
    return phi0 * eta_v * v_rep


def derive_regime_table(epsilon_phi_list: list = None) -> list:
    """Produce derivation table: epsilon_phi → a_eta → k_max^peak → regime label.

    Parameters
    ----------
    epsilon_phi_list : list, optional
        Truncation tolerances to evaluate; defaults to [1e-3, 5e-3, 1e-2].

    Returns
    -------
    list of dict with keys: epsilon_phi, a_eta, kmax_peak, regime_label.
    """
    if epsilon_phi_list is None:
        epsilon_phi_list = [1e-3, 5e-3, 1e-2]
    label_map = {0: 'kappa_2 ≈ 0.81', 1: 'kappa_3 ≈ 1.155', 2: 'kappa_4 ≈ 1.347'}
    rows = []
    for idx, eps in enumerate(epsilon_phi_list):
        ae = admissible_argument(eps)
        km = kmax_peak(ae)
        rows.append({
            'epsilon_phi': eps,
            'a_eta': ae,
            'kmax_peak': km,
            'regime_label': label_map.get(idx, f'eps={eps:.1e}'),
        })
    return rows


# Alias used by tests and external callers
derive_boundary_table = derive_regime_table


def kappa_star_derivation() -> dict:
    """Derive kappa* = 0.5 from k_max^peak = 0.5 ⟹ a_eta = 1.

    If k_max^peak = a_eta^2/2 = 0.5, then a_eta = sqrt(1) = 1.
    Then epsilon_Phi = |a9| * a_eta^9 = |a9| ≈ 1.154e-4,
    which motivates kappa* = 0.5 as the near-OTM tail-guard threshold.

    Returns
    -------
    dict with keys: kmax_target, a_eta, epsilon_phi, kappa_star.
    """
    kmax_target = 0.5
    a_eta = np.sqrt(2.0 * kmax_target)   # = 1.0
    eps_phi = abs(a9) * a_eta**9          # = |a9|
    return {
        'kmax_target': kmax_target,
        'a_eta': a_eta,
        'epsilon_phi': eps_phi,
        'kappa_star': kmax_target,
    }
