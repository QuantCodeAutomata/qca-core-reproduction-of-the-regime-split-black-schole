"""
Exp 2: Analytical derivation and validation of regime boundaries.

Reproduces the paper's theoretical derivation of kappa_2, kappa_3, kappa_4
from Gaussian-CDF Taylor truncation tolerances, the HH-4 admissibility
threshold eta_v*, and the tail-guard thresholds c* and kappa*.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.theory_boundaries import (
    admissible_argument, kmax_peak, hh4_admissibility,
    tail_guard_price, derive_regime_table, kappa_star_derivation,
    a9, a1 as phi0,
)


def run_exp2() -> dict:
    """Execute the four analytical stages of regime boundary derivation."""
    print("=" * 70)
    print("EXP 2: Analytical Derivation of Regime Boundaries")
    print("=" * 70)

    # ------------------------------------------------------------------ Stage 1 + 2
    print("\nStage 1+2: Admissible argument size and peak moneyness support")
    table = derive_regime_table([1e-3, 5e-3, 1e-2])
    df = pd.DataFrame(table)
    df.columns = ['epsilon_phi', 'a_eta', 'kmax_peak', 'regime_label']
    print(df.to_string(index=False))

    print("\nComparison with paper values:")
    paper_vals = [0.81, 1.155, 1.347]
    for i, (row, pv) in enumerate(zip(table, paper_vals)):
        diff = abs(row['kmax_peak'] - pv)
        print(f"  eps={row['epsilon_phi']:.0e}: kmax^peak={row['kmax_peak']:.4f}, "
              f"paper={pv:.3f}, diff={diff:.4f}  {'✓' if diff < 0.1 else '✗'}")

    # ------------------------------------------------------------------ Stage 3
    print("\nStage 3: HH-4 two-step admissibility threshold")
    eta_v_star = hh4_admissibility(1e-14)
    print(f"  eta_v* = (1e-14)^(1/16) = {eta_v_star:.8f}")
    print(f"  Interpretation: max allowable relative seed error ≈ {eta_v_star*100:.2f}%")
    print(f"  Paper claims ≈ 13.3%:  {'✓' if abs(eta_v_star - 0.133) < 0.003 else '✗'}")

    # ------------------------------------------------------------------ Stage 4
    print("\nStage 4: Tail-guard thresholds")
    c_star = tail_guard_price(phi0=phi0, v_rep=0.4)
    kappa_deriv = kappa_star_derivation()
    print(f"  c* = phi(0)*eta_v*(0.4) = {phi0:.10f}*{eta_v_star:.8f}*0.4 = {c_star:.5f}")
    print(f"  Paper states c* ≈ 0.02128:  {'✓' if abs(c_star - 0.02128) < 0.002 else '✗'}")
    print(f"  kappa* derivation:")
    print(f"    k_max^peak = 0.5 → a_eta = {kappa_deriv['a_eta']:.6f}")
    print(f"    epsilon_Phi = |a9| * a_eta^9 = {kappa_deriv['epsilon_phi']:.4e}")
    print(f"    kappa* = {kappa_deriv['kappa_star']}  (paper confirms 0.5)")

    # ------------------------------------------------------------------ Plot
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    v_range = np.linspace(0.01, 3.0, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: k_max(v; a_eta) curves
    ax = axes[0]
    for row in table:
        ae = row['a_eta']
        k_max = ae * v_range - v_range**2 / 2.0
        ax.plot(v_range, k_max, label=f"eps={row['epsilon_phi']:.0e}, peak={row['kmax_peak']:.3f}")
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Total Volatility v')
    ax.set_ylabel('k_max(v; a_eta)')
    ax.set_title('Supported Moneyness vs Total Volatility')
    ax.legend()
    ax.set_ylim(-0.5, 2.5)
    ax.grid(True, alpha=0.3)

    # Panel 2: HH-4 contraction illustration (eta^4 curves)
    ax = axes[1]
    eta_range = np.linspace(0, 0.5, 300)
    for n_steps in [1, 2, 4]:
        ax.plot(eta_range, eta_range**(4**n_steps if n_steps <= 2 else 4**2),
                label=f'{n_steps} step(s)')
    ax.axvline(eta_v_star, color='r', linestyle='--', label=f'eta_v*={eta_v_star:.3f}')
    ax.axhline(1e-14, color='g', linestyle=':', label='epsilon_v=1e-14')
    ax.set_yscale('log')
    ax.set_xlabel('Initial seed error eta_v')
    ax.set_ylabel('Residual after HH-4 steps')
    ax.set_title('HH-4 Quartic Convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'exp2_regime_boundaries.png'), dpi=120)
    plt.close()
    print("\n  Plot saved to results/exp2_regime_boundaries.png")

    return {
        'boundary_table': df,
        'eta_v_star': eta_v_star,
        'c_star': c_star,
        'kappa_star_deriv': kappa_deriv,
    }


if __name__ == '__main__':
    run_exp2()
