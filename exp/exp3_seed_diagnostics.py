"""
Exp 3: Seed-diagnostics reproduction for ATM, transition, and deep-OTM regimes.

Evaluates raw initial seeds before HH-4 polishing:
  - ATM family: BS, Taylor2, Taylor3, Taylor4
  - Transition region: v_P7, (v_P3+v_P7)/2, v_P3
  - Deep-OTM: v_q vs v_q,1
"""
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.seeds import (
    atm_seed, brenner_subrahmanyam_seed, taylor2_seed, taylor3_seed,
    taylor4_seed, vP1_seed, vP3_seed, vP7_seed, deep_otm_seed,
    otm_equivalent_price,
)
from src.black_scholes import bs_call
from src.theory_boundaries import hh4_admissibility

HH4_THRESH = hh4_admissibility(1e-14)   # ≈ 0.133


def run_atm_diagnostics() -> pd.DataFrame:
    """Compare ATM seed family at k=0 over a range of total volatilities."""
    print("\n--- ATM Seed Diagnostics (k=0) ---")
    v_range = np.concatenate([[0.05, 0.10, 0.20, 0.30],
                               np.arange(0.5, 2.01, 0.5)])
    rows = []
    for v_true in v_range:
        c = bs_call(0.0, v_true)
        c_tv = c   # intrinsic = 0 at k=0
        v_bs = brenner_subrahmanyam_seed(c_tv)
        v_t2 = taylor2_seed(c_tv)
        v_t3 = taylor3_seed(c_tv)
        v_t4 = taylor4_seed(c_tv)
        rows.append({
            'v_true': v_true,
            'v_BS': v_bs, 'eta_BS': abs(v_bs/v_true - 1),
            'v_T2': v_t2, 'eta_T2': abs(v_t2/v_true - 1),
            'v_T3': v_t3, 'eta_T3': abs(v_t3/v_true - 1),
            'v_T4': v_t4, 'eta_T4': abs(v_t4/v_true - 1),
        })
    df = pd.DataFrame(rows)
    print(df[['v_true', 'eta_BS', 'eta_T2', 'eta_T3', 'eta_T4']].to_string(index=False))
    return df


def run_transition_diagnostics() -> pd.DataFrame:
    """Compare polynomial seeds in transition region 0.81 < kappa <= 1.347.

    Only options with c_seed >= 0.02128 are included (tail-guard cases redirect
    to the deep-OTM seed and are therefore outside the polynomial regime).
    The paper's specific representative point k=1.280, v=1.130 is appended.
    """
    print("\n--- Transition-Region Seed Diagnostics (0.81 < kappa <= 1.347, c_seed >= c*) ---")
    TAIL_GUARD = 0.02128  # c* threshold; cases below are redirected to deep-OTM seed
    # Use delta in a range that produces moderate-kappa options where the polynomial works
    delta_vals = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    v_vals = np.arange(0.5, 2.6, 0.1)
    rows = []
    for v_true in v_vals:
        for delta in delta_vals:
            d1_true = norm.ppf(delta)
            k = v_true * (v_true / 2.0 - d1_true)
            kappa = abs(k)
            if not (0.81 < kappa <= 1.347):
                continue
            d2 = d1_true - v_true
            c = norm.cdf(d1_true) - np.exp(k) * norm.cdf(d2)
            c_seed = otm_equivalent_price(k, c)
            # Exclude tail-guard cases — polynomial seeds are not used there
            if c_seed < TAIL_GUARD:
                continue
            vp7 = vP7_seed(c_seed, kappa)
            vp3 = vP3_seed(c_seed, kappa)
            v_avg = (vp3 + vp7) / 2.0
            rows.append({
                'v_true': v_true, 'k': round(k, 4), 'kappa': round(kappa, 4),
                'c_seed': round(c_seed, 6),
                'v_P7': vp7, 'eta_P7': abs(vp7 / v_true - 1),
                'v_avg': v_avg, 'eta_avg': abs(v_avg / v_true - 1),
                'v_P3': vp3, 'eta_P3': abs(vp3 / v_true - 1),
                'subregion': 'lower (0.81-1.155)' if kappa <= 1.155 else 'upper (1.155-1.347)',
            })

    # Paper representative point: k=1.280, v=1.130  (upper-transition, kappa=1.280)
    for v_true, k in [(1.130, 1.280)]:
        kappa = abs(k)
        d1_true = -k / v_true + v_true / 2.0
        d2 = d1_true - v_true
        c = norm.cdf(d1_true) - np.exp(k) * norm.cdf(d2)
        c_seed = otm_equivalent_price(k, c)
        vp7 = vP7_seed(c_seed, kappa)
        vp3 = vP3_seed(c_seed, kappa)
        v_avg = (vp3 + vp7) / 2.0
        rows.append({
            'v_true': v_true, 'k': round(k, 4), 'kappa': round(kappa, 4),
            'c_seed': round(c_seed, 6),
            'v_P7': vp7, 'eta_P7': abs(vp7 / v_true - 1),
            'v_avg': v_avg, 'eta_avg': abs(v_avg / v_true - 1),
            'v_P3': vp3, 'eta_P3': abs(vp3 / v_true - 1),
            'subregion': 'paper ref (upper, k=1.280)',
        })

    df = pd.DataFrame(rows)
    if len(df):
        print(df[['v_true', 'kappa', 'eta_P7', 'eta_avg', 'eta_P3', 'subregion']]
              .to_string(index=False))
    return df


def run_deep_otm_diagnostics() -> pd.DataFrame:
    """Compare v_q vs v_q,1 in deep-OTM regime."""
    print("\n--- Deep-OTM Seed Diagnostics ---")
    delta_vals = np.array([0.01, 0.02, 0.03, 0.05, 0.08])
    v_vals = np.arange(0.5, 2.51, 0.25)
    rows = []
    for v_true in v_vals:
        for delta in delta_vals:
            d1_true = norm.ppf(delta)
            k = v_true * (v_true / 2.0 - d1_true)
            kappa = abs(k)
            d2 = d1_true - v_true
            c = norm.cdf(d1_true) - np.exp(k) * norm.cdf(d2)
            c_seed = otm_equivalent_price(k, c)
            if c_seed <= 1e-15 or c_seed >= 1.0:
                continue
            v_q, v_q1 = deep_otm_seed(c_seed, kappa)
            if v_q > 0 and np.isfinite(v_q) and np.isfinite(v_q1):
                eta_q = abs(v_q / v_true - 1)
                eta_q1 = abs(v_q1 / v_true - 1)
                reduction = (eta_q - eta_q1) / eta_q * 100 if eta_q > 1e-15 else 0.0
                rows.append({
                    'v_true': v_true, 'kappa': round(kappa, 4),
                    'c_seed': c_seed,
                    'v_q': v_q, 'eta_q': eta_q,
                    'v_q1': v_q1, 'eta_q1': eta_q1,
                    'reduction_%': reduction,
                    'v_q1_below_thresh': eta_q1 < HH4_THRESH,
                })
    df = pd.DataFrame(rows)
    if len(df):
        print(df[['v_true', 'kappa', 'eta_q', 'eta_q1', 'reduction_%', 'v_q1_below_thresh']]
              .head(12).to_string(index=False))
    return df


def run_exp3() -> dict:
    """Execute all seed diagnostic sub-experiments."""
    print("=" * 70)
    print("EXP 3: Seed Diagnostics (ATM / Transition / Deep-OTM)")
    print("=" * 70)

    df_atm = run_atm_diagnostics()
    df_trans = run_transition_diagnostics()
    df_deep = run_deep_otm_diagnostics()

    # ------------------------------------------------------------------ Summaries
    print("\n=== SUMMARY ===")

    print("\nATM Seed Family — max relative error by seed:")
    for col in ['eta_BS', 'eta_T2', 'eta_T3', 'eta_T4']:
        mx = df_atm[col].max()
        at_v = df_atm.loc[df_atm[col].idxmax(), 'v_true']
        print(f"  {col}: max={mx:.4e}  at v={at_v:.2f}")

    if len(df_trans):
        lower = df_trans[df_trans['subregion'].str.startswith('lower')]
        upper = df_trans[df_trans['subregion'].str.startswith('upper')]
        paper_ref = df_trans[df_trans['subregion'].str.startswith('paper')]
        print("\nTransition Seeds (c_seed >= c* cases only):")
        if len(lower):
            print(f"  Lower (0.81-1.155): max eta_P7={lower['eta_P7'].max():.4e}, "
                  f"max eta_avg={lower['eta_avg'].max():.4e}")
        if len(upper):
            print(f"  Upper (1.155-1.347): max eta_avg={upper['eta_avg'].max():.4e}, "
                  f"max eta_P3={upper['eta_P3'].max():.4e}")
        if len(paper_ref):
            row = paper_ref.iloc[0]
            print(f"  Paper ref k=1.280,v=1.130: "
                  f"eta_P7={row['eta_P7']:.4e}, eta_avg={row['eta_avg']:.4e}, "
                  f"eta_P3={row['eta_P3']:.4e}  (paper reports max eta_avg≈8.67%)")

    if len(df_deep):
        print("\nDeep-OTM Corrected Seed:")
        print(f"  Mean error reduction: {df_deep['reduction_%'].mean():.1f}%")
        print(f"  Max eta_q1:           {df_deep['eta_q1'].max():.4e}")
        print(f"  Frac v_q1 < 13.3%:   {df_deep['v_q1_below_thresh'].mean():.2%}")

    # ------------------------------------------------------------------ Plots
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ATM seed errors vs v
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, label in [('eta_BS', 'BS'), ('eta_T2', 'Taylor2'),
                       ('eta_T3', 'Taylor3'), ('eta_T4', 'Taylor4 (ATM)')]:
        ax.semilogy(df_atm['v_true'], df_atm[col], marker='o', markersize=4, label=label)
    ax.set_xlabel('Total Volatility v')
    ax.set_ylabel('Relative Seed Error eta_v')
    ax.set_title('ATM Seed Family Relative Errors')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'exp3_atm_seeds.png'), dpi=120)
    plt.close()

    # Deep-OTM seed errors
    if len(df_deep):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(df_deep['kappa'], df_deep['eta_q'], label='v_q (plain)', alpha=0.7, s=25)
        ax.scatter(df_deep['kappa'], df_deep['eta_q1'], label='v_q,1 (corrected)', alpha=0.7, s=25)
        ax.axhline(HH4_THRESH, color='r', linestyle='--', label=f'HH-4 threshold {HH4_THRESH:.3f}')
        ax.set_yscale('log')
        ax.set_xlabel('Absolute Moneyness kappa = |k|')
        ax.set_ylabel('Relative Seed Error eta_v')
        ax.set_title('Deep-OTM Seed Comparison')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'exp3_deep_otm_seeds.png'), dpi=120)
        plt.close()

    print("\n  Plots saved to results/")

    return {'atm': df_atm, 'transition': df_trans, 'deep_otm': df_deep}


if __name__ == '__main__':
    run_exp3()
