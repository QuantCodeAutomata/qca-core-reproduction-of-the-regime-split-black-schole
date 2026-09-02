"""
Exp 4: Inverse-CDF robustness study for the deep-OTM seed.

Compares deep-OTM Mills-ratio-corrected seed v_q,1 when computed with:
  Version A: Acklam's piecewise rational approximation
  Version B: Soranzo-Epure fully algebraic approximation

Verifies that differences are negligible and do not affect final accuracy.
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
from src.inverse_normal import acklam_ppf, soranzo_epure_ppf, clip_prob
from src.seeds import deep_otm_seed, otm_equivalent_price
from src.refinement import hh4_refine


def compare_ppf_accuracy() -> dict:
    """Compare both inverse-CDF approximations vs scipy reference."""
    p_vals = np.concatenate([
        np.logspace(-8, np.log10(0.02424), 60),
        np.linspace(0.025, 0.975, 100),
        1.0 - np.logspace(-8, np.log10(0.02424), 60),
    ])
    p_vals = np.clip(p_vals, 1e-12, 1 - 1e-12)
    ref = norm.ppf(p_vals)
    ack = np.array([acklam_ppf(p) for p in p_vals])
    se = np.array([soranzo_epure_ppf(p) for p in p_vals])
    return {
        'p_vals': p_vals,
        'ref': ref,
        'acklam': ack,
        'se': se,
        'acklam_max_err': float(np.max(np.abs(ack - ref))),
        'se_max_err': float(np.max(np.abs(se - ref))),
    }


def compare_deep_otm_seeds() -> pd.DataFrame:
    """Compare v_q,1 using Acklam vs Soranzo-Epure on deep-OTM cases."""
    delta_vals = np.array([0.01, 0.015, 0.02, 0.03, 0.05, 0.08])
    v_vals = np.arange(0.3, 3.01, 0.15)
    rows = []
    for v_true in v_vals:
        for delta in delta_vals:
            d1_true = norm.ppf(delta)
            k = v_true * (v_true / 2.0 - d1_true)
            kappa = abs(k)
            d2 = d1_true - v_true
            c = norm.cdf(d1_true) - np.exp(k) * norm.cdf(d2)
            c_seed = otm_equivalent_price(k, c)
            if c_seed <= 1e-15 or c_seed >= 1.0 or kappa < 0.5:
                continue
            # Version A: Acklam
            v_qa, v_q1a = deep_otm_seed(c_seed, kappa, ppf_func=acklam_ppf)
            # Version B: Soranzo-Epure
            v_qs, v_q1s = deep_otm_seed(c_seed, kappa, ppf_func=soranzo_epure_ppf)
            if not (np.isfinite(v_q1a) and np.isfinite(v_q1s)):
                continue
            # HH-4 polish both
            v_hat_a, n_a, _ = hh4_refine(k, v_q1a, c)
            v_hat_s, n_s, _ = hh4_refine(k, v_q1s, c)
            rows.append({
                'v_true': v_true, 'k': round(k, 4), 'kappa': round(kappa, 4),
                'c_seed': c_seed,
                'v_q1_acklam': v_q1a,
                'v_q1_se': v_q1s,
                'diff_seed': abs(v_q1a - v_q1s),
                'eta_q1_acklam': abs(v_q1a / v_true - 1),
                'eta_q1_se': abs(v_q1s / v_true - 1),
                'v_hat_acklam': v_hat_a,
                'v_hat_se': v_hat_s,
                'abs_err_acklam': abs(v_hat_a - v_true),
                'abs_err_se': abs(v_hat_s - v_true),
                'n_upd_acklam': n_a,
                'n_upd_se': n_s,
            })
    return pd.DataFrame(rows)


def run_exp4() -> dict:
    """Execute inverse-CDF robustness comparison."""
    print("=" * 70)
    print("EXP 4: Acklam vs Soranzo-Epure Inverse-CDF Robustness")
    print("=" * 70)

    # PPF accuracy
    ppf_res = compare_ppf_accuracy()
    p_vals = ppf_res['p_vals']
    ref_vals = ppf_res['ref']
    # Operational deep-OTM seed range: c_seed in (c*, 0.1) → p in (0.001, 0.10)
    op_mask = (p_vals >= 0.001) & (p_vals <= 0.10)
    print(f"\nInverse CDF accuracy vs scipy.stats.norm.ppf (full range p in ~[1e-8, 1-1e-8]):")
    print(f"  Acklam   max |err|: {ppf_res['acklam_max_err']:.4e}")
    print(f"  Soranzo  max |err|: {ppf_res['se_max_err']:.4e}")
    print(f"  Note: Soranzo-Epure is a fully algebraic *tail* approximation.")
    print(f"        For very extreme p (|p|<1e-4), it can return 0 (known limitation).")
    print(f"        For the operational deep-OTM seed input range p in (0.001, 0.10):")
    if op_mask.any():
        se_op_err = np.max(np.abs(ppf_res['se'][op_mask] - ref_vals[op_mask]))
        ack_op_err = np.max(np.abs(ppf_res['acklam'][op_mask] - ref_vals[op_mask]))
        print(f"    Acklam  max |err|: {ack_op_err:.4e}")
        print(f"    Soranzo max |err|: {se_op_err:.4e}")

    # Deep-OTM seed comparison
    df = compare_deep_otm_seeds()

    if len(df):
        print(f"\nDeep-OTM Seed Comparison ({len(df)} test cases):")
        print(f"  Seed discrepancy  |v_q1_A - v_q1_SE|:  max={df['diff_seed'].max():.4e}, "
              f"mean={df['diff_seed'].mean():.4e}")
        polished_diff = (df['v_hat_acklam'] - df['v_hat_se']).abs()
        print(f"  Polished diff     |v_hat_A - v_hat_SE|: max={polished_diff.max():.4e}, "
              f"mean={polished_diff.mean():.4e}")
        print(f"  Acklam  max abs error: {df['abs_err_acklam'].max():.4e}")
        print(f"  SE      max abs error: {df['abs_err_se'].max():.4e}")
        print(f"\n  Sample rows:")
        print(df[['v_true', 'kappa', 'diff_seed', 'eta_q1_acklam', 'eta_q1_se']]
              .head(10).to_string(index=False))
    else:
        print("  No valid deep-OTM test cases generated.")

    # ------------------------------------------------------------------ Plots
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(ppf_res['p_vals'], np.abs(ppf_res['acklam'] - ppf_res['ref']),
                label='Acklam', alpha=0.8)
    ax.semilogy(ppf_res['p_vals'], np.abs(ppf_res['se'] - ppf_res['ref']),
                label='Soranzo-Epure', alpha=0.8)
    ax.set_xlabel('Probability p')
    ax.set_ylabel('|approx - scipy.ppf|')
    ax.set_title('Inverse Normal CDF Accuracy')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    if len(df):
        ax = axes[1]
        ax.scatter(df['kappa'], df['diff_seed'], s=15, alpha=0.7)
        ax.set_yscale('log')
        ax.set_xlabel('kappa = |k|')
        ax.set_ylabel('|v_q1_Acklam - v_q1_SE|')
        ax.set_title('Seed Discrepancy vs Moneyness')
        ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'exp4_robustness.png'), dpi=120)
    plt.close()
    print("\n  Plot saved to results/exp4_robustness.png")

    return {'ppf_accuracy': ppf_res, 'seed_comparison': df}


if __name__ == '__main__':
    run_exp4()
