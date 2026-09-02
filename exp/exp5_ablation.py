"""
Exp 5: Architecture ablation — near-OTM tail guard contribution.

Compares two solver variants:
  Variant A: three-band baseline (no tail guard)
  Variant B: four-band improved (with near-OTM tail guard)

Quantifies reduction in HH-4 update counts from the tail guard.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.black_scholes import make_grid1, make_grid2
from src.solver import solve_grid, aggregate_metrics


def run_ablation(grid: dict, grid_name: str) -> dict:
    """Run ablation for a single grid."""
    print(f"\n{'='*60}")
    print(f"Grid: {grid_name}  ({len(grid['k'])} options)")
    print(f"{'='*60}")

    # Without tail guard (Variant A)
    res_no = solve_grid(grid, use_tail_guard=False)
    m_no = aggregate_metrics(res_no, filter_near_otm=False)

    # With tail guard (Variant B)
    res_yes = solve_grid(grid, use_tail_guard=True)
    m_yes = aggregate_metrics(res_yes, filter_near_otm=False)

    print(f"\n  Variant A (no tail guard):")
    print(f"    mean_updates = {m_no['mean_updates']:.4f}, "
          f"n1={m_no['n_1_update']}, n2={m_no['n_2_updates']}, "
          f"n>2={m_no['n_gt2_updates']}, frac<=2={m_no['frac_le2']:.4f}")

    print(f"\n  Variant B (with tail guard):")
    print(f"    mean_updates = {m_yes['mean_updates']:.4f}, "
          f"n1={m_yes['n_1_update']}, n2={m_yes['n_2_updates']}, "
          f"n>2={m_yes['n_gt2_updates']}, frac<=2={m_yes['frac_le2']:.4f}")

    print(f"\n  Improvement: delta_mean_updates = "
          f"{m_yes['mean_updates'] - m_no['mean_updates']:+.4f}")

    # Options changed by tail guard
    changed = res_no['regime'] != res_yes['regime']
    n_changed = int(np.sum(changed))
    print(f"\n  Options redirected by tail guard: {n_changed}")

    if n_changed > 0:
        nu_before = res_no['n_updates'][changed]
        nu_after = res_yes['n_updates'][changed]
        print(f"    Mean updates before: {nu_before.mean():.3f}")
        print(f"    Mean updates after:  {nu_after.mean():.3f}")
        print(f"    Regime before: {dict(zip(*np.unique(res_no['regime'][changed], return_counts=True)))}")
        print(f"    Regime after:  {dict(zip(*np.unique(res_yes['regime'][changed], return_counts=True)))}")

        # Show changed options
        idx_changed = np.where(changed)[0][:10]
        chg_df = pd.DataFrame({
            'k': grid['k'][idx_changed],
            'c_seed_approx': grid['c'][idx_changed],
            'v_true': grid['v_true'][idx_changed],
            'regime_before': res_no['regime'][idx_changed],
            'regime_after': res_yes['regime'][idx_changed],
            'upd_before': res_no['n_updates'][idx_changed],
            'upd_after': res_yes['n_updates'][idx_changed],
            'abs_err_before': res_no['abs_error'][idx_changed],
            'abs_err_after': res_yes['abs_error'][idx_changed],
        })
        print("\n  Changed-option sample:")
        print(chg_df.to_string(index=False))

    return {
        'no_guard': m_no,
        'with_guard': m_yes,
        'n_changed': n_changed,
        'res_no': res_no,
        'res_yes': res_yes,
    }


def run_exp5() -> dict:
    """Execute ablation on both benchmark grids."""
    print("=" * 70)
    print("EXP 5: Ablation — Near-OTM Tail Guard Contribution")
    print("=" * 70)

    g1 = make_grid1()
    g2 = make_grid2()

    ab1 = run_ablation(g1, 'Grid 1 (328 options)')
    ab2 = run_ablation(g2, 'Grid 2 (1970 options)')

    # ------------------------------------------------------------------ Plot
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (name, ab, res) in zip(axes, [
            ('Grid 1 (328)', ab1, ab1['res_no']),
            ('Grid 2 (1970)', ab2, ab2['res_no'])]):
        max_nu = max(res['n_updates'].max(), ab['res_yes']['n_updates'].max())
        bins = np.arange(0.5, max_nu + 1.5, 1)
        ax.hist(res['n_updates'], bins=bins, alpha=0.7, label='No tail guard', color='steelblue')
        ax.hist(ab['res_yes']['n_updates'], bins=bins, alpha=0.7, label='With tail guard', color='darkorange')
        ax.set_xlabel('HH-4 Update Count')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name}: Update Count Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'exp5_ablation.png'), dpi=120)
    plt.close()
    print("\n  Plot saved to results/exp5_ablation.png")

    return {'grid1': ab1, 'grid2': ab2}


if __name__ == '__main__':
    run_exp5()
