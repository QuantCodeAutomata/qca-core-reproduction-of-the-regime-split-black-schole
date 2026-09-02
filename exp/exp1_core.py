"""
Exp 1: Core reproduction of the regime-split Black-Scholes implied-total-volatility
solver with HH-4 refinement.

Validates:
  - Total volatility recovery accuracy (max |v_hat - v_true|)
  - Regime-split seed architecture with exact cutoffs
  - HH-4 update count distributions on Grid 1 and Grid 2
  - Timing benchmark (solver throughput in ns/IV)
"""
import sys
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.black_scholes import make_grid1, make_grid2, bs_call_vec
from src.solver import solve_grid, aggregate_metrics, KAPPA_1, KAPPA_2, KAPPA_3, KAPPA_4


def print_grid_summary(grid_name: str, grid: dict, results: dict) -> dict:
    """Print and return aggregate metrics for a solved grid."""
    m_all = aggregate_metrics(results, filter_near_otm=False)
    m_filt = aggregate_metrics(results, filter_near_otm=True)

    print(f"\n{'='*60}")
    print(f"{grid_name} — {m_all['n_options']} options")
    print(f"{'='*60}")
    print(f"  Max |v_hat - v_true|:    {m_all['max_abs_error']:.4e}")
    print(f"\n  --- ALL options (no near-OTM filter) ---")
    print(f"  Mean HH-4 updates:       {m_all['mean_updates']:.4f}")
    print(f"  Updates == 1:            {m_all['n_1_update']}")
    print(f"  Updates == 2:            {m_all['n_2_updates']}")
    print(f"  Updates >  2:            {m_all['n_gt2_updates']}")
    print(f"  Fraction ≤ 2:            {m_all['frac_le2']:.4f}")

    print(f"\n  --- With near-OTM filter (excl. regime B) ---")
    print(f"  Options after filter:    {m_filt['n_options']}")
    print(f"  Mean HH-4 updates:       {m_filt['mean_updates']:.4f}")
    print(f"  Updates == 1:            {m_filt['n_1_update']}")
    print(f"  Updates == 2:            {m_filt['n_2_updates']}")
    print(f"  Updates >  2:            {m_filt['n_gt2_updates']}")
    print(f"  Fraction ≤ 2:            {m_filt['frac_le2']:.4f}")

    # Regime distribution
    regimes, counts = np.unique(results['regime'], return_counts=True)
    print(f"\n  Regime distribution: {dict(zip(regimes, counts))}")

    return {'all': m_all, 'filtered': m_filt}


def timing_benchmark(grid: dict, n_tiles: int = 10) -> float:
    """Time the solver on tiled grid; return ns per inversion (best of 3 runs).

    Note: uses a small tile count (n_tiles=10) because the Python/scipy
    implementation is not optimised for throughput.  The paper's reference
    values (~134 ns/IV) are from a Numba-JIT or C implementation.
    """
    k_tiled = np.tile(grid['k'], n_tiles)
    c_tiled = np.tile(grid['c'], n_tiles)
    v_tiled = np.tile(grid['v_true'], n_tiles)
    n_total = len(k_tiled)

    tiled_grid = {'k': k_tiled, 'c': c_tiled, 'v_true': v_tiled}

    best_ns = float('inf')
    for _ in range(3):
        t0 = time.perf_counter()
        solve_grid(tiled_grid)
        t1 = time.perf_counter()
        ns = (t1 - t0) * 1e9 / n_total
        if ns < best_ns:
            best_ns = ns

    return best_ns, n_total


def plot_error_heatmap(grid: dict, results: dict, title: str,
                       save_path: str = None) -> None:
    """Plot heatmap of log10 absolute errors on the grid."""
    # Identify v and delta unique values
    v_u = np.unique(np.round(grid['v_true'], 6))
    d_u = np.unique(np.round(grid['delta'], 6))

    err_mat = np.zeros((len(v_u), len(d_u)))
    for i, vi in enumerate(v_u):
        for j, dj in enumerate(d_u):
            mask = (np.abs(grid['v_true'] - vi) < 1e-9) & (np.abs(grid['delta'] - dj) < 1e-9)
            if mask.any():
                err_mat[i, j] = np.log10(max(results['abs_error'][mask][0], 1e-20))

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(err_mat.T, aspect='auto', origin='lower',
                   extent=[v_u.min(), v_u.max(), d_u.min(), d_u.max()],
                   vmin=-16, vmax=-10, cmap='viridis')
    plt.colorbar(im, ax=ax, label='log10(|v_hat - v_true|)')
    ax.set_xlabel('Total Volatility v_true')
    ax.set_ylabel('Call Delta')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        plt.close()


def plot_update_distribution(results_dict: dict, save_path: str = None) -> None:
    """Plot histogram of HH-4 update counts for each grid."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, res) in zip(axes, results_dict.items()):
        nu = res['n_updates']
        counts = [int(np.sum(nu == i)) for i in range(1, int(nu.max()) + 2)]
        ax.bar(range(1, len(counts) + 1), counts, color='steelblue', edgecolor='k')
        ax.set_xlabel('Number of HH-4 Updates')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}: Update Distribution')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        plt.close()


def run_exp1() -> dict:
    """Execute all Exp 1 sub-tasks and return results dictionary."""
    print("=" * 70)
    print("EXP 1: Core Regime-Split Implied-Volatility Solver")
    print("=" * 70)

    results_all = {}

    # ------------------------------------------------------------------ Grid 1
    print("\n[1] Generating Grid 1 (41 x 8 = 328 options)...")
    g1 = make_grid1()
    assert len(g1['k']) == 328, "Grid 1 size mismatch"

    # Verify BS consistency
    c_recomp = bs_call_vec(g1['k'], g1['v_true'])
    max_bs_err = np.max(np.abs(c_recomp - g1['c']))
    print(f"  BS consistency check: max |c_recomputed - c_generated| = {max_bs_err:.4e}")
    assert max_bs_err < 1e-13

    print("[2] Solving Grid 1...")
    res1 = solve_grid(g1)
    m1 = print_grid_summary("Grid 1 (328 options)", g1, res1)
    results_all['grid1'] = res1
    results_all['metrics1'] = m1

    # ------------------------------------------------------------------ Grid 2
    print("\n[3] Generating Grid 2 (197 x 10 = 1970 options)...")
    g2 = make_grid2()
    assert len(g2['k']) == 1970, "Grid 2 size mismatch"

    print("[4] Solving Grid 2...")
    res2 = solve_grid(g2)
    m2 = print_grid_summary("Grid 2 (1970 options)", g2, res2)
    results_all['grid2'] = res2
    results_all['metrics2'] = m2

    # ------------------------------------------------------------------ Timing
    # Pure Python/scipy loop — uses small tile count; paper reference values
    # (~134 ns/IV single-thread) are from C/Numba implementations.
    TILES = 10
    print(f"\n[5] Timing benchmark (Grid 1, {TILES} tiles = {328*TILES:,} inversions)...")
    ns1, n1 = timing_benchmark(g1, n_tiles=TILES)
    print(f"  Grid 1: {ns1:.1f} ns/IV  ({n1:,} total inversions, Python/scipy)")
    print(f"  Paper reference (Numba/C): ~134 ns/IV  — Python overhead expected")

    print(f"\n[6] Timing benchmark (Grid 2, {TILES} tiles = {1970*TILES:,} inversions)...")
    ns2, n2 = timing_benchmark(g2, n_tiles=TILES)
    print(f"  Grid 2: {ns2:.1f} ns/IV  ({n2:,} total inversions, Python/scipy)")

    results_all['timing'] = {'grid1_ns_per_iv': ns1, 'grid2_ns_per_iv': ns2}

    # ------------------------------------------------------------------ Sample table
    print("\n[7] Sample per-option table (first 10 options, Grid 1):")
    sample = pd.DataFrame({
        'k': g1['k'][:10],
        'c': g1['c'][:10],
        'v_true': g1['v_true'][:10],
        'v_hat': res1['v_hat'][:10],
        'abs_err': res1['abs_error'][:10],
        'seed_ratio': res1['seed_ratio'][:10],
        'eta_v': res1['eta_v'][:10],
        'regime': res1['regime'][:10],
        'n_upd': res1['n_updates'][:10],
    })
    print(sample.to_string(index=False))

    # ------------------------------------------------------------------ Plots
    os.makedirs('../results', exist_ok=True)
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    plot_error_heatmap(g1, res1, 'Grid 1: log10 absolute error',
                       os.path.join(results_dir, 'exp1_grid1_error_heatmap.png'))
    plot_error_heatmap(g2, res2, 'Grid 2: log10 absolute error',
                       os.path.join(results_dir, 'exp1_grid2_error_heatmap.png'))
    plot_update_distribution({'Grid 1 (328)': res1, 'Grid 2 (1970)': res2},
                             os.path.join(results_dir, 'exp1_update_distribution.png'))
    print("\n  Plots saved to results/")

    return results_all


if __name__ == '__main__':
    run_exp1()
