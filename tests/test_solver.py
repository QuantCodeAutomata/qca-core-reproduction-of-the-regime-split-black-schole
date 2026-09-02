"""
Tests for src/solver.py — full implied total-vol solver and grid aggregation.
"""
import numpy as np
import pytest
from scipy.stats import norm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.solver import solve_iv, solve_grid, aggregate_metrics
from src.black_scholes import bs_call, make_grid1, make_grid2


class TestSolveIV:
    def test_atm_accuracy(self):
        k, v_true = 0.0, 0.5
        c = bs_call(k, v_true)
        res = solve_iv(k, c)
        assert abs(res['v_hat'] - v_true) < 1e-12

    def test_otm_accuracy(self):
        k, v_true = 0.5, 0.8
        c = bs_call(k, v_true)
        res = solve_iv(k, c)
        assert abs(res['v_hat'] - v_true) < 1e-12

    def test_itm_accuracy(self):
        k, v_true = -1.0, 1.0
        c = bs_call(k, v_true)
        res = solve_iv(k, c)
        assert abs(res['v_hat'] - v_true) < 1e-12

    def test_result_keys(self):
        c = bs_call(0.0, 0.5)
        res = solve_iv(0.0, c)
        for key in ['v_hat', 'v0', 'v_ATM', 'regime', 'n_updates', 'f_final', 'converged']:
            assert key in res

    def test_regime_labels(self):
        """Regime label must be one of A, B, C, D."""
        for k in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            c = bs_call(k, 0.5)
            res = solve_iv(k, c)
            assert res['regime'] in ('A', 'B', 'C', 'D')

    def test_convergence_flag(self):
        c = bs_call(0.0, 0.5)
        res = solve_iv(0.0, c)
        assert res['converged'] is True

    def test_below_intrinsic_raises(self):
        with pytest.raises(ValueError):
            solve_iv(0.0, -0.1)

    def test_at_or_above_one_raises(self):
        with pytest.raises(ValueError):
            solve_iv(0.0, 1.0)

    def test_deep_otm_accuracy(self):
        """Deep OTM: c_seed < c_star and kappa > kappa_star."""
        delta = 0.02
        v_true = 0.5
        d1 = norm.ppf(delta)
        k = v_true * (v_true / 2.0 - d1)
        d2 = d1 - v_true
        c = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
        res = solve_iv(k, c)
        assert abs(res['v_hat'] - v_true) < 1e-11

    def test_atm_regime_A(self):
        """Near k=0, regime should be A."""
        c = bs_call(1e-5, 0.5)
        res = solve_iv(1e-5, c)
        assert res['regime'] == 'A'

    def test_residual_below_tol(self):
        tol = 1e-14
        for k in [-1.0, 0.0, 1.0]:
            c = bs_call(k, 0.8)
            res = solve_iv(k, c, tol=tol)
            if res['converged']:
                assert abs(res['f_final']) < tol

    def test_without_tail_guard(self):
        """Without tail guard, should still converge accurately."""
        for k in [-2.0, -1.0, 0.0, 0.5, 1.5]:
            c = bs_call(k, 0.6)
            res = solve_iv(k, c, use_tail_guard=False)
            assert abs(res['v_hat'] - 0.6) < 1e-11


class TestSolveGrid:
    def test_grid1_max_error(self):
        """Max absolute error on grid1 must be < 1e-12."""
        g = make_grid1()
        results = solve_grid(g)
        assert np.max(results['abs_error']) < 1e-12

    def test_grid2_max_error(self):
        """Max absolute error on grid2 must be < 1e-12."""
        g = make_grid2()
        results = solve_grid(g)
        assert np.max(results['abs_error']) < 1e-12

    def test_output_keys(self):
        g = make_grid1()
        results = solve_grid(g)
        for key in ['v_hat', 'v0', 'regime', 'n_updates', 'abs_error', 'seed_ratio', 'eta_v']:
            assert key in results

    def test_all_converge_grid1(self):
        g = make_grid1()
        results = solve_grid(g)
        tol = 1e-13
        max_err = np.max(results['abs_error'])
        assert max_err < tol, f"Not converged: max_error={max_err:.2e}"

    def test_seed_ratio_array(self):
        g = make_grid1()
        results = solve_grid(g)
        assert results['seed_ratio'].shape == (328,)
        assert np.all(np.isfinite(results['seed_ratio']))


class TestAggregateMetrics:
    def test_mean_updates_grid1(self):
        """Mean updates on grid1 should be below 2.1."""
        g = make_grid1()
        results = solve_grid(g)
        metrics = aggregate_metrics(results)
        assert metrics['mean_updates'] < 2.1

    def test_mean_updates_grid2(self):
        g = make_grid2()
        results = solve_grid(g)
        metrics = aggregate_metrics(results)
        assert metrics['mean_updates'] < 2.1

    def test_n_options_grid1(self):
        g = make_grid1()
        results = solve_grid(g)
        m = aggregate_metrics(results)
        assert m['n_options'] == 328

    def test_frac_le2_high(self):
        """At least 90% of options should converge within 2 updates."""
        g = make_grid1()
        results = solve_grid(g)
        m = aggregate_metrics(results, filter_near_otm=True)
        assert m['frac_le2'] >= 0.90

    def test_max_abs_error_tiny(self):
        g = make_grid1()
        results = solve_grid(g)
        m = aggregate_metrics(results)
        assert m['max_abs_error'] < 1e-12
