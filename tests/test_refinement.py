"""
Tests for src/refinement.py — HH-4 single update and iteration loop.
"""
import numpy as np
import pytest
from scipy.stats import norm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.refinement import hh4_update, hh4_refine
from src.black_scholes import bs_call


class TestHH4Update:
    def test_single_update_reduces_residual(self):
        k, v_true = 0.0, 0.5
        c = bs_call(k, v_true)
        v0 = 0.6   # off by 20%
        v1, f, _ = hh4_update(k, v0, c)
        f0 = bs_call(k, v0) - c
        assert abs(f) <= abs(f0) + 1e-14

    def test_returns_three_tuple(self):
        k, v, c = 0.0, 0.5, bs_call(0.0, 0.5)
        result = hh4_update(k, v, c)
        assert len(result) == 3

    def test_update_count_always_one(self):
        k, v, c = 0.0, 0.5, bs_call(0.0, 0.5)
        _, _, n = hh4_update(k, v, c)
        assert n == 1


class TestHH4Refine:
    def test_atm_convergence(self):
        """ATM case should converge within a few updates."""
        k, v_true = 0.0, 0.5
        c = bs_call(k, v_true)
        v0 = 0.7
        v_hat, n, converged = hh4_refine(k, v0, c, tol=1e-14)
        assert converged
        assert abs(v_hat - v_true) < 1e-12

    def test_otm_convergence(self):
        k, v_true = 0.5, 0.8
        c = bs_call(k, v_true)
        v0 = 0.5
        v_hat, n, converged = hh4_refine(k, v0, c, tol=1e-14)
        assert converged
        assert abs(v_hat - v_true) < 1e-12

    def test_itm_convergence(self):
        k, v_true = -0.5, 0.8
        c = bs_call(k, v_true)
        v0 = 0.5
        v_hat, n, converged = hh4_refine(k, v0, c, tol=1e-14)
        assert converged
        assert abs(v_hat - v_true) < 1e-12

    def test_minimum_one_update(self):
        """At least one update must always be performed (repeat-until)."""
        k, v_true = 0.0, 0.5
        c = bs_call(k, v_true)
        # Even when starting from perfect seed, one update must run
        _, n, _ = hh4_refine(k, v_true, c, tol=1e-14)
        assert n >= 1

    def test_final_residual_below_tol(self):
        tol = 1e-14
        for k in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for v_true in [0.2, 0.5, 1.0]:
                c = bs_call(k, v_true)
                v_hat, _, converged = hh4_refine(k, v_true * 0.8, c, tol=tol)
                if converged:
                    d1 = -k / v_hat + v_hat / 2.0
                    d2 = d1 - v_hat
                    f = norm.cdf(d1) - np.exp(k) * norm.cdf(d2) - c
                    assert abs(f) < tol

    def test_convergence_within_few_updates(self):
        """With a good seed, convergence should happen in <= 10 updates."""
        k, v_true = 0.3, 0.7
        c = bs_call(k, v_true)
        _, n, converged = hh4_refine(k, v_true * 0.9, c, tol=1e-14)
        assert converged
        assert n <= 10

    def test_deep_otm(self):
        """Deep OTM case (large k and small v) should still converge."""
        from scipy.stats import norm
        delta = 0.05
        v_true = 0.4
        d1 = norm.ppf(delta)
        k = v_true * (v_true / 2.0 - d1)
        d2 = d1 - v_true
        c = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
        v_hat, n, converged = hh4_refine(k, v_true * 1.2, c, tol=1e-14)
        assert converged
        assert abs(v_hat - v_true) < 1e-11
