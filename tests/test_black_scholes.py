"""
Tests for src/black_scholes.py — normalized BS call pricing and grid generation.
"""
import numpy as np
import pytest
from scipy.stats import norm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.black_scholes import (
    bs_call, bs_call_vec, bs_call_intrinsic, bs_call_time_value,
    generate_grid, make_grid1, make_grid2,
)


class TestBSCall:
    def test_atm_bounds(self):
        """ATM call price (k=0) should be in (0, 1) for v > 0."""
        for v in [0.1, 0.5, 1.0, 2.0]:
            c = bs_call(0.0, v)
            assert 0.0 < c < 1.0

    def test_atm_formula(self):
        """ATM: c = 2*Phi(v/2) - 1 = erf(v/(2*sqrt(2)))."""
        from scipy.special import erf
        for v in [0.2, 0.5, 1.0]:
            c = bs_call(0.0, v)
            expected = erf(v / (2.0 * np.sqrt(2.0)))
            assert abs(c - expected) < 1e-12

    def test_otm_below_atm(self):
        """OTM call (k > 0) cheaper than ATM at same v."""
        v = 0.5
        assert bs_call(0.5, v) < bs_call(0.0, v)

    def test_intrinsic_lower_bound(self):
        """c_BS(k,v) >= max(1 - exp(k), 0)."""
        for k in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]:
            for v in [0.1, 0.5, 1.0]:
                c = bs_call(k, v)
                assert c >= max(1.0 - np.exp(k), 0.0) - 1e-13

    def test_upper_bound(self):
        """c_BS(k,v) < 1 for all finite k, v > 0."""
        for k in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            for v in [0.01, 0.5, 2.0]:
                assert bs_call(k, v) < 1.0

    def test_zero_vol(self):
        """Zero vol gives intrinsic value."""
        assert abs(bs_call(0.0, 0.0) - 0.0) < 1e-12
        assert abs(bs_call(-1.0, 0.0) - (1.0 - np.exp(-1.0))) < 1e-12
        assert abs(bs_call(1.0, 0.0) - 0.0) < 1e-12

    def test_put_call_parity(self):
        """c - p = 1 - exp(k) where p = exp(k)*Phi(-d2) - Phi(-d1)."""
        for k in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for v in [0.3, 0.8]:
                d1 = -k / v + v / 2.0
                d2 = d1 - v
                c = bs_call(k, v)
                p = norm.cdf(-d2) * np.exp(k) - norm.cdf(-d1)
                parity = c - p - (1.0 - np.exp(k))
                assert abs(parity) < 1e-12


class TestBSCallVec:
    def test_matches_scalar(self):
        k = np.array([-1.0, 0.0, 0.5])
        v = np.array([0.5, 0.5, 0.5])
        cvec = bs_call_vec(k, v)
        for i in range(len(k)):
            assert abs(cvec[i] - bs_call(k[i], v[i])) < 1e-14

    def test_shape(self):
        k = np.linspace(-2, 2, 20)
        v = np.ones(20) * 0.5
        c = bs_call_vec(k, v)
        assert c.shape == (20,)


class TestGridGeneration:
    def test_grid1_size(self):
        g = make_grid1()
        assert len(g['k']) == 328

    def test_grid2_size(self):
        g = make_grid2()
        assert len(g['k']) == 1970

    def test_price_consistency(self):
        """Generated prices satisfy BS formula."""
        g = make_grid1()
        c_recomp = bs_call_vec(g['k'], g['v_true'])
        max_err = np.max(np.abs(c_recomp - g['c']))
        assert max_err < 1e-13, f"BS consistency error: {max_err:.2e}"

    def test_positive_vols(self):
        g = make_grid2()
        assert np.all(g['v_true'] > 0)

    def test_delta_bounds(self):
        g = make_grid1()
        assert np.all(g['delta'] > 0)
        assert np.all(g['delta'] < 1)

    def test_c_above_intrinsic(self):
        g = make_grid1()
        intrinsic = np.maximum(1.0 - np.exp(g['k']), 0.0)
        assert np.all(g['c'] >= intrinsic - 1e-12)

    def test_c_below_one(self):
        g = make_grid2()
        assert np.all(g['c'] < 1.0)
