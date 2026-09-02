"""
Tests for src/seeds.py — initial seed functions.
"""
import numpy as np
import pytest
from scipy.stats import norm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.seeds import (
    atm_seed, brenner_subrahmanyam_seed, taylor2_seed, taylor3_seed,
    taylor4_seed, vP1_seed, vP3_seed, vP7_seed, deep_otm_seed,
    otm_equivalent_price,
)
from src.black_scholes import bs_call


class TestATMSeeds:
    """ATM seed family (k=0 exact, c_tv = c for k=0)."""

    def _c_tv_atm(self, v):
        """c_tv = c - max(1-1,0) = c at k=0."""
        return bs_call(0.0, v)

    def test_atm_seed_positive(self):
        for v in [0.1, 0.5, 1.0, 2.0]:
            c_tv = self._c_tv_atm(v)
            assert atm_seed(c_tv) > 0.0

    def test_brenner_positive(self):
        for v in [0.1, 0.5, 1.0]:
            c_tv = self._c_tv_atm(v)
            assert brenner_subrahmanyam_seed(c_tv) > 0.0

    def test_taylor_order_improves(self):
        """Taylor4 should be closer to truth than Taylor2 at moderate v."""
        v_true = 0.8
        c_tv = self._c_tv_atm(v_true)
        err_t2 = abs(taylor2_seed(c_tv) / v_true - 1.0)
        err_t4 = abs(taylor4_seed(c_tv) / v_true - 1.0)
        assert err_t4 <= err_t2 + 1e-10

    def test_taylor4_equals_atm(self):
        """taylor4_seed and atm_seed should be identical."""
        c_tv = 0.1
        assert taylor4_seed(c_tv) == atm_seed(c_tv)

    def test_atm_small_ctv(self):
        """Very small time value gives small seed."""
        assert atm_seed(1e-8) < 1e-6


class TestVP1Seed:
    def test_positive(self):
        assert vP1_seed(0.05, 0.5) > 0.0

    def test_nonzero_moneyness(self):
        """vP1 at various kappa values should be positive."""
        for kappa in [0.01, 0.1, 0.5, 1.0]:
            val = vP1_seed(0.1, kappa)
            assert val > 0.0

    def test_negative_discriminant_clipped(self):
        """Should not raise even when discriminant would be negative."""
        # Tiny c_seed with large kappa — may hit clipping
        val = vP1_seed(1e-8, 2.0)
        assert np.isfinite(val)
        assert val >= 0.0


class TestVP7Seed:
    def test_positive(self):
        val = vP7_seed(0.05, 0.5)
        assert val > 0.0

    def test_close_to_truth_moderate(self):
        """v_P7 seed error at moderate moneyness should be reasonable (<50%)."""
        for delta in [0.1, 0.2, 0.3]:
            for v_true in [0.3, 0.6, 1.0]:
                d1 = norm.ppf(delta)
                k = v_true * (v_true / 2.0 - d1)
                kappa = abs(k)
                if kappa > 1.4:
                    continue
                d2 = d1 - v_true
                c = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
                c_seed = otm_equivalent_price(k, c)
                vp7 = vP7_seed(c_seed, kappa)
                if vp7 > 0:
                    err = abs(vp7 / v_true - 1.0)
                    assert err < 0.5, f"vP7 err={err:.3f} at v={v_true}, kappa={kappa:.3f}"


class TestVP3Seed:
    def test_positive(self):
        val = vP3_seed(0.05, 0.5)
        assert val > 0.0

    def test_finite(self):
        for kappa in [0.1, 0.5, 1.0, 1.3]:
            val = vP3_seed(0.05, kappa)
            assert np.isfinite(val)


class TestDeepOTMSeed:
    def test_returns_tuple(self):
        v_q, v_q1 = deep_otm_seed(0.01, 1.0)
        assert isinstance(v_q, float) and isinstance(v_q1, float)

    def test_v_q1_better_than_v_q(self):
        """v_q1 should typically be closer to truth than v_q on deep-OTM cases."""
        improved_count = 0
        total = 0
        for delta in [0.01, 0.02, 0.05]:
            for v_true in [0.5, 1.0, 1.5]:
                d1 = norm.ppf(delta)
                k = v_true * (v_true / 2.0 - d1)
                kappa = abs(k)
                d2 = d1 - v_true
                c = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
                c_seed = otm_equivalent_price(k, c)
                if c_seed <= 1e-15 or c_seed >= 1.0:
                    continue
                v_q, v_q1 = deep_otm_seed(c_seed, kappa)
                if v_q > 0 and np.isfinite(v_q) and np.isfinite(v_q1):
                    if abs(v_q1 / v_true - 1) <= abs(v_q / v_true - 1):
                        improved_count += 1
                    total += 1
        assert total > 0, "No valid test cases generated"
        assert improved_count / total >= 0.7, (
            f"v_q1 improved only {improved_count}/{total} cases"
        )

    def test_finite_output(self):
        v_q, v_q1 = deep_otm_seed(0.001, 2.0)
        assert np.isfinite(v_q) and np.isfinite(v_q1)


class TestOTMEquivalentPrice:
    def test_k_positive_unchanged(self):
        assert otm_equivalent_price(0.5, 0.1) == 0.1

    def test_k_zero_unchanged(self):
        assert otm_equivalent_price(0.0, 0.2) == 0.2

    def test_k_negative_transformation(self):
        k = -0.5
        c = 0.7
        c_seed = otm_equivalent_price(k, c)
        # Should equal put price (via put-call parity)
        expected = np.exp(-k) * (c - 1.0 + np.exp(k))
        assert abs(c_seed - expected) < 1e-15

    def test_c_seed_non_negative(self):
        """OTM-equivalent price should be non-negative."""
        for k in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]:
            for v in [0.3, 0.8]:
                c = bs_call(k, v)
                c_seed = otm_equivalent_price(k, c)
                assert c_seed >= -1e-12
