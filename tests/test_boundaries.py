"""
Tests for src/theory_boundaries.py — analytical regime boundary derivation.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.theory_boundaries import (
    admissible_argument, kmax_peak, hh4_admissibility, tail_guard_price,
    derive_boundary_table,
)


class TestAdmissibleArgument:
    A9 = 0.0001154347  # |a9|

    def test_positive(self):
        assert admissible_argument(1e-3, self.A9) > 0.0

    def test_larger_eps_larger_aeta(self):
        a1 = admissible_argument(1e-3, self.A9)
        a2 = admissible_argument(5e-3, self.A9)
        a3 = admissible_argument(1e-2, self.A9)
        assert a1 < a2 < a3

    def test_formula(self):
        eps = 1e-3
        a_eta = admissible_argument(eps, self.A9)
        assert abs(a_eta - (eps / self.A9) ** (1.0 / 9.0)) < 1e-12


class TestKmaxPeak:
    def test_formula(self):
        """k_max^peak = a_eta^2 / 2."""
        for a_eta in [1.0, 1.27, 1.52]:
            assert abs(kmax_peak(a_eta) - a_eta**2 / 2.0) < 1e-15

    def test_kappa2_boundary(self):
        """a_eta from epsilon_Phi=1e-3 should give kmax_peak ≈ 0.81."""
        a9 = 0.0001154347
        a_eta = admissible_argument(1e-3, a9)
        km = kmax_peak(a_eta)
        assert abs(km - 0.81) < 0.05

    def test_kappa3_boundary(self):
        a9 = 0.0001154347
        a_eta = admissible_argument(5e-3, a9)
        km = kmax_peak(a_eta)
        assert abs(km - 1.155) < 0.05

    def test_kappa4_boundary(self):
        a9 = 0.0001154347
        a_eta = admissible_argument(1e-2, a9)
        km = kmax_peak(a_eta)
        assert abs(km - 1.347) < 0.05


class TestHH4Admissibility:
    def test_formula(self):
        """eta_v* = epsilon_v^(1/16)."""
        eps_v = 1e-14
        eta = hh4_admissibility(eps_v)
        assert abs(eta - eps_v ** (1.0 / 16.0)) < 1e-15

    def test_value(self):
        """With epsilon_v=1e-14, eta* ≈ 0.133."""
        eta = hh4_admissibility(1e-14)
        assert abs(eta - 0.133) < 0.01

    def test_positive(self):
        assert hh4_admissibility(1e-14) > 0.0


class TestTailGuardPrice:
    def test_value(self):
        """phi(0)*eta_v*v_rep ≈ 0.02128 with phi(0)=0.3989422804, eta*=0.133, v=0.4."""
        phi0 = 0.3989422804
        eta = hh4_admissibility(1e-14)
        c_star = tail_guard_price(phi0, eta, 0.4)
        assert abs(c_star - 0.02128) < 5e-4

    def test_positive(self):
        phi0 = 0.3989422804
        eta = hh4_admissibility(1e-14)
        assert tail_guard_price(phi0, eta, 0.4) > 0.0


class TestBoundaryTable:
    def test_returns_dict(self):
        table = derive_boundary_table()
        assert isinstance(table, (dict, list))

    def test_three_rows(self):
        """Table should have one row per tolerance value (3 entries)."""
        table = derive_boundary_table()
        if isinstance(table, list):
            assert len(table) == 3
        else:
            assert len(next(iter(table.values()))) == 3

    def test_kappa_values_ordered(self):
        """kmax_peak values should be increasing with epsilon_Phi."""
        table = derive_boundary_table()
        if isinstance(table, list):
            kp = [row['kmax_peak'] for row in table]
        else:
            kp = list(table['kmax_peak'])
        assert kp[0] < kp[1] < kp[2]
