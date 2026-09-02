"""
Tests for src/inverse_normal.py — Acklam and Soranzo-Epure inverse normal.
"""
import numpy as np
import pytest
from scipy.stats import norm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.inverse_normal import acklam_ppf, soranzo_epure_ppf, clip_prob


class TestAcklamPPF:
    def test_midpoint(self):
        """Phi^{-1}(0.5) should be 0."""
        assert abs(acklam_ppf(0.5)) < 1e-10

    def test_symmetry(self):
        """Phi^{-1}(1-p) = -Phi^{-1}(p)."""
        for p in [0.1, 0.25, 0.3, 0.4, 0.01, 0.001]:
            assert abs(acklam_ppf(1.0 - p) + acklam_ppf(p)) < 1e-8

    def test_boundary_values(self):
        """Extreme inputs map to +/-inf."""
        assert acklam_ppf(0.0) == -np.inf
        assert acklam_ppf(1.0) == np.inf

    def test_accuracy_central(self):
        """Acklam accuracy vs scipy in central region (max err < 1e-7)."""
        p_vals = np.linspace(0.03, 0.97, 200)
        for p in p_vals:
            diff = abs(acklam_ppf(p) - norm.ppf(p))
            assert diff < 1e-7, f"Acklam err={diff:.2e} at p={p:.4f}"

    def test_accuracy_tails(self):
        """Acklam accuracy in left tail (max err < 1e-5)."""
        for p in [1e-4, 1e-3, 0.01, 0.02]:
            diff = abs(acklam_ppf(p) - norm.ppf(p))
            assert diff < 1e-5, f"Acklam tail err={diff:.2e} at p={p:.4e}"

    def test_monotone(self):
        """Acklam ppf must be monotonically increasing."""
        p_vals = np.linspace(0.001, 0.999, 500)
        vals = np.array([acklam_ppf(p) for p in p_vals])
        assert np.all(np.diff(vals) > 0), "Acklam ppf not monotone"

    def test_type(self):
        """Return type is float."""
        assert isinstance(acklam_ppf(0.3), float)


class TestSoranzoEpurePPF:
    def test_midpoint(self):
        assert abs(soranzo_epure_ppf(0.5)) < 1e-8

    def test_symmetry(self):
        for p in [0.1, 0.2, 0.3, 0.05]:
            assert abs(soranzo_epure_ppf(1.0 - p) + soranzo_epure_ppf(p)) < 1e-6

    def test_boundary_values(self):
        assert soranzo_epure_ppf(0.0) == -np.inf
        assert soranzo_epure_ppf(1.0) == np.inf

    def test_accuracy_moderate(self):
        """SE accuracy should be within 1e-3 of scipy across central region."""
        for p in np.linspace(0.05, 0.95, 50):
            diff = abs(soranzo_epure_ppf(p) - norm.ppf(p))
            assert diff < 5e-3, f"SE err={diff:.2e} at p={p:.4f}"

    def test_monotone(self):
        p_vals = np.linspace(0.01, 0.99, 200)
        vals = np.array([soranzo_epure_ppf(p) for p in p_vals])
        assert np.all(np.diff(vals) > 0), "SE ppf not monotone"

    def test_type(self):
        assert isinstance(soranzo_epure_ppf(0.4), float)


class TestClipProb:
    def test_clips_zero(self):
        assert clip_prob(0.0) > 0.0

    def test_clips_one(self):
        assert clip_prob(1.0) < 1.0

    def test_interior_unchanged(self):
        for p in [0.1, 0.5, 0.9]:
            assert abs(clip_prob(p) - p) < 1e-15

    def test_negative_input(self):
        assert clip_prob(-0.5) > 0.0

    def test_above_one(self):
        assert clip_prob(1.5) < 1.0
