# Regime-Split Black-Scholes Implied Total-Volatility Solver

Reproduction of the analytical regime-split implied total-volatility (IV) solver
described in the paper.  The solver inverts normalised Black-Scholes call prices
to recover **total volatility `v = σ√T`** using analytically derived seed
functions and fourth-order Householder (HH-4) iterative refinement.

## Algorithm Summary

```
Branch A  (κ < 0.001)        → Taylor-4 ATM seed
Branch B  (c_seed < c★, κ > κ★) → Mills-ratio deep-OTM seed  [tail guard]
Branch C  (κ ≤ 1.347)        → polynomial seeds vP7 / avg(vP3,vP7) / vP3
Branch D  (κ > 1.347)        → Mills-ratio deep-OTM seed
              ↓
         HH-4 refinement  (repeat-until, ≥1 update, tol = 1e-14)
```

Regime cutoffs:  κ₁=0.001, κ₂=0.81, κ₃=1.155, κ₄=1.347, c★=0.02128, κ★=0.5.

## Repository Structure

```
src/                 Core modules
  black_scholes.py   Normalised BS call price and synthetic grid generators
  inverse_normal.py  Acklam's and Soranzo-Epure inverse-CDF approximations
  seeds.py           ATM Taylor-4, vP1/vP3/vP7 polynomial, deep-OTM seeds
  refinement.py      HH-4 (Householder order-4) iterative refinement
  solver.py          Regime-split solver: solve_iv, solve_grid, aggregate_metrics
  theory_boundaries.py  Analytical derivation of regime cutoffs (Exp 2)

exp/                 Experiment scripts
  exp1_core.py       Core reproduction on 328-pt and 1970-pt synthetic grids
  exp2_boundaries.py Analytical boundary derivation and validation
  exp3_seed_diagnostics.py  Seed-quality diagnostics per regime
  exp4_robustness.py Deep-OTM seed: Acklam vs Soranzo-Epure comparison
  exp5_ablation.py   Near-OTM tail-guard architecture ablation

tests/               pytest test suite (100 tests)
data/                Synthetic data directory (generated on-the-fly)
results/             Plots and RESULTS.md summary
```

## Installation

```bash
pip install numpy scipy matplotlib pandas
```

Optional for timing benchmarks:
```bash
pip install numba
```

## Running Experiments

```bash
# All five experiments
python exp/exp1_core.py
python exp/exp2_boundaries.py
python exp/exp3_seed_diagnostics.py
python exp/exp4_robustness.py
python exp/exp5_ablation.py

# Full test suite
pytest tests/ -v
```

## Key Results

| Metric | Grid 1 (328) | Grid 2 (1970) |
|--------|-------------|--------------|
| Max \|v̂ − v_true\| | 9.66 × 10⁻¹⁴ | 2.74 × 10⁻¹³ |
| Mean HH-4 updates | 2.040 | 2.046 |
| Fraction ≤ 2 updates | 93.9% | 92.9% |

Regime boundaries derived from theory:
κ₂ ≈ 0.8079, κ₃ ≈ 1.1552, κ₄ ≈ 1.3476, η★ ≈ 13.3%, c★ ≈ 0.02128.

See `results/RESULTS.md` for full experiment findings.

## No Market Data Required

All experiments use purely synthetic benchmark data generated from known
total volatilities and Black-Scholes call deltas.
