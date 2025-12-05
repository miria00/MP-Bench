# MP-Bench

Mathematical Programming Benchmark Suite. This repo is designed to fetch, standardize, and evaluate Linear Programming (LP) solvers. Expansion will include further curated optimization resources & general instances. 

It aggregates historically significant datasets—including Netlib, Mittelmann, and MIPLIB 2017—into a standard NumPy (.npz) format, specifically tailored for testing both GPU-accelerated solvers and established CPU baselines.

**Functionality**

Automated Fetching: Downloads raw MPS files from reliable mirrors (ZIB, COIN-OR).

Standardization: Parses .mps files using highspy and converts them to a consistent Standard Form LP ($min \ c^T x$ s.t. $l \le Ax \le u$).

**Comprehensive Suites**

Netlib LP: The classic 1980s standard for correctness.

Mittelmann: Hard, large-scale stress tests for numerical stability.

MIPLIB 2017: Real-world LP relaxations of mixed-integer problems.

CVXPY Unit Tests: Ported edge cases (unbounded, infeasible, redundant) for API verification.

Comparison Harness currently built-in main.py to run CORGIsolver against SciPy (HiGHS) and calculate relative error.


## Installation and Usage

Requires Python 3.8+. Recommended to use a virtual environment (e.g., conda or venv).

pip install numpy scipy requests highspy jax jaxlib


(Note: highspy is the Python binding for the HiGHS solver, used here for robust MPS parsing).

Usage

1. Fetch and Process Data

Run the aggregator to download benchmarks and convert them to .npz:

python aggregate_test_suite.py


This will create the lp_benchmark_data directory structure.

2. Run the Test Harness

The main.py script runs your JAX solver against the downloaded benchmarks.

To run a specific benchmark directory:
Edit main.py configuration:

TEST_TYPE = "./lp_benchmark_data/processed/Netlib_LP/small"


Then run:

python main.py


To run a procedural random test:
Edit main.py configuration:

TEST_TYPE = "random"


📊 Benchmark Collections

Collection

Description

Characteristics

Count

Netlib LP

The historic standard for LP solvers.

Varied density, includes pathological cases (cycling, degeneracy).

~98

Mittelmann

Hans Mittelmann's "Hard" benchmarks.

Large, extremely sparse network and scheduling problems.

~14

MIPLIB 2017

LP relaxations of MIPs.

Highly structured real-world problems (airline, logistics).

~54

Synthetic

CVXPY ports & Hinder-style matrices.

Edge cases (unbounded/infeasible) and procedural scaling tests.

~6

## Directory Structure

After running the aggregator, the data is organized by Collection and Scale:

lp_benchmark_data/
├── raw/                  # Original .mps.gz downloads
└── processed/            # Standardized .npz files
    ├── Netlib_LP/
    │   └── small/        # e.g., afiro.npz, pilot87.npz
    ├── Mittelmann_Benchmark/
    │   └── medium/       # e.g., rail507.npz
    └── MIPLIB2017_Relaxations/
        └── medium/       # e.g., air03.npz



## References & Sources

This suite aggregates data from the following open-source repositories and archives:

Netlib LP Benchmark: https://www.netlib.org/lp/

GitHub Mirror (COIN-OR): https://github.com/coin-or-tools/Data-Netlib

Description: Gay, D. M. (1985). Electronic mail distribution of linear programming test problems.

Hans Mittelmann's Benchmarks: http://plato.asu.edu/bench.html

Mirrors: Hosted via ZIB (Zuse Institute Berlin)

Description: Benchmark of Simplex LP solvers & Large Network-LP Benchmark.

MIPLIB 2017: https://miplib.zib.de/

Paper: Gleixner, A., et al. (2021). MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library.

CVXPY: https://github.com/cvxpy/cvxpy

Description: Diamond, S., & Boyd, S. (2016). CVXPY: A Python-embedded modeling language for convex optimization.