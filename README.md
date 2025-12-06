# MP-Bench

Mathematical Programming Benchmark Suite. This repo is designed to fetch, standardize, and download problems. It aggregates historically significant and modern datasets into a standard form (.npz), specifically tailored for testing both GPU-accelerated solvers and established CPU baselines.

**Functionality:**

Automated Fetching: Downloads raw MPS files from reliable mirrors (ZIB, COIN-OR).

Standardization: Parses .mps files using highspy and converts them to a consistent Standard Form. 

Suggested setup: Requires Python 3.8+ in a virtual environment (conda or venv). 

```pip install -r requirements.txt```

Note: highspy is the Python binding for the HiGHS solver, used here for robust MPS parsing.

## Data Sources & Characteristics

Netlib LP: The classic 1980s standard for correctness.

Mittelmann: Hard, large-scale stress tests for numerical stability.

MIPLIB 2017: Real-world LP relaxations of mixed-integer problems.

CVXPY Unit Tests: Ported edge cases (unbounded, infeasible, redundant) for API verification.

1. Netlib LP Benchmark (Classic)
   - Count: ~98 instances
   - Type:  Classic LP problems from the 1980s.
   - Scale: Small to Medium (Hundreds to a few thousand variables).
   - Characteristics: 
     - Varied density (some dense, some sparse).
     - Historic standard for simplex/interior-point correctness.
     - Includes pathological cases (cycling, degeneracy) like `cycle`, `greenbea`, `pilot`.

2. Hans Mittelmann's Benchmark (Stress Tests)
   - Count: ~14 selected instances
   - Type:  Hard instances from the "Benchmark of Simplex LP Solvers" and "Large Network-LP".
   - Scale: Large to Very Large.
   - Characteristics:
     - Extremely sparse (e.g., `rail507`, `pds` series).
     - Designed to stress-test solver stability, pivot selection, and numerical precision.
     - Includes network flow problems (`fome`, `pds`) and hard scheduling (`rail`).

3. MIPLIB 2017 Relaxations (Modern Real-World)
   - Count: ~54 representative instances (Subset of the full 383)
   - Type:  LP relaxations of Mixed-Integer Programming (MIP) problems.
   - Scale: Medium to Large.
   - Characteristics:
     - Highly structured (arising from combinatorial problems like Airline Crew Scheduling).
     - Real-world applications: Logistics, telecommunications, bin packing.
     - Often exhibit specific block structures or massive degeneracy.

4. CVXPY Synthetic Generators & Unit Tests
   - CVXPY Test Suite: Ported directly from CVXPY's standard LP test cases. Covers edge cases 
     (unbounded, infeasible, redundant constraints) and basic correctness.
     
5. Large Scale: Procedurally generated sparse matrices (proposed by Oliver Hinder) for memory scaling tests.
   - TODO: Documentation.

## Output Format

When running this script, you will see output rows like:
```  afiro           | small  | 83       nnz```


The columns correspond to:
1. Instance Name (e.g., 'afiro'): 
   The unique identifier for the problem instance.
2. Scale (e.g., 'small'): 
   A size category based on the number of variables (n).
   - small:  n < 10,000
   - medium: 10,000 <= n < 100,000
   - large:  n >= 100,000
3. Non-Zeros (e.g., '83 nnz'): 
   The total count of non-zero entries in the constraint matrix A. 

Output Directory: ```./lp_benchmark_data/processed/{Category}/{Scale}/{InstanceName}.npz```

The output directory is organized by Collection and Scale:

```
lp_benchmark_data/
├── raw/                  # Original .mps.gz downloads
└── processed/            # Standardized .npz files
    ├── Netlib_LP/
    │   └── small/        # e.g., afiro.npz, pilot87.npz
    ├── Mittelmann_Benchmark/
    │   └── medium/       # e.g., rail507.npz
    └── MIPLIB2017_Relaxations/
        └── medium/       # e.g., air03.npz
```


The output is a collection of `.npz` files. For LPs, each file contains the form: 

$$\begin{aligned}
\min_x \quad & c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c \\
                  & \ell_v \le x \le u_v
\end{aligned}$$

NPZ Keys:
- 'c': Linear objective coefficients (1D array)
- 'A_data', 'A_indices': Sparse matrix data (coordinate format)
- 'A_shape': Shape of A (m, n)
- 'l_c', 'u_c': Lower and upper bounds on constraints (rows)
- 'l_v', 'u_v': Lower and upper bounds on variables (columns)


## Usage

1. Fetch and Process Data

Run the aggregator to download benchmarks and saves them to .npz:

```python aggregate_test_suite.py```


This will create the lp_benchmark_data directory structure.

2. Runner

The main.py script runs the JAX solver against the downloaded benchmarks.

TEST_TYPE = "./lp_benchmark_data/processed/Netlib_LP/small"

Note: The random test type just generates a random matrix and tests it once. 

TEST_TYPE = "random"

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

Large Scale LP Test Problems: https://github.com/ohinder/large-scale-LP-test-problems

Description: PDLP: A Practical First-Order Method for Large-Scale Linear Programming test cases hosted by Oliver Hinder