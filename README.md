# MP-Bench

Mathematical Programming Benchmark Suite. This repo is designed to fetch, standardize, and download problems. It aggregates historically significant and modern datasets into a standard form (.npz), specifically tailored for testing both GPU-accelerated solvers and established CPU baselines.

**Functionality:**

Automated Fetching: Downloads raw MPS files from reliable mirrors (ZIB, COIN-OR).

Standardization: Parses .mps files using highspy and converts them to a consistent Standard Form. 
Such as LP form: ($min \ c^T x$ s.t. $l \le Ax \le u$).

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
     
5. Large Scale: Procedurally generated sparse matrices (Hinder-style) for memory scaling tests.
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


The output is a collection of `.npz` files. For LPs, each file contains the LP in Standard Form:
$$
\begin{aligned}
\text{minimize} \quad & c^T x \\
\text{subject to} \quad & l_c \leq Ax \leq u_c \\
& l_v \leq x \leq u_v
\end{aligned}
$$

NPZ Keys:
- 'c': Linear objective coefficients (1D array)
- 'A_data', 'A_indices': Sparse matrix data (coordinate format)
- 'A_shape': Shape of A (m, n)
- 'l_c', 'u_c': Lower and upper bounds on constraints (rows)
- 'l_v', 'u_v': Lower and upper bounds on variables (columns)


## How to Use for Solver Testing

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




1. Iterate through the `processed` directory.
2. Load an instance: `data = np.load("path/to/instance.npz")`
3. Reconstruct the sparse matrix:
   `A = scipy.sparse.coo_matrix((data['A_data'], data['A_indices'].T), shape=data['A_shape'])`
4. Pass (c, A, bounds) to your custom solver.
5. Compare your result against Highs or SciPy (using `scipy.optimize.linprog`).



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

https://github.com/ohinder/large-scale-LP-test-problems