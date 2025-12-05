"""
CORGI LP Benchmark Suite Aggregator
===================================

This script aggregates, downloads, and standardizes a comprehensive suite of Linear Programming (LP) 
benchmarks for testing solver performance, stability, and correctness. It unifies three historically 
significant datasets into a single, clean directory structure of standardized NumPy (.npz) files.

--- DATA SOURCES & CHARACTERISTICS ---

1. Netlib LP Benchmark (Classic)
   - Count: ~98 instances
   - Type:  Classic LP problems from the 1980s.
   - Scale: Small to Medium (Hundreds to a few thousand variables).
   - Characteristics: 
     - Varied density (some dense, some sparse).
     - Historic standard for simplex/interior-point correctness.
     - Includes "pathological" cases (cycling, degeneracy) like `cycle`, `greenbea`, `pilot`.

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

4. Synthetic Generators & Unit Tests
   - CVXPY Test Suite: Ported directly from CVXPY's standard LP test cases. Covers edge cases 
     (unbounded, infeasible, redundant constraints) and basic correctness.
   - Large Scale: Procedurally generated sparse matrices (Hinder-style) for memory scaling tests.

--- COMMAND LINE OUTPUT GUIDE ---

When running this script, you will see output rows like:
  afiro           | small  | 83       nnz

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
   Note: Pure bound-constrained problems (like some CVXPY tests) may have 0 nnz 

--- OUTPUT FORMAT & USAGE ---

Output Directory: ./lp_benchmark_data/processed/{Category}/{Scale}/{InstanceName}.npz

The output is a collection of `.npz` files. Each file contains the LP in Standard Form:
    minimize c^T x
    subject to l_c <= A x <= u_c
               l_v <= x <= u_v

NPZ Keys:
    - 'c': Linear objective coefficients (1D array)
    - 'A_data', 'A_indices': Sparse matrix data (coordinate format)
    - 'A_shape': Shape of A (m, n)
    - 'l_c', 'u_c': Lower and upper bounds on constraints (rows)
    - 'l_v', 'u_v': Lower and upper bounds on variables (columns)

--- HOW TO USE FOR SOLVER TESTING ---

1. Iterate through the `processed` directory.
2. Load an instance: `data = np.load("path/to/instance.npz")`
3. Reconstruct the sparse matrix:
   `A = scipy.sparse.coo_matrix((data['A_data'], data['A_indices'].T), shape=data['A_shape'])`
4. Pass (c, A, bounds) to your custom solver.
5. Compare your result against Highs or SciPy (using `scipy.optimize.linprog`).

Dependencies: requests, numpy, scipy, highspy
"""
import os
import requests
import gzip
import shutil
import numpy as np
import highspy
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Optional, List, Dict

# --- Configuration ---
DATA_DIR = "./lp_benchmark_data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

SCALE_SMALL = 10_000
SCALE_MEDIUM = 100_000

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Disable warnings for verify=False
requests.packages.urllib3.disable_warnings()

@dataclass
class LPStandardForm:
    name: str
    source_url: str
    c: np.ndarray
    A: sp.coo_matrix
    l_c: np.ndarray
    u_c: np.ndarray
    l_v: np.ndarray
    u_v: np.ndarray
    
    @property
    def m(self): return self.A.shape[0]
    @property
    def n(self): return self.A.shape[1]
    @property
    def nnz(self): return self.A.nnz

class BenchmarkFetcher:
    """
    Downloads raw MPS files using high-reliability mirrors.
    """
    
    # --- Instance Lists ---
    
    # 1. Netlib LP (The Classic ~98)
    NETLIB_INSTANCES = [
        "25fv47", "80bau3b", "adlittle", "afiro", "agg", "agg2", "agg3",
        "bandm", "beaconfd", "blend", "bnl1", "bnl2", "boeing1", "boeing2",
        "bore3d", "brandy", "capri", "cycle", "czprob", "d2q06c", "d6cube",
        "degen2", "degen3", "dfl001", "e226", "etamacro", "fffff800",
        "finnis", "fit1d", "fit1p", "fit2d", "fit2p", "forplan", "ganges",
        "gfrd-pnc", "greenbea", "grow15", "grow22", "grow7", "israel",
        "kb2", "lotfi", "maros", "maros-r7", "modszk1", "nesm", "perold",
        "pilot", "pilot4", "pilot87", "pilotnov", "recipe", "sc105",
        "sc205", "sc50a", "sc50b", "scagr25", "scagr7", "scfxm1", "scfxm2",
        "scfxm3", "scorpion", "scrs8", "scsd1", "scsd6", "scsd8", "sctap1",
        "seba", "share1b", "share2b", "shell", "ship04l", "ship04s",
        "ship08l", "ship08s", "ship12l", "ship12s", "sierra", "stair",
        "standata", "standgub", "standmps", "stocfor1", "stocfor2",
        "stocfor3", "truss", "tuff", "vtpbase", "wood1p", "woodw"
    ]

    # 2. Mittelmann Benchmark (Simplex/Barrier/Network)
    # Selected from "Benchmark of Simplex LP solvers" and "Large Network-LP"
    MITTELMANN_INSTANCES = [
        "rail507", "rail582",   # Railway scheduling
        "pds-02", "pds-06", "pds-10", "pds-20", # Patient Distribution System
        "fome12", "fome13", "fome21", # Network flow
        "watson_1", "watson_2",
        "neos-3083819-nubu", "neos18", # NEOS instances often used in his suite
        "world"
    ]

    # 3. MIPLIB 2017 Relaxations (Representative Subset of the 383)
    # Covering diverse application areas (Air, Scheduling, Network, Packing)
    MIPLIB_INSTANCES = [
        "air03", "air04", "air05", 
        "30n20b8", "50v-10", 
        "bell3a", "bell5", 
        "blend2", "binkar10_1",
        "cap6000", "car", "cov1075",
        "dano3_3", "dano3_5", "danoint",
        "enlight8", "ex72a", "ex73a",
        "fast0507", "fiber", "fixnet6",
        "gesa2", "gesa2_o", "gmu-35-40", "gmu-35-50",
        "harp2", "iis-100-0-cov",
        "l1000-15", "leo1", "leo2",
        "mas74", "mas76", "mkc", "modglob",
        "n3seq24", "n4-3", "neos-5223519-amur",
        "nw04", 
        "p200x1188c", "pk1", "pp08a", "pp08aCUTS",
        "qap15", "qiu",
        "rgn", "rout",
        "s100", "sp97ar", "swath1", "swath3",
        "timtab1", "timtab2", 
        "vpm2"
    ]

    # --- Source Maps ---
    
    SOURCES_CONFIG = {
        "Netlib_LP": {
            "base": "https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/master/",
            "suffix": ".mps.gz",
            "files": NETLIB_INSTANCES
        },
        "Mittelmann_Benchmark": {
            "base": "https://miplib.zib.de/WebData/instances/",
            "suffix": ".mps.gz",
            "files": MITTELMANN_INSTANCES
        },
        "MIPLIB2017_Relaxations": {
            "base": "https://miplib.zib.de/WebData/instances/",
            "suffix": ".mps.gz",
            "files": MIPLIB_INSTANCES
        }
    }

    @staticmethod
    def download_file(url: str, dest_folder: str) -> Optional[str]:
        filename = url.split("/")[-1]
        dest_path = os.path.join(dest_folder, filename)
        
        # Check if file exists
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            # Quick check for HTML error pages without reading whole file
            with open(dest_path, 'rb') as f:
                header = f.read(4)
            if header.startswith(b'<'):
                print(f"    [Corrupt] {filename} (HTML/404). Retrying...")
                os.remove(dest_path)
            else:
                return dest_path

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Connection': 'keep-alive',
        }

        try:
            r = requests.get(url, headers=headers, stream=True, verify=False, timeout=20)
            if r.status_code != 200:
                print(f"    [Missing] {filename} (HTTP {r.status_code})")
                return None
            
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Post-Validation
            with open(dest_path, 'rb') as f:
                header = f.read(4)
            if header.startswith(b'<!') or header.startswith(b'<htm'):
                os.remove(dest_path)
                return None

            print(f"    [Downloaded] {filename}")
            return dest_path
        except Exception as e:
            print(f"    [Error] {filename}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return None

    def fetch_all(self):
        print(f"\n=== 1. Downloading Raw MPS Files (Comprehensive Suite) ===")
        downloaded_map = {} 
        
        for category, config in self.SOURCES_CONFIG.items():
            print(f"  > Category: {category} ({len(config['files'])} files)")
            cat_dir = os.path.join(RAW_DIR, category)
            os.makedirs(cat_dir, exist_ok=True)
            
            base = config["base"]
            suffix = config["suffix"]
            
            for name in config["files"]:
                url = f"{base}{name}{suffix}"
                path = self.download_file(url, cat_dir)
                if path:
                    downloaded_map[path] = url
                    
        return downloaded_map

class SyntheticGenerator:
    """
    Generates code-based test instances.
    Includes standard unit tests ported from CVXPY and Synthetic Sparse matrices.
    """
    
    @staticmethod
    def generate_cvxpy_lp_1() -> LPStandardForm:
        """
        Port of CVXPY lp_1():
        min -4x[0] - 5x[1]
        s.t. 2x[0] + x[1] <= 3
             x[0] + 2x[1] <= 3
             x >= 0
        """
        c = np.array([-4.0, -5.0])
        A = sp.coo_matrix(np.array([[2.0, 1.0], [1.0, 2.0]]))
        u_c = np.array([3.0, 3.0])
        l_c = np.full(2, -np.inf)
        l_v = np.array([0.0, 0.0])
        u_v = np.full(2, np.inf)
        return LPStandardForm("cvxpy_lp_1", "CVXPY Test Suite", c, A, l_c, u_c, l_v, u_v)

    @staticmethod
    def generate_cvxpy_lp_2() -> LPStandardForm:
        """
        Port of CVXPY lp_2():
        min x[0] + 0.5x[1]
        s.t. x[0] >= -100, x[0] <= -10
             x[1] == 1
        """
        c = np.array([1.0, 0.5])
        # No matrix constraints, purely variable bounds
        A = sp.coo_matrix((0, 2)) 
        l_c = np.array([])
        u_c = np.array([])
        l_v = np.array([-100.0, 1.0])
        u_v = np.array([-10.0, 1.0])
        return LPStandardForm("cvxpy_lp_2", "CVXPY Test Suite", c, A, l_c, u_c, l_v, u_v)

    @staticmethod
    def generate_cvxpy_lp_3() -> LPStandardForm:
        """
        Port of CVXPY lp_3(): UNBOUNDED problem
        min sum(x) s.t. x <= 1
        (x is size 5)
        """
        n = 5
        c = np.ones(n)
        A = sp.coo_matrix((0, n))
        l_c = np.array([])
        u_c = np.array([])
        l_v = np.full(n, -np.inf)
        u_v = np.ones(n) # x <= 1
        return LPStandardForm("cvxpy_lp_3_unbounded", "CVXPY Test Suite", c, A, l_c, u_c, l_v, u_v)

    @staticmethod
    def generate_cvxpy_lp_4() -> LPStandardForm:
        """
        Port of CVXPY lp_4(): INFEASIBLE problem
        min sum(x) s.t. x <= 0 AND x >= 1
        """
        n = 5
        c = np.ones(n)
        # We can implement this via conflicting bounds
        A = sp.coo_matrix((0, n))
        l_c = np.array([])
        u_c = np.array([])
        l_v = np.ones(n)   # x >= 1
        u_v = np.zeros(n)  # x <= 0 (Impossible)
        return LPStandardForm("cvxpy_lp_4_infeasible", "CVXPY Test Suite", c, A, l_c, u_c, l_v, u_v)

    @staticmethod
    def generate_cvxpy_lp_5() -> LPStandardForm:
        """
        Port of CVXPY lp_5(): Redundant Constraints
        10 variables, 6 equality constraints (two redundant)
        """
        np.random.seed(0)
        x0 = np.array([0, 1, 0, 2, 0, 4, 0, 5, 6, 7])
        mu0 = np.array([-2, -1, 0, 1, 2, 3.5])
        A_min = np.random.randn(4, 10)
        A_red = A_min.T @ np.random.rand(4, 2)
        A_red = A_red.T
        A_dense = np.vstack((A_min, A_red))
        b = A_dense @ x0
        c = A_dense.T @ mu0
        c[[0, 2, 4, 6]] += np.random.rand(4)
        
        # Constraints: x >= 0, A @ x == b
        m, n = A_dense.shape
        A = sp.coo_matrix(A_dense)
        l_c = b
        u_c = b
        l_v = np.zeros(n)
        u_v = np.full(n, np.inf)
        
        return LPStandardForm("cvxpy_lp_5_redundant", "CVXPY Test Suite", c, A, l_c, u_c, l_v, u_v)

    @staticmethod
    def generate_hinder_style_sparse(n=5000, density=0.001) -> LPStandardForm:
        # print(f"  [Generator]: Hinder-style Sparse LP (n={n})...")
        m = int(n / 2)
        np.random.seed(42)
        A = sp.random(m, n, density=density, format='coo')
        x0 = np.random.uniform(0, 1, n)
        b = A @ x0
        c = np.random.uniform(-1, 1, n)
        return LPStandardForm(
            f"hinder_synthetic_n{n}", "Synthetic_LargeScale",
            c, A, b, b, np.zeros(n), np.full(n, np.inf)
        )

class LPProcessor:
    CATEGORY_DESCRIPTIONS = {
        "MIPLIB2017_Relaxations": "LP relaxations from MIPLIB 2017 (Real-world, diverse, often structured).",
        "Mittelmann_Benchmark": "Hans Mittelmann's Hard Benchmarks (Stress tests, large sparse networks).",
        "Netlib_LP": "Classic Netlib LP Suite (Historic standard, small to medium scale).",
        "CVXPY_Test_LPs": "Standard Unit Tests ported from CVXPY (Edge cases: unbounded, infeasible, redundant).",
        "Synthetic_LargeScale": "Procedurally generated sparse matrices (Hinder-style).",
        "General": "General LP instances."
    }

    def __init__(self):
        self.seen_categories = set()

    @staticmethod
    def determine_scale_label(n_vars: int) -> str:
        if n_vars < SCALE_SMALL: return "small"
        if n_vars < SCALE_MEDIUM: return "medium"
        return "large"

    @staticmethod
    def load_mps(mps_path: str, source_url: str) -> Optional[LPStandardForm]:
        name = os.path.basename(mps_path).split(".")[0]
        
        use_path = mps_path
        if mps_path.endswith(".gz"):
            temp_path = mps_path[:-3]
            # Unzip only if needed
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                try:
                    with gzip.open(mps_path, 'rb') as f_in:
                        with open(temp_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                except Exception as e:
                    print(f"    [Error] Corrupt GZIP: {name}")
                    return None
            use_path = temp_path

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        
        if not os.path.exists(use_path) or os.path.getsize(use_path) == 0:
            return None

        status = h.readModel(use_path)
        if status != highspy.HighsStatus.kOk:
            print(f"    [Error] Parse failed: {name}")
            return None
        
        lp = h.getLp()
        rows = lp.a_matrix_.num_row_
        cols = lp.a_matrix_.num_col_
        
        if rows == 0 and cols == 0:
             return None

        A_coo = sp.csc_matrix(
            (np.array(lp.a_matrix_.value_), 
             np.array(lp.a_matrix_.index_), 
             np.array(lp.a_matrix_.start_)),
            shape=(rows, cols)
        ).tocoo()

        inf = 1e20
        l_v = np.array(lp.col_lower_); u_v = np.array(lp.col_upper_)
        l_c = np.array(lp.row_lower_); u_c = np.array(lp.row_upper_)
        
        l_v[l_v <= -inf] = -np.inf; u_v[u_v >= inf] = np.inf
        l_c[l_c <= -inf] = -np.inf; u_c[u_c >= inf] = np.inf
        
        return LPStandardForm(name, source_url, np.array(lp.col_cost_), A_coo, l_c, u_c, l_v, u_v)

    def process_and_save(self, instance: LPStandardForm, category_hint: str = "General"):
        if category_hint not in self.seen_categories:
            desc = self.CATEGORY_DESCRIPTIONS.get(category_hint, "General LP instances.")
            print(f"\n>>> [{category_hint}]: {desc}")
            self.seen_categories.add(category_hint)

        scale = self.determine_scale_label(instance.n)
        
        save_dir = os.path.join(PROCESSED_DIR, category_hint, scale)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{instance.name}.npz")
        
        # Only print if we are actually doing work (e.g. not re-saving identical)
        # For this script, we always overwrite, so we print a compact status
        print(f"  {instance.name:<15} | {scale:<6} | {instance.nnz:<8} nnz")
        
        np.savez_compressed(
            save_path,
            name=instance.name,
            source=instance.source_url,
            c=instance.c,
            A_data=instance.A.data,
            A_indices=np.vstack((instance.A.row, instance.A.col)).T,
            A_shape=instance.A.shape,
            l_c=instance.l_c, u_c=instance.u_c,
            l_v=instance.l_v, u_v=instance.u_v
        )

def main():
    fetcher = BenchmarkFetcher()
    downloaded_files = fetcher.fetch_all()
    processor = LPProcessor()
    
    print("\n=== 2. Processing Files ===")
    for path, url in downloaded_files.items():
        category = os.path.basename(os.path.dirname(path))
        try:
            instance = processor.load_mps(path, url)
            if instance:
                processor.process_and_save(instance, category_hint=category)
        except Exception as e:
            pass # Skip broken files silently in large batch
            
    print("\n=== 3. Generators ===")
    try:
        # Run the CVXPY Ported Tests
        cvxpy_tests = [
            SyntheticGenerator.generate_cvxpy_lp_1(),
            SyntheticGenerator.generate_cvxpy_lp_2(),
            SyntheticGenerator.generate_cvxpy_lp_3(),
            SyntheticGenerator.generate_cvxpy_lp_4(),
            SyntheticGenerator.generate_cvxpy_lp_5(),
        ]
        
        for test_case in cvxpy_tests:
             processor.process_and_save(test_case, "CVXPY_Test_LPs")

        # Run Large Scale Synthetic
        processor.process_and_save(SyntheticGenerator.generate_hinder_style_sparse(15000, 0.0005), "Synthetic_LargeScale")
        
    except Exception as e:
        print(f"  [CRITICAL] Generator failed: {e}")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()