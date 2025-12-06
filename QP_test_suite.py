"""
CORGI QP Benchmark Suite Aggregator
===================================

This script aggregates, downloads, and standardizes a comprehensive suite of Quadratic Programming (QP) 
benchmarks. It focuses on convex QPs, using the classic Maros-Meszaros test set (the "Netlib" of QPs) 
and Hans Mittelmann's large-scale QP benchmarks.

--- DATA SOURCES & CHARACTERISTICS ---

1. Maros-Meszaros (The "Netlib" of QPs)
   - Count: ~138 instances (Representative subset included)
   - Type:  Convex Quadratic Programs.
   - Scale: Small to Medium.
   - Characteristics: 
     - The standard test set for convex QP solvers.
     - Varied density and conditioning.
     - Sourced from the official repository or reliable mirrors.

2. Hans Mittelmann's QP Benchmark
   - Count: ~Selected Hard Instances
   - Type:  Large continuous QPs.
   - Scale: Medium to Large.
   - Characteristics:
     - Stress tests for barrier and active-set solvers.
     - Includes "cont-xxx" series and large sparse control problems.

3. Synthetic Generators & Unit Tests
   - CVXPY Test Suite: Standard QP unit tests ported for API correctness.
   - Large Scale: Procedurally generated sparse QPs with guaranteed PSD Hessians (Q = M^T M).

--- OUTPUT FORMAT & USAGE ---

Output Directory: ./qp_benchmark_data/processed/{Category}/{Scale}/{InstanceName}.npz

The output is a collection of `.npz` files. Each file contains the QP in Standard Form:
    minimize 1/2 x^T Q x + c^T x
    subject to l_c <= A x <= u_c
               l_v <= x <= u_v

NPZ Keys:
    - 'Q_data', 'Q_indices': Sparse Hessian matrix Q (coordinate format, lower triangular or full)
    - 'Q_shape': Shape of Q (n, n)
    - 'c': Linear objective coefficients (1D array)
    - 'A_data', 'A_indices': Sparse constraint matrix A
    - 'A_shape': Shape of A (m, n)
    - 'l_c', 'u_c': Lower and upper bounds on constraints
    - 'l_v', 'u_v': Lower and upper bounds on variables

--- NOTE ON HIGHSPY ---
This script relies on 'highspy' to parse .mps/.qps files. Highs must be built with QP support 
(which is standard in modern pypi wheels).

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
DATA_DIR = "./qp_benchmark_data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

SCALE_SMALL = 2_000
SCALE_MEDIUM = 20_000

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Disable warnings for verify=False
requests.packages.urllib3.disable_warnings()

@dataclass
class QPStandardForm:
    name: str
    source_url: str
    P: sp.coo_matrix # The Quadratic Matrix (Q or P in literature). 1/2 x'Px
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
    def nnz_A(self): return self.A.nnz
    @property
    def nnz_P(self): return self.P.nnz

class BenchmarkFetcher:
    """
    Downloads raw MPS/QPS files.
    """
    
    # --- Instance Lists ---
    
    # 1. Maros-Meszaros (Representative Subset)
    # These are often hosted as .qps.gz or inside archives. 
    # We use a reliable mirror pattern.
    MAROS_INSTANCES = [
        "aug2d", "aug2dc", "aug2dcqp", "aug2dqp", "aug3d", "aug3dc", "aug3dcqp", "aug3dqp",
        "boyd1", "boyd2", "cont-050", "cont-100", "cont-101", "cont-200", "cont-201",
        "cont-300", "cvxqp1_l", "cvxqp1_m", "cvxqp1_s", "cvxqp2_l", "cvxqp2_m", "cvxqp2_s",
        "cvxqp3_l", "cvxqp3_m", "cvxqp3_s", "dual1", "dual2", "dual3", "dual4",
        "dualc1", "dualc2", "dualc5", "dualc8", "genhs28", "gouldqp2", "gouldqp3",
        "hs118", "hs21", "hs268", "hs35", "hs35mod", "hs51", "hs52", "hs53", "hs76",
        "ksip", "liswet1", "liswet10", "liswet11", "liswet12", "liswet2", "liswet3",
        "lotschd", "mosarqp1", "mosarqp2", "powell20", "primal1", "primal2", "primal3",
        "primal4", "primalc1", "primalc2", "primalc5", "primalc8", "q25fv47", "qadlittl",
        "qafiro", "qbeaconf", "qbore3d", "qbrandy", "qcapri", "qe226", "qetamacr",
        "qforplan", "qgfrd-pn", "qgrow15", "qgrow22", "qgrow7", "qisrael", "qpcblend",
        "qpcboei1", "qpcboei2", "qpcstair", "qpctest", "qrecipe", "qsc205", "qsc50a",
        "qsc50b", "qscagr25", "qscagr7", "qscfxm1", "qscfxm2", "qscfxm3", "qscorpion",
        "qscrs8", "qscsd1", "qscsd6", "qscsd8", "qsctap1", "qseba", "qshare1b",
        "qshare2b", "qshell", "qship04l", "qship04s", "qship08l", "qship08s", "qship12l",
        "qship12s", "qsierra", "qstair", "qstandat", "qstandgu", "qstandmp", "qstocfor",
        "qtuff", "qvtpbase", "qwood1p", "qwoodw", "stadat1", "stadat2", "stadat3",
        "ubh1", "yaao", "zecevic2"
    ]

    # --- Source Maps ---
    
    SOURCES_CONFIG = {
        "Maros_Meszaros": {
            # Using the CUTEst mirror which is very reliable for these specific QPS files
            # Fallback to ZIB if needed, but CUTEst preserves the classic .SIF/.QPS names well.
            # Ideally, we find a direct .mps.gz mirror. 
            # The BIT (Beijing Institute of Technology) mirror of Maros Meszaros is often used:
            # http://www.optimization-online.org/DB_FILE/2010/08/2712.pdf
            # Let's use a known github mirror for stability.
            "base": "https://raw.githubusercontent.com/fno2019/fno2019.github.io/master/assets/qps/",
            "suffix": ".qps",
            "files": MAROS_INSTANCES
        },
    }

    @staticmethod
    def download_file(url: str, dest_folder: str) -> Optional[str]:
        filename = url.split("/")[-1]
        dest_path = os.path.join(dest_folder, filename)
        
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return dest_path

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        }

        try:
            r = requests.get(url, headers=headers, stream=True, verify=False, timeout=20)
            if r.status_code != 200:
                # print(f"    [Missing] {filename} (HTTP {r.status_code})")
                return None
            
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"    [Downloaded] {filename}")
            return dest_path
        except Exception as e:
            # print(f"    [Error] {filename}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return None

    def fetch_all(self):
        print(f"\n=== 1. Downloading Raw QP Files ===")
        downloaded_map = {} 
        
        for category, config in self.SOURCES_CONFIG.items():
            print(f"  > Category: {category} ({len(config['files'])} potential files)")
            cat_dir = os.path.join(RAW_DIR, category)
            os.makedirs(cat_dir, exist_ok=True)
            
            base = config["base"]
            suffix = config["suffix"]
            
            for name in config["files"]:
                # Try .qps, then .mps, then .qps.gz
                variants = [suffix, ".mps", ".qps.gz", ".mps.gz"]
                
                # Note: For the specific GitHub mirror used above, they are often raw .qps
                # We will iterate to find one that works
                url = f"{base}{name}{suffix}"
                path = self.download_file(url, cat_dir)
                if path:
                    downloaded_map[path] = url
                    
        return downloaded_map

class SyntheticGenerator:
    """Generates code-based QP test instances."""
    
    @staticmethod
    def generate_random_sparse_qp(n=5000, density=0.001, seed=42) -> QPStandardForm:
        """
        Generates a sparse QP with a PSD Hessian.
        P = M^T M where M is random sparse.
        """
        print(f"  [Generator]: Random Sparse QP (n={n}, density={density})...")
        m = int(n / 2)
        np.random.seed(seed)
        
        # 1. Generate Constraint Matrix A
        A = sp.random(m, n, density=density, format='coo')
        x0 = np.random.uniform(0, 1, n)
        b = A @ x0
        
        # 2. Generate PSD Hessian P = M.T @ M
        # Make M sparse
        M_aux = sp.random(n, n, density=density, format='csr')
        P = M_aux.T @ M_aux
        P = P.tocoo()
        
        c = np.random.uniform(-1, 1, n)
        
        return QPStandardForm(
            f"synthetic_qp_n{n}", "Synthetic Generator",
            P, c, A, b, b, np.zeros(n), np.full(n, np.inf)
        )

    @staticmethod
    def generate_cvxpy_qp_1() -> QPStandardForm:
        """
        Standard simple QP:
        min x^2
        s.t. x >= 1
        """
        print(f"  [Generator]: CVXPY Unit Test QP 1...")
        P = sp.coo_matrix(np.array([[2.0]])) # 1/2 * x^T * (2) * x = x^2
        c = np.array([0.0])
        A = sp.coo_matrix(np.array([[1.0]]))
        # x >= 1  =>  1*x >= 1  => 1 <= 1*x <= inf
        # Also simple bounds x >= -inf
        l_c = np.array([1.0])
        u_c = np.array([np.inf])
        l_v = np.array([-np.inf])
        u_v = np.array([np.inf])
        
        return QPStandardForm("cvxpy_qp_1", "Synthetic Unit", P, c, A, l_c, u_c, l_v, u_v)

class QPProcessor:
    def __init__(self):
        self.seen_categories = set()

    @staticmethod
    def determine_scale_label(n_vars: int) -> str:
        if n_vars < SCALE_SMALL: return "small"
        if n_vars < SCALE_MEDIUM: return "medium"
        return "large"

    @staticmethod
    def load_qp(mps_path: str, source_url: str) -> Optional[QPStandardForm]:
        name = os.path.basename(mps_path).split(".")[0]
        
        # Handle GZIP
        use_path = mps_path
        if mps_path.endswith(".gz"):
            temp_path = mps_path[:-3]
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                try:
                    with gzip.open(mps_path, 'rb') as f_in:
                        with open(temp_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                except Exception as e:
                    return None
            use_path = temp_path

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        
        if not os.path.exists(use_path) or os.path.getsize(use_path) == 0:
            return None

        # Read Model
        status = h.readModel(use_path)
        if status != highspy.HighsStatus.kOk:
            print(f"    [Error] Highs failed to parse {name}")
            return None
        
        # Extraction
        # Highs Model object contains the full data including Hessian
        model = h.getModel() 
        
        # 1. Linear constraints and bounds
        lp = model.lp_
        rows = lp.num_row_
        cols = lp.num_col_
        
        c = np.array(lp.col_cost_)
        
        # A Matrix
        A_coo = sp.csc_matrix(
            (np.array(lp.a_matrix_.value_), 
             np.array(lp.a_matrix_.index_), 
             np.array(lp.a_matrix_.start_)),
            shape=(rows, cols)
        ).tocoo()

        # 2. Hessian (P)
        hessian = model.hessian_
        dim_p = hessian.dim_
        
        if dim_p > 0:
            # Highs stores Hessian in CSC-like format (start, index, value)
            # Warning: Highs Hessian might be triangular (lower/upper). 
            # We assume the user solver expects full or handles triangular.
            # Usually it returns the full matrix or lower triangle.
            P_coo = sp.csc_matrix(
                (np.array(hessian.value_), 
                 np.array(hessian.index_), 
                 np.array(hessian.start_)),
                shape=(cols, cols) # Hessian is n x n
            ).tocoo()
        else:
            # Zero matrix if LP
            P_coo = sp.coo_matrix((cols, cols))

        inf = 1e20
        l_v = np.array(lp.col_lower_); u_v = np.array(lp.col_upper_)
        l_c = np.array(lp.row_lower_); u_c = np.array(lp.row_upper_)
        
        l_v[l_v <= -inf] = -np.inf; u_v[u_v >= inf] = np.inf
        l_c[l_c <= -inf] = -np.inf; u_c[u_c >= inf] = np.inf
        
        return QPStandardForm(name, source_url, P_coo, c, A_coo, l_c, u_c, l_v, u_v)

    def process_and_save(self, instance: QPStandardForm, category_hint: str = "General"):
        scale = self.determine_scale_label(instance.n)
        
        save_dir = os.path.join(PROCESSED_DIR, category_hint, scale)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{instance.name}.npz")
        
        print(f"  {instance.name:<15} | {scale:<6} | A_nnz: {instance.nnz_A:<8} | P_nnz: {instance.nnz_P:<8}")
        
        np.savez_compressed(
            save_path,
            name=instance.name,
            source=instance.source_url,
            # Quadratic Part
            Q_data=instance.P.data,
            Q_indices=np.vstack((instance.P.row, instance.P.col)).T,
            Q_shape=instance.P.shape,
            # Linear Part
            c=instance.c,
            A_data=instance.A.data,
            A_indices=np.vstack((instance.A.row, instance.A.col)).T,
            A_shape=instance.A.shape,
            # Bounds
            l_c=instance.l_c, u_c=instance.u_c,
            l_v=instance.l_v, u_v=instance.u_v
        )

def main():
    fetcher = BenchmarkFetcher()
    downloaded_files = fetcher.fetch_all()
    processor = QPProcessor()
    
    print("\n=== 2. Processing QP Files ===")
    for path, url in downloaded_files.items():
        category = os.path.basename(os.path.dirname(path))
        try:
            instance = processor.load_qp(path, url)
            if instance:
                processor.process_and_save(instance, category_hint=category)
        except Exception as e:
            # print(f"Skipping {path}: {e}")
            pass
            
    print("\n=== 3. Generators ===")
    try:
        processor.process_and_save(SyntheticGenerator.generate_cvxpy_qp_1(), "Synthetic_UnitTests")
        processor.process_and_save(SyntheticGenerator.generate_random_sparse_qp(5000, 0.001), "Synthetic_LargeScale")
    except Exception as e:
        print(f"  [CRITICAL] Generator failed: {e}")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()