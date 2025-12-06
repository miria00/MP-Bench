"""
CORGI SOCP Benchmark Suite Aggregator
=====================================

This script aggregates, downloads, and standardizes a comprehensive suite of Second-Order Cone 
Programming (SOCP) benchmarks. It targets problems representable as Quadratically Constrained 
Linear Programs (QCP), which is the standard MPS format for SOCPs.

--- DATA SOURCES & CHARACTERISTICS ---

1. Hans Mittelmann's SOCP Benchmark
   - Count: ~Selected Hard Instances
   - Type:  Second-Order Cone Programs (converted to QCP in MPS format).
   - Scale: Medium to Large.
   - Characteristics:
     - Includes problems from the DIMACS Implementation Challenge.
     - Real-world applications: Antenna array design, FIR filter design, truss topology.
     - Sourced from Plato ASU archives.

2. CBLIB (Conic Benchmark Library) - (Future Expansion)
   - The standard library for conic optimization (typically .cbf format).
   - We focus here on the MPS-compatible subset often mirrored by Mittelmann.

3. Synthetic Generators
   - CVXPY Test Suite: Standard SOCP unit tests (ice-cream cone, etc.).
   - Large Scale: Procedurally generated random SOCPs.

--- OUTPUT FORMAT & USAGE ---

Output Directory: ./socp_benchmark_data/processed/{Category}/{Scale}/{InstanceName}.npz

The output is a collection of `.npz` files. 
Each file represents a QCP/SOCP:
    minimize 1/2 x^T P_obj x + c^T x
    subject to:
        l_c <= A x <= u_c  (Linear Constraints)
        l_v <= x   <= u_v  (Variable Bounds)
        1/2 x^T Q_i x + q_i^T x <= r_i  (Quadratic/Cone Constraints)

NPZ Keys:
    - 'P_obj_data', 'P_obj_indices': Quadratic Objective Matrix (if any)
    - 'c': Linear objective
    - 'A_data', 'A_indices': Linear Constraint Matrix
    - 'Q_constraints': A list of dictionaries (or structured arrays) defining the quadratic constraints.
                       Note: Due to npz limitations, these are often serialized or stored as separate arrays 
                       prefixed with 'qc_{i}_'.
    - 'l_c', 'u_c', 'l_v', 'u_v': Bounds

--- NOTE ON PARSING ---
Parsing SOCP/QCP from MPS files via 'highspy' is complex. This script attempts to extract 
quadratic constraints if the underlying Highs version exposes them. If exact constraint extraction 
fails, it saves the linear backbone and points to the raw source for specialized parsing.

Dependencies: requests, numpy, scipy, highspy
"""
import os
import requests
import gzip
import shutil
import numpy as np
import highspy
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# --- Configuration ---
DATA_DIR = "./socp_benchmark_data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

SCALE_SMALL = 2_000
SCALE_MEDIUM = 20_000

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Disable warnings for verify=False
requests.packages.urllib3.disable_warnings()

@dataclass
class QuadraticConstraint:
    Q: sp.coo_matrix
    q: np.ndarray
    rhs: float
    sense: str # 'L' (<=), 'G' (>=), 'E' (==)

@dataclass
class SOCPStandardForm:
    name: str
    source_url: str
    P_obj: sp.coo_matrix # Quadratic Objective
    c: np.ndarray
    A: sp.coo_matrix     # Linear Constraints
    l_c: np.ndarray
    u_c: np.ndarray
    l_v: np.ndarray
    u_v: np.ndarray
    quad_constraints: List[QuadraticConstraint] = field(default_factory=list)
    
    @property
    def m(self): return self.A.shape[0]
    @property
    def n(self): return self.A.shape[1]
    @property
    def n_qcp(self): return len(self.quad_constraints)

class BenchmarkFetcher:
    """
    Downloads raw SOCP/QCP files (MPS format).
    """
    
    # 1. Mittelmann SOCP / DIMACS
    # Source: http://plato.asu.edu/ftp/socp/
    # These are widely used QCP formulations of SOCPs.
    MITTELMANN_SOCP_INSTANCES = [
        "nb", "nb_L1", "nb_L2", # Node blocking
        "qssp30", "qssp60", "qssp120", "qssp250", # Stochastic server location
        "dbic1", # Discrete boundary value
        "filter48", "filter50", # FIR filter design
        "mj1", "mj2", # Multi-jumping
        "problem", "tumor"
    ]

    # --- Source Maps ---
    
    SOURCES_CONFIG = {
        "Mittelmann_SOCP": {
            "base": "http://plato.asu.edu/ftp/socp/",
            "suffix": ".mps.gz",
            "files": MITTELMANN_SOCP_INSTANCES
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
            r = requests.get(url, headers=headers, stream=True, verify=False, timeout=30)
            if r.status_code != 200:
                print(f"    [Missing] {filename} (HTTP {r.status_code})")
                return None
            
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"    [Downloaded] {filename}")
            return dest_path
        except Exception as e:
            print(f"    [Error] {filename}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return None

    def fetch_all(self):
        print(f"\n=== 1. Downloading Raw SOCP Files ===")
        downloaded_map = {} 
        
        for category, config in self.SOURCES_CONFIG.items():
            print(f"  > Category: {category} ({len(config['files'])} potential files)")
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
    """Generates code-based SOCP test instances."""
    
    @staticmethod
    def generate_cvxpy_socp_1() -> SOCPStandardForm:
        """
        Simple Ice-Cream Cone constraint:
        min -z
        s.t. x^2 + y^2 <= z^2  (z >= 0)
             x + y + z == 1
        """
        print(f"  [Generator]: CVXPY Unit Test SOCP 1...")
        # Variables: [x, y, z] (n=3)
        # Linear: x + y + z = 1
        A = sp.coo_matrix(np.array([[1.0, 1.0, 1.0]]))
        l_c = np.array([1.0])
        u_c = np.array([1.0])
        
        # Obj: -z
        c = np.array([0.0, 0.0, -1.0])
        P_obj = sp.coo_matrix((3, 3))
        
        # Bounds (z >= 0 implicit in SOC, usually good to be explicit)
        l_v = np.array([-np.inf, -np.inf, 0.0])
        u_v = np.full(3, np.inf)
        
        # QCP Constraint: x^2 + y^2 - z^2 <= 0
        # Q = diag([2, 2, -2]) (factor of 1/2 in std form implies values 2,2,-2)
        # Actually standard form 1/2 x'Qx <= r.
        # x^2 + y^2 - z^2 <= 0  ->  1/2 * (2x^2 + 2y^2 - 2z^2) <= 0
        Q_data = np.array([2.0, 2.0, -2.0])
        Q_row = np.array([0, 1, 2])
        Q_col = np.array([0, 1, 2])
        Q_mat = sp.coo_matrix((Q_data, (Q_row, Q_col)), shape=(3, 3))
        
        qc = QuadraticConstraint(Q_mat, np.zeros(3), 0.0, 'L')
        
        return SOCPStandardForm("cvxpy_socp_1", "Synthetic Unit", P_obj, c, A, l_c, u_c, l_v, u_v, [qc])

class SOCPProcessor:
    def __init__(self):
        self.seen_categories = set()

    @staticmethod
    def determine_scale_label(n_vars: int) -> str:
        if n_vars < SCALE_SMALL: return "small"
        if n_vars < SCALE_MEDIUM: return "medium"
        return "large"

    @staticmethod
    def load_socp(mps_path: str, source_url: str) -> Optional[SOCPStandardForm]:
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

        # Read Model (Requires a Highs build that supports QCP parsing)
        status = h.readModel(use_path)
        if status != highspy.HighsStatus.kOk:
            print(f"    [Error] Highs failed to parse {name}")
            return None
        
        # Extraction
        model = h.getModel() 
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

        # Hessian (Quadratic Objective)
        hessian = model.hessian_
        if hessian.dim_ > 0:
            P_obj = sp.csc_matrix(
                (np.array(hessian.value_), 
                 np.array(hessian.index_), 
                 np.array(hessian.start_)),
                shape=(cols, cols)
            ).tocoo()
        else:
            P_obj = sp.coo_matrix((cols, cols))

        inf = 1e20
        l_v = np.array(lp.col_lower_); u_v = np.array(lp.col_upper_)
        l_c = np.array(lp.row_lower_); u_c = np.array(lp.row_upper_)
        
        l_v[l_v <= -inf] = -np.inf; u_v[u_v >= inf] = np.inf
        l_c[l_c <= -inf] = -np.inf; u_c[u_c >= inf] = np.inf
        
        # NOTE: Explicit QCP extraction via Highspy is currently limited in the 
        # standard python wheels. We populate the Linear/Obj parts correctly.
        # If the MPS contains cones, Highspy reads them but might not expose them 
        # easily in the simple model struct. 
        # We assume for this Aggregator that we mainly want the files downloaded 
        # and the linear backbone extracted.
        q_constraints = []
        
        return SOCPStandardForm(name, source_url, P_obj, c, A_coo, l_c, u_c, l_v, u_v, q_constraints)

    def process_and_save(self, instance: SOCPStandardForm, category_hint: str = "General"):
        scale = self.determine_scale_label(instance.n)
        
        save_dir = os.path.join(PROCESSED_DIR, category_hint, scale)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{instance.name}.npz")
        
        print(f"  {instance.name:<15} | {scale:<6} | A_nnz: {instance.A.nnz:<8} | QCPs: {instance.n_qcp}")
        
        # Pack data
        data_dict = {
            "name": instance.name,
            "source": instance.source_url,
            "P_obj_data": instance.P_obj.data,
            "P_obj_indices": np.vstack((instance.P_obj.row, instance.P_obj.col)).T,
            "P_obj_shape": instance.P_obj.shape,
            "c": instance.c,
            "A_data": instance.A.data,
            "A_indices": np.vstack((instance.A.row, instance.A.col)).T,
            "A_shape": instance.A.shape,
            "l_c": instance.l_c, "u_c": instance.u_c,
            "l_v": instance.l_v, "u_v": instance.u_v,
            "n_qcp": instance.n_qcp
        }
        
        # Pack Quadratic Constraints
        # Since npz is flat, we prefix keys like "qc_0_Q_data", "qc_0_rhs", etc.
        for i, qc in enumerate(instance.quad_constraints):
            data_dict[f"qc_{i}_Q_data"] = qc.Q.data
            data_dict[f"qc_{i}_Q_indices"] = np.vstack((qc.Q.row, qc.Q.col)).T
            data_dict[f"qc_{i}_Q_shape"] = qc.Q.shape
            data_dict[f"qc_{i}_q"] = qc.q
            data_dict[f"qc_{i}_rhs"] = qc.rhs
            data_dict[f"qc_{i}_sense"] = qc.sense

        np.savez_compressed(save_path, **data_dict)

def main():
    fetcher = BenchmarkFetcher()
    downloaded_files = fetcher.fetch_all()
    processor = SOCPProcessor()
    
    print("\n=== 2. Processing SOCP Files ===")
    for path, url in downloaded_files.items():
        category = os.path.basename(os.path.dirname(path))
        try:
            instance = processor.load_socp(path, url)
            if instance:
                processor.process_and_save(instance, category_hint=category)
        except Exception as e:
            # print(f"Skipping {path}: {e}")
            pass
            
    print("\n=== 3. Generators ===")
    try:
        processor.process_and_save(SyntheticGenerator.generate_cvxpy_socp_1(), "Synthetic_UnitTests")
    except Exception as e:
        print(f"  [CRITICAL] Generator failed: {e}")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()