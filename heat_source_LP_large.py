'''Heat Source Location (Inverse PDE)
File: generate_heat_source_location.py This requires solving a PDE (Poisson equation) first to generate "Ground Truth" data, then building the optimization model.'''

import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import pyomo.environ as pyo
import h5py
import time
import sys

def build_laplacian_stencil(N):
    """Builds a sparse matrix for the 3D Laplacian on an N^3 grid."""
    # This constructs the standard 7-point stencil finite difference matrix
    grid_points = N**3
    h = 1.0 / N
    factor = 1.0 / (h**2)
    
    # We construct this using Kronecker products for efficiency
    # D2 is 1D 2nd derivative matrix
    ones = np.ones(N)
    D2 = sp.diags([ones[:-1], -2*ones, ones[:-1]], [-1, 0, 1], shape=(N, N))
    I = sp.eye(N)
    
    # Laplacian in 3D: Dxx + Dyy + Dzz
    # L = D2 (x) I (x) I  +  I (x) D2 (x) I  +  I (x) I (x) D2
    L = sp.kronsum(sp.kronsum(D2, D2), D2) * factor
    return L

def solve_ground_truth(args):
    """Solves the forward PDE to get u_true."""
    N = args.grid_size
    # Adjust N because Julia code includes boundary in indices but fixes them to 0
    # We will simulate the internal grid points (N-2)^3 
    # But for simplicity, let's treat grid_size as the internal mesh count
    
    # Source generation
    np.random.seed(args.seed)
    num_src = args.num_source_locations
    
    # Random locations in 3D grid
    # We treat indices 0 to N-1
    indices = np.random.randint(0, N, size=(num_src, 3))
    rates = np.random.rand(num_src)
    
    # Create RHS vector q (negative of heat source)
    q = np.zeros((N, N, N))
    for i in range(num_src):
        # Scale factor from Julia code: grid_size^3
        q[tuple(indices[i])] = (N**3) * rates[i]
        
    q_vec = -q.flatten() # -q because equation is Laplacian(u) = -q
    
    print("Building Laplacian...")
    A = build_laplacian_stencil(N)
    
    print("Solving forward PDE (Ground Truth)...")
    # Using Minres or Splu. Splu is often faster for this size if memory allows.
    try:
        u_vec = sp.linalg.spsolve(A, q_vec)
    except:
        # Fallback to iterative if memory is tight
        u_vec, info = sp.linalg.minres(A, q_vec, tol=args.pde_solve_tolerance)
        
    u_true = u_vec.reshape((N, N, N))
    return u_true, indices, rates

def build_inverse_model(args, u_true, true_src_indices):
    N = args.grid_size
    model = pyo.ConcreteModel()
    
    print("Building Optimization Model...")
    
    # We pad the grid with zeros for boundaries to match stencil logic easily in Pyomo
    # 0..N+1 range
    model.I = pyo.RangeSet(0, N+1)
    model.J = pyo.RangeSet(0, N+1)
    model.K = pyo.RangeSet(0, N+1)
    
    # Variables: u (temperature), q (source)
    model.u = pyo.Var(model.I, model.J, model.K, domain=pyo.Reals)
    # q is only non-zero at internal points
    model.q = pyo.Var(pyo.RangeSet(1, N), pyo.RangeSet(1, N), pyo.RangeSet(1, N), 
                      domain=pyo.NonNegativeReals) # Implicitly q <= 0 in Julia logic? 
                      # Julia: 0 <= q <= 0 initially, then unbounds specific spots.
                      # Actually Julia sets -q in equation. Let's stick to standard q >= 0.
    
    # Objective: Minimize sum(q) (Sparsity inducing L1 norm equivalent since q >= 0)
    model.Obj = pyo.Objective(expr=sum(model.q[i,j,k] for i in range(1,N+1) for j in range(1,N+1) for k in range(1,N+1)), 
                              sense=pyo.minimize)
    
    # Constraints: PDE
    h = 1.0 / N
    h2 = h**2
    
    def pde_rule(m, i, j, k):
        # Apply only to internal points
        laplacian = (
            (m.u[i+1,j,k] - 2*m.u[i,j,k] + m.u[i-1,j,k]) +
            (m.u[i,j+1,k] - 2*m.u[i,j,k] + m.u[i,j-1,k]) +
            (m.u[i,j,k+1] - 2*m.u[i,j,k] + m.u[i,j,k-1])
        ) / h2
        return laplacian == -m.q[i,j,k]
        
    model.PDE = pyo.Constraint(pyo.RangeSet(1, N), pyo.RangeSet(1, N), pyo.RangeSet(1, N), rule=pde_rule)
    
    # Boundary Conditions (Dirichlet = 0)
    # Fix boundaries
    for i in [0, N+1]:
        for j in range(N+2):
            for k in range(N+2):
                model.u[i,j,k].fix(0.0)
                model.u[j,i,k].fix(0.0)
                model.u[j,k,i].fix(0.0)

    # Constraint: q is zero almost everywhere, except potential locations
    # In Julia code, they initialize q=0 and unfix at candidate locations.
    # Here, we fix all q to 0, then unfix candidates.
    for i in range(1, N+1):
        for j in range(1, N+1):
            for k in range(1, N+1):
                model.q[i,j,k].fix(0.0)

    # Generate Candidates
    num_poss = args.num_possible_source_locations
    # Ensure true sources are in candidates
    candidates = set(tuple(x + 1) for x in true_src_indices) # +1 offset for Pyomo 1-based internal
    
    while len(candidates) < num_poss:
        cand = tuple(np.random.randint(1, N+1, size=3))
        candidates.add(cand)
        
    for (i,j,k) in candidates:
        model.q[i,j,k].unfix()
        
    # Measurement Constraints
    # Fix u at measurement locations to ground truth +/- error
    num_meas = args.num_measurement_locations
    for _ in range(num_meas):
        # Random measurement point (internal)
        idx = tuple(np.random.randint(0, N, size=3))
        pyomo_idx = (idx[0]+1, idx[1]+1, idx[2]+1)
        
        val = u_true[idx]
        err = args.maximum_relative_measurement_error
        
        # In Julia: val / (1+err) <= u <= val * (1+err)
        lb = val / (1 + err)
        ub = val * (1 + err)
        
        model.u[pyomo_idx].setlb(lb)
        model.u[pyomo_idx].setub(ub)

    return model, u_true

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--ground_truth_file", required=True)
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--num_source_locations", type=int, default=2)
    parser.add_argument("--num_possible_source_locations", type=int, default=50)
    parser.add_argument("--num_measurement_locations", type=int, default=20)
    parser.add_argument("--maximum_relative_measurement_error", type=float, default=1e-6)
    parser.add_argument("--pde_solve_tolerance", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # 1. Solve Forward Problem
    u_true, true_locs, _ = solve_ground_truth(args)
    
    # 2. Save Ground Truth
    with h5py.File(args.ground_truth_file, 'w') as f:
        f.create_dataset("u_true", data=u_true)
    
    # 3. Build LP
    model, _ = build_inverse_model(args, u_true, true_locs)
    
    print("Writing MPS file...")
    model.write(args.output_file, format='mps', io_options={'symbolic_solver_labels': True})