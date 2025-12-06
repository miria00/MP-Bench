import argparse
import numpy as np
import scipy.spatial
import pyomo.environ as pyo
import time

def build_matching_problem(args):
    np.random.seed(args.seed)
    n = args.num_treatment_samples
    m = args.num_control_samples
    d = args.num_covariates
    k_edges = args.num_edges_per_treatment
    
    print("Generating data...")
    A_data = np.random.randn(n, d)
    shift = args.control_shift_magnitude * np.random.randn(d)
    B_data = np.random.randn(m, d) + shift
    
    print("Building KDTree and finding neighbors...")
    # Find k closest controls for each treatment
    tree = scipy.spatial.cKDTree(B_data)
    dists, indices = tree.query(A_data, k=k_edges)
    
    # Prepare edge data: list of (treatment_idx, control_idx, cost)
    # treatment i (0..n-1), control j (0..m-1)
    edges = []
    for i in range(n):
        for k_idx in range(k_edges):
            j = indices[i, k_idx]
            cost = dists[i, k_idx] # Euclidean distance
            edges.append((i, j, cost))
            
    num_edges = len(edges)
    
    print("Building Pyomo model...")
    model = pyo.ConcreteModel()
    
    # Variables
    # x: weight on edge (i,j)
    model.E = pyo.RangeSet(0, num_edges - 1)
    model.x = pyo.Var(model.E, bounds=(0, 1))
    
    # w: weight on control j used
    model.C = pyo.RangeSet(0, m - 1)
    model.w = pyo.Var(model.C, bounds=(0, 1))
    
    # Objective: Minimize weighted distance
    model.Obj = pyo.Objective(expr=sum(edges[e][2] * model.x[e] for e in model.E), sense=pyo.minimize)
    
    # Constraints
    # 1. Treatment Assignment: sum_j x_ij = 1 for all i
    # Pre-calculate edge maps for speed
    edges_from_treatment = {i: [] for i in range(n)}
    edges_into_control = {j: [] for j in range(m)}
    
    for e_idx, (i, j, cost) in enumerate(edges):
        edges_from_treatment[i].append(e_idx)
        edges_into_control[j].append(e_idx)
        
    def treatment_rule(m, i):
        return sum(m.x[e] for e in edges_from_treatment[i]) == 1.0
    model.TreatAssign = pyo.Constraint(pyo.RangeSet(0, n-1), rule=treatment_rule)
    
    # 2. Control Assignment: sum_i x_ij = w_j
    def control_rule(m, j):
        if not edges_into_control[j]:
            return m.w[j] == 0 # Optimize out unused controls
        return sum(m.x[e] for e in edges_into_control[j]) == m.w[j]
    model.ControlAssign = pyo.Constraint(model.C, rule=control_rule)
    
    # 3. Moment Matching (Balance Constraints)
    epsilon = args.epsilon
    
    # First Moment (Mean)
    # Target: Mean(Treatment)
    target_mu = np.mean(A_data, axis=0)
    
    # Check: Mean(Control_Weighted) approx Target
    # sum(B[j] * w[j]) / n  (Note divided by n, size of treatment group)
    
    def moment1_rule_lb(m, k):
        # dimension k
        weighted_sum = sum(B_data[j, k] * m.w[j] for j in model.C)
        return weighted_sum / n >= target_mu[k] - epsilon

    def moment1_rule_ub(m, k):
        weighted_sum = sum(B_data[j, k] * m.w[j] for j in model.C)
        return weighted_sum / n <= target_mu[k] + epsilon
        
    model.Moment1_LB = pyo.Constraint(pyo.RangeSet(0, d-1), rule=moment1_rule_lb)
    model.Moment1_UB = pyo.Constraint(pyo.RangeSet(0, d-1), rule=moment1_rule_ub)
    
    # Second Moment (Covariance / Cross terms)
    # Only creating interactions k <= l
    interaction_indices = [(k, l) for k in range(d) for l in range(k, d)]
    
    # Precompute target second moments
    target_M2 = {}
    for (k, l) in interaction_indices:
        target_M2[(k,l)] = np.mean(A_data[:, k] * A_data[:, l])
        
    def moment2_rule_lb(m, idx):
        k, l = interaction_indices[idx]
        weighted_sum = sum(B_data[j, k] * B_data[j, l] * m.w[j] for j in model.C)
        return weighted_sum / n >= target_M2[(k,l)] - epsilon

    def moment2_rule_ub(m, idx):
        k, l = interaction_indices[idx]
        weighted_sum = sum(B_data[j, k] * B_data[j, l] * m.w[j] for j in model.C)
        return weighted_sum / n <= target_M2[(k,l)] + epsilon
        
    model.Moment2_LB = pyo.Constraint(pyo.RangeSet(0, len(interaction_indices)-1), rule=moment2_rule_lb)
    model.Moment2_UB = pyo.Constraint(pyo.RangeSet(0, len(interaction_indices)-1), rule=moment2_rule_ub)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--num_treatment_samples", type=int, default=5000)
    parser.add_argument("--num_control_samples", type=int, default=20000)
    parser.add_argument("--num_covariates", type=int, default=8)
    parser.add_argument("--num_edges_per_treatment", type=int, default=10)
    parser.add_argument("--control_shift_magnitude", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    model = build_matching_problem(args)
    
    print("Writing MPS file...")
    model.write(args.output_file, format='mps', io_options={'symbolic_solver_labels': True})