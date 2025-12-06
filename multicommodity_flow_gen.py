# multicomodity flow generator 

import argparse
import numpy as np
import pyomo.environ as pyo
import os
import time

def build_multicommodity_flow_problem(args):
    np.random.seed(args.seed)
    
    # Parameters
    num_factories_per_comm = args.num_factories_per_commodity
    num_comm = args.num_commodities
    num_warehouses = args.num_warehouses
    num_stores = args.num_stores
    
    print("Generating instance data...")
    start_time = time.time()
    
    # Demand generation
    avg_demand = 100.0 * np.exp(np.random.randn(num_comm))
    # Poisson distribution for demand
    demand_per_comm_store = np.array([np.random.poisson(lam, num_stores) for lam in avg_demand])
    
    total_demand_per_comm = np.sum(demand_per_comm_store, axis=1)
    tolerance = 1e-8
    supply_per_factory = (1.0 + tolerance) * total_demand_per_comm / num_factories_per_comm
    total_demand = np.sum(demand_per_comm_store)
    
    warehouse_normal_capacity = 0.95 * total_demand / num_warehouses
    
    # Locations
    factory_locs = np.random.rand(num_comm, num_factories_per_comm, 2)
    warehouse_locs = np.random.rand(num_warehouses, 2)
    store_locs = np.random.rand(num_stores, 2)
    
    # Costs (Euclidean distance)
    # Shape: (num_comm, num_factories, num_warehouses)
    cost_f_w = np.zeros((num_comm, num_factories_per_comm, num_warehouses))
    for k in range(num_comm):
        for f in range(num_factories_per_comm):
            dist = np.linalg.norm(factory_locs[k, f, :] - warehouse_locs, axis=1)
            cost_f_w[k, f, :] = dist
            
    # Shape: (num_warehouses, num_stores)
    cost_w_s = np.linalg.norm(warehouse_locs[:, np.newaxis, :] - store_locs[np.newaxis, :, :], axis=2)
    
    print(f"Data generated in {time.time() - start_time:.2f}s")
    
    # Model Building
    print("Building Pyomo model...")
    model = pyo.ConcreteModel()
    
    # Sets
    model.K = pyo.RangeSet(0, num_comm - 1)
    model.F = pyo.RangeSet(0, num_factories_per_comm - 1)
    model.W = pyo.RangeSet(0, num_warehouses - 1)
    model.S = pyo.RangeSet(0, num_stores - 1)
    
    # Variables
    model.flow_fw = pyo.Var(model.K, model.F, model.W, domain=pyo.NonNegativeReals)
    model.flow_ws = pyo.Var(model.K, model.W, model.S, domain=pyo.NonNegativeReals)
    model.overtime = pyo.Var(model.W, domain=pyo.NonNegativeReals)
    
    # Objective
    def obj_rule(m):
        cost1 = sum(cost_f_w[k,f,w] * m.flow_fw[k,f,w] for k in m.K for f in m.F for w in m.W)
        cost2 = sum(cost_w_s[w,s] * m.flow_ws[k,w,s] for k in m.K for w in m.W for s in m.S)
        overtime = args.additional_overtime_cost * sum(m.overtime[w] for w in m.W)
        return cost1 + cost2 + overtime
    model.Obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Constraints
    # 1. Supply Constraints
    def supply_rule(m, k, f):
        return sum(m.flow_fw[k,f,w] for w in m.W) <= supply_per_factory[k]
    model.Supply = pyo.Constraint(model.K, model.F, rule=supply_rule)
    
    # 2. Overtime Constraints
    def capacity_rule(m, w):
        total_inflow = sum(m.flow_fw[k,f,w] for k in m.K for f in m.F)
        return total_inflow <= warehouse_normal_capacity + m.overtime[w]
    model.Capacity = pyo.Constraint(model.W, rule=capacity_rule)
    
    # 3. Flow Conservation (Warehouse)
    def conservation_rule(m, w, k):
        inflow = sum(m.flow_fw[k,f,w] for f in m.F)
        outflow = sum(m.flow_ws[k,w,s] for s in m.S)
        return inflow == outflow
    model.Conservation = pyo.Constraint(model.W, model.K, rule=conservation_rule)
    
    # 4. Demand Constraints
    def demand_rule(m, k, s):
        return sum(m.flow_ws[k,w,s] for w in m.W) == demand_per_comm_store[k,s]
    model.Demand = pyo.Constraint(model.K, model.S, rule=demand_rule)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_commodities", type=int, default=10)
    parser.add_argument("--num_factories_per_commodity", type=int, default=5)
    parser.add_argument("--num_warehouses", type=int, default=10)
    parser.add_argument("--num_stores", type=int, default=100)
    parser.add_argument("--additional_overtime_cost", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    model = build_multicommodity_flow_problem(args)
    
    print("Writing MPS file...")
    # io_options symbolic_solver_labels=True preserves variable names roughly
    model.write(args.output_file, format='mps', io_options={'symbolic_solver_labels': True})
    print(f"Done. Saved to {args.output_file}")