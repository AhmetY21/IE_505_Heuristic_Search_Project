from ortools.linear_solver import pywraplp
import numpy as np
import random


def solve_vehicle_routing_problem(N, C, H, K, demand, vehicle_capacity, vehicle_max_distance,
                                  vehicle_available, distance_matrix, cost_matrix, coordinates, M=1e5):
    # Solver implementation as previously described
    # [Complete function code as provided above]
    # Create index mappings for nodes and vehicle types
    node_indices = {node: idx for idx, node in enumerate(N)}
    customer_indices = {customer: idx for idx, customer in enumerate(C)}
    hub_indices = {hub: idx for idx, hub in enumerate(H)}
    vehicle_type_indices = {k: idx for idx, k in enumerate(K)}

    num_nodes = len(N)
    num_customers = len(C)
    num_hubs = len(H)
    num_vehicle_types = len(K)

    # Create the solver
    solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')

    if not solver:
        print('Solver not found.')
        return None

    # Decision Variables
    x = [[[solver.BoolVar(f'x[{i},{j},{k}]') for k in range(num_vehicle_types)]
          for j in range(num_nodes)] for i in range(num_nodes)]
    l = [[solver.NumVar(0, solver.infinity(), f'l[{i},{k}]') for k in range(num_vehicle_types)]
         for i in range(num_nodes)]
    s = [[solver.NumVar(0, solver.infinity(), f's[{i},{k}]') for k in range(num_vehicle_types)]
         for i in range(num_nodes)]

    # Objective Function
    total_transport_cost = solver.Sum([
        cost_matrix[i][j][k] * x[i][j][k]
        for i in range(num_nodes)
        for j in range(num_nodes)
        for k in range(num_vehicle_types)
    ])
    solver.Minimize(total_transport_cost)

    # Pre-assignment of customers to hubs

    # Pre-assignment of customers to hubs (assign each customer to the closest hub)
    customer_to_hub = []
    for idx_c, c in enumerate(C):
        min_distance = float('inf')
        assigned_hub_idx = None
        c_idx = node_indices[c]
        for h in H:
            h_idx = node_indices[h]
            if distance_matrix[c_idx][h_idx] < min_distance:
                min_distance = distance_matrix[c_idx][h_idx]
                assigned_hub_idx = h_idx
        customer_to_hub.append(assigned_hub_idx)

    # 1. Vehicle Flow Constraints

    # 1a. Flow Conservation at Customer Nodes
    for idx_c, c in enumerate(C):
        c_idx = node_indices[c]
        for k in range(num_vehicle_types):
            inflow = solver.Sum([x[j][c_idx][k] for j in range(num_nodes)])
            outflow = solver.Sum([x[c_idx][j][k] for j in range(num_nodes)])
            solver.Add(inflow - outflow == 0)

    # 1b. Vehicles Start from Assigned Hubs
    for idx_c, c in enumerate(C):
        c_idx = node_indices[c]
        h_idx = customer_to_hub[idx_c]
        for k in range(num_vehicle_types):
            solver.Add(x[h_idx][c_idx][k] == 1)

    # 1c. Vehicles Return to Hubs
    for idx_c, c in enumerate(C):
        c_idx = node_indices[c]
        h_idx = customer_to_hub[idx_c]
        for k in range(num_vehicle_types):
            solver.Add(x[c_idx][h_idx][k] == 1)

    # 1d. Vehicle Availability Constraints
    for k in range(num_vehicle_types):
        total_vehicles_used = solver.Sum([
            x[h_idx][c_idx][k]
            for idx_c, c in enumerate(C)
            for h_idx in [customer_to_hub[idx_c]]
        ])
        solver.Add(total_vehicles_used <= vehicle_available[k])

    # 2. Load Constraints

    # 2a. Load Update Constraints
    for i in range(num_nodes):
        for j in range(num_nodes):
            for k in range(num_vehicle_types):
                if i != j:
                    demand_j = demand[customer_indices[N[j]]] if N[j] in C else 0
                    solver.Add(l[j][k] >= l[i][k] + demand_j - vehicle_capacity[k] * (1 - x[i][j][k]))

    # 2b. Load Limits at Nodes
    for i in range(num_nodes):
        for k in range(num_vehicle_types):
            solver.Add(l[i][k] <= vehicle_capacity[k])

    # 2c. Initialize Load at Hubs
    for h in H:
        h_idx = node_indices[h]
        for k in range(num_vehicle_types):
            solver.Add(l[h_idx][k] == 0)

    # 3. Subtour Elimination Constraints (Using MTZ Formulation)

    # 3a. Subtour Elimination
    for i in range(num_nodes):
        for j in range(num_nodes):
            for k in range(num_vehicle_types):
                if i != j:
                    solver.Add(s[j][k] >= s[i][k] + distance_matrix[i][j] * x[i][j][k] - M * (1 - x[i][j][k]))

    # 3b. Maximum Distance Constraints
    for i in range(num_nodes):
        for k in range(num_vehicle_types):
            solver.Add(s[i][k] <= vehicle_max_distance[k])

    # 3c. Initialize Distance at Hubs
    for h in H:
        h_idx = node_indices[h]
        for k in range(num_vehicle_types):
            solver.Add(s[h_idx][k] == 0)

    # Solve the Model
    status = solver.Solve()

    # Prepare the results
    results = {}

    if status == pywraplp.Solver.OPTIMAL:
        results['Optimal Objective Value'] = solver.Objective().Value()
        hubs_used = set(H)
        results['Hubs Used'] = hubs_used
        customer_assignments = {}
        for idx_c, c in enumerate(C):
            h_idx = customer_to_hub[idx_c]
            h = N[h_idx]
            customer_assignments[c] = h
        results['Customer Assignments'] = customer_assignments
        vehicle_routes = {}
        for k in range(num_vehicle_types):
            vehicle_type = K[k]
            vehicle_routes[vehicle_type] = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if x[i][j][k].solution_value() > 0.5:
                        route_info = {
                            'from': N[i],
                            'to': N[j],
                            'load': l[j][k].solution_value(),
                            'distance': s[j][k].solution_value(),
                            'vehicle_type': vehicle_type
                        }
                        vehicle_routes[vehicle_type].append(route_info)
        results['Vehicle Routes'] = vehicle_routes
        results['Coordinates'] = coordinates
        return results
    else:
        print("The problem does not have an optimal solution.")
        return None
