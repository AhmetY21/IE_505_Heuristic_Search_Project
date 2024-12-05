import random
import numpy as np


class GeneticAlgorithmVRP:
    def __init__(self, distance_matrix, demands, vehicle_capacity, customer_assignments, hubs, customers,
                    time_windows, vehicle_speed, max_hubs_to_open,
                    population_size=50, num_generations=100, mutation_rate=0.1, crossover_rate=0.8,fixed_costs = None):
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.customer_assignments = customer_assignments
        self.hubs = hubs
        self.customers = customers
        self.time_windows = time_windows
        #self.fixed_costs = fixed_costs
        self.vehicle_speed = vehicle_speed
        self.max_hubs_to_open = max_hubs_to_open
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_solution = None
        self.best_cost = float('inf')

    def initialize_population(self):
        """Generate initial population of solutions."""
        for _ in range(self.population_size):
            solution = {}
            for hub in self.hubs:
                assigned_customers = [c for c in self.customers if self.customer_assignments[c] == hub]
                routes = self.create_initial_routes(assigned_customers, hub)
                solution[hub] = routes
            if self.validate_solution(solution):
                self.population.append(solution)

    def create_initial_routes(self, customers, hub):
        """Create initial routes for a hub."""
        random.shuffle(customers)
        routes = []
        current_route = [hub]  # Start with the hub
        current_load = 0
        for customer in customers:
            demand = self.demands[customer]
            if current_load + demand <= self.vehicle_capacity:
                current_route.append(customer)
                current_load += demand
            else:
                current_route.append(hub)  # End the route at the hub
                routes.append(current_route)
                current_route = [hub, customer]  # Start new route from the hub
                current_load = demand
        current_route.append(hub)  # End the last route at the hub
        if len(current_route) > 2:
            routes.append(current_route)
        return routes

    def calculate_route_distance(self, route):
        """Calculate the total distance of a route."""
        distance = 0
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            distance += self.distance_matrix[from_node][to_node]
        return distance

    def validate_solution(self, solution):
        """Ensure all constraints are satisfied."""
        # Check that all customers are served exactly once
        served_customers = set()
        for hub, routes in solution.items():
            for route in routes:
                route_demand = sum(self.demands[c] for c in route if c != hub)
                if route_demand > self.vehicle_capacity:
                    return False  # Exceeds vehicle capacity
                for customer in route:
                    if customer != hub:
                        if customer in served_customers:
                            return False  # Duplicate visit
                        served_customers.add(customer)
        return served_customers == set(self.customers)

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        child = {}
        for hub in self.hubs:
            routes1 = parent1.get(hub, [])
            routes2 = parent2.get(hub, [])
            # Simple one-point crossover for routes
            if len(routes1) > 1 and len(routes2) > 1:
                split_point = random.randint(1, min(len(routes1), len(routes2)) - 1)
                child_routes = routes1[:split_point] + routes2[split_point:]
            else:
                # If not enough routes for crossover, copy routes from one parent
                child_routes = routes1 if len(routes1) >= len(routes2) else routes2
            child[hub] = child_routes
        return child


    def mutate(self, solution):
        """Apply mutation to a solution."""
        new_solution = {}
        for hub, routes in solution.items():
            mutated_routes = []
            for route in routes:
                route = route.copy()
                if random.random() < self.mutation_rate:
                    random.shuffle(route[1:-1])  # Mutate route (exclude hubs)
                mutated_routes.append(route)
            new_solution[hub] = mutated_routes
        return new_solution



    def validate_solution(self, solution):
        """Ensure all constraints are satisfied."""
        # Check that all customers are served exactly once
        served_customers = set()
        for hub, routes in solution.items():
            for route in routes:
                route_demand = sum(self.demands[c] for c in route if c != hub)
                if route_demand > self.vehicle_capacity:
                    return False  # Exceeds vehicle capacity
                for customer in route:
                    if customer != hub:
                        if customer in served_customers:
                            return False  # Duplicate visit
                        served_customers.add(customer)
        return served_customers == set(self.customers)

    def repair_routes(self, routes, hub):
        """Ensure that all assigned customers are included and no duplicates."""
        assigned_customers = [c for c in self.customers if self.customer_assignments[c] == hub]
        customers_in_routes = [customer for route in routes for customer in route if customer != hub]
        missing_customers = set(assigned_customers) - set(customers_in_routes)
        extra_customers = set(customers_in_routes) - set(assigned_customers)
        # Remove extra customers
        for route in routes:
            route[:] = [c for c in route if c not in extra_customers or c == hub]
        # Add missing customers
        for customer in missing_customers:
            # Add to a random route or create a new route
            added = False
            random.shuffle(routes)
            for route in routes:
                if sum(self.demands[c] for c in route if c != hub) + self.demands[customer] <= self.vehicle_capacity:
                    # Insert customer into route before the end hub
                    route.insert(-1, customer)
                    added = True
                    break
            if not added:
                # Create a new route starting and ending at the hub
                routes.append([hub, customer, hub])
        return routes

    def repair_solution(self, solution):
        """Ensure that all customers are included exactly once and no duplicates exist."""
        # Build a set of all customers that should be assigned
        all_customers = set(self.customers)
        # Build a mapping of customers already assigned
        assigned_customers = set()
        for hub in self.hubs:
            routes = solution[hub]
            for route in routes:
                assigned_customers.update([c for c in route if c != hub])
        # Identify missing and duplicate customers
        missing_customers = all_customers - assigned_customers
        duplicate_customers = assigned_customers - all_customers  # Should be empty

        # Remove duplicates
        customer_counts = {}
        for hub in self.hubs:
            routes = solution[hub]
            for route in routes:
                for customer in route:
                    if customer != hub:
                        customer_counts[customer] = customer_counts.get(customer, 0) + 1
        # Remove duplicate customers from routes
        for customer, count in customer_counts.items():
            if count > 1:
                # Remove extra occurrences
                occurrences = 0
                for hub in self.hubs:
                    routes = solution[hub]
                    for route in routes:
                        if customer in route:
                            if occurrences < 1:
                                occurrences += 1
                            else:
                                route.remove(customer)
        # Add missing customers
        for customer in missing_customers:
            assigned_hub = self.customer_assignments[customer]
            # Add to a random route of the assigned hub or create a new route
            routes = solution[assigned_hub]
            added = False
            random.shuffle(routes)
            for route in routes:
                if sum(self.demands[c] for c in route if c != assigned_hub) + self.demands[customer] <= self.vehicle_capacity:
                    # Insert customer into route before the end hub
                    route.insert(-1, customer)
                    added = True
                    break
            if not added:
                # Create a new route starting and ending at the hub
                routes.append([assigned_hub, customer, assigned_hub])
        return solution

    def swap_mutation(self, route, hub):
        """Swap two customers in a route."""
        customer_indices = [i for i in range(1, len(route) - 1)]  # Exclude hubs at start and end
        if len(customer_indices) < 2:
            return route
        idx1, idx2 = random.sample(customer_indices, 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
        return route
    
    def evaluate_solution(self, solution):
        """Calculate total cost of a solution, including travel distance and fixed hub costs."""
        total_distance = 0
        #total_fixed_costs = sum(self.fixed_costs[hub] for hub in solution.keys())
        for hub, routes in solution.items():
            for route in routes:
                total_distance += self.calculate_route_distance(route)
        return total_distance #+ total_fixed_costs

    def run(self):
        """Run the genetic algorithm."""
        # Step 1: Initialize the population
        self.initialize_population()
        if not self.population:
            print("Failed to initialize a valid population.")
            return None, float('inf')

        # Step 2: Iterate through generations
        for generation in range(self.num_generations):
            # Step 3: Evaluate fitness of solutions
            fitness_scores = [self.evaluate_solution(sol) for sol in self.population]
            best_index = fitness_scores.index(min(fitness_scores))

            # Update the best solution found so far
            if fitness_scores[best_index] < self.best_cost:
                self.best_cost = fitness_scores[best_index]
                self.best_solution = self.population[best_index]
                print(f"Generation {generation}: Best Cost = {self.best_cost}")

            # Step 4: Create the next generation
            new_population = []
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = random.sample(self.population, 2)

                # Apply crossover
                if random.random() < self.crossover_rate:
                    child1 = self.crossover(parent1, parent2)
                    child2 = self.crossover(parent2, parent1)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Apply mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                # Repair solutions to ensure feasibility
                child1 = {hub: self.repair_routes(child1[hub], hub) for hub in self.hubs}
                child2 = {hub: self.repair_routes(child2[hub], hub) for hub in self.hubs}

                # Repair entire solution to remove duplicates and ensure all customers are served
                child1 = self.repair_solution(child1)
                child2 = self.repair_solution(child2)

                # Validate and add to new population
                if self.validate_solution(child1):
                    new_population.append(child1)
                if self.validate_solution(child2):
                    new_population.append(child2)

                # Break if population is full
                if len(new_population) >= self.population_size:
                    break

            # Update the population
            self.population = new_population[:self.population_size]

            # If population dies out, stop early
            if not self.population:
                print("Population died out.")
                break

        # Step 5: Return the best solution found
        return self.best_solution, self.best_cost
