import random
import numpy as np

class GeneticAlgorithmVRP:
    def __init__(self, distance_matrix, demands, vehicle_capacity, customer_assignments, hubs, customers,
                 population_size=50, num_generations=100, mutation_rate=0.1, crossover_rate=0.8):
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.customer_assignments = customer_assignments  # customer index -> hub index
        self.hubs = hubs  # List of hub indices
        self.customers = customers  # List of customer indices
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

    def evaluate_solution(self, solution):
        """Calculate total distance of all routes in the solution."""
        total_distance = 0
        for hub in self.hubs:
            hub_routes = solution[hub]
            for route in hub_routes:
                route_distance = self.calculate_route_distance(route)
                total_distance += route_distance
        return total_distance

    def validate_route(self, route, hub):
        """Check if a route is feasible (capacity constraints)."""
        total_demand = sum(self.demands[customer] for customer in route if customer != hub)
        return total_demand <= self.vehicle_capacity

    def calculate_route_distance(self, route):
        """Calculate the total distance of a route."""
        distance = 0
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            distance += self.distance_matrix[from_node][to_node]
        return distance

    def validate_solution(self, solution):
        """Ensure all customers are served exactly once and capacity constraints are satisfied."""
        served_customers = set()
        for hub in self.hubs:
            hub_routes = solution[hub]
            for route in hub_routes:
                # Check for capacity constraints
                if not self.validate_route(route, hub):
                    return False
                for customer in route:
                    if customer != hub:
                        if customer in served_customers:
                            return False  # Duplicate visit detected
                        served_customers.add(customer)
        return served_customers == set(self.customers)  # Ensure all customers are visited

    def mutate(self, solution):
        """Apply mutation to a solution."""
        new_solution = {}
        for hub in self.hubs:
            routes = solution[hub]
            mutated_routes = []
            for route in routes:
                route = route.copy()
                if random.random() < self.mutation_rate:
                    route = self.swap_mutation(route, hub)
                mutated_routes.append(route)
            new_solution[hub] = mutated_routes
        # Repair solution to remove duplicates and ensure all customers are assigned
        new_solution = self.repair_solution(new_solution)
        return new_solution

    def swap_mutation(self, route, hub):
        """Swap two customers in a route."""
        customer_indices = [i for i in range(1, len(route) - 1)]  # Exclude hubs at start and end
        if len(customer_indices) < 2:
            return route
        idx1, idx2 = random.sample(customer_indices, 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
        return route

    def select_parents(self, fitness_scores):
        """Select two parents using tournament selection."""
        tournament_size = 5
        selected = random.sample(list(zip(self.population, fitness_scores)), tournament_size)
        parent1 = min(selected, key=lambda x: x[1])[0]
        selected = random.sample(list(zip(self.population, fitness_scores)), tournament_size)
        parent2 = min(selected, key=lambda x: x[1])[0]
        return parent1, parent2

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        child1 = {}
        child2 = {}
        for hub in self.hubs:
            routes1 = parent1[hub]
            routes2 = parent2[hub]
            # For simplicity, perform one-point crossover on the routes list
            if len(routes1) > 1 and len(routes2) > 1:
                crossover_point = random.randint(1, min(len(routes1), len(routes2)) - 1)
                child1_routes = routes1[:crossover_point] + routes2[crossover_point:]
                child2_routes = routes2[:crossover_point] + routes1[crossover_point:]
            else:
                child1_routes = [route.copy() for route in routes1]
                child2_routes = [route.copy() for route in routes2]
            # Repair routes to ensure all customers are included and no duplicates
            child1[hub] = self.repair_routes(child1_routes, hub)
            child2[hub] = self.repair_routes(child2_routes, hub)
        return child1, child2

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

    def run(self):
        """Run the genetic algorithm."""
        self.initialize_population()
        if not self.population:
            print("Failed to initialize a valid population.")
            return None, float('inf')
        for generation in range(self.num_generations):
            fitness_scores = []
            for solution in self.population:
                if not self.validate_solution(solution):
                    fitness_scores.append(float('inf'))
                else:
                    fitness_scores.append(self.evaluate_solution(solution))
            best_index = fitness_scores.index(min(fitness_scores))
            if fitness_scores[best_index] < self.best_cost:
                self.best_cost = fitness_scores[best_index]
                self.best_solution = self.population[best_index]
                print(f"Generation {generation}: Best Cost = {self.best_cost}")
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(fitness_scores)
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                # Validate and add to new population
                if self.validate_solution(child1):
                    new_population.append(child1)
                if self.validate_solution(child2):
                    new_population.append(child2)
                if len(new_population) >= self.population_size:
                    break
            self.population = new_population[:self.population_size]
            if not self.population:
                print("Population died out.")
                break
        return self.best_solution, self.best_cost
