import numpy as np
import random

class GeneticAlgorithmHSA:
    def __init__(self, cost_matrix, num_nodes, num_hubs, 
                 pop_size=50, num_generations=100, mutation_rate=0.05, crossover_rate=0.8):
        self.cost_matrix = cost_matrix
        self.num_nodes = num_nodes
        self.num_hubs = num_hubs 
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_hub_selection = None

    def initialize_population(self):
        """Generate initial population with distinct hubs and valid assignments."""
        population = []
        for _ in range(self.pop_size):
            hubs = random.sample(range(self.num_nodes), self.num_hubs)
            y = [False] * self.num_nodes
            for hub in hubs:
                y[hub] = True
            z = [random.choice(hubs) for _ in range(self.num_nodes)]
            population.append((y, z))
        self.population = population

    def evaluate_solution(self, y, z):
        """Calculate total distance between customers and assigned hubs."""
        # Apply death penalty if any customer is assigned to a closed hub
        if any(not y[assigned_hub] for assigned_hub in z):
            return float('inf')  # Death penalty
        else:
            return sum(self.cost_matrix[i, z[i]] for i in range(len(z)))

    def validate_solution(self, y, z):
        """Ensure the solution has exactly num_hubs open hubs and customers are assigned to open hubs."""
        open_hubs = [i for i, hub_open in enumerate(y) if hub_open]
        closed_hubs = [i for i, hub_open in enumerate(y) if not hub_open]

        # Adjust the number of open hubs to exactly num_hubs
        if len(open_hubs) > self.num_hubs:
            hubs_to_close = random.sample(open_hubs, len(open_hubs) - self.num_hubs)
            for hub in hubs_to_close:
                y[hub] = False
        elif len(open_hubs) < self.num_hubs:
            hubs_to_open = random.sample(closed_hubs, self.num_hubs - len(open_hubs))
            for hub in hubs_to_open:
                y[hub] = True
        open_hubs = [i for i, hub_open in enumerate(y) if hub_open]

        # Assign customers only to open hubs
        z = [assigned_hub if y[assigned_hub] else random.choice(open_hubs) for assigned_hub in z]

        return y, z

    def mutate(self, solution):
        """Apply mutation to a solution while maintaining distinct hubs and valid assignments."""
        y, z = solution
        y = y.copy()
        z = z.copy()

        # Ensure there are exactly `num_hubs` open hubs
        open_hubs = [i for i, hub_open in enumerate(y) if hub_open]
        closed_hubs = [i for i, hub_open in enumerate(y) if not hub_open]

        # Swap an open hub with a closed hub
        if random.random() < self.mutation_rate and open_hubs and closed_hubs:
            hub_to_close = random.choice(open_hubs)
            hub_to_open = random.choice(closed_hubs)

            # Perform the swap
            y[hub_to_close] = False
            y[hub_to_open] = True

            # Update the list of open hubs
            open_hubs.remove(hub_to_close)
            open_hubs.append(hub_to_open)

            # Reassign customers connected to the closed hub
            z = [random.choice(open_hubs) if assigned_hub == hub_to_close else assigned_hub for assigned_hub in z]

        # Mutate customer assignments
        if random.random() < self.mutation_rate:
            for i in range(self.num_nodes):
                if random.random() < self.mutation_rate:
                    z[i] = random.choice(open_hubs)

        # Validate the solution
        return self.validate_solution(y, z)

    def select_parents(self, fitness_scores):
        """Select parents using roulette-wheel selection."""
        # Since lower cost is better, invert the scores
        inverted_scores = [1 / (score + 1e-6) for score in fitness_scores]  # Add a small value to avoid division by zero
        total_fitness = sum(inverted_scores)
        if total_fitness == 0:
            probabilities = [1 / len(fitness_scores)] * len(fitness_scores)
        else:
            probabilities = [f / total_fitness for f in inverted_scores]
        return random.choices(self.population, weights=probabilities, k=2)

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        y1, z1 = parent1
        y2, z2 = parent2

        # Combine hubs from both parents
        hubs1 = [i for i, hub_open in enumerate(y1) if hub_open]
        hubs2 = [i for i, hub_open in enumerate(y2) if hub_open]
        combined_hubs = list(set(hubs1 + hubs2))

        # Ensure that we have enough hubs to choose from
        if len(combined_hubs) < self.num_hubs:
            combined_hubs += random.sample([i for i in range(self.num_nodes) if i not in combined_hubs], self.num_hubs - len(combined_hubs))

        # Randomly select num_hubs hubs for each child
        child1_hubs = random.sample(combined_hubs, self.num_hubs)
        child2_hubs = random.sample(combined_hubs, self.num_hubs)

        # Create y for each child
        child1_y = [False] * self.num_nodes
        child2_y = [False] * self.num_nodes
        for hub in child1_hubs:
            child1_y[hub] = True
        for hub in child2_hubs:
            child2_y[hub] = True

        # For customer assignments, perform crossover
        crossover_point = random.randint(1, self.num_nodes - 1)
        child1_z = z1[:crossover_point] + z2[crossover_point:]
        child2_z = z2[:crossover_point] + z1[crossover_point:]

        # Validate children
        return self.validate_solution(child1_y, child1_z), self.validate_solution(child2_y, child2_z)

    def run(self):
        """Run the genetic algorithm."""
        self.initialize_population()

        for generation in range(self.num_generations):
            # Evaluate fitness of each solution
            fitness_scores = [self.evaluate_solution(y, z) for y, z in self.population]

            # Update the best solution
            for idx, cost in enumerate(fitness_scores):
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = self.population[idx]
                    self.best_hub_selection = self.population[idx][0]
                    print(f"Generation {generation}: Best Cost = {self.best_cost}")

            # Generate next generation
            new_population = []
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.select_parents(fitness_scores)
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])



            self.population = new_population[:self.pop_size]

        return  self.best_solution[0],self.best_solution[1], self.best_cost
