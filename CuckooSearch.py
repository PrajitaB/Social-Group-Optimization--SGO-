# Cuckoo Search (CS) Algorithm

import numpy as np
from math import gamma

# Input instructions
pop = int(input("Enter the Number of Population : "))
dim = int(input("Enter the Number of Dimension : "))
r1 = int(input("Enter the Range : "))
r2 = int(input())
itr = int(input("Enter the Number of Iteration : "))
beta = 1.5

# Define Objective Function (Sphere Function)
def objective_Fun(x):
    return np.sum(x**2)

# Initialize Population
def initial_Population(pop, dim, r1, r2):
    return np.random.uniform(r1, r2, (pop, dim))

# Calculate Levy Flight
def Levy_Flight(beta, dim):
    sigma = ((gamma(1+beta)*np.sin(np.pi*beta/2)/(gamma((1+beta)/2)*beta*2**((beta-1)/2))))**(1/beta)
    u = np.random.normal(0, sigma, size=dim)
    v = np.random.normal(0, 1, size=dim)
    step = u / np.abs(v) ** (1/beta)
    return step

# Cuckoo Search Algorithm
def CS(objective_Fun, pop, dim, itr, pa = 0.25):
    population = initial_Population(pop, dim, r1, r2)
    fitness = np.array([objective_Fun(nest) for nest in population])
    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    # Main Loop
    for i in range(itr):

        # Generate New Solution by Levy Flight
        new_population = np.empty_like(population)
        for j, nest in enumerate(population):
            Step_size = Levy_Flight(beta, dim)
            Step_Direction = np.random.uniform(-1, 1, size=dim)
            new_nest = nest + Step_Direction * 0.01 * Step_size * (nest - best_solution)
            new_population[j] = new_nest

            # Check Bounds [r1, r2]
            new_population[j]  = np.clip(new_population[j], r1, r2)

        # Calculate New Solution Fitness
        new_fitness = np.array([objective_Fun(nest) for nest in new_population])

        # Compare & Replace The Solutions
        replace_soln = np.where(new_fitness < fitness)[0]
        population[replace_soln] = new_population[replace_soln]
        fitness[replace_soln] = new_fitness[replace_soln]

        # Update Best Solution
        current_best_index = np.argmin(fitness)
        if fitness[current_best_index] < best_fitness :
            best_solution = population[current_best_index]
            best_fitness = fitness[current_best_index]

        # Abandon Egg With Probability (pa) and lay new eggs
        abandon_egg = int(pa * pop)
        abandon_soln = np.random.choice(pop, size=abandon_egg, replace=False)
        population[abandon_soln] = initial_Population(abandon_egg, dim, r1, r2)
        fitness[abandon_soln] = np.array([objective_Fun(nest) for nest in population[abandon_soln]])

        print(f"Iteration {i+1} : GBest = {best_fitness}")

    return best_solution, best_fitness

best_solution, best_fitness = CS(objective_Fun, pop, dim, itr, pa = 0.25)
print("Best Solution : ", best_solution)
print("Best Fitness : ", best_fitness)