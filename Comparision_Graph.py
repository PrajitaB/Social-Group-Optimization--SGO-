# SGO COMPARISION GRAPH (SPHERE FUNCTION : f(x) = Xi^2, 1<i<n, here f(x) = X1^2 + X2^2)
import random
import numpy as np
from math import gamma
import matplotlib.pyplot as plt

# Input instructions
pop = int(input("Enter the Number of Population : "))
dim = int(input("Enter the Number of Dimension : "))
r1 = int(input("Enter the Range : "))
r2 = int(input())
itr = int(input("Enter the Number of Iteration : "))


# || CUCKOO SEARCH ALGORITHM ||

def CuckooSearch_SPHERE(pop, dim, r1, r2, itr, beta=1.5):

    def objective_Fun(x):
        return np.sum(x ** 2)

    def initial_Population(pop, dim, r1, r2):
        return np.random.uniform(r1, r2, (pop, dim))

    def Levy_Flight(beta, dim):
        sigma = ((gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                    gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def CS(objective_Fun, pop, dim, itr, pa=0.25):
        population = initial_Population(pop, dim, r1, r2)
        fitness = np.array([objective_Fun(nest) for nest in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        GBest_values = [best_fitness]

        for i in range(itr):
            new_population = np.zeros_like(population)
            for j in range(pop):
                step_size = 0.01 * Levy_Flight(beta, dim)
                new_nest = population[j] + step_size * (population[j] - best_solution)
                new_fitness = objective_Fun(new_nest)
                if new_fitness < fitness[j]:
                    new_population[j] = new_nest
                    fitness[j] = new_fitness
                else:
                    new_population[j] = population[j]

            for j in range(pop):
                if np.random.rand() < pa:
                    new_population[j] = initial_Population(1, dim, r1, r2)

            fitness = np.array([objective_Fun(nest) for nest in new_population])
            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_solution = new_population[np.argmin(fitness)]
                best_fitness = current_best_fitness

            GBest_values.append(best_fitness)
            population = new_population

        return GBest_values

    return CS(objective_Fun, pop, dim, itr)

fitness_values_CuckooSearch = CuckooSearch_SPHERE(pop, dim, r1, r2, itr)

print("\n|| CUCKOO SEARCH ||")
for idx, fitness in enumerate(fitness_values_CuckooSearch):
    print(f"Iteration {idx}: GBest = {fitness}")




# || ARTIFICIAL BEE COLONY ALGORITHM (ABC) ||

def ABC_SPHERE(pop, dim, r1, r2, itr):

    def sphere(x):
        return sum(i ** 2 for i in x)

    def EBee(X, f, Trials):
        for i in range(len(X)):
            V = []
            R = X.copy()
            R.remove(X[i])
            r = random.choice(R)
            for j in range(len(X[0])):
                V.append(X[i][j] + random.uniform(-1, 1) * (X[i][j] - r[j]))
            if f(V) < f(X[i]):
                X[i] = V
                Trials[i] = 0
            else:
                Trials[i] += 1
        return X, Trials

    def P(X, f):
        P = []
        sP = sum([(1 / (1 + f(i))) for i in X])
        for i in range(len(X)):
            P.append((1 / (1 + f(X[i]))) / sP)
        return P

    def OBee(X, f, Trials):
        Pi = P(X, f)
        for i in range(len(X)):
            if random.random() < Pi[i]:
                V = []
                R = X.copy()
                R.remove(X[i])
                r = random.choice(R)
                for j in range(len(X[0])):
                    V.append(X[i][j] + random.uniform(-1, 1) * (X[i][j] - r[j]))
                if f(V) < f(X[i]):
                    X[i] = V
                    Trials[i] = 0
                else:
                    Trials[i] += 1
        return X, Trials

    def SBee(X, Trials, bounds, limit=50):
        for i in range(len(X)):
            if Trials[i] > limit:
                Trials[i] = 0
                X[i] = [(bounds[j][0] + random.uniform(0, 1) * (bounds[j][1] - bounds[j][0])) for j in range(len(X[0]))]
        return X

    def ABC(dims, bounds, f, limit, pop, runs):
        bounds = [(bounds[0], bounds[1]) for _ in range(dims)]
        X = [[bounds[j][0] + random.uniform(0, 1) * (bounds[j][1] - bounds[j][0]) for j in range(dims)] for _ in range(pop)]
        Trials = [0 for _ in range(pop)]

        fx = [f(i) for i in X]
        GBest = min(fx)
        GBest_values = [GBest]

        for iteration in range(1, runs + 1):
            X, Trials = EBee(X, f, Trials)
            X, Trials = OBee(X, f, Trials)
            X = SBee(X, Trials, bounds, limit)
            fx = [f(i) for i in X]
            current_GBest = min(fx)
            if current_GBest < GBest:
                GBest = current_GBest
            GBest_values.append(GBest)

        return GBest_values

    return ABC(dim, (r1, r2), sphere, limit=50, pop=pop, runs=itr)


fitness_values_ABC = ABC_SPHERE(pop, dim, r1, r2, itr)

print("\n|| ARTIFICIAL BEE COLONY ||")
for idx, fitness in enumerate(fitness_values_ABC):
    print(f"Iteration {idx}: GBest = {fitness}")

# SGO COMPARISON GRAPH (SPHERE FUNCTION: f(x) = Σ(Xi^2))

# || SOCIAL GROUP OPTIMIZATION ||
# || PARTICLE SWARM OPTIMIZATION ||
# || FIREFLY ALGORITHM ||

# Remaining algorithms (SGO, PSO, Firefly) go here...
# SGO COMPARISION GRAPH (SPHERE FUNCTION : f(x) = Xi^2, 1<i<n, here f(x) = X1^2 + X2^2)


# || SOCIAL GROUP OPTIMIZATION ||

def SGO_SPHERE(pop, dim, r1, r2, itr):
    constant = 0.5

    # Sphere function: Objective Function
    def calculate_fx(row):
        return sum(x ** 2 for x in row)

    # Update population with f(x) column
    def fx_col_pop(matrix):
        modified_pop_fx = np.zeros((matrix.shape[0], matrix.shape[1] + 1))
        modified_pop_fx[:, :-1] = matrix
        for i in range(matrix.shape[0]):
            modified_pop_fx[i][-1] = calculate_fx(matrix[i])
        return modified_pop_fx

    # Random population generation
    initial_pop = np.random.uniform(r1, r2, size=(pop, dim))
    initial_fx_pop = fx_col_pop(initial_pop)
    experimental_pop = initial_pop.copy()

    # Fitness Function to find gbest (Population Fitness)
    def find_gbest(matrix, col):
        gbest = matrix[0][col]
        row = 0
        for i in range(1, len(matrix)):
            if matrix[i][col] < gbest:
                gbest = matrix[i][col]
                row = i
        return gbest, row

    gbest0, row = find_gbest(initial_fx_pop, dim)
    gbest_values = [gbest0]

    for count in range(itr):
        # Phase I: Improving
        z = 0
        while z == 0:
            def improving_phase(experimental_pop, pop, dim):
                modified_pop = np.copy(experimental_pop)
                for i in range(pop):
                    for j in range(dim):
                        rdm = random.random()
                        modified_pop[i, j] = constant * experimental_pop[i, j] + rdm * (experimental_pop[row, j] - experimental_pop[i, j])
                return modified_pop

            modified_pop = improving_phase(experimental_pop, pop, dim)
            modified_fx_pop = fx_col_pop(modified_pop)
            gbest_ip, row = find_gbest(modified_fx_pop, dim)
            if gbest_ip < gbest0:
                gbest0 = gbest_ip
                z = 1

        # Phase II: Acquiring
        z = 0
        while z == 0:
            def acquiring_phase(modified_pop, pop, dim):
                modified_popp = np.copy(modified_pop)
                for i in range(pop):
                    other_Xr = [index for index in range(pop) if index != i]
                    Xr_row = random.choice(other_Xr)
                    Xr = random.choice(modified_pop[Xr_row])
                    fXr = calculate_fx(modified_pop[Xr_row])
                    fXi = calculate_fx(modified_pop[i])
                    if fXi < fXr:
                        for j in range(dim):
                            rdm1 = random.random()
                            rdm2 = random.random()
                            modified_popp[i, j] = modified_pop[i, j] + rdm1 * (modified_pop[i, j] - modified_pop[Xr_row, j]) + rdm2 * (modified_pop[row, j] - modified_pop[i, j])
                    else:
                        for j in range(dim):
                            rdm1 = random.random()
                            rdm2 = random.random()
                            modified_popp[i, j] = modified_pop[i, j] + rdm1 * (modified_pop[Xr_row, j] - modified_pop[i, j]) + rdm2 * (modified_pop[row, j] - modified_pop[i, j])
                return modified_popp

            modified_popp = acquiring_phase(modified_pop, pop, dim)
            modified_fx_popp = fx_col_pop(modified_popp)
            gbest_ap, row = find_gbest(modified_fx_popp, dim)
            if gbest_ap < gbest_ip:
                gbest_ip = gbest_ap
                z = 1

        gbest_values.append(gbest_ap)
        experimental_pop = modified_popp

    return gbest_values

fitness_values_SGO = SGO_SPHERE(pop, dim, r1, r2, itr)
print("\n|| SOCIAL GROUP OPTIMIZATION ||")
for idx, fitness in enumerate(fitness_values_SGO):
    print(f"Iteration {idx}: GBest = {fitness}")





# || PARTICLE SWARM OPTIMIZATION ||

# Define Class Particle
class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.zeros_like(position)
        self.best_position = position
        self.best_fitness = float('inf')

def PSO_SPHERE(pop, dim, r1, r2, itr):
    def sphere_function(x):
        return np.sum(x**2)

    swarm_best_position = None
    swarm_best_fitness = float('inf')
    particles = []

    # Initialization of Position of Each Particle
    for i in range(pop):
        position = np.random.uniform(r1, r2, dim)
        particle = Particle(position)
        particles.append(particle)

        # Fitness Calculation
        fitness = sphere_function(position)
        if fitness < swarm_best_fitness:
            swarm_best_fitness = fitness
            swarm_best_position = position

            particle.best_position = position
            particle.best_fitness = fitness

    gbest_values = [swarm_best_fitness]

    # PSO Main Loop
    for j in range(itr):
        for particle in particles:
            # PSO parameters
            w = 0.8
            c1 = 1.2
            c2 = 1.2

            r1 = random.random()
            r2 = random.random()

            # Velocity Calculation
            particle.velocity = (w * particle.velocity + c1 * r1 * (particle.best_position - particle.position) + c2 * r2 * (swarm_best_position - particle.position))

            # New Position
            particle.position += particle.velocity

            # Fitness Calculation
            fitness = sphere_function(particle.position)

            # Update PBest
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            # Update GBest
            if fitness < swarm_best_fitness:
                swarm_best_fitness = fitness
                swarm_best_position = particle.position

        gbest_values.append(swarm_best_fitness)

    return gbest_values

fitness_values_PSO = PSO_SPHERE(pop, dim, r1, r2, itr)
print("\n|| PARTICLE SWARM OPTIMIZATION ||")
for idx, fitness in enumerate(fitness_values_PSO):
    print(f"Iteration {idx}: GBest = {fitness}")





# || FIREFLY ALGORITHM ||

alpha = 0.1  # Mutation Coefficient
beta0 = 1  # Attraction Coefficient
gamma = 1  # Light Absorption Coefficient
alpha_damp = 0.95  # Damping Ratio

def Fireflies_SPHERE(pop, dim, r1, r2, itr, alpha=0.2, gamma=1, beta0=2, alpha_damp=0.98):

    # Define Objective Function (Sphere Function)
    def Objective_Function(x):
        return sum(x ** 2)

    # Initialize Best Solution
    Best_Sol = {"Position": None, "Cost": np.inf}

    # Store Best Fitness
    Best_Cost = [np.inf]  # Initialize with infinite for iteration 0

    # Calculate Maximum Distance (Based on Fitness Value)
    dmax = np.sqrt(dim) * np.sum((r2 - r1) ** 2)

    # Initialize Fireflies Population
    fireflies = [{"Position": np.random.uniform(r1, r2, dim), "Cost": None} for _ in range(pop)]

    # Calculate Fitness Values
    for i in range(pop):
        fireflies[i]["Cost"] = Objective_Function(fireflies[i]["Position"])

    # Firefly Main Loop
    for it in range(1, itr + 1):  # Start iterations from 1
        New_Pop = [{"Position": None, "Cost": None} for _ in range(pop)]
        for i in range(pop):
            for j in range(pop):
                if fireflies[j]["Cost"] < fireflies[i]["Cost"]:
                    distance = np.linalg.norm(fireflies[i]["Position"] - fireflies[j]["Position"])
                    beta = beta0 * np.exp(-gamma * (distance / dmax) ** 2)
                    movement_vector = alpha * (np.random.rand(dim) - 0.5) * (r2 - r1)  # Exploration vector

                    New_Sol = {"Position": fireflies[i]["Position"] + beta * (fireflies[j]["Position"] - fireflies[i]["Position"]) + movement_vector, "Cost": None}
                    New_Sol["Position"] = np.maximum(New_Sol["Position"], r1)
                    New_Sol["Position"] = np.minimum(New_Sol["Position"], r2)
                    New_Sol["Cost"] = Objective_Function(New_Sol["Position"])

                    if New_Sol["Cost"] < fireflies[i]["Cost"]:
                        fireflies[i] = New_Sol
                        if fireflies[i]["Cost"] < Best_Sol["Cost"]:
                            Best_Sol = fireflies[i].copy()
                        # Firefly Replacement in New Population
                        New_Pop[i] = fireflies[i].copy()

        # Merge Population
        population = sorted([individual for individual in fireflies + New_Pop if individual["Cost"] is not None], key=lambda x: x["Cost"])
        population = population[:pop]

        # Sort Population (Based on Fitness Value)
        fireflies = sorted(population, key=lambda x: x["Cost"])

        alpha *= alpha_damp

        # Store Best Solution
        Best_Cost.append(Best_Sol["Cost"])

    return Best_Cost

fitness_values_FIREFLY = Fireflies_SPHERE(pop, dim, r1, r2, itr, alpha, gamma, beta0, alpha_damp)

print("\n|| FIREFLY ALGORITHM ||")
for idx, fitness in enumerate(fitness_values_FIREFLY):
    print(f"Iteration {idx}: GBest = {fitness}")


# || TEACHING LEARNING BASED ALGORITHM ||
def TLBO_SPHERE(nPop, nDim, r1, r2, itr):
    def objective_Fun(x):
        return np.sum(x ** 2)

    class Individual:
        def __init__(self):
            self.Position = np.random.uniform(r1, r2, nDim)
            self.Cost = objective_Fun(self.Position)

    pop = [Individual() for _ in range(nPop)]
    BestSol = min(pop, key=lambda ind: ind.Cost)

    BestCosts = np.zeros(itr + 1)
    BestCosts[0] = BestSol.Cost

    print(f"Iteration 0: GBest = {BestCosts[0]}")

    for it in range(1, itr + 1):
        Mean = np.mean([ind.Position for ind in pop], axis=0)
        Teacher = min(pop, key=lambda ind: ind.Cost)

        for i in range(nPop):
            newsol = Individual()
            TF = np.round(1 + np.random.random())
            newsol.Position = pop[i].Position + np.random.random() * (Teacher.Position - TF * Mean)
            newsol.Position = np.clip(newsol.Position, r1, r2)
            newsol.Cost = objective_Fun(newsol.Position)

            if newsol.Cost < pop[i].Cost:
                pop[i] = newsol
                if pop[i].Cost < BestSol.Cost:
                    BestSol = pop[i]

        for i in range(nPop):
            j = np.random.choice([idx for idx in range(nPop) if idx != i])
            Step = pop[i].Position - pop[j].Position
            if pop[j].Cost < pop[i].Cost:
                Step = -Step

            newsol = Individual()
            newsol.Position = pop[i].Position + np.random.random() * Step
            newsol.Position = np.clip(newsol.Position, r1, r2)
            newsol.Cost = objective_Fun(newsol.Position)

            if newsol.Cost < pop[i].Cost:
                pop[i] = newsol
                if pop[i].Cost < BestSol.Cost:
                    BestSol = pop[i]

        BestCosts[it] = BestSol.Cost
        print(f"Iteration {it}: GBest = {BestCosts[it]}")

    return BestCosts
fitness_values_TLBO = TLBO_SPHERE(pop, dim, r1, r2, itr)



# || COMPARISON GRAPH ||

iterations = list(range(itr + 1))

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 7

smoothed_SGO = moving_average(fitness_values_SGO, window_size)
smoothed_PSO = moving_average(fitness_values_PSO, window_size)
smoothed_FIREFLY = moving_average(fitness_values_FIREFLY, window_size)
smoothed_ABC = moving_average(fitness_values_ABC, window_size)
smoothed_TLBO = moving_average(fitness_values_TLBO, window_size)
smoothed_CuckooSearch = moving_average(fitness_values_CuckooSearch, window_size)

smoothed_iterations = iterations[:len(smoothed_SGO)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(smoothed_iterations, smoothed_SGO, label='SGO')
plt.plot(smoothed_iterations, smoothed_PSO, label='PSO')
plt.plot(smoothed_iterations, smoothed_FIREFLY, label='Firefly')
plt.plot(smoothed_iterations, smoothed_ABC, label='ABC')
plt.plot(smoothed_iterations, smoothed_TLBO, label='TLBO')
plt.plot(smoothed_iterations, smoothed_CuckooSearch, label='Cuckoo Search')

plt.title('Iteration vs Fitness Value')
plt.xlabel('Iteration')
plt.ylabel('Fitness Value')
plt.legend()
plt.grid(True)
plt.show()