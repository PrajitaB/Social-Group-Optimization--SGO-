# TEACHING LEARNING BASED OPTIMIZATION ALGORITHM

import numpy as np

# Objective Function (Sphere)
def objective_Fun(x):
    return np.sum(x**2)

# TLBO Parameters
nPop = int(input("Enter the Number of Population: "))
nDim = int(input("Enter the Number of Dimensions: "))
r1 = float(input("Enter the lower bound of the range: "))
r2 = float(input("Enter the upper bound of the range: "))
itr = int(input("Enter the Number of Iterations: "))

# Individual Class
class Individual:
    def __init__(self):
        self.Position = None
        self.Cost = None

# Initialize Population
def initial_Population(pop_size, dim, r1, r2):
    population = np.empty(pop_size, dtype=object)
    for i in range(pop_size):
        individual = Individual()
        individual.Position = np.random.uniform(r1, r2, dim)
        individual.Cost = objective_Fun(individual.Position)
        population[i] = individual
    return population

pop = initial_Population(nPop, nDim, r1, r2)

# Initialize Best Solution
BestSol = Individual()
BestSol.Cost = np.inf

# Update Best Solution
for i in range(nPop):
    if pop[i].Cost < BestSol.Cost:
        BestSol.Position = np.copy(pop[i].Position)
        BestSol.Cost = pop[i].Cost

# Initialize Best Cost Record
BestCosts = np.zeros(itr)
BestCosts[0] = BestSol.Cost

print('Iteration 0 : GBest =', BestCosts[0])

# TLBO Main Loop
for it in range(1, itr+1):

    # Calculate Population Mean
    Mean = np.zeros(nDim)
    for i in range(nPop):
        Mean += pop[i].Position
    Mean /= nPop

    # Select Teacher
    Teacher = pop[0]
    for i in range(1, nPop):
        if pop[i].Cost < Teacher.Cost:
            Teacher = pop[i]

    # Teacher Phase
    for i in range(nPop):

        # Create Empty Solution
        newsol = Individual()

        # Teaching Factor
        TF = np.round(1 + np.random.random())

        # Teaching (moving towards teacher)
        newsol.Position = pop[i].Position + np.random.random() * (Teacher.Position - TF * Mean)

        # Clipping
        newsol.Position = np.clip(newsol.Position, r1, r2)

        # Evaluation
        newsol.Cost = objective_Fun(newsol.Position)

        # Comparison
        if newsol.Cost < pop[i].Cost:
            pop[i] = newsol
            if pop[i].Cost < BestSol.Cost:
                BestSol.Position = np.copy(pop[i].Position)
                BestSol.Cost = pop[i].Cost

    # Learner Phase
    for i in range(nPop):
        A = np.arange(nPop)
        A = np.delete(A, i)
        j = np.random.choice(A)
        Step = pop[i].Position - pop[j].Position
        if pop[j].Cost < pop[i].Cost:
            Step = -Step

        # Create Empty Solution
        newsol = Individual()

        # Learning (moving towards a better individual)
        newsol.Position = pop[i].Position + np.random.random() * Step

        # Clipping
        newsol.Position = np.clip(newsol.Position, r1, r2)

        # Evaluation
        newsol.Cost = objective_Fun(newsol.Position)

        # Comparison
        if newsol.Cost < pop[i].Cost:
            pop[i] = newsol
            if pop[i].Cost < BestSol.Cost:
                BestSol.Position = np.copy(pop[i].Position)
                BestSol.Cost = pop[i].Cost

    # Store Record for Current Iteration
    BestCosts[it-1] = BestSol.Cost

    # Show Iteration Information
    print('Iteration', it, ': GBest =', BestCosts[it-1])