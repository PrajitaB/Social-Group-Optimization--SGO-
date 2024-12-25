# Artificial Bee Colony (Sphere)

import random

# Input instructions
pop = int(input("Enter the Number of Population: "))
dim = int(input("Enter the Number of Dimensions: "))
r1 = int(input("Enter the Range : "))
r2 = int(input())
itr = int(input("Enter the Number of Iterations: "))

# Define Objective Function (Sphere)
def sphere(x):
    return sum(i ** 2 for i in x)

# Employed Phase
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

# Probability Factor : P(i) = (1 / (1 + f(x))) / (sum(1...n) 1 / (1 + f(x(n))))
def P(X, f):
    P = []
    sP = sum([(1 / (1 + f(i))) for i in X])
    for i in range(len(X)):
        P.append((1 / (1 + f(X[i]))) / sP)
    return P

# Onlooker Phase
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

# Scout Phase
def SBee(X, Trials, bounds, limit=3):
    for i in range(len(X)):
        if Trials[i] > limit:
            Trials[i] = 0
            X[i] = [(bounds[j][0] + random.uniform(0, 1) * (bounds[j][1] - bounds[j][0])) for j in range(len(X[0]))]
    return X

# ABC Main Loop
def ABC(dims, bounds, f, limit, pop, runs):
    bounds = [(bounds[0], bounds[1]) for _ in range(dims)]
    X = [[bounds[j][0] + random.uniform(0, 1) * (bounds[j][1] - bounds[j][0]) for j in range(dims)] for _ in range(pop)]
    Trials = [0 for _ in range(pop)]

    fx = [f(i) for i in X]
    GBest = min(fx)
    print(f"Iteration 0: GBest = {GBest}")

    for iteration in range(1, runs + 1):
        X, Trials = EBee(X, f, Trials)
        X, Trials = OBee(X, f, Trials)
        X = SBee(X, Trials, bounds, limit)
        fx = [f(i) for i in X]
        current_GBest = min(fx)
        if current_GBest < GBest:
            GBest = current_GBest
        print(f"Iteration {iteration}: GBest = {GBest}")

    I = fx.index(GBest)
    print(f"Best Solution : {X[I]}")
    print(f"Best Fitness : {GBest}")
    return GBest

ABC(dim, (r1, r2), sphere, limit=50, pop=pop, runs=itr)