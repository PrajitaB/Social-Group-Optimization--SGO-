# SOCIAL GROUP OPTIMISATION

import random
import numpy as np

# Input instructions
pop = int(input("Enter the Number of Population : "))
dim = int(input("Enter the Number of Dimension : "))
r1 = int(input("Enter the Range : "))
r2 = int(input())
itr = int(input("Enter the Number of Iteration : "))
constant = 0.5

# 2. Update population with f(x) column
def calculate_fx(row):
    return sum(x for x in row)     # Objective Function (Summation)
def fx_col_pop(matrix):
    modified_pop_fx = np.zeros((matrix.shape[0], matrix.shape[1] + 1))
    modified_pop_fx[:, :-1] = matrix
    for i in range(matrix.shape[0]):
        modified_pop_fx[i][-1] = calculate_fx(matrix[i])
    return modified_pop_fx

# 1. Random population generation
initial_pop = np.random.uniform(r1, r2, size=(pop, dim))
for i in range(pop):
    while initial_pop[i].sum() < 0:
        initial_pop[i] = np.random.uniform(r1, r2, size=(dim,))
np.set_printoptions(precision=2)
print("\nInitial Population :")
print(initial_pop)
initial_fx_pop = fx_col_pop(initial_pop)
experimental_pop = initial_pop.copy()

# 3. Fitness Function to Find gbest (Population Fitness)

def find_gbest(matrix, col):
    gbest = matrix[0][col]
    row = 0
    for i in range(1, len(matrix)):
        if matrix[i][col] < gbest:
            gbest = matrix[i][col]
            row = i
    return gbest, row
gbest0, row = find_gbest(initial_fx_pop, dim)
print(f"gbest for 0th Iteration (Initialization) : {gbest0}")

for count in range(itr):
    print(f"ITERATION : {count + 1}")

    # 4. Phase I : Improving

    z = 0
    c = 0
    while z == 0:
        c = c + 1

        def improving_phase(experimental_pop, pop, dim):
            modified_pop = np.copy(experimental_pop)
            for i in range(pop):
                for j in range(dim):
                    rdm = random.random()
                    modified_pop[i, j] = constant * experimental_pop[i, j] + rdm * (
                            experimental_pop[row, j] - experimental_pop[i, j])
            return modified_pop

        modified_pop = improving_phase(experimental_pop, pop, dim)
        modified_fx_pop = fx_col_pop(modified_pop)
        gbest_ip, row = find_gbest(modified_fx_pop, dim)
        if gbest_ip < gbest0:
            print(f"Improving Phase Accepted after {c} Attempt and gbest at This Stage : {gbest_ip}")
            z = 1
        else:
            print("Improving Phase Rejected.")
            z = 0

    # 5. Phase II : Acquiring

    z = 0
    c = 0
    while z == 0:
        c = c + 1

        def acquring_phase(modified_pop, pop, dim):
            modified_popp = np.copy(modified_pop)
            for i in range(pop):
                other_Xr = [index for index in range(pop) if index != i]
                Xr_row = random.choice(other_Xr)
                Xr = random.choice(modified_pop[Xr_row])
                fXr = (lambda row: sum(x ** 2 for x in row))(modified_pop[Xr_row])
                fXi = (lambda row: sum(x ** 2 for x in row))(modified_pop[i])
                if fXi < fXr:
                    for j in range(dim):
                        rdm1 = random.random()
                        rdm2 = random.random()
                        modified_popp[i, j] = modified_pop[i, j] + rdm1 * (
                                modified_pop[i, j] - modified_pop[Xr_row, j]) + rdm2 * (
                                                      modified_pop[row, j] - modified_pop[i, j])
                else:
                    for j in range(dim):
                        rdm1 = random.random()
                        rdm2 = random.random()
                        modified_popp[i, j] = modified_pop[i, j] + rdm1 * (
                                modified_pop[Xr_row, j] - modified_pop[i, j]) + rdm2 * (
                                                      modified_pop[row, j] - modified_pop[i, j])
            return modified_popp

        modified_popp = acquring_phase(modified_pop, pop, dim)
        modified_fx_popp = fx_col_pop(modified_popp)
        gbest_ap, row = find_gbest(modified_fx_popp, dim)
        if gbest_ap < gbest_ip:
            print(f"Acquiring Phase Accepted after {c} Attempt and gbest at This Stage : {gbest_ap}")
            z = 1
        else:
            print("Acquiring Phase Rejected.")
            z = 0

    print(f"gbest at Iteration {count + 1} : {gbest_ap}")
    experimental_pop = modified_popp