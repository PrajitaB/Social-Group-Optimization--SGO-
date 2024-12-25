# FIREFLY ALGORITHM
import numpy as np

# 1. Define Objective Function (Sphere Here)
def Objective_Function(x):
    return sum(x**2)

# 2. Define Firefly Algorithm
def FireFly(Obj_Fun, r1, r2, pop, dim, itr, alpha = 0.2, gamma = 1, beta0 = 2, alpha_damp = 0.98):

    # Initialize Best Solution
    Best_Sol = {"Position" : None, "Cost" : np.inf}

    # Store Best Fitness
    Best_Cost = np.zeros(itr)

    # Calculate Maximum Distance (Based on Fitness Value)
    dmax = np.sqrt(dim) * np.sum((r2 - r1)**2)

    # Initialize Fireflies Population
    fireflies = [{"Position" : np.random.uniform(r1, r2, dim), "Cost" : None} for _ in range(pop)]

    # Calculate Fitness Values
    for i in range(pop):
        fireflies[i]["Cost"] = Obj_Fun(fireflies[i]["Position"])

    # Check Infinite Cost
    if any(not np.isfinite(firefly["Cost"]) for firefly in fireflies):
        print("Solution Has Infinite Cost.")
        return Best_Sol, Best_Cost

    # Firefly Main Loop
    for it in range(itr):
        New_Pop = [{"Position" : None, "Cost" : None} for _ in range(pop)]
        for i in range(pop):
            for j in range(pop):
                if fireflies[j]["Cost"] < fireflies[i]["Cost"] :
                    distance = np.linalg.norm(fireflies[i]["Position"] - fireflies[j]["Position"])
                    beta = beta0 * np.exp(-gamma * (distance / dmax)**2)
                    movement_vector = alpha * (np.random.rand(dim) - 0.5) * (r2 - r1)     # This is an Exploration which is generated within the SearchSpace Boundary

                    New_Sol = {"Position" : fireflies[i]["Position"] + beta * (fireflies[j]["Position"] - fireflies[i]["Position"]) + movement_vector, "Cost" : None}
                    New_Sol["Position"] = np.maximum(New_Sol["Position"], r1)
                    New_Sol["Position"] = np.minimum(New_Sol["Position"], r2)
                    New_Sol["Cost"] = Obj_Fun(New_Sol["Position"])

                    if New_Sol["Cost"] < fireflies[i]["Cost"]:
                        fireflies[i] = New_Sol
                        if fireflies[i]["Cost"] < Best_Sol["Cost"]:
                            Best_Sol = fireflies[i].copy()

                        # Firefly Replacement in New Population
                        New_Pop[i] = fireflies[i].copy()

        # Merge Population
        population = sorted([individual for individual in fireflies + New_Pop if individual["Cost"] is not None], key = lambda x : x["Cost"])
        population = population[:pop]

        # Sort Population (Based on Fitness Value)
        fireflies = sorted(population, key = lambda x : x["Cost"])

        alpha *= alpha_damp

        # Store Best Solution
        Best_Cost[it] = Best_Sol["Cost"]

        # Display Iteration wise Solution
        print(f"Best Cost at Iteration {it+1} : {Best_Cost[it]}")

        # Display Best Solution
        print(f"Best Solution at Iteration {it+1} : {Best_Sol['Position']} (Cost : {Best_Sol['Cost']}")

    return Best_Sol, Best_Cost


# 3. Initialize Parameters
pop = int(input("Enter the Size of the Population: "))
dim = int(input("Enter the Number of Dimension : "))
r1 = int(input("Enter the Range : "))
r2 = int(input())
itr = int(input("Enter the Number of Iterations: "))

alpha = 0.1          # Mutation Coefficient
beta0 = 1             # Attraction Coefficient
gamma = 1            # Light Absorption Coefficient
alpha_damp = 0.95    # Damping Ratio

# 4. Run Firefly Algorithm
Best_Sol, Best_Cost = FireFly(Objective_Function, r1, r2, pop, dim, itr, alpha, gamma, beta0, alpha_damp)

# Display Best Solution and Fitness
print("\nBest Solution : ", Best_Sol["Position"])
print("Best Fitness Value : ", Best_Sol["Cost"])