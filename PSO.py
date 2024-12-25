# PARTICLE SWARM OPTIMIZATION

import random
import numpy as np

# Define Class Particles
class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.zeros_like(position)
        self.best_position = position
        self.best_fitness = float('inf')

def PSO(ObjF, pop, dim, r1, r2, itr):
    swarm_best_position = None
    swarm_best_fitness = float('inf')
    particles = []

    # Initialization of Position of Each Particle
    for i in range(pop):
        position = np.random.uniform(r1, r2, dim)
        particle = Particle(position)
        particles.append(particle)

        # Fitness Calculation
        fitness = ObjF(position)
        if fitness < swarm_best_fitness:
            swarm_best_fitness = fitness
            swarm_best_position = position

            particle.best_position = position
            particle.best_fitness = fitness

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
            fitness = ObjF(particle.position)

            # Update PBest
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            # Update GBest
            if fitness < swarm_best_fitness:
                swarm_best_fitness = fitness
                swarm_best_position = particle.position

        print(f"GBest at Iteration {j+1} : {swarm_best_fitness}")

    return swarm_best_position, swarm_best_fitness

# Parameters
pop = int(input("Enter the Size of the Population: "))
dim = int(input("Enter the Number of Dimension : "))
r1 = int(input("Enter the Range : "))
r2 = int(input())
itr = int(input("Enter the Number of Iterations: "))

# Define ObjFunctions (Sphere Here)
def F1(x):
    return np.sum(x**2)

Objective_Functions = {
    'F1': F1,
}

# Iteration Over ObjF using PSO
for FuncName, ObjF in Objective_Functions.items():
    best_position, best_fitness = PSO(ObjF, pop, dim, r1, r2, itr)
    print(f"Best Position: {best_position}")
    print(f"Best Fitness: {best_fitness}")