# Python3 program to create target string, starting from 
# random string using Genetic Algorithm 

import random
import numpy as np
import matplotlib.pyplot as plt
import time

POPULATION_SIZE = 1000
GENES = '''abcdefghijklmnopqrstuvwxyzčćđšžČĆĐŠŽABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890, .-;:_!"#%&/()=?@${[]}<>'''
TARGET = "Ja volim Operacijska istraživanja II <3"

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        return random.choice(GENES)

    @classmethod
    def generate_mutated_pool(cls, pool_size=1000):
        mutated_pool = [cls.mutated_genes() for _ in range(pool_size)]
        return np.array(mutated_pool, dtype=np.dtype('U1'))

    @classmethod
    def create_gnome(cls):
        return np.array([cls.mutated_genes() for _ in range(len(TARGET))], dtype=np.dtype('U1'))

    def mate(self, partner, mutated_pool):
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, partner.chromosome):
            prob = random.random()
            if prob < 0.45:
                child_chromosome.append(gp1)
            elif prob < 0.90:
                child_chromosome.append(gp2)
            else:
                child_chromosome.append(random.choice(mutated_pool))  # Using pre-generated mutated_pool
        return Individual(np.array(child_chromosome, dtype=np.dtype('U1')))

    def cal_fitness(self):
        return np.sum(self.chromosome != np.array(list(TARGET), dtype=np.dtype('U1')))

def main():
    generation = 1
    found = False
    population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]
    fitness_progress = []

    # Generate a pool of mutated genes
    mutated_pool = Individual.generate_mutated_pool()

    while not found:
        population = sorted(population, key=lambda x: x.fitness)

        if population[0].fitness <= 0:
            found = True
            break

        new_generation = []
        s = int((10 * POPULATION_SIZE) / 100)
        new_generation.extend(population[:s])

        s = int((90 * POPULATION_SIZE) / 100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2, mutated_pool)  # Pass the pre-generated mutated_pool
            new_generation.append(child)

        population = new_generation
        fitness_progress.append(population[0].fitness)

        print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")
        generation += 1

    print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")

    # Visualization using Matplotlib
    plt.plot(fitness_progress)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progression')
    plt.show()

if __name__ == '__main__':
    main()