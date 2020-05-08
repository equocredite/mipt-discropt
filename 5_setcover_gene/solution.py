import numpy as np
from numpy.random import uniform
from numpy.random import shuffle
import time


start_time = time.time()


class SetCoverGeneSolver:
    def __init__(self, n_elems, n_sets, weights, inc_matrix, population_size=20, tol=0.01):
        self.n_elems = n_elems
        self.n_sets = n_sets
        self.weights = weights
        self.inc_matric = inc_matrix
        self.population_size = population_size
        self.tol = tol
        self.solution = None
        self.weight_order = None

    def get_weight(self, solution):
        return self.weights.dot(solution)

    def is_feasible(self, solution):
        return (self.inc_matric.dot(solution) >= np.ones(self.n_elems)).all()

    def crossover(self, a, b):
        w_a = self.get_weight(a)
        w_b = self.get_weight(b)
        threshold = w_b / (w_a + w_b)
        offspring = np.array([a[i] if uniform() <= threshold else b[i] for i in range(self.n_sets)])
        return self.make_feasible(offspring)

    def mutate(self, solution):
        mutant = np.array([gene if uniform() <= 1 / self.n_sets else 1 - gene for gene in solution])
        return self.make_feasible(mutant)

    # add sets greedily
    def make_feasible(self, base):
        solution = base.copy()
        if not self.is_feasible(solution):
            for set_id in self.weight_order:
                if solution[set_id] == 0:
                    solution[set_id] = 1
                    if uniform() < 1/10 and self.is_feasible(solution):
                        break
        return solution

    def create_initial_population(self):
        population = []
        base = np.array([1 if uniform() <= 2 / 3 else 0 for i in range(self.n_sets)])
        while len(population) < self.population_size:
            shuffle(base)
            population.append(self.make_feasible(base))
        return np.array(population)

    def run(self):
        self.weight_order = np.argsort(weights)
        population = self.create_initial_population()
        while True:
            offsprings = [self.crossover(population[i], population[j])
                          for i in range(self.population_size) for j in range(i)]
            mutants = [self.mutate(solution) for solution in population]
            candidates = np.concatenate((population, offsprings, mutants))
            candidate_weights = [self.get_weight(solution) for solution in candidates]
            new_population = candidates[np.argsort(candidate_weights)][:self.population_size]
            if time.time() - start_time > 29. or \
                    self.get_weight(new_population[0]) <= (1 + self.tol) * self.get_weight(population[0]):
                break
            population = new_population
        self.solution = population[0]

    def get_covering(self):
        covering = []
        for set_id, indicator in enumerate(self.solution):
            if indicator == 1:
                covering.append(set_id)
        return np.array(covering)


n_elems, n_sets = map(int, input().split())
weights = np.zeros(n_sets)
inc_matrix = np.zeros((n_elems, n_sets))

for set_id in range(n_sets):
    data = list(map(int, input().split()))
    weights[set_id] = data[0]
    for elem_id in data[1:]:
        inc_matrix[elem_id, set_id] = 1

solver = SetCoverGeneSolver(n_elems, n_sets, weights, inc_matrix)
solver.run()
for set_id in solver.get_covering():
    print(set_id + 1, end=' ')
