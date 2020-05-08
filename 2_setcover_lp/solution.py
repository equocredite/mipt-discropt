import numpy as np
from numpy.random import uniform
from scipy.optimize import linprog


class RowCoverSolver:
    def __init__(self, n_cols, n_rows, weights, adj_matrix, rounds=40):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.weights = weights
        self.adj_matrix = adj_matrix
        self.rounds = rounds  # rounds number for randomized rounding
        self.solution = None

    def get_weight(self, x):
        return self.weights.dot(x)

    def is_feasible(self, x):
        return (self.adj_matrix.dot(x) >= np.ones(n_cols)).all()

    @staticmethod
    def round_randomized(x):
        return np.array([1 if uniform() <= xi else 0 for xi in x])

    def generate_ilp_solutions(self, lp_solution, n):
        ilp_solutions = []
        while len(ilp_solutions) < n:
            ilp_solution = self.round_randomized(lp_solution)
            while not self.is_feasible(ilp_solution):
                ilp_solution = self.round_randomized(lp_solution)
            ilp_solutions.append(ilp_solution)
        return np.array(ilp_solutions)

    def run(self):
        lp_solution = linprog(c=self.weights, A_ub=-self.adj_matrix, b_ub=np.full(self.n_cols, -1), bounds=(0, 1)).x
        ilp_solutions = self.generate_ilp_solutions(lp_solution, self.rounds)
        self.solution = ilp_solutions[np.argmin(np.apply_along_axis(self.get_weight, 1, ilp_solutions))]

    def get_covering(self):
        covering = []
        for row, indicator in enumerate(self.solution):
            if indicator == 1:
                covering.append(row)
        return np.array(covering)


n_cols, n_rows = map(int, input().split())
weights = np.zeros(n_rows)
adj_matrix = np.zeros((n_cols, n_rows))

for row in range(n_rows):
    data = list(map(int, input().split()))
    weights[row] = data[0]
    for col in data[1:]:
        adj_matrix[col, row] = 1

solver = RowCoverSolver(n_cols, n_rows, weights, adj_matrix)
solver.run()
for row in solver.get_covering():
    print(row + 1, end=' ')