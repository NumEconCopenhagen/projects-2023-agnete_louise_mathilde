import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Exercise3_1:

    # Set the random seed
    np.random.seed(2023)

    def __init__(self):
        self.bounds = [-600, 600]
        self.tolerance = 1e-8
        self.K_warmup = 10
        self.K_max = 1000
        self.x_star = None
        self.initial_guesses = None

    def griewank(self, x):
        return self.griewank_(x[0], x[1])

    def griewank_(self, x1, x2):
        A = x1**2 / 4000 + x2**2 / 4000
        B = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
        return A - B + 1

    def refined_global_optimizer(self):
        self.x_star = None
        x_ast = None
        chi = None
        self.initial_guesses = []

        for k in range(self.K_max):
            x_k = np.random.uniform(self.bounds[0], self.bounds[1], size=2)
            self.initial_guesses.append(x_k)

            if k >= self.K_warmup:
                chi_k = 0.5 * 2 / (1 + np.exp((k - self.K_warmup) / 100))
                x_k0 = chi_k * x_k + (1 - chi_k) * x_ast
            else:
                x_k0 = x_k

            result = minimize(self.griewank, x_k0, method='BFGS', tol=self.tolerance)
            x_k_ast = result.x

            if x_ast is None or self.griewank(x_k_ast) < self.griewank(x_ast):
                x_ast = x_k_ast
                self.x_star = x_k_ast

            if self.griewank(x_ast) < self.tolerance:
                break

            # Print iteration details
            print(f'{k:4d}: x0 = ({x_k0[0]:7.2f}, {x_k0[1]:7.2f})', end='')
            print(f' -> converged at ({x_k_ast[0]:7.2f}, {x_k_ast[1]:7.2f}) with f = {self.griewank(x_k_ast):12.8f}')

        # Print the best solution
        print(f'\nbest solution:\n x = ({x_ast[0]:7.2f}, {x_ast[1]:7.2f}) -> f = {self.griewank(x_ast):12.8f}')

    def plot_initial_guesses(self):
        if self.initial_guesses is None:
            print("Optimizer hasn't been run yet.")
            return

        initial_guesses = np.array(self.initial_guesses)
        plt.scatter(initial_guesses[:, 0], initial_guesses[:, 1], c='b', label='Initial Guesses')
        plt.scatter(self.x_star[0], self.x_star[1], c='r', marker='x', label='Optimal Solution')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Effective Initial Guesses')
        plt.legend()
        plt.show()


class Exercise3_2:

    # Set the random seed
    np.random.seed(2023)

    def __init__(self):
        self.bounds = [-600, 600]
        self.tolerance = 1e-8
        self.K_warmup = 100
        self.K_max = 1000
        self.x_star = None
        self.initial_guesses = None

    def griewank(self, x):
        return self.griewank_(x[0], x[1])

    def griewank_(self, x1, x2):
        A = x1**2 / 4000 + x2**2 / 4000
        B = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
        return A - B + 1

    def refined_global_optimizer(self):
        self.x_star = None
        x_ast = None
        chi = None
        self.initial_guesses = []

        for k in range(self.K_max):
            x_k = np.random.uniform(self.bounds[0], self.bounds[1], size=2)
            self.initial_guesses.append(x_k)

            if k >= self.K_warmup:
                chi_k = 0.5 * 2 / (1 + np.exp((k - self.K_warmup) / 100))
                x_k0 = chi_k * x_k + (1 - chi_k) * x_ast
            else:
                x_k0 = x_k

            result = minimize(self.griewank, x_k0, method='BFGS', tol=self.tolerance)
            x_k_ast = result.x

            if x_ast is None or self.griewank(x_k_ast) < self.griewank(x_ast):
                x_ast = x_k_ast
                self.x_star = x_k_ast

            if self.griewank(x_ast) < self.tolerance:
                break

            # Print iteration details
            print(f'{k:4d}: x0 = ({x_k0[0]:7.2f}, {x_k0[1]:7.2f})', end='')
            print(f' -> converged at ({x_k_ast[0]:7.2f}, {x_k_ast[1]:7.2f}) with f = {self.griewank(x_k_ast):12.8f}')

        # Print the best solution
        print(f'\nbest solution:\n x = ({x_ast[0]:7.2f}, {x_ast[1]:7.2f}) -> f = {self.griewank(x_ast):12.8f}')

    def plot_initial_guesses(self):
        if self.initial_guesses is None:
            print("Optimizer hasn't been run yet.")
            return

        initial_guesses = np.array(self.initial_guesses)
        plt.scatter(initial_guesses[:, 0], initial_guesses[:, 1], c='b', label='Initial Guesses')
        plt.scatter(self.x_star[0], self.x_star[1], c='r', marker='x', label='Optimal Solution')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Effective Initial Guesses')
        plt.legend()
        plt.show()