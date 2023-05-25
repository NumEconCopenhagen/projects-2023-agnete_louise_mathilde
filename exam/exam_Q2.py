import numpy as np
from types import SimpleNamespace
from scipy.optimize import minimize
import matplotlib.pyplot as plt 


class Exercise2_1:
    "Class setting up the model from exercise 2_1"

    def __init__(self):
        """ setup model """
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()

        # Baseline parameters exercise 1
        self.par.eta = 0.5
        self.par.w = 1.0
        self.par.kappa = np.array([1, 2])

        # Solutions exercise 1
        self.sol.ell = np.zeros(self.par.kappa.size)
        self.sol.P = np.zeros(self.par.kappa.size)

    def profitmaximization(self, ell, kappa):
        P = kappa * np.power(ell, 1 - self.par.eta) - self.par.w * ell
        return -np.max(P)

    def maximize_profit(self):
        for i in range(len(self.par.kappa)):
            result = minimize(self.profitmaximization, x0=0.0, args=(self.par.kappa[i],), bounds=[(0, None)])
            self.sol.ell[i] = result.x[0]
            self.sol.P[i] = -result.fun

    def print_results(self):
        for i in range(len(self.par.kappa)):
            print(f"For kappa = {self.par.kappa[i]}:")
            print(f"Ell: {self.sol.ell[i]}")
            print(f"P: {self.sol.P[i]}\n")


class Exercise2_2:
    "Class setting up the model from exercise 2_2"

    def __init__(self):
        """ setup model """
        self.par = SimpleNamespace()

        # Baseline parameters
        self.par.eta=0.5
        self.par.w=1.0
        self.par.rho = 0.90
        self.par.iota = 0.01
        self.par.sigma = 0.10
        self.par.R = np.power(1+0.01, 1/12)

    def calculate_expected_value(self, eta):
        # Set the initial values
        kappa_t_minus_1 = 1
        ell_t_minus_1 = 0

        # Set the number of shock series
        K = 10000  

        # Initialize the sum variable
        H = 0

        # Set the random seed
        np.random.seed(2023)  

        # Generate K shock series
        for k in range(K):
            sum_h = 0

            # Generate 120 random shocks
            epsilon = np.random.normal(loc=-0.5 * self.par.sigma ** 2, scale=self.par.sigma, size=120)

            for t in range(120):
                # Calculate kappa_t using the AR(1) process
                log_kappa_t = self.par.rho * np.log(kappa_t_minus_1) + epsilon[t]
                kappa_t = np.exp(log_kappa_t)

                # Calculate ell_t using the policy
                ell_t = ((1 - self.par.eta) * kappa_t / self.par.w) ** (1 / self.par.eta)

                # Calculate the profit for the current time period
                h_t = self.par.R * (-t) * (kappa_t * ell_t * (1 - self.par.eta) - self.par.w * ell_t - (ell_t != ell_t_minus_1) * self.par.iota)

                # Update the sum variable
                sum_h += h_t

                # Update the values for the next iteration
                kappa_t_minus_1 = kappa_t
                ell_t_minus_1 = ell_t

            # Calculate the average for the current shock series at t=120, i.e. the average of the ex post value of the salon conditional on the shock series
            avg_h = sum_h / 120

            # Add the average to the overall sum variable, i.e. calculate the ex anta expected value of the salon 
            H += avg_h

        # Calculate the final ex ante expected value of the salon
        H = H / K

        return H


class Exercise2_3:
    "Class setting up the model from exercise 2_3"

    def __init__(self):
        """ setup model """
        self.par = SimpleNamespace()

        # Baseline parameters
        self.par.eta=0.5
        self.par.w=1.0
        self.par.rho = 0.90
        self.par.iota = 0.01
        self.par.sigma = 0.10
        self.par.R = np.power(1+0.01, 1/12)
        self.par.delta = 0.05

    def calculate_expected_value(self, eta):
        # Set the initial values
        kappa_t_minus_1 = 1
        ell_t_minus_1 = 0

        # Set the number of shock series
        K = 10000  

        # Initialize the sum variable
        H = 0

        # Set the random seed
        np.random.seed(2023) 

        # Generate K shock series
        for k in range(K):
            sum_h = 0

            # Generate 120 random shocks
            epsilon = np.random.normal(loc=-0.5 * self.par.sigma ** 2, scale=self.par.sigma, size=120)

            for t in range(120):
                # Calculate kappa_t using the AR(1) process
                log_kappa_t = self.par.rho * np.log(kappa_t_minus_1) + epsilon[t]
                kappa_t = np.exp(log_kappa_t)

                # Calculate ell_t using the policy
                ell_t = ((1 - self.par.eta) * kappa_t / self.par.w) ** (1 / self.par.eta)

                # Calculate the profit for the current time period
                if np.abs(ell_t_minus_1 - ell_t) > self.par.delta:
                    h_t = self.par.R * (-t) * (kappa_t * ell_t * (1 - self.par.eta) - self.par.w * ell_t - (ell_t != ell_t_minus_1) * self.par.iota)
                else:
                    h_t = self.par.R * (-t) * (kappa_t * ell_t_minus_1 * (1 - self.par.eta) - self.par.w * ell_t_minus_1)

                # Update the sum variable
                sum_h += h_t

                # Update the values for the next iteration
                kappa_t_minus_1 = kappa_t
                ell_t_minus_1 = ell_t

            # Calculate the average for the current shock series at t=120, i.e. the average of the ex post value of the salon conditional on the shock series
            avg_h = sum_h / 120

            # Add the average to the overall sum variable, i.e. calculate the ex anta expected value of the salon 
            H += avg_h

        # Calculate the final ex ante expected value of the salon
        H = H / K

        return H


class Exercise2_4:
    "Class setting up the model from exercise 2_4"

    def __init__(self):
        """ setup model """
        self.par = SimpleNamespace()

        # Baseline parameters
        self.par.eta = 0.5
        self.par.w = 1.0
        self.par.rho = 0.90
        self.par.iota = 0.01
        self.par.sigma = 0.10
        self.par.R = np.power(1 + 0.01, 1 / 12)
        self.par.delta = 0.05

    def calculate_expected_value(self, eta, delta_values):
        # Set the initial values
        kappa_t_minus_1 = 1
        ell_t_minus_1 = 0

        # Set the number of shock series
        K = 100 

        # Initialize the maximum expected value and corresponding delta
        max_H = -float('inf')
        max_delta = None

        # Set the random seed
        np.random.seed(2023)

        # Empty list to store H values
        H_values = []

        # Generate K shock series
        for delta in delta_values:
            H = 0

            for k in range(K):
                sum_h = 0

                # Generate 120 random shocks
                epsilon = np.random.normal(loc=-0.5 * self.par.sigma ** 2, scale=self.par.sigma, size=120)

                for t in range(120):
                    # Calculate kappa_t using the AR(1) process
                    log_kappa_t = self.par.rho * np.log(kappa_t_minus_1) + epsilon[t]
                    kappa_t = np.exp(log_kappa_t)

                    # Calculate ell_t using the policy
                    ell_t = ((1 - self.par.eta) * kappa_t / self.par.w) ** (1 / self.par.eta)

                    # Calculate the profit for the current time period
                    if np.abs(ell_t_minus_1 - ell_t) > delta:
                        h_t = self.par.R * (-t) * (kappa_t * ell_t * (1 - self.par.eta) - self.par.w * ell_t - (ell_t != ell_t_minus_1) * self.par.iota)
                    else:
                        h_t = self.par.R * (-t) * (kappa_t * ell_t_minus_1 * (1 - self.par.eta) - self.par.w * ell_t_minus_1)

                    # Update the sum variable
                    sum_h += h_t

                    # Update the values for the next iteration
                    kappa_t_minus_1 = kappa_t
                    ell_t_minus_1 = ell_t

                # Calculate the average for the current shock series at t=120
                avg_h = sum_h / 120

                # Add the average to the overall sum variable
                H += avg_h

            # Calculate the final ex ante expected value of the salon for the current delta value
            H = H / K

            # Check if the current delta value maximizes the expected value
            if H > max_H:
                max_H = H
                max_delta = delta

            # Collect the H value for the current delta
            H_values.append(H)

        return max_H, max_delta, H_values
    

class Exercise2_5:
    "Class setting up the model from exercise 2_5"

    def __init__(self):
        """ setup model """
        self.par = SimpleNamespace()

        # Baseline parameters
        self.par.eta = 0.5
        self.par.w = 1.0
        self.par.rho = 0.90
        self.par.iota = 0.01
        self.par.sigma = 0.10
        self.par.R = np.power(1 + 0.01, 1 / 12)
        self.par.delta = 0.05

    def calculate_expected_value(self, eta):
        # Set the initial values
        kappa_t_minus_1 = 1
        ell_t_minus_1 = 0

        # Set the number of shock series
        K = 10000

        # Initialize the sum variable
        H_values = []

        # Set the random seed
        np.random.seed(2023)

        # Generate K shock series
        for k in range(K):
            sum_h = 0

            # Generate 120 random shocks
            epsilon = np.random.normal(loc=-0.5 * self.par.sigma ** 2, scale=self.par.sigma, size=120)

            # Initialize the array to store all ell_t values
            ell_t_array = np.zeros(120)

            for t in range(120):
                # Calculate kappa_t using the AR(1) process
                log_kappa_t = self.par.rho * np.log(kappa_t_minus_1) + epsilon[t]
                kappa_t = np.exp(log_kappa_t)

                # Calculate ell_t using the policy
                ell_t = ((1 - self.par.eta) * kappa_t / self.par.w) ** (1 / self.par.eta)

                # Update ell_t_array with the current ell_t value
                ell_t_array[t] = ell_t

                # Calculate the profit for the current time period
                if np.abs(np.mean(ell_t_array[:t]) - ell_t) > self.par.delta:
                    h_t = self.par.R * (-t) * (
                            kappa_t * ell_t * (1 - self.par.eta) - self.par.w * ell_t - (
                            ell_t != ell_t_minus_1) * self.par.iota)
                else:
                    h_t = self.par.R * (-t) * (
                            kappa_t * ell_t_minus_1 * (1 - self.par.eta) - self.par.w * ell_t_minus_1)

                # Update the sum variable
                sum_h += h_t

                # Update the values for the next iteration
                kappa_t_minus_1 = kappa_t
                ell_t_minus_1 = ell_t

            # Calculate the average for the current shock series at t=120, i.e. the average of the ex post value of the salon conditional on the shock series
            avg_h = sum_h / 120

            # Add the average to the list of H values
            H_values.append(avg_h)

        # Calculate the final ex ante expected value of the salon
        H = np.mean(H_values)

        return H