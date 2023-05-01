import numpy as np
import sympy as sm

from scipy import linalg
from scipy import optimize
from scipy.optimize import minimize


# local modules
import modelproject

from types import SimpleNamespace
import time

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

from scipy.optimize import minimize

def portugal_production():
        # Define the utility function
    def utility(x):
        alpha = 0.5
        u = (x[0]**alpha) * (x[1]**(1-alpha))
        return -u

    # The aviailable hours are 1000, and the productivity of Portugal is as stated in the modelproject.ipynb file
    def budget_constraint(x):
        return x[0] - 1000/80 + 90/80 * x[1]

    # Define the bounds on x
    bounds = ((0, 1000), (0, 1000))

    # Define the initial guess for x
    x0 = [500, 500]

    # Define the constraints dictionary
    cons = [{'type': 'eq', 'fun': budget_constraint}]

    # Minimize the negative utility function subject to the budget constraint using the SLSQP algorithm
    result = minimize(utility, x0, method='SLSQP', bounds=bounds, constraints=cons)

    # Extract the optimal values of x
    x_opt = result.x
    u_opt = -result.fun

    # Print the results
    return print("Portugal produces {:.2f} units of wine and {:.2f} units of cloth. And, the resulting utility level is {:.2f}".format(x_opt[0], x_opt[1], u_opt))



def england_production():

   # Define the utility function
    def utility(x):
        alpha = 0.5
        u = (x[0]**alpha) * (x[1]**(1-alpha))
        return -u

    # The aviailable hours are 1000, and the productivity of Portugal is as stated in the modelproject.ipynb file
    def budget_constraint(x):
        return x[0] - 1000/120 + 100/120 * x[1]

    # Define the bounds on x
    bounds = ((0, 1000), (0, 1000))

    # Define the initial guess for x
    x0 = [500, 500]

    # Define the constraints dictionary
    cons = [{'type': 'eq', 'fun': budget_constraint}]

    # Minimize the negative utility function subject to the budget constraint using the SLSQP algorithm
    result = minimize(utility, x0, method='SLSQP', bounds=bounds, constraints=cons)

    # Extract the optimal values of x
    x_opt = result.x
    u_opt = -result.fun

    # Print the results
    return print("England produces {:.2f} units of wine and {:.2f} units of cloth. And, the resulting utility level is {:.2f}".format(x_opt[0], x_opt[1], u_opt))

