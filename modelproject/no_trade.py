import numpy as np
import sympy as sm

from scipy import linalg
from scipy import optimize
from scipy.optimize import minimize

from types import SimpleNamespace
import time

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

from scipy.optimize import minimize


class no_trade_class():

    def portugal_production(Option = False, do_print = True):
            # Define the utility function
        def utility(x):
            alpha = 0.5
            u = (x[0]**alpha) * (x[1]**(1-alpha))
            return -u

        # The aviailable hours are 8760, and the productivity of Portugal is as stated in the modelproject.ipynb file
        def budget_constraint(x):
            return x[0] - 8760/80 + 90/80 * x[1]

        # Define the bounds on x
        bounds = ((0, 8760), (0, 8760))

        # Define the initial guess for x
        x0 = [500, 500]

        # Define the constraints dictionary
        cons = [{'type': 'eq', 'fun': budget_constraint}]

        # Minimize the negative utility function subject to the budget constraint using the SLSQP algorithm
        result = optimize.minimize(utility, x0,
                                method='SLSQP',
                                constraints=cons,
                                bounds=bounds,
                                options={'disp':Option},
                                tol=1e-8)
        
        # Extract the optimal values of x
        x_opt = result.x
        u_opt = -result.fun

        wine_p=x_opt[0]
        cloth_p=x_opt[1]

        # Print the results
        if do_print:
            print("Portugal produces {:.2f} units of wine and {:.2f} units of cloth.".format(wine_p, cloth_p))
            print("And, the resulting utility level is {:.2f}".format(u_opt))

        return wine_p, cloth_p, u_opt
    
    def england_production(Option = False, do_print = True):

    # Define the utility function
        def utility(x):
            alpha = 0.5
            u = (x[0]**alpha) * (x[1]**(1-alpha))
            return -u

        # The aviailable hours are 1000, and the productivity of Portugal is as stated in the modelproject.ipynb file
        def budget_constraint(x):
            return x[0] - 8760/120 + 100/120 * x[1]

        # Define the bounds on x
        bounds = ((0, 8760), (0, 8760))

        # Define the initial guess for x
        x0 = [500, 500]

        # Define the constraints dictionary
        cons = [{'type': 'eq', 'fun': budget_constraint}]

        # Minimize the negative utility function subject to the budget constraint using the SLSQP algorithm
        result = optimize.minimize(utility, x0,
                                method='SLSQP',
                                constraints=cons,
                                bounds=bounds,
                                options={'disp':Option},
                                tol=1e-8)

        # Extract the optimal values of x
        x_opt = result.x
        u_opt = -result.fun

        wine_e=x_opt[0]
        cloth_e=x_opt[1]

        # Print the results
        if do_print:
            print("England produces {:.2f} units of wine and {:.2f} units of cloth.".format(wine_e, cloth_e))
            print("And, the resulting utility level is {:.2f}".format(u_opt))

        return wine_e, cloth_e, u_opt



