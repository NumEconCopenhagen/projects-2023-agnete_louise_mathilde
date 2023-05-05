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

def optimal_trade():

    # Define the utility function to be maximized 
    def utility_p(x):
        alpha = 0.5
        u_p = ((x[0]+x[2])**alpha) * ((x[1]+x[3])**(1-alpha))
        return -u_p
    
    def utility_e(x):
        alpha = 0.5
        u_e = ((x[0]+x[2])**alpha) * ((x[1]+x[3])**(1-alpha))
        return -u_e
    
    def _utility(x):
        return utility_p(x) + utility_e(x)
         
    # Define the constraints dictionary
    cons = [] 

    # Define the budget constraint for Portugal
    cons.append({'type': 'eq', 'fun': lambda x: x[0] - 1000/80 + 90/80 * x[1]})
    
    # Define the budget constraint for England
    cons.append({'type': 'eq', 'fun': lambda x: x[2] - 1000/120 + 100/120 * x[3]})
    
    # Define the bounds on x
    bounds = ((0, 1000), (0, 1000), (0, 1000), (0, 1000))

    # Define the initial guess for x
    x0 = [20, 20, 20, 20]

    # Minimize the negative utility function subject to the budget constraint using the SLSQP algorithm
    result = optimize.minimize(_utility, x0,
                               method='SLSQP',
                               constraints=cons,
                               bounds=bounds,
                               options={'disp':True},
                               tol=1e-8)

    # Extract the optimal values of x
    x_opt = result.x
    u_opt = -result.fun

    # Print the results
    print("The optimal production levels for Portugal are {:.2f} units of wine and {:.2f} units of cloth".format(x_opt[0], x_opt[1]))
    print("The optimal production levels for England are {:.2f} units of wine and {:.2f} units of cloth".format(x_opt[2], x_opt[3]))
    print("The resulting utility level is {:.2f}".format(u_opt))
