import numpy as np
import sympy as sm

from scipy import linalg
from scipy import optimize
from scipy.optimize import minimize


# local modules
import modelproject
from no_trade import no_trade_class as ntc

from types import SimpleNamespace
import time

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

from scipy.optimize import minimize

class PortugalEnglandTradeModel:
    """ Class for solving the Portugal-England trade model """

    def __init__(self):
        """ setup model """

        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()

        # productivity
        self.par.w_p = 80
        self.par.w_e = 120
        self.par.c_p = 90
        self.par.c_e = 100 

        # relative productivity
        self.par.wc_p = self.par.w_p/self.par.c_p
        self.par.wc_e = self.par.w_e/self.par.c_e
        
        # hours
        self.par.hours = 8760

        # opportunity cost
        self.par.temp_w_p = self.par.hours/self.par.w_p
        self.par.temp_c_p = self.par.hours/self.par.c_p
        self.par.temp_w_e = self.par.hours/self.par.w_e
        self.par.temp_c_e = self.par.hours/self.par.c_e

        self.par.oc_w_p = self.par.temp_c_p/self.par.temp_w_p
        self.par.oc_c_p = self.par.temp_w_p/self.par.temp_c_p
        self.par.oc_w_e = self.par.temp_c_e/self.par.temp_w_e
        self.par.oc_c_e = self.par.temp_w_e/self.par.temp_c_e

        # utilty 
        self.par.alpha_p = 0.5
        self.par.alpha_p_vec = np.linspace(0,1,100)
        self.par.alpha_e = 0.5

        
def optimal_trade():
    """ Solve the Portugal-England trade model """

    model = PortugalEnglandTradeModel()
    x_opt_p = None

    # Define the utility function to be maximized 
    def utility_p(x):
        u_p = ((x[0]+x[5])** model.par.alpha_p) * ((x[2]+x[7])**(1-model.par.alpha_p))
        return u_p
        
    def utility_e(x):
        u_e = ((x[4]+x[1])**model.par.alpha_e) * ((x[6]+x[3])**(1-model.par.alpha_e))
        return u_e
        
    def utility(x):
        u_e = utility_e(x)
        u_p = utility_p(x)
        return -(u_e + u_p)
    
    # Define the constraints dictionary
    cons = [] 

    # Define the budget constraint for Portugal
    cons.append({'type': 'eq', 'fun': lambda x: 
                 (x[0] + x[5]) + model.par.c_e/model.par.w_e * (x[2] + x[7])- model.par.hours/model.par.w_p})

    # Define the budget constraint for England
    cons.append({'type': 'eq', 'fun': lambda x: 
                 model.par.w_p / model.par.c_p * (x[4] + x[1]) + (x[6] + x[3]) - model.par.hours/model.par.c_e})

    cons.append({'type': 'ineq', 'fun': lambda x: (x[0]+x[1]) - 1/model.par.oc_w_p * (x[0]+x[5])})
    cons.append({'type': 'ineq', 'fun': lambda x: (x[6]+x[7]) - 1/model.par.oc_c_e * (x[4]+x[1])})
    
    # The x[5] and the x[7] are the exports of wine and cloth from Portugal to England.
    # The x[1] and the x[3] are the exports of wine and cloth from England to Portugal.

    # Define the bounds on x
    bounds = ((0, 100), (0, 100), (0, 100), (0, 100),
              (0, 100), (0, 100), (0, 100), (0, 100))

    # Define the initial guess for x
    x0 = [2, 2, 2, 2,
          2, 2, 2, 2]
    
    # Minimize the negative utility function subject to the budget constraint using the SLSQP algorithm
    result = optimize.minimize(utility, x0,
                            method='SLSQP',
                            constraints=cons,
                            bounds=bounds,
                            options={'disp':True},
                            tol=1e-8)

    # Extract the optimal values of x
    x_opt_p = result.x

    # produciton
    wine_p_pro = x_opt_p[0]+x_opt_p[1]
    cloth_p_pro = x_opt_p[2]+x_opt_p[3]
    wine_e_pro = x_opt_p[4]+x_opt_p[5]
    cloth_e_pro = x_opt_p[6]+x_opt_p[7]

    # consumption 
    wine_p_consum = x_opt_p[0]+x_opt_p[5]
    cloth_p_consum = x_opt_p[2]+x_opt_p[7]
    wine_e_consum = x_opt_p[4]+x_opt_p[1]
    cloth_e_consum = x_opt_p[6]+x_opt_p[3]

    # utility 
    u_p = ((x_opt_p[0]+x_opt_p[5])**0.5) * ((x_opt_p[2]+x_opt_p[7])**(1-0.5))
    u_e = ((x_opt_p[4]+x_opt_p[1])**0.5) * ((x_opt_p[6]+x_opt_p[3])**(1-0.5))
    
    # Print the results
    print("The optimal production levels for Portugal are {:.2f} units of wine and {:.2f} units of cloth".format(wine_p_pro, cloth_p_pro))
    print("The export of wine from Portugal to England is {:.2f} units".format(x_opt_p[1]))
    print("The export of cloth from Portugal to England is {:.2f} units".format(x_opt_p[3]))
    print("The consumption levels for Portugal are {:.2f} units of wine and {:.2f} units of cloth".format(wine_p_consum, cloth_p_consum))
    print("The utility for Portugal is {:.2f}".format(u_p))
    print('\n')
    print("The optimal production levels for England are {:.2f} units of wine and {:.2f} units of cloth".format(wine_e_pro, cloth_e_pro))
    print("The export of wine from England to Portugal is {:.2f} units".format(x_opt_p[5]))
    print("The export of cloth from England to Portugal is {:.2f} units".format(x_opt_p[7]))
    print("The consumption levels for England are {:.2f} units of wine and {:.2f} units of cloth".format(wine_e_consum, cloth_e_consum))
    print("The utility for England is {:.2f}".format(u_e))



    def figure():
        """Varying the value of alpha_p holding alpha_e fixed at 0.5"""
        """Figure that shows producion and consumption of both England and Portugal"""
        """On the x-axis we have the value of alpha_p and on the y-axis we have the production and consumption of both England and Portugal"""
    
    model = PortugalEnglandTradeModel()
    x_opt_p = None



    # foreach alpha in alpha_p_vec perform the optmization in optimal_trade()
    for alpha in model.par.alpha_p_vec:

        pass



