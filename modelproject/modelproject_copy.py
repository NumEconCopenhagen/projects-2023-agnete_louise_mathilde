import numpy as np
import sympy as sm

from scipy import linalg
from scipy import optimize
from scipy.optimize import minimize


# local modules
import modelproject
import no_trade as nt

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
        
        # hours
        self.par.hours = 1000


def optimal_trade():
    """ Solve the Portugal-England trade model """
    model = PortugalEnglandTradeModel()
    x_opt_p = None

    # Define the utility function to be maximized 
    def utility_p(x):
        u_p = ((x[0]+x[5])**0.5) * ((x[2]+x[7])**(1-0.5))
        return u_p
    
    #  # The Marginal Rate of Substitution (MRS) between wine / cloth for Portugal
    # def MRS_p(x):
    #     return (x[0]+x[5]) / ((x[2]+x[7]))
    
    def utility_e(x):
        u_e = ((x[4]+x[1])**0.5) * ((x[6]+x[3])**(1-0.5))
        return u_e
    
    # # The Marginal Rate of Substitution (MRS) between wine / cloth for England
    # def MRS_e(x):
    #     return (x[4]+x[1]) / ((x[6]+x[3]))
    
    def utility(x):
        u_e = utility_e(x)
        u_p = utility_p(x)
        return -(u_e * u_p)
      
    # Define the constraints dictionary
    cons = [] 

    # Define the budget constraint for Portugal
    cons.append({'type': 'eq', 'fun': lambda x: 80 * (x[0] + x[1]) - model.par.hours + 90 * (x[2] + x[3])})
   
    # Define the budget constraint for England
    cons.append({'type': 'eq', 'fun': lambda x: 120 * (x[4] + x[5]) - model.par.hours + 100 * (x[6] + x[7])})

    # The x[5] and the x[7] are the exports of wine and cloth from Portugal to England.
    # The x[1] and the x[3] are the exports of wine and cloth from England to Portugal.

    # # England will not buy cloth from Portugal for more than 1.2 wine. 
    # cons.append({'type': 'ineq', 'fun': lambda x: x[7] - 120/100 * x[1]})
    
    # # Portugal will not buy cloth from England for more than 8/9 wine.
    # cons.append({'type': 'ineq', 'fun': lambda x: x[3] - 80/90 * x[5] })
          
    # Define the optimal trade decision based on MRS_p
    # def trade_p():
    #     if MRS_p(x_opt_p) > 90/80:
    #         x_opt_p[5] = 0
    #         x_opt_p[7] = model.par.hours * 0.5 / 100
    #     elif MRS_p(x_opt_p) < 80/90:
    #         x_opt_p[5] = model.par.hours * 0.5 / 120
    #         x_opt_p[7] = 0

    # # Define the optimal trade decision based on MRS_e
    # def trade_e():
    #     if MRS_e(x_opt_p) > 100/120:
    #         x_opt_p[1] = 0
    #         x_opt_p[3] = model.par.hours * 0.5 / 100
    #     elif MRS_e(x_opt_p) < 120/100:
    #         x_opt_p[1] = model.par.hours * 0.5 / 80
    #         x_opt_p[3] = 0
    
    # cons.append({'type': 'ineq', 'fun': lambda x: trade_p()})
    # cons.append({'type': 'ineq', 'fun': lambda x: trade_e()})

    # # Consumption of wine and cloth must be greater than or equal to what it is without trade
    # def consumption_england(x):
    #     wine_e, cloth_e = nt.england_production()
    #     return (x[0] - x[1] + x[5] - wine_e), (x[2] - x[3] + x[7] - cloth_e)
    
    # def consumption_portugal(x):
    #     wine_p, cloth_p = nt.portugal_production()
    #     return (x[4] - x[5] + x[1] - wine_p), (x[6] - x[7] + x[3] - cloth_p)
    
    # cons.append({'type': 'ineq', 'fun': consumption_england})
    # cons.append({'type': 'ineq', 'fun': consumption_portugal})
    
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
    print("The consumption levels for Portugal are {:.2f} units of wine and {:.2f} units of cloth".format(wine_p_consum, cloth_p_consum))
    print("The utility for Portugal is {:.2f}".format(u_p))
    print('\n')
    print("The optimal production levels for England are {:.2f} units of wine and {:.2f} units of cloth".format(wine_e_pro, cloth_e_pro))
    print("The consumption levels for England are {:.2f} units of wine and {:.2f} units of cloth".format(wine_e_consum, cloth_e_consum))
    print("The utility for England is {:.2f}".format(u_e))
  
  