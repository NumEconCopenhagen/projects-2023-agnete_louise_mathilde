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


class Production_Possibility_Frontiers():
    
    def __init__(self):
        # Define the parameters
        self.par = {}

        # Hours it takes to produce one unit:
        self.par['w_p'] = 80
        self.par['w_e'] = 120
        self.par['c_p'] = 90
        self.par['c_e'] = 100 
        
        # Hours
        self.par['hours'] = 8760

        # Units produced in one year (8760 hours): 
        self.par['temp_w_p'] = self.par['hours'] / self.par['w_p'] # = 109.5 (intercept with x-axis for Portugal)
        self.par['temp_c_p'] = self.par['hours'] / self.par['c_p'] # = 97.3 (intercept with y-axis for Portugal)
        self.par['temp_w_e'] = self.par['hours'] / self.par['w_e'] # = 73 (intercept with x-axis for England)
        self.par['temp_c_e'] = self.par['hours'] / self.par['c_e'] # = 87.6 (intercept with y-axis for England)

        # Opportunity costs (OC): 
        self.par['oc_w_p'] = self.par['temp_c_p'] / self.par['temp_w_p'] # = 0.89 (slope for Portugal before trade
        self.par['oc_c_p'] = self.par['temp_w_p'] / self.par['temp_c_p'] # = 1.13 
        self.par['oc_w_e'] = self.par['temp_c_e'] / self.par['temp_w_e'] # = 1.20 (slope for England before trade
        self.par['oc_c_e'] = self.par['temp_w_e'] / self.par['temp_c_e'] # = 0.83 

    def plot_production_graphs(self):
        #model = Production_Possibility_Frontiers()

        # Define the x (wine) and y (cloth) values for Portugal
        x_p = [0, self.par['temp_w_p']]
        y_p = [self.par['temp_c_p'],0]

        # Define the x (wine) and y (cloth) values for England
        x_e = [0, self.par['temp_w_e']]
        y_e = [self.par['temp_c_e'],0]

        # Plot the graphs
        plt.plot(x_p, y_p, label='Portugal', linestyle='-', color='blue')  # Set linestyle and color for Portugal
        plt.plot(x_e, y_e, label='England', linestyle='-', color='green')  # Set linestyle and color for England
        
        # Highlight intercepts with axes
        plt.scatter(self.par['temp_w_p'], 0, color='blue', marker='o')  # Highlight Portugal intercept on x-axis
        plt.scatter(0, self.par['temp_c_p'], color='blue', marker='o')  # Highlight Portugal intercept on y-axis
        plt.scatter(self.par['temp_w_e'],0 , color='green', marker='o')  # Highlight England intercept on x-axis
        plt.scatter(0, self.par['temp_c_e'], color='green', marker='o')  # Highlight England intercept on y-axis

        # Highlight intercept values on x-axis and y-axis
        plt.text(self.par['temp_w_p'], -2, str(round(self.par['temp_w_p'], 1)), ha='left', va='center', color='blue', fontsize=8)  # Portugal intercept on x-axis
        plt.text(-2, self.par['temp_c_p'], str(round(self.par['temp_c_p'], 1)), ha='right', va='center', color='blue', fontsize=8)  # Portugal intercept on y-axis

        plt.text(self.par['temp_w_e'], -2, str(round(self.par['temp_w_e'], 1)), ha='left', va='center', color='green', fontsize=8)  # England intercept on x-axis
        plt.text(-2, self.par['temp_c_e'], str(round(self.par['temp_c_e'], 1)), ha='right', va='center', color='green', fontsize=8)  # England intercept on y-axis


        plt.title('Production Possibility Frontiers before trade')
        plt.xlabel('Wine')
        plt.ylabel('Cloth')
        plt.legend()
        plt.show()


        

class no_trade_class():

    def portugal_production():
            # Define the utility function
        def utility(x):
            alpha = 0.5
            u = (x[0]**alpha) * (x[1]**(1-alpha))
            return -u

        # The aviailable hours are 1000, and the productivity of Portugal is as stated in the modelproject.ipynb file
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
                                options={'disp':True},
                                tol=1e-8)
        
        # Extract the optimal values of x
        x_opt = result.x
        u_opt = -result.fun

        wine_p=x_opt[0]
        cloth_p=x_opt[1]

        # Print the results
        print("Portugal produces {:.2f} units of wine and {:.2f} units of cloth.".format(wine_p, cloth_p))
        print("And, the resulting utility level is {:.2f}".format(u_opt))

    def england_production():

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
                                options={'disp':True},
                                tol=1e-8)

        # Extract the optimal values of x
        x_opt = result.x
        u_opt = -result.fun

        wine_e=x_opt[0]
        cloth_e=x_opt[1]

        # Print the results
        print("England produces {:.2f} units of wine and {:.2f} units of cloth.".format(wine_e, cloth_e))
        print("And, the resulting utility level is {:.2f}".format(u_opt))



