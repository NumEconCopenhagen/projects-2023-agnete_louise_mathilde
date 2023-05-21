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

     
class Plot_before_trade():
    
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


    def plot_bt_graphs(self):
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
        
       # Plot the indifference curves passing through the given points
        u_p = lambda x: (x[0] * 0.5) * (x[1] * 0.5)  # Utility function for Portugal
        u_e = lambda x: (x[0] * 0.5) * (x[1] * 0.5)  # Utility function for England

        # Calculate the indifference curve passing through the point for Portugal
        u_p = (54.75 * 0.5) * (48.67 * 0.5)
        wine_p = np.linspace(0, self.par['temp_w_p'], 100)
        cloth_p = (u_p / ((wine_p + 1e-8) * 0.5)) * 2

        # Calculate the indifference curve passing through the point for England
        u_e = (36.50 * 0.5) * (43.80 * 0.5)
        wine_e = np.linspace(0, self.par['temp_w_e'], 100)
        cloth_e = (u_e / ((wine_e + 1e-8) * 0.5)) * 2

        # Plot the indifference curves passing through the points
        plt.plot(wine_p, cloth_p, label='Portugal Indifference Curve', linestyle='--', color='blue')
        plt.plot(wine_e, cloth_e, label='England Indifference Curve', linestyle='--', color='green')

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

        # Portugal produces 54.75 units of wine and 48.67 units of cloth
        plt.scatter(54.75, 48.67, color='blue', marker='X')

        # England produces 36.50 units of wine and 43.80 units of cloth.
        plt.scatter(36.50, 43.80, color='green', marker='X')

        plt.ylim(0, 100)

        plt.title('Production Possibility Frontiers before trade')
        plt.xlabel('Wine')
        plt.ylabel('Cloth')
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)
        plt.show()