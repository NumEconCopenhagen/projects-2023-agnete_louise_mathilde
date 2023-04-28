# This py contains the code for question 5

from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt
import ipywidgets 

class Qfive:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5
        par.new_nu = 0.001
        
        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan


    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b1. define power - without these lines, we might devide by zero in the b2. home production
        if par.sigma == 0:
            s_power = (par.sigma-1)/(par.sigma+0.001)
        elif par.sigma == 1:
            s_power = (par.sigma-1)/(par.sigma+0.001)
        else:
            s_power = (par.sigma-1)/(par.sigma)

        # b2. home production
        if par.sigma == 0:
            H = pd.min(HM,HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha + 1e-8)*HF**(par.alpha + 1e-8)
        else:
            H = ((1-par.alpha)*HM**(s_power)+par.alpha*HF**(s_power))**(1/s_power)


        # c1. define power - without these lines, we might devide by zero in the c2. total consumption utility
        if par.rho == 1:
            r_power = (1-par.rho+0.001)
        elif par.rho == 0:
            r_power = (1-par.rho+0.001)
        else:
            r_power = (1-par.rho)     

        # c2. total consumption utility
        Q = C**(par.omega+1e-8)*H**(1-par.omega+1e-8)
        utility = np.fmax(Q,1e-8)**(r_power)/(r_power)


        # d1. define power - without these lines, we might devide by zero in the d2. disutility of work
        if par.epsilon == 0:
            e_power = 1+1/(par.epsilon+0.001)
        elif par.epsilon == 1:
            e_power = 1+1/(par.epsilon+0.001)
        else:
            e_power = 1+1/par.epsilon

        # d2. disutlity of work
        TM = LM+HM
        TF = LF+HF

        if par.new_nu > 1:
            par.new_nu = 1

        disutility = par.nu*(TM**e_power/e_power+TF**e_power/e_power) + par.new_nu*HM + (1-par.new_nu)*HF

        return utility - disutility


    
    def solve_continous(self,do_print=False):
        """ solve model continously """

        par = self.par 
        sol = self.sol 
        opt = SimpleNamespace()

        # a. The initial guess
        LM_guess, LF_guess, HM_guess, HF_guess = 6, 6, 6, 6
        x_guess = np.array([LM_guess, HM_guess, LF_guess, HF_guess])

        # b. Define the Objective Function 
        # We use a lambda function and the calc_utility defined above.
        obj = lambda x: -self.calc_utility(x[0],x[1],x[2],x[3])

        # c. Bounds
        bounds = ((1e-8,24-1e-8), 
                  (1e-8,24-1e-8),
                  (1e-8,24-1e-8), 
                  (1e-8,24-1e-8))

        # d. Optimization using Nelder-Mead 
        result = optimize.minimize(obj,x_guess,method='Nelder-Mead',bounds=bounds, tol=1e-8)

        opt.LM = result.x[0]
        opt.HM = result.x[1]
        opt.LF = result.x[2]
        opt.HF = result.x[3]

        # Print. 
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')
    
        return opt  
    
    
    def solution_wF_vec(self, do_plot=False, do_print=False):
        """ solve model (discrete or contionus) for a range of wF"""

        par = self.par
        sol = self.sol
    
            # Solving for wF_vec
        for i, wF in enumerate(self.par.wF_vec):
            # Set wF for this iteration
            self.par.wF = wF
            # Solve for optimal choices
            opt = self.solve_continous()
            # Store results in solution arrays
            sol.HM_vec[i] = opt.HM
            sol.HF_vec[i] = opt.HF
        pass
        
                       
        # Create the figure 
        if do_plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            # plot
            ax.plot(np.log(par.wF_vec), np.log(sol.HF_vec/sol.HM_vec), color='black', lw=2)
            ax.scatter(np.log(par.wF_vec), np.log(sol.HF_vec/sol.HM_vec), color='blue')
            ax.grid(True)
            # add labels
            plt.xlabel('$\log(w_F / w_M)$')
            plt.ylabel('$\log(H_F/H_M)$')
            ax.set_title("Plot of log of relative hours at home against log wF")
            plt.show()

        return

      
    def run_regression(self, print_beta=False):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
      

    def estimateQfive(self,new_nu=None,sigma=None, do_print=False, do_plotb=False):
        """ estimate alpha and sigma """

        sol = self.sol
        par = self.par

        # Initial guess
        nu_guess = 0.5
        sigma_guess = 1
        ns_guess = (nu_guess, sigma_guess)

        # Bounds
        bounds = ((1e-8,1), (1e-8,1.5))

        # Empty list to store values tried out in optimization
        new_nu_list = []
        sigma_list = []
        # Empty list to store the objective function value
        obj_list = []

        # Defining the objective function
        def obj(x):
            par.alpha = 0.5
            par.new_nu, par.sigma = x  # unpack into the list
            new_nu_list.append(x[0])  # Append input values to the list
            sigma_list.append(x[1])  # Append input values to the list
            self.solution_wF_vec()
            self.run_regression()
            value = (abs((par.beta0_target-sol.beta0)**2
                +(par.beta1_target-sol.beta1)**2))  
            obj_list.append(value)  # Append the objective function value to the list          
            return value

        # Minimizing the R-squared value with scipy
        result = optimize.minimize(obj, ns_guess, method="Nelder-Mead", 
                                bounds=bounds, tol=0.0001)
        pass
    
        # Printing the optimized values
        if  do_print == True:
            par.alpha = 0.5
            par.new_nu, par.sigma = result.x
            self.solution_wF_vec()
            self.run_regression()
            value = (abs((par.beta0_target-sol.beta0)**2
                +(par.beta1_target-sol.beta1)**2)) 
            
            print("Optimized Parameter Values:")
            print(f"Alpha = {par.alpha:6.4f}")
            print(f"Sigma = {par.sigma:6.4f}")
            print(f"New_nu = {par.new_nu:6.4f}")

            print("The Resulting Estimate Values:")
            print(f"Beta0 = {sol.beta0:6.4f}")
            print(f"Beta1 = {sol.beta1:6.4f}")

            print("The optimal value of H_F/H_M is:")
            print(f"H_F/H_M = {np.exp(sol.beta0):6.4f}")
            
            print("The Minimized Value Resulting from the Optimization:")
            print(f"Squared Residual = {value:6.4f}")

            pass

        # Plotting the values tried out in the optimization
        if do_plotb == True:
            par.alpha = 0.5
            par.new_nu, par.sigma = result.x 
            new_nu_list.append(result.x[0])  
            sigma_list.append(result.x[1])  
            self.solution_wF_vec()
            self.run_regression()
            value = (abs((par.beta0_target-sol.beta0)**2
                +(par.beta1_target-sol.beta1)**2))  
            obj_list.append(value)

            fig = plt.figure(figsize=(15, 6))
            ax = fig.add_subplot(111, projection='3d')
            # plot all the values tried out in the optimization in blue
            ax.scatter(new_nu_list, sigma_list, obj_list, c='blue', alpha=0.5, s=10)
            # plot the optimized value in red
            ax.scatter(par.new_nu, par.sigma, value, c='red', marker='o', s=50)
            # add labels
            ax.set_xlabel('Nu_New')
            ax.set_xlim(0.2, 0.8)
            ax.set_ylabel('Sigma')
            ax.set_ylim(1, 1.75)
            ax.set_zlabel('Objective Function Value')
            ax.zaxis.labelpad=1/10 # this alters where the photo is cropped.
            ax.set_zlim(0, 0.07)

            plt.show()




        pass


    