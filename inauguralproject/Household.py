
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

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
        sol.LM_vec = np.zeros(par.wF_vec.size) # vector of LM. size is the same as wF_vec. 
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

        # b1. define power
        if par.sigma == 0:
            power = (par.sigma-1)/(par.sigma+1e-8)
        else:
            power = (par.sigma-1)/(par.sigma)

        # b2. home production
        if par.sigma == 0:
            H = pd.min(HM,HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha)*HM**(power)+par.alpha*HF**(power))**(1/power)

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility
    


    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
   
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
 
        LM = LM.ravel() # vector 
        HM = HM.ravel() # ravel orders the elements 
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf 
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt


    def solve_continuous(self, do_print=False):
        """ solve model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        def objective(x):
            LM, HM, LF, HF = x
            return -self.calc_utility(LM, HM, LF, HF)

        def constraintM(x):
            LM, HM, LF, HF = x
            return 24-(LM+HM)
        
        def constraintF(x):
            LM, HM, LF, HF = x
            return 24-(LF+HF)
        
        constraints = [{"type":"ineq","fun":constraintF},
                       {"type":"ineq","fun":constraintM}]
        intial_guess = [12,12,12,12]

        result = optimize.minimize(objective, intial_guess, 
                                   constraints=constraints, method = "SLSQP")

        sol.LM = opt.LM = result.x[0]    
        sol.HM = opt.HM = result.x[1]
        sol.LF = opt.LF = result.x[2]
        sol.HF = opt.HF = result.x[3]
        opt.util = self.calc_utility(opt.LM, opt.HM, opt.LF, opt.HF)

        return opt

   
    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol
        
        for n, i in enumerate(par.wF_vec):

            par.wF = i
            
            los = self.solve_continuous()

            sol.LF_vec[n] = los.LF
            sol.HF_vec[n] = los.HF
            sol.LM_vec[n] = los.LM
            sol.HM_vec[n] = los.HM


        self.run_regression()    

       

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]


    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        par = self.par
        sol = self.sol

        def obj(x):
            self.par.alpha = x[0]
            self.par.sigma = x[1]
            self.solve_wF_vec()
            return np.sum((sol.beta1-sol.beta1_target)**2)
    
        x0 = [0.5, 0.5]
        res = optimize.minimize(obj, x0, tol = 1e-8, method = "Nelder-Mead", bounds = [(0.01, 0.99),(0.01, 1.0)])

        return res.x
       