# This py contains the code for question 1

import numpy as np
import pandas as pd 
from scipy import optimize
import matplotlib.pyplot as plt

# Import your own code
import Household

def Q_1plot(alpha_l, sigma_l):
    "Make a plot of the optimal House_work and work for the household"    
    
    # initialize dictionary to store results
    resultsHF = {}
    resultsHM = {}
    resultsHFHNM = {}

    # foreach combination of alpha and sigma, print the ratio of optimal HF/HM
    for alpha in alpha_l:    
        for sigma in sigma_l:
            # set new values for alpha and sigma
            model = Household.HouseholdSpecializationModelClass()
            model.par.alpha = alpha
            model.par.sigma = sigma

            # solve the model
            opt = model.solve_discrete()

            # store the result
            resultsHF[(alpha, sigma)] = opt.HF
            resultsHM[(alpha, sigma)] = opt.HM
            resultsHFHNM[(alpha, sigma)] = opt.HF/opt.HM

            # print results
            print(f"with an alpha = {alpha} and a sigma = {sigma}, we get that HF/HM = {opt.HF:.2f} / {opt.HM:.2f} = {opt.HF/opt.HM:.2f}")
                        
    # plot the results
    fig, ax = plt.subplots()
    for alpha in alpha_l:
        y = [resultsHFHNM[(alpha, sigma)] for sigma in sigma_l]
        ax.plot(sigma_l, y, label=f"alpha={alpha}")
    ax.set_xlabel("sigma")
    ax.set_ylabel("HF/HM")
    ax.legend()
    plt.show()

    # plot the results 2.0
    fig1, ax = plt.subplots()
    for sigma in sigma_l:
        y = [resultsHFHNM[(alpha, sigma)] for alpha in alpha_l]
        ax.plot(alpha_l, y, label=f"sigma={sigma}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("HF/HM")
    ax.legend()
    plt.show()

    return y