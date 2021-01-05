# this script simulates stage-structured population sizes in space and time for given parameters and forcin
# functions
# Initially this was called simulation_population
from __future__ import print_function, division

import numpy as np
import math as math
import random as random
import sys
import copy as copy
from scipy.stats import norm
import statsmodels.api as sm
from scipy import stats
from numpy.linalg import inv
import pymc3
import scipy.stats as sst
from sklearn import preprocessing
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.kernel_ridge import KernelRidge
############################################################################################
######################
def temp_dependence(temperature, Topt, width, kopt):
    """ compute growth rate as a function of temperature, were kopt is the optimal growth rate, Topt, optimal temperature, width, the standard deviation from the optimal temperature.
        """
    #theta = -width*(temperature-Topt)*(temperature-Topt) + kopt
    theta = kopt*np.exp(-0.5*np.square((temperature-Topt)/width))
    #theta=((temperature-Tmin)*(temperature-Tmax))/(((temperature-Tmin)*(temperature-Tmax))-(temperature-Topt))
    return theta
#####################################################################################
def species1(params, temperatures, T_FINAL, no_patches, N_J0, N_Y0,N_A0):
    """ Takes in the initial population sizes and simulates the population size moving forward """
    # Set starting numbers for population and allocate space for population sizes
    rows=T_FINAL
    cols=no_patches
    N_J = np.ndarray(shape=(rows, cols), dtype=float, order='F')
    alpha=N_J.copy()
    N_Y=N_J.copy()
    N_A=N_J.copy()
    N_J[0] = N_J0
    N_Y[0] = N_Y0
    N_A[0] = N_A0
    for x in range(0, cols):
        alpha[:,x]=temp_dependence(temperatures[:,x],  params["Topt"], params["width"], params["kopt"])
    #N_Y = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
    #N_Y[0] = N_Y0
    #N_A = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
    #N_A[0] = N_A0
    for t in range(0,rows-1):
        #boundary values
        N_J[t+1,0]=max(0,(1-params["m_J"])*N_J[t,0]-params["g_J"]*N_J[t,0]+N_A[t,0]*np.exp(alpha[t,0]*(1-N_A[t,0]/params["K"])))
        N_J[t+1,cols-1]=max(0,(1-params["m_J"])*N_J[t,cols-1]-params["g_J"]*N_J[t,cols-1]+N_A[t,cols-1]*np.exp(alpha[t,cols-1]*(1-N_A[t,cols-1]/params["K"])))
        N_Y[t+1, 0] = max(0,(1-params["m_Y"])*N_Y[t,0]+params["g_J"]*N_J[t,0]-params["g_Y"]*N_Y[t,0])
        N_Y[t+1, cols-1] = max(0,(1-params["m_Y"])*N_Y[t,cols-1]+params["g_J"]*N_J[t,cols-1]-params["g_Y"]*N_Y[t,cols-1])
        N_A[t+1, 0]=max(0,(1-params["m_A"])*N_A[t,0]+params["g_Y"]*N_Y[t,0]-params["xi"]*N_A[t,0]+ params["xi"]*N_A[t,1])
        N_A[t+1, cols-1]=max(0,(1-params["m_A"])*N_A[t,cols-1]+params["g_Y"]*N_Y[t,cols-1]-params["xi"]*N_A[t,cols-1]+ params["xi"]*N_A[t,cols-2])
        for x in range(1, cols-1):
            N_J[t+1,x]=max(0,(1-params["m_J"])*N_J[t,x]-params["g_J"]*N_J[t,x]+N_A[t,x]*np.exp(alpha[t,x]*(1-N_A[t,x]/params["K"])))
            N_Y[t+1, x] = max(0,(1-params["m_Y"])*N_Y[t,x]+params["g_J"]*N_J[t,x]-params["g_Y"]*N_Y[t,x])
            N_A[t+1, x]=max(0,(1-params["m_A"])*N_A[t,x]+params["g_Y"]*N_Y[t,x]-2*params["xi"]*N_A[t,x]+ params["xi"]*N_A[t,x+1]+params["xi"]*N_A[t,x-1])
    return N_J, N_Y, N_A


#############################################################################

def species2(params, temperatures, T_FINAL, no_patches, N_J0, N_Y0,N_A0):
    """ Takes in the initial population sizes and simulates the population size moving forward """
    # Set starting numbers for population and allocate space for population sizes
    rows=T_FINAL
    cols=no_patches
    N_J = np.ndarray(shape=(rows, cols), dtype=float, order='F')
    alpha=N_J.copy()
    N_Y=N_J.copy()
    N_A=N_J.copy()
    N_J[0] = N_J0
    N_Y[0] = N_Y0
    N_A[0] = N_A0
    for x in range(0, cols):
        alpha[:,x]=temp_dependence(temperatures[:,x],  params["Topt"], params["width"], params["kopt"])
    #N_Y = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
    #N_Y[0] = N_Y0
    #N_A = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
    #N_A[0] = N_A0
    for t in range(0,rows-1):
        #boundary values
        N_J[t+1,0]=max(0,(1-params["m_J"])*N_J[t,0]-params["g_J"]*N_J[t,0]+(1-params["xi"])*N_A[t,0]*alpha[t,0]*(1-(N_A[t,0]+N_Y[t,0]+N_J[t,0])/params["K"])+ params["xi"]*N_A[t,1]*alpha[t,1]*(1-(N_A[t,1]+N_Y[t,1]+N_J[t,1])/params["K"]))
        N_J[t+1,cols-1]=max(0,(1-params["m_J"])*N_J[t,cols-1]-params["g_J"]*N_J[t,cols-1]+(1-params["xi"])*N_A[t,cols-1]*alpha[t,cols-1]*(1-(N_A[t,cols-1]+N_Y[t,cols-1]+N_J[t,cols-1])/params["K"])+ params["xi"]*N_A[t,cols-2]*alpha[t,cols-2]*(1-(N_A[t,cols-2]+N_Y[t,cols-2]+N_J[t,cols-2])/params["K"]))
        N_Y[t+1, 0] = max(0,(1-params["m_Y"])*N_Y[t,0]+params["g_J"]*N_J[t,0]-params["g_Y"]*N_Y[t,0])
        N_Y[t+1, cols-1] = max(0,(1-params["m_Y"])*N_Y[t,cols-1]+params["g_J"]*N_J[t,cols-1]-params["g_Y"]*N_Y[t,cols-1])
        N_A[t+1, 0]=max(0,(1-params["m_A"])*N_A[t,0]+params["g_Y"]*N_Y[t,0])
        N_A[t+1, cols-1]=max(0,(1-params["m_A"])*N_A[t,cols-1]+params["g_Y"]*N_Y[t,cols-1])
        for x in range(1, cols-1):
            N_J[t+1,x]=max(0,(1-params["m_J"])*N_J[t,x]-params["g_J"]*N_J[t,x]+(1-2*params["xi"])*N_A[t,x]*alpha[t,x]*(1-(N_A[t,x]+N_Y[t,x]+N_J[t,x])/params["K"])+ params["xi"]*N_A[t,x+1]*alpha[t,x+1]*(1-(N_A[t,x+1]+N_Y[t,x+1]+N_J[t,x+1])/params["K"])+ params["xi"]*N_A[t,x-1]*alpha[t,x-1]*(1-(N_A[t,x-1]+N_Y[t,x-1]+N_J[t,x-1])/params["K"]))
            N_Y[t+1, x] = max(0,(1-params["m_Y"])*N_Y[t,x]+params["g_J"]*N_J[t,x]-params["g_Y"]*N_Y[t,x])
            N_A[t+1, x]=max(0,(1-params["m_A"])*N_A[t,x]+params["g_Y"]*N_Y[t,x])
    return N_J, N_Y, N_A

##################################################################################################################################################################
##Species 3
def species3(params, temperatures, T_FINAL, no_patches, N_J0, N_Y0,N_A0):
    """ Takes in the initial population sizes and simulates the population size moving forward """
    # Set starting numbers for population and allocate space for population sizes
    rows=T_FINAL
    cols=no_patches
    N_J = np.ndarray(shape=(rows, cols), dtype=float, order='F')
    alpha=N_J.copy()
    N_Y=N_J.copy()
    N_A=N_J.copy()
    N_J[0] = N_J0
    N_Y[0] = N_Y0
    N_A[0] = N_A0
    for x in range(0, cols):
        alpha[:,x]=temp_dependence(temperatures[:,x],  params["Topt"], params["width"], params["kopt"])
    #N_Y = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
    #N_Y[0] = N_Y0
    #N_A = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
    #N_A[0] = N_A0
    L_bJ=params["L_inf"]-(params["L_inf"]-params["L_J"])*np.exp(alpha)
    L_bY=params["L_inf"]-(params["L_inf"]-params["L_Y"])*np.exp(alpha)
    g_J=(params["L_J"]-L_bJ)/(params["L_J"]-params["L_0"])
    g_Y=(params["L_Y"]-L_bY)/(params["L_Y"]-params["L_J"])
    for t in range(0,rows-1):
        #boundary values
        #N_J[t+1,0]=max(0,(1-params["m_J"])*N_J[t,0]+N_A[t,0]*np.exp(params["r"]*(1-N_A[t,0]/params  ["K"]))-g_J[t,0]*N_J[t,0])
        N_J[t+1,0]=max(0,(1-params["m_J"])*N_J[t,0]+N_A[t,0]*params["r"]-g_J[t,0]*N_J[t,0])
        #N_J[t+1,cols-1]=max(0,(1-params["m_J"])*N_J[t,cols-1]+N_A[t,cols-1]*np.exp(params ["r"]*(1-N_A[t,cols-1]/params["K"]))-g_J[t,cols-1]*N_J[t,cols-1])
        N_J[t+1,cols-1]=max(0,(1-params["m_J"])*N_J[t,cols-1]+N_A[t,cols-1]*params ["r"]-g_J[t,cols-1]*N_J[t,cols-1])
        N_Y[t+1, 0] = max(0,(1-params["m_Y"])*N_Y[t,0]+g_J[t,0]*N_J[t,0]-g_Y[t,0]*N_Y[t,0])
        N_Y[t+1, cols-1] = max(0,(1-params["m_Y"])*N_Y[t,cols-1]+g_J[t, cols-1]*N_J[t,cols-1]-g_Y[t,cols-1]*N_Y[t,cols-1])
        
        N_A[t+1, 0]=max(0,(1-params["m_A"])*N_A[t,0]+g_Y[t,0]*N_Y[t,0]-params["xi"]*N_A[t,0]+ params["xi"]*N_A[t,1])
        N_A[t+1, cols-1]=max(0,(1-params["m_A"])*N_A[t,cols-1]+g_Y[t,cols-1]*N_Y[t,cols-1]-params["xi"]*N_A[t,cols-1]+ params["xi"]*N_A[t,cols-2])
        for x in range(1, cols-1):
            #N_J[t+1, x]=max(0,(1-params["m_J"])*N_J[t,x]+N_A[t,x]*np.exp(params["r"]*(1-N_A[t,x]/params["K"]))-g_J[t,x]*N_J[t,x])
            N_J[t+1, x]=max(0,(1-params["m_J"])*N_J[t,x]+N_A[t,x]*params["r"]-g_J[t,x]*N_J[t,x])
            N_Y[t+1, x] = max(0,(1-params["m_Y"])*N_Y[t,x]+g_J[t,x]*N_J[t,x]-g_Y[t,x]*N_Y[t,x])
            N_A[t+1, x]=max(0,(1-params["m_A"])*N_A[t,x]+g_Y[t,x]*N_Y[t,x]-2*params["xi"]*N_A[t,x]+ params["xi"]*N_A[t,x+1]+params["xi"]*N_A[t,x-1])
    return N_J, N_Y, N_A

