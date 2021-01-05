# this script simulates stage-structured population sizes in space and time for given parameters and forcin
# functions
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
from pygam import PoissonGAM, s, te, LinearGAM, GAM
import pandas as pd
############################################################################################
######################


def temp_dependence(temperature, Topt, width, kopt):
    """ compute growth rate as a function of temperature, were kopt is the optimal growth rate, Topt, optimal temperature, width, the standard deviation from the optimal temperature.
        """
    #theta = -width*(temperature-Topt)*(temperature-Topt) + kopt
    theta = kopt*np.exp(-0.5*np.square((temperature-Topt)/width))
    #theta=((temperature-Tmin)*(temperature-Tmax))/(((temperature-Tmin)*(temperature-Tmax))-(temperature-Topt))
    return theta
#############################################################################
#Solving the systemof equations
def simulation_population(params, time_general, temp):
    """ Takes in the initial population sizes and simulates the population size moving forward """
    # Set starting numbers for population and allocate space for population sizes
    rows=time_general
    cols=no_patches
    N_J = np.ndarray(shape=(rows, cols), dtype=float, order='F')
    alpha=N_J.copy()
    N_Y=N_J.copy()
    N_A=N_J.copy()
    N_J[0] = N_J0
    N_Y[0] = N_Y0
    N_A[0] = N_A0
    for x in range(0, cols):
        alpha[:,x]=temp_dependence(temp[:,x],  params["Topt"], params["width"], params["kopt"])
    #N_Y = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
    #N_Y[0] = N_Y0
    #N_A = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
    #N_A[0] = N_A0
    for t in range(0,rows-1):
        #boundary values
        N_J[t+1,0]=max(0,(1-params["m_J"])*N_J[t,0]-params["g_J"]*N_J[t,0]+(1-params["xi"])*N_A[t,0]*alpha[t,0]*(1-(N_J[t,0]+N_Y[t,0]+N_A[t,0])/params["K"])+ params["xi"]*N_A[t,1]*alpha[t,1]*(1-(N_J[t,1]+N_Y[t,1]+N_A[t,1])/params["K"]))
        N_J[t+1,cols-1]=max(0,(1-params["m_J"])*N_J[t,cols-1]-params["g_J"]*N_J[t,cols-1]+(1-params["xi"])*N_A[t,cols-1]*alpha[t,cols-1]*(1-(N_J[t,cols-1]+N_Y[t,cols-1]+N_A[t,cols-1])/params["K"])+ params["xi"]*N_A[t,cols-2]*alpha[t,cols-2]*(1-(N_J[t,cols-2]+N_Y[t,cols-2]+N_A[t,cols-2])/params["K"]))
        N_Y[t+1, 0] = max(0,(1-params["m_Y"])*N_Y[t,0]+params["g_J"]*N_J[t,0]-params["g_Y"]*N_Y[t,0])
        N_Y[t+1, cols-1] = max(0,(1-params["m_Y"])*N_Y[t,cols-1]+params["g_J"]*N_J[t,cols-1]-params["g_Y"]*N_Y[t,cols-1])
        N_A[t+1, 0]=max(0,(1-params["m_A"])*N_A[t,0]+params["g_Y"]*N_Y[t,0])
        N_A[t+1, cols-1]=max(0,(1-params["m_A"])*N_A[t,cols-1]+params["g_Y"]*N_Y[t,cols-1])
        for x in range(1, cols-1):
            N_J[t+1,x]=max(0,(1-params["m_J"])*N_J[t,x]-params["g_J"]*N_J[t,x]+(1-2*params["xi"])*N_A[t,x]*alpha[t,x]*(1-(N_J[t,x]+N_Y[t,x]+N_A[t,x])/params["K"])+ params["xi"]*N_A[t,x+1]*alpha[t,x+1]*(1-(N_J[t,x+1]+N_Y[t,x+1]+N_A[t,x+1])/params["K"])+ params["xi"]*N_A[t,x-1]*alpha[t,x-1]*(1-(N_J[t,x-1]+N_Y[t,x-1]+N_A[t,x-1])/params["K"]))
            N_Y[t+1, x] = max(0,(1-params["m_Y"])*N_Y[t,x]+params["g_J"]*N_J[t,x]-params["g_Y"]*N_Y[t,x])
            N_A[t+1, x]=max(0,(1-params["m_A"])*N_A[t,x]+params["g_Y"]*N_Y[t,x])
    return N_J, N_Y, N_A,  alpha

####################################################################################################################################################
def simulation_populationnew(params, time_general, temp):
    """ Takes in the initial population sizes and simulates the population size moving forward """
    # Set starting numbers for population and allocate space for population sizes
    rows=time_general
    cols=no_patches
    NNS = np.ndarray(shape=(rows, cols), dtype=float, order='F')
    alphanew=NNS.copy()
    NNS[0,:]=NNS[0,:]
    for x in range(0, cols):
        alphanew[:,x]=temp_dependence(temp[:,x],  params["ToptN"], params["widthN"], params["koptN"])
    #N_Y = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
    #N_Y[0] = N_Y0
    #N_A = np.ndarray(shape=(T_FINAL+1, 2), dtype=float, order='F')
    #N_A[0] = N_A0
    for t in range(0,rows-1):
        #boundary values
        NNS[t+1,0]=max(0,(1-params["mN"])*NNS[t,0]+(1-params["xiN"])*NNS[t,0]*alphanew[t,0]*(1-NNS[t,0]/params["KN"])+ params["xiN"]*NNS[t,1]*alphanew[t,1]*(1-NNS[t,1]/params["KN"]))
        NNS[t+1,cols-1]=max(0,(1-params["mN"])*NNS[t,cols-1]+(1-params["xiN"])*NNS[t,cols-1]*alphanew[t,cols-1]*(1-NNS[t,cols-1]/params["KN"])+ params["xiN"]*NNS[t,cols-2]*alphanew[t,cols-2]*(1-NNS[t,cols-2]/params["KN"]))
        for x in range(1, cols-1):
            NNS[t+1,x]=max(0,(1-params["mN"])*NNS[t,x]+(1-2*params["xiN"])*NNS[t,x]*alphanew[t,x]*(1-NNS[t,x]/params["KN"])+ params["xiN"]*NNS[t,x+1]*alphanew[t,x+1]*(1-NNS[t,x+1]/params["KN"])+ params["xiN"]*NNS[t,x-1]*alphanew[t,x-1]*(1-NNS[t,x-1]/params["KN"]))
    return NNS, alphanew##############

def calculate_summary_stats(N_J, N_Y, N_A):
    """Takes in a matrix of time x place population sizes for each stage and calculates summary statistics"""
    time=range(T_FINAL)
    total_adult = N_A.sum(axis=1) # total population in each stage, summed over space
    total_young = N_Y.sum(axis=1)
    total_juv   = N_J.sum(axis=1)
    #lquartile_adult=np.percentile(total_adult, 25)
    L_Q1=np.percentile(time, 5, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    L_Q=np.percentile(time, 25, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    M_Q=np.percentile(time, 50, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    M_Q1=np.percentile(time,60, axis=None, out=None, overwrite_input=False,interpolation='nearest')
    U_Q=np.percentile(time, 75, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    U_Q1=np.percentile(time, 95, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    #print('time:', time)
    #print('LQ:', L_Q, ' MQ:', M_Q, ' UQ:', U_Q)
    lquartile_adult=total_adult[L_Q]#np.percentile(total_adult, 25)
    median_adult=total_adult[M_Q]#np.percentile(total_adult, 50)
    uquartile_adult=total_adult[U_Q]#np.percentile(total_adult, 75)
    mean_adult=np.mean(total_adult)
    std_adult=np.std(total_adult)
    lquartile_young=total_young[L_Q]#np.percentile(total_juv, 25)
    median_young=total_young[M_Q]#np.percentile(total_juv, 50)
    uquartile_young=total_young[U_Q]#np.percentile(total_juv, 75)
    mean_young=np.mean(total_young)
    std_young=np.std(total_young)
    lquartile_juv=total_juv[L_Q]#np.percentile(total_larv, 25)
    median_juv=total_juv[M_Q]#np.percentile(total_larv, 50)
    uquartile_juv=total_juv[U_Q]#np.percentile(total_larv, 75)
    mean_juv=np.mean(total_juv)
    std_juv=np.std(total_juv)
    #print('total_adult:', total_adult)
    #print('N_A:', N_A)
    SS_adult=np.hstack((lquartile_adult, median_adult, uquartile_adult))
    SS_young=np.hstack((lquartile_young, median_young, uquartile_young))
    SS_juv=np.hstack((lquartile_juv, median_juv, uquartile_juv))
    SS_adult1=np.hstack((N_A[L_Q1], N_A[L_Q], N_A[M_Q],N_A[M_Q1], N_A[U_Q], N_A[U_Q1]))
    SS_young1=np.hstack((N_Y[L_Q1], N_Y[L_Q], N_Y[M_Q],N_Y[M_Q1], N_Y[U_Q], N_Y[U_Q1]))
    SS_juv1=np.hstack((N_J[L_Q1], N_J[L_Q], N_J[M_Q], N_J[M_Q1],N_J[U_Q], N_J[U_Q1]))
    #total_population = total_adult + total_juv + total_larv # total population size in each time
    #print(total_adult)
    # print(lquartile_adult)
    #print(SS_adult)
    #print(mean_adult)
    #sys.exit()
    return SS_adult1, SS_young1, SS_juv1
##############################################################################################################################
def calculate_summary_statsnew(Nnew):
    """Takes in a matrix of time x place population sizes for each stage and calculates summary statistics"""
    time=range(T_FINAL)
    L_Q1=np.percentile(time, 5, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    L_Q=np.percentile(time, 25, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    M_Q=np.percentile(time, 50, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    M_Q1=np.percentile(time,60, axis=None, out=None, overwrite_input=False,interpolation='nearest')
    U_Q=np.percentile(time, 75, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    U_Q1=np.percentile(time, 80, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    U_Q2=np.percentile(time, 85, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    U_Q3=np.percentile(time, 90, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    U_Q4=np.percentile(time,95, axis=None, out=None, overwrite_input=False,interpolation='nearest')
    U_Q5=np.percentile(time, 97, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    U_Q6=np.percentile(time, 99, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    
    SS_NN1=np.hstack((Nnew[L_Q1], Nnew[L_Q], Nnew[M_Q],Nnew[M_Q1], Nnew[U_Q],Nnew[U_Q1]))#, np.std(N_A, axis=0), np.mean(N_A, axis=0)
    return SS_NN1
##################################################################################################################

def small_percent(vector, percent):
    """ Takes a vector and returns the indexes of the elements within the smallest (percent) percent of the vector"""
    sorted_vector = sorted(vector)
    cutoff = math.floor(len(vector)*percent/100) # finds the value which (percent) percent are below
    indexes = []
    print('cutoff:',cutoff)
    cutoff = int(cutoff)
    for i in range(0,len(vector)):
        if vector[i] < sorted_vector[cutoff]: # looks for values below the found cutoff
            indexes.append(i)

    return indexes, sorted_vector[cutoff]


def z_score(x):
    """Takes a list and returns a 0 centered, std = 1 scaled version of the list"""
    st_dev = np.std(x,axis=0)
    mu = np.mean(x,axis=0)
    rescaled_values = []
    for element in range(0,len(x)):
        rescaled_values[element] = (x[element] - mu) / st_dev

    return rescaled_values

############################
# this function transform the paramters using logit function. the aim is to ensure that we do not end up with a parameter out of the prior
def do_logit_transformation(library, param_bound):
    for i in range(len(library[0,:])):
        library[:,i]=(library[:,i]-param_bound[i,0])/(param_bound[i,1]-param_bound[i,0])
        library[:,i]=np.log(library[:,i]/(1-library[:,i]))
    return library
###########################
#this function back transform parameter values
def do_ivlogit_transformation(para_reg, param_bound):
    for i in range(len(library[0,:])):
        para_reg[:,i]=np.exp(para_reg[:,i])/(1+np.exp(para_reg[:,i]))
        para_reg[:,i]=para_reg[:,i]*(param_bound[i,1]-param_bound[i,0])+param_bound[i,0]
    return para_reg
############################

def do_kernel_ridge(stats, library, param_bound):
    #print('X:', X.shape)
    #print('Y:', Y.shape)
    #'rbf'
    X = sm.add_constant(stats)
    Y=library
    clf     = KernelRidge(alpha=1.0, kernel='rbf', coef0=1)
    resul   = clf.fit(X, Y)
    resul_coef=np.dot(X.transpose(), resul.dual_coef_)
    coefficients =resul_coef[1:]
    #mean_conf=confidence_interval_Kridge(logit(library), weights, stats,resul_coef)
    para_reg   =Y- stats.dot(coefficients)
    para_reg=do_ivlogit_transformation(para_reg, param_bound)
    #param_SS[:,ii]   =Y[:,ii]- inv_logit(res_wls_SS.params[1:])+inv_logit(res_wls_OS.params[1:])
    parameter_estimate = np.average(para_reg, axis=0)
    HPDR=pymc3.stats.hpd(para_reg)
    return parameter_estimate, HPDR
    #NMSE_ridreg=1-(((np.linalg.norm(actual-coefficients , ord=2))**2)/((np.linalg.norm(actual- np.mean(actual), ord=2))**2))
    #print('Estimates from regression abc using Kernel ridge regression is :', parameter_estimate)
#print('Estimates HPDR using Kernel ridge regression is :', HPDR)
# print('NMSE for kernel Ridge regression  is :', NMSE_ridreg)
    #print('coefficients:', coefficients)
##############################################################################################
def do_rejection(library):
    parameter_estimate = np.average(library, axis=0)
    HPDR=pymc3.stats.hpd(library)
    return parameter_estimate, HPDR
    # print('library is:', library)
    #print('Estimates from rejection is:', parameter_estimate)
#print('Estimates HPDR from rejection is :', HPDR)

    #####################################################################################
#return all the observe summaery statitics (OS)and simulated summary statistics (SS) in a matrix with first row corresponding to OS and the rest of the rows to SS
def run_sim(param_actual):
    PARAMS_ABC = copy.deepcopy(param_actual) # copies parameters so new values can be generated; FIX ME! this is a redirect, not a copy?
    param_save = [] # sets an initial 0; fixed to [] because [[]] made the mean go poorly (averaging in an [] at start?)
    
    #print_parameters(PARAMS, prefix='True')
    N_J, N_Y, N_A, grate = simulation_population(param_actual, T_FINAL, temp1)
    SS_adult, SS_young, SS_juv= calculate_summary_stats(N_J, N_Y, N_A)
    SO=np.hstack((SS_adult, SS_young, SS_juv))
    Obs_Sim=np.zeros((NUMBER_SIMS+1,len(SO)))
    Obs_Sim[0,:]=SO
    for i in range(0,NUMBER_SIMS):
        g_J_theta    = np.random.uniform(0,1)#np.random.normal(0.4,0.3) #np.random.beta(2,2)
        g_Y_theta    =np.random.uniform(0,1) #np.random.uniform(0,1)#np.random.beta(2,2)
        Topt_theta =np.random.uniform(1,9)#np.random.normal(6.5,2) #np.random.uniform(1,12) #np.random.lognormal(1,1)
        width_theta  =np.random.uniform(1,20)#np.random.normal(2,1)
        ##np.random.lognormal(1,1)
        kopt_theta    =np.random.uniform(0,1)#np.random.normal(0.5,0.4)# np.random.u(0,1)
        xi_theta     =np.random.uniform(0,0.5/2)#np.random.normal(0.1,0.09) #np.random.normal(0,1)#np.random.normal(0,0.5)
        m_J_theta    =np.random.uniform(0,1)#np.random.normal(0.04,0.04) # #np.random.beta(2,2)
        m_Y_theta    =np.random.uniform(0,1)#np.random.normal(0.05,0.04) #np.random.uniform(0,1) #np.random.beta(2,2)
        m_A_theta    =np.random.uniform(0,1)#np.random.normal(0.05,0.05)# np.random.uniform(0,1)#np.random.beta(2,2)
        K_theta= np.random.uniform(100,3000)
        PARAMS_ABC["g_J"]    = g_J_theta # sets the g_J parameter to our random guess
        PARAMS_ABC["g_Y"]    = g_Y_theta
        PARAMS_ABC["Topt"] = Topt_theta
        PARAMS_ABC["width"]  = width_theta
        PARAMS_ABC["kopt"]    = kopt_theta
        PARAMS_ABC["xi"]     = xi_theta
        PARAMS_ABC["m_J"]    = m_J_theta
        PARAMS_ABC["m_Y"]    = m_Y_theta
        PARAMS_ABC["m_A"]    = m_A_theta
        PARAMS_ABC["K"]    = K_theta
        # Simulate population for new parameters
        N_J_sim, N_Y_sim, N_A_sim, gratesim = simulation_population(PARAMS_ABC, T_FINAL, temp1) # simulates population with g_J value
        # Calculate the summary statistics for the simulation
        Sim_SS_adult, Sim_SS_young, Sim_SS_juv= calculate_summary_stats(N_J_sim, N_Y_sim, N_A_sim)
        SS=np.hstack((Sim_SS_adult, Sim_SS_young, Sim_SS_juv))
        Obs_Sim[i+1,:]=SS
        
        param_save.append([g_J_theta, g_Y_theta, Topt_theta, width_theta, kopt_theta, xi_theta, m_J_theta,m_Y_theta, m_A_theta, K_theta])
    
    return np.asarray(param_save), Obs_Sim, N_J, N_Y, N_A

#########################################################################################
#####################################################################################
#return all the observe summaery statitics (OS)and simulated summary statistics (SS) in a matrix with first row corresponding to OS and the rest of the rows to SS
def run_simnew():
    PARAMS_ABC = {}# copies parameters so new values can be generated; FIX ME! this is a redirect, not a copy?
    param_save = [] # sets an initial 0; fixed to [] because [[]] made the mean go poorly (averaging in an [] at start?
    #print_parameters(PARAMS, prefix='True')
    SS_NN= calculate_summary_statsnew(NN)
    SO=np.hstack((SS_NN))
    Obs_Sim=np.zeros((NUMBER_SIMS+1,len(SO)))
    Obs_Sim[0,:]=SO
    for i in range(0,NUMBER_SIMS):
        ToptN_theta =np.random.uniform(1, 9)#np.random.normal(6.5,2) #np.random.uniform(1,12) #np.random.lognormal(1,1)
        widthN_theta  =np.random.uniform(1,20)#np.random.normal(2,1)
        ##np.random.lognormal(1,1)
        koptN_theta    =np.random.uniform(0,1)#np.random.normal(0.5,0.4)# np.random.u(0,1)
        xiN_theta     =np.random.uniform(0,0.5/2)#np.random.normal(0.1,0.09) #np.random.normal(0,1)#np.random.normal(0,0.5)
        mN_theta    =np.random.uniform(0,1)#np.random.normal(0.04,0.04) # #np.random.beta(2,2)
        KN_theta= np.random.uniform(100, 3000)
        PARAMS_ABC["ToptN"] = ToptN_theta
        PARAMS_ABC["widthN"]  = widthN_theta
        PARAMS_ABC["koptN"]    = koptN_theta
        PARAMS_ABC["xiN"]     = xiN_theta
        PARAMS_ABC["mN"]    = mN_theta
        PARAMS_ABC["KN"]    = KN_theta
        # Simulate population for new parameters
        NN_sim, gratesim = simulation_populationnew(PARAMS_ABC, T_FINAL, temp1)  # simulates population with g_J value
        # Calculate the summary statistics for the simulation
        Sim_SS= calculate_summary_statsnew(NN_sim)
        SS=np.hstack((Sim_SS))
        Obs_Sim[i+1,:]=SS
        
        param_save.append([ToptN_theta, widthN_theta, koptN_theta, xiN_theta, mN_theta, KN_theta])
    
    return np.asarray(param_save), Obs_Sim

#########################################################################################
def compute_scores(dists, param_save, difference,Sim_SS):
    eps=0.01
    library_index, NSS_cutoff = small_percent(dists, eps)
    n                = len(library_index)
    library = np.empty((n, param_save.shape[1]))
    stats            = np.empty((n, difference.shape[1]))
    stats_SS            = np.empty((n, difference.shape[1]))
    

    for i in range(0,len(library_index)):
        j = library_index[i]
        library[i] = param_save[j]
        stats[i]   = difference[j]
        stats_SS[i]   = Sim_SS[j]
    return library, stats, NSS_cutoff, library_index, stats_SS
##########################################################################################
#computes weights for local regression

def compute_weight(kernel,t, eps, index):
     weights=np.empty(len(index))
     if (kernel == "epanechnikov"):
         for i in range(0,len(library_index)):
             j = library_index[i]
     #weights[i]= (1. - (t[j] / eps)**2)
             weights[i]=(1. - (t[j] / eps)**2)
     elif(kernel == "rectangular"):
          for i in range(0,len(library_index)):
              j = library_index[i]
              weights[i]=t[j] / eps
     elif (kernel == "gaussian"):
          for i in range(0,len(library_index)):
              j = library_index[i]
              weights[i]= 1/np.sqrt(2*np.pi)*np.exp(-0.5*(t[j]/(eps/2))**2)
            
     elif (kernel == "triangular"):
          for i in range(0,len(library_index)):
              j = library_index[i]
              weights[i]= 1 - np.abs(t[j]/eps)
     elif (kernel == "biweight"):
          for i in range(0,len(library_index)):
              j = library_index[i]
              weights[i]=(1 - (t[j]/eps)**2)**2
     else:
          for i in range(0,len(library_index)):
              j = library_index[i]
              weights[i]= np.cos(np.pi/2*t[j]/eps)
     return weights
############################################################################################
def actual_params(PARAMS):
    PARAMS["g_J"]   = np.random.uniform(0,0.7)#np.random.normal(0.4,0.3) #np.random.beta(2,2)
    PARAMS["g_Y"]    =np.random.uniform(0,0.7) #np.random.uniform(0,1)#np.random.beta(2,2)
    PARAMS["Topt"]  =np.random.uniform(10, 20)#np.random.normal(6.5,2) #np.random.uniform(1,12) #np.random.lognormal(1,1)
    PARAMS["width"]  =np.random.uniform(1,5)#np.random.normal(2,1)
    ##np.random.lognormal(1,1)
    PARAMS["kopt"]     =np.random.uniform(0,0.7)#np.random.normal(0.5,0.4)# np.random.u(0,1)
    PARAMS["xi"]     =np.random.uniform(0,0.5/4)#np.random.normal(0.1,0.09) #np.random.normal(0,1)#np.random.normal(0,0.5)
    PARAMS["m_J"]     =np.random.uniform(0,0.7)#np.random.normal(0.04,0.04) # #np.random.beta(2,2)
    PARAMS["m_Y"]   =np.random.uniform(0,0.7)#np.random.normal(0.05,0.04) #np.random.uniform(0,1) #np.random.beta(2,2)
    PARAMS["m_A"]   =np.random.uniform(0,0.7)#np.random.normal(0.05,0.05)# np.random.uniform(0,1)#np.random.beta(2,2)
    #PARAMS["delta_t"]   =0.1
    PARAMS["K"]   =np.random.uniform(500,2000)
    return PARAMS
##########################################################################################


def sum_stats(Obs_Sim, param_save):
    dists = np.zeros((NUMBER_SIMS,1))
    #Obs_Sim_scale=np.nan_to_num(sst.zscore(Obs_Sim, axis=0,ddof=1),copy=True)
    Obs_Sim_scale=np.nan_to_num(preprocessing.normalize(Obs_Sim, axis=0),copy=True)
    #Substract each row of teh array from row 1
    Sim_SS=Obs_Sim_scale[1:NUMBER_SIMS+1,: ]
    Obs_SS=Obs_Sim_scale[0,:]
    difference=Obs_Sim_scale[1:NUMBER_SIMS+1,: ]-Obs_Sim_scale[0,:]
    #c=np.std(Obs_Sim_scale[1:NUMBER_SIMS+1,: ], axis=1)
    # compute the norm 2 of each row
    dists = np.linalg.norm(difference, axis=1)

    library, stats, NSS_cutoff, library_index, stats_SS = compute_scores(dists, param_save, difference,Sim_SS)
    # print(library)
    return library, dists, stats,stats_SS,   NSS_cutoff, library_index
################################################################################################
###################################################################################################

def do_regression(library, stats, PARAMS):

    # REJECTION
    print('\nDo a rejection ABC:')
    do_rejection(library, PARAMS)
    do_local_linear(stats, library, weights,KK)
    #print('\nStats:', stats.shape)
    #print('\nStats:', stats)
    #print('\nLibar:', library.shape)
    #print('\nLibar:', library)

    do_kernel_ridge(stats, library)
    do_ridge(stats, library)
##################################################################################################
def do_goodness_fit(result,HPDR, actual, n, i):
    for j in range(0,n):
        if HPDR[j][0]<=actual[j]<=HPDR[j][1]:
           coverage[i,j]=1
        else:
           coverage[i,j]=0
    resultsbias[i,:] = (result - actual)/actual
    return coverage,resultsbias

#############################################################################################

if __name__ == '__main__':
    ############################################################################################
    # exact parameter values
    PARAMS = {"g_J": 0.4, "g_Y": 0.3, "Topt": 5, "width": 2, "kopt": 0.6,"xi":0.1, "m_J": .05, "m_Y": .05, "m_A": .05, "K":100}
    #final time
    param_bound=np.array([[0,1],[0,1],[3,6],[1,4],[0,1],[0,0.5],[0,1],[0,1],[0,1],[100, 500]])
    T_FINAL = 30
    T_forcast=70
    #number of patches
    #initial abundance in each patch and for each stage
    N_J0=200
    N_Y0=200
    N_A0= 200
    no_patches=10
    temp_max=24
    #Number of iteration
    rSize   = len(PARAMS)
    NUMBER_SIMS = 200000
    N_Species  = 1
    rows=T_forcast
    cols=no_patches
    coverage=np.empty((N_Species, rSize))
    resultsbias = np.empty((N_Species, rSize))
    temperatures = np.ndarray(shape=(rows, cols), dtype=float, order='F')
    #temperatures[:, 0]=np.linspace(0, temp_max,T_forcast)
    q=10
    for x in range(0, cols):
        #p=3*(x+1)
        temperatures[:, x]=np.linspace(q, q+5,T_forcast)
        q=q+1
    temp1=temperatures[0:T_FINAL,:]
# for q in range(1,no_patches+1):
#   D['A_bias'+ str(q)] = np.empty((N_Species, len(temperatures)))
#   D['Y_bias'+ str(q)] = np.empty((N_Species, len(temperatures)))
#    D['J_bias'+ str(q)] = np.empty((N_Species, len(temperatures)))
    for i in range(N_Species ):
        #param_actual = {"g_J": 0.49, "g_Y": 0.57, "Topt":16.21, "width":2.72, "kopt": 0.36,"xi":0.060, "m_J": 0.057, "m_Y": 0.055, "m_A":  0.073, "K":29991.02}
        param_actual=actual_params(PARAMS)
        actual=[param_actual["g_J"], param_actual["g_Y"], param_actual["Topt"], param_actual["width"], param_actual["kopt"],param_actual["xi"], param_actual["m_J"], param_actual["m_Y"], param_actual["m_A"], param_actual["K"]]
        #actual=[0.49, 0.57, 16.21, 2.72, 0.36, 0.060, 0.057, 0.055, 0.073, 29991.02]
    #############################################################################
    #simulating summary statistics and retaining a matrix with first row observed summar (OS) statistics and the remaining rows simulated summary (SS) statistics for NUMBER_SIMS iterations.It equally retain the simulated parameters. i'e retain all (theta_i, S_i) for i=0:NUMBER_SIMS.  theta_i and s_i from the joint distribution.
        param_save, Obs_Sim , N_J, N_Y, N_A        = run_sim(param_actual)
        NN= N_J+N_Y+N_A
        param_savenew, Obs_Simnew        = run_simnew()
    ######################################################################################################
        library, dists, stats,stats_SS,  NSS_cutoff, library_index   = sum_stats(Obs_Sim, param_save)
        result, HPDR =do_rejection(library)
        print('Estimates from rejection is:', result)
        print('Estimated HPDR from rejection is :', HPDR)
        librarynew, distsnew, statsnew,stats_SSnew,  NSS_cutoffnew, library_indexnew   = sum_stats(Obs_Simnew, param_savenew)
        resultnew, HPDRnew =do_rejection(librarynew)
        print('Estimates from 1 stage rejection is:', resultnew)
        print('Estimated HPDR from 1 stage  rejection is :', HPDRnew)
        #coverage_rej,resultsbias_rej=do_goodness_fit(result,HPDR, actual, len(PARAMS), i)
        #coverage_rej_percen=(np.array(coverage_rej).sum(axis=0)/N_Species*100)
        # print("The Coverage probabilityfrom rejection  is:", coverage_rej_percen)
        #print("The bias is:",resultsbias)
        print("The actual paramter estimate are:", actual)
        #library_reg=do_logit_transformation(library, param_bound)
        #result_reg, HPDR_reg=do_kernel_ridge(stats, library_reg, param_bound)
        #[7.46523971e-01 5.24399658e-01 6.19278685e+00 2.31461743e+00
        #5.42012598e-01 1.17146062e-01 5.61542583e-01 2.10611610e-01
        #   5.11170030e-01 6.07582106e+02]
        PARAMS1  = {}
        PARAMS1 = {"g_J": param_actual["g_J"], "g_Y": param_actual["g_Y"], "Topt":param_actual["Topt"], "width":param_actual["width"], "kopt": param_actual["kopt"],"xi":param_actual["xi"], "m_J": param_actual["m_J"], "m_Y": param_actual["m_Y"], "m_A": param_actual["m_A"], "K":param_actual["K"]}
        # print("the actual parameter under PARAMS 1 is:", PARAMS1)
#PARAMS1 = {"g_J": 7.46523971e-01, "g_Y": 5.24399658e-01, "Topt":6.19278685e+00, "width":2.31461743e+00, "kopt": 5.42012598e-01,"xi":1.17146062e-01, "m_J":5.61542583e-01, "m_Y": 2.10611610e-01, "m_A":  5.11170030e-01, "delta_t": param_actual["delta_t"], "K":6.07582106e+02}
        PARAMS2={}
        PARAMS2 = {"g_J": result[0], "g_Y": result[1], "Topt":result[2], "width":result[3], "kopt": result[4],"xi":result[5], "m_J": result[6], "m_Y": result[7], "m_A": result[8], "K":result[9]}
        PARAMS3={}
        PARAMS3 = {"ToptN":resultnew[0], "widthN":resultnew[1], "koptN": resultnew[2],"xiN":resultnew[3], "mN": resultnew[4], "KN":resultnew[5]}
#PARAMS2 = {"g_J": 7.96523971e-01, "g_Y": 5.24399658e-01, "Topt":6.29278685e+00, "width":2.31461743e+00, "kopt": 5.42012598e-01,"xi":1.17146062e-01, "m_J":5.61542583e-01, "m_Y": 2.10611610e-01, "m_A":  5.11170030e-01, "delta_t": param_actual["delta_t"], "K":6.07582106e+02}
        N_J1, N_Y1, N_A1, alpha1 = simulation_population( PARAMS1, T_forcast, temperatures)
        N_J2, N_Y2, N_A2, alpha2 = simulation_population( PARAMS2, T_forcast, temperatures)
        NNN=N_J1+N_Y1+N_A1
        NNS=N_J2+N_Y2+N_A2
        NN1, alphaalha1 = simulation_populationnew( PARAMS3, T_forcast, temperatures)
    time=np.array(range(0,70))
    abun=[]
    stage=[]
    for q in range(1,no_patches+1):
        fig1, ax1=plt.subplots()
        plt.plot(temperatures[:,q-1], alpha1[:,q-1], 'r--', linewidth=2)
        ax1.set_ylabel('growth rate', fontsize=18)
        ax1.set_xlabel('temperature',fontsize=18)
        ax1.tick_params(width = 2, direction = "out")
        fig1.savefig('growthrate'+ str(q))
        plt.close()
    import plot
    df=pd.DataFrame({'patch1_trueA':N_A1[:,0] , 'patch2_trueA':N_A1[:,1] ,  'patch3_trueA':N_A1[:,2] ,  'patch4_trueA':N_A1[:,3] , 'patch5_trueA':N_A1[:,4] , 'patch6_trueA':N_A1[:,5] , 'patch7_trueA':N_A1[:,6] , 'patch8_trueA':N_A1[:,7] , 'patch9_trueA':N_A1[:,8] , 'patch10_trueA':N_A1[:,9] ,'patch1_trueY':N_Y1[:,0] , 'patch2_trueY':N_Y1[:,1] ,  'patch3_trueY':N_Y1[:,2] ,  'patch4_trueY':N_Y1[:,3] , 'patch5_trueY':N_Y1[:,4] , 'patch6_trueY':N_Y1[:,5] , 'patch7_trueY':N_Y1[:,6] , 'patch8_trueY':N_Y1[:,7] , 'patch9_trueY':N_Y1[:,8] , 'patch10_trueY':N_Y1[:,9] , 'patch1_trueJ':N_J1[:,0] , 'patch2_trueJ':N_J1[:,1] ,  'patch3_trueJ':N_J1[:,2] ,  'patch4_trueJ':N_J1[:,3] , 'patch5_trueJ':N_J1[:,4] , 'patch6_trueJ':N_J1[:,5] , 'patch7_trueJ':N_J1[:,6] , 'patch8_trueJ':N_J1[:,7] , 'patch9_trueJ':N_J1[:,8] , 'patch10_trueJ':N_J1[:,9] ,
        'patch1_trueAS':N_A2[:,0] , 'patch2_trueAS':N_A2[:,1] ,  'patch3_trueAS':N_A2[:,2] ,  'patch4_trueAS':N_A2[:,3] , 'patch5_trueAS':N_A2[:,4] , 'patch6_trueAS':N_A2[:,5] , 'patch7_trueAS':N_A2[:,6] , 'patch8_trueAS':N_A2[:,7] , 'patch9_trueAS':N_A2[:,8] , 'patch10_trueAS':N_A2[:,9], 'patch1_trueYS':N_Y2[:,0] , 'patch2_trueYS':N_Y2[:,1] ,  'patch3_trueYS':N_Y2[:,2] ,  'patch4_trueYS':N_Y2[:,3] , 'patch5_trueYS':N_Y2[:,4] , 'patch6_trueYS':N_Y2[:,5] , 'patch7_trueYS':N_Y2[:,6] , 'patch8_trueYS':N_Y2[:,7] , 'patch9_trueYS':N_Y2[:,8] , 'patch10_trueYS':N_Y2[:,9], 'patch1_trueJS':N_J2[:,0] , 'patch2_trueJS':N_J2[:,1] ,  'patch3_trueJS':N_J2[:,2] ,  'patch4_trueJS':N_J2[:,3] , 'patch5_trueJS':N_J2[:,4] , 'patch6_trueJS':N_J2[:,5] , 'patch7_trueJS':N_J2[:,6] , 'patch8_trueJS':N_J2[:,7] , 'patch9_trueJS':N_J2[:,8] , 'patch10_trueJS':N_J2[:,9],
                    'patch1_onestage':NN1[:,0] , 'patch2_onestage':NN1[:,1] ,  'patch3_onestage':NN1[:,2],  'patch4_onestage':NN1[:,3], 'patch5_onestage':NN1[:,4], 'patch6_onestage':NN1[:,5] , 'patch7_onestage':NN1[:,6] , 'patch8_onestage':NN1[:,7] , 'patch9_onestage':NN1[:,8], 'patch10_onestage':NN1[:,9]})#,
    df1=pd.DataFrame({'patch1_30days':NN[:,0] , 'patch2_30days':NN[:,1] ,  'patch3_30days':NN[:,2],  'patch4_30days':NN[:,3], 'patch5_30days':NN[:,4], 'patch6_30days':NN[:,5] , 'patch7_30days':NN[:,6] , 'patch8_30days':NN[:,7] , 'patch9_30days':NN[:,8], 'patch10_30days':NN[:,9]})
    writer=species_distribution= pd.ExcelWriter('Gam_Single_Stage_comparism.xlsx')
    df.to_excel(writer, sheet_name='line_plot_comparism', index=False)
    writer.save()
    writer.close()
    writer=species_distribution= pd.ExcelWriter('ActualTotal_pop_templess70.xlsx')
    df.to_excel(writer, sheet_name='pop', index=False)
    writer.save()
    writer.close()
    for q in range(1,no_patches+1):
        plot.do_lineplot(N_A2[:,q-1], N_A1[:,q-1],  'species2_A_abun'+ str(q))
        plot.do_lineplot(N_Y2[:,q-1], N_Y1[:,q-1], 'species2_Y_abun'+ str(q))
        plot.do_lineplot(N_J2[:,q-1], N_J1[:,q-1], 'species2_J_abun'+ str(q))
    for q in range(1,no_patches+1):
        plot.do_lineplot(NN1[:,q-1], NNN[:,q-1],  'onestage'+ str(q))
    XGam=temp1[:,0]
    NNGam=NN[:,0]
    NNNGam=NNN[:,0]
    Nsimnew=np.ndarray(shape=(rows, cols), dtype=float, order='F')
    XXGam=temperatures[:,0]
    for q in range(1,no_patches):
        XGam=np.hstack(( XGam, temp1[:,q]))
        NNGam=np.hstack((NNGam, NN[:,q]))
        NNNGam=np.hstack((NNNGam, NNN[:,q]))
        #Z=temp1[:,q-1]
        XXGam=np.hstack((XXGam, temperatures[:,q]))
    gam = GAM().fit(XGam, NNGam)
    ZZ=gam.predict(XXGam)
    for q in range(1,no_patches+1):
        yy=NNN[:,q-1]
        #print(NNN[:,q-1])
        #XX=temperatures[:,q-1]
        Nsimnew[:,q-1]=ZZ[(q-1)*70:((q-1)*70)+70]
        #print(NNN[:,q-1])
        NNNsim=Nsimnew[:,q-1]
        ZZsim=pd.Series(NNNsim,index=pd.Series(range(0,70)))
        NO=pd.Series(yy,index=pd.Series(range(0,70)))
        fig, ax=plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.plot(ZZsim.index, ZZsim, 'r--', linewidth=2)
        plt.scatter(NO.index, NO, linewidths=2, facecolor='gray', edgecolors='none')
#plt.legend(loc='best')
        ax.set_ylabel('Abundance', fontsize=20)
        ax.set_xlabel('Time (years)',fontsize=20)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.tick_params(width = 2, direction = "out")
        fig.savefig('GAMmodel'+ str(q))
        plt.close()
        fig, ax=plt.subplots()
        NOT=pd.Series(NNN[:,q-1],index=pd.Series(range(0,70)))
        NSST=NNS[:,q-1]
        NSO=NN1[:,q-1]
        NSST=pd.Series(NSST[30:],index=pd.Series(range(30,70)))
        NSO=pd.Series(NSO[30:],index=pd.Series(range(30,70)))
        NGAM=pd.Series(NNNsim[30:],index=pd.Series(range(30,70)))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.scatter(NOT.index, NOT, linewidths=2, facecolor='gray', edgecolors='none')
        plt.plot(NSST.index, NSST, 'k', linewidth=2)
        plt.plot(NSO.index, NSO, 'r', linewidth=2)
        plt.plot(NGAM.index, NGAM, 'b', linewidth=2)
        ax.set_ylabel('Abundance', fontsize=20)
        ax.set_xlabel('Time (years)',fontsize=20)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.tick_params(width = 2, direction = "out")
        fig.savefig('Total'+ str(q), bbox_inches='tight')
        plt.close()



    


