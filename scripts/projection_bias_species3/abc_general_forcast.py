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
        N_J[t+1,0]=max(0,(1-params["m_J"])*N_J[t,0]-params["g_J"]*N_J[t,0]+(1-params["xi"])*N_A[t,0]*np.exp(alpha[t,0]*(1-N_A[t,0]/params["K"]))+ params["xi"]*N_A[t,1]*np.exp(alpha[t,1]*(1-N_A[t,1]/params["K"])))
        N_J[t+1,cols-1]=max(0,(1-params["m_J"])*N_J[t,cols-1]-params["g_J"]*N_J[t,cols-1]+(1-params["xi"])*N_A[t,cols-1]*np.exp(alpha[t,cols-1]*(1-N_A[t,cols-1]/params["K"]))+ params["xi"]*N_A[t,cols-2]*np.exp(alpha[t,cols-2]*(1-N_A[t,cols-2]/params["K"])))
        N_Y[t+1, 0] = max(0,(1-params["m_Y"])*N_Y[t,0]+params["g_J"]*N_J[t,0]-params["g_Y"]*N_Y[t,0])
        N_Y[t+1, cols-1] = max(0,(1-params["m_Y"])*N_Y[t,cols-1]+params["g_J"]*N_J[t,cols-1]-params["g_Y"]*N_Y[t,cols-1])
        N_A[t+1, 0]=max(0,(1-params["m_A"])*N_A[t,0]+params["g_Y"]*N_Y[t,0])
        N_A[t+1, cols-1]=max(0,(1-params["m_A"])*N_A[t,cols-1]+params["g_Y"]*N_Y[t,cols-1])
        for x in range(1, cols-1):
            N_J[t+1,x]=max(0,(1-params["m_J"])*N_J[t,x]-params["g_J"]*N_J[t,x]+(1-2*params["xi"])*N_A[t,x]*np.exp(alpha[t,x]*(1-N_A[t,x]/params["K"]))+ params["xi"]*N_A[t,x+1]*np.exp(alpha[t,x+1]*(1-N_A[t,x+1]/params["K"]))+ params["xi"]*N_A[t,x-1]*np.exp(alpha[t,x-1]*(1-N_A[t,x-1]/params["K"])))
            N_Y[t+1, x] = max(0,(1-params["m_Y"])*N_Y[t,x]+params["g_J"]*N_J[t,x]-params["g_Y"]*N_Y[t,x])
            N_A[t+1, x]=max(0,(1-params["m_A"])*N_A[t,x]+params["g_Y"]*N_Y[t,x])
    return N_J, N_Y, N_A

##################################################################################################################################################################

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

def small_percent(vector, percent):
    """ Takes a vector and returns the indexes of the elements within the smallest (percent) percent of the vector"""
    sorted_vector = sorted(vector)
    cutoff = math.floor(len(vector)*percent/100) # finds the value which (percent) percent are below
    indexes = []
    #print('cutoff:',cutoff)
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
    # HPDR=pymc3.stats.hpd(library)
    return parameter_estimate
    # print('library is:', library)
    #print('Estimates from rejection is:', parameter_estimate)
#print('Estimates HPDR from rejection is :', HPDR)

    #####################################################################################
#return all the observe summaery statitics (OS)and simulated summary statistics (SS) in a matrix with first row corresponding to OS and the rest of the rows to SS
def run_sim(param_actual):
    PARAMS_ABC = copy.deepcopy(param_actual) # copies parameters so new values can be generated; FIX ME! this is a redirect, not a copy?
    param_save = [] # sets an initial 0; fixed to [] because [[]] made the mean go poorly (averaging in an [] at start?)
    
    #print_parameters(PARAMS, prefix='True')
    N_J, N_Y, N_A = abc_example.species3(param_actual,temp1, T_FINAL, no_patches,N_J0, N_Y0,N_A0)
    SS_adult, SS_young, SS_juv= calculate_summary_stats(N_J, N_Y, N_A)
    SO=np.hstack((SS_adult, SS_young, SS_juv))
    Obs_Sim=np.zeros((NUMBER_SIMS+1,len(SO)))
    Obs_Sim[0,:]=SO
    for i in range(0,NUMBER_SIMS):
        L_0_theta    = np.random.uniform(0,5)#np.random.normal(0.4,0.3) #np.random.beta(2,2)
        L_inf_theta    =np.random.uniform(17,100)
        #L_J_theta=np.random.uniform(4, 6)
        #L_Y_theta=np.random.uniform(8, 12)
        #np.random.uniform(0,1)#np.random.beta(2,2)
        Topt_theta =np.random.uniform(1,9)#np.random.normal(6.5,2) #np.random.uniform(1,12) #np.random.lognormal(1,1)
        width_theta  =np.random.uniform(1,20)#np.random.normal(2,1)
        ##np.random.lognormal(1,1)
        kopt_theta    =np.random.uniform(0,1)#np.random.normal(0.5,0.4)# np.random.u(0,1)
        xi_theta     =np.random.uniform(0,0.5/2)#np.random.normal(0.1,0.09) #np.random.normal(0,1)#np.random.normal(0,0.5)
        r_theta     =np.random.uniform(0,1)
        m_J_theta    =np.random.uniform(0,1)#np.random.normal(0.04,0.04) # #np.random.beta(2,2)
        m_Y_theta    =np.random.uniform(0,1)#np.random.normal(0.05,0.04) #np.random.uniform(0,1) #np.random.beta(2,2)
        m_A_theta    =np.random.uniform(0,1)#np.random.normal(0.05,0.05)# np.random.uniform(0,1)#np.random.beta(2,2)
        K_theta= np.random.uniform(100,3000)
        PARAMS_ABC["L_0"]    = L_0_theta # sets the g_J parameter to our random guess
        PARAMS_ABC["L_inf"]    = L_inf_theta
        #PARAMS_ABC["L_J"]    = L_J_theta
        #PARAMS_ABC["L_Y"]    = L_Y_theta
        PARAMS_ABC["Topt"] = Topt_theta
        PARAMS_ABC["width"]  = width_theta
        PARAMS_ABC["kopt"]    = kopt_theta
        PARAMS_ABC["xi"]     = xi_theta
        PARAMS_ABC["r"]     = r_theta
        PARAMS_ABC["m_J"]    = m_J_theta
        PARAMS_ABC["m_Y"]    = m_Y_theta
        PARAMS_ABC["m_A"]    = m_A_theta
        PARAMS_ABC["K"]    = K_theta
        # Simulate population for new parameters
        N_J_sim, N_Y_sim, N_A_sim = abc_example.species3(PARAMS_ABC, temp1, T_FINAL, no_patches,N_J0, N_Y0,N_A0) # simulates population with g_J value
        
        # Calculate the summary statistics for the simulation
        Sim_SS_adult, Sim_SS_young, Sim_SS_juv= calculate_summary_stats(N_J_sim, N_Y_sim, N_A_sim)
        SS=np.hstack((Sim_SS_adult, Sim_SS_young, Sim_SS_juv))
        Obs_Sim[i+1,:]=SS
        
        param_save.append([L_0_theta, L_inf_theta, Topt_theta, width_theta, kopt_theta, xi_theta, r_theta, m_J_theta,m_Y_theta, m_A_theta, K_theta])
    
    return np.asarray(param_save), Obs_Sim
#########################################################################################
#return all  all parameters (library) and simulated  NSS (stats) corresponding to d ≤ δ(eps).
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
def actual_params(param):
    param["L_0"]   = np.random.uniform(0,3)#np.random.normal(0.4,0.3) #np.random.beta(2,2)
    param["L_inf"]    =np.random.uniform(17,23) #np.random.uniform(0,1)#np.random.beta(2,2)
    param["L_J"]    =6
    param["L_Y"]    =10
    param["Topt"]  =np.random.uniform(3,6)#np.random.normal(6.5,2) #np.random.uniform(1,12) #np.random.lognormal(1,1)
    param["width"]  =np.random.uniform(1,6)#np.random.normal(2,1)
    ##np.random.lognormal(1,1)
    param["kopt"]     =np.random.uniform(0,0.7)#np.random.normal(0.5,0.4)# np.random.u(0,1)
    param["xi"]     =np.random.uniform(0,0.5/4)#np.random.normal(0.1,0.09)
    param["r"]=np.random.uniform(0,0.7)
    param["m_J"]     =np.random.uniform(0,0.7)#np.random.normal(0.04,0.04) # #np.random.beta(2,2)
    param["m_Y"]   =np.random.uniform(0,0.7)#np.random.normal(0.05,0.04) #np.random.uniform(0,1) #np.random.beta(2,2)
    param["m_A"]   =np.random.uniform(0,0.7)#np.random.normal(0.05,0.05)# np.random.uniform(0,1)#np.random.beta(2,2)
    param["delta_t"]   =0.1
    param["K"]   =np.random.uniform(500,2000)
    return param
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
    PARAMS = {"L_0": 1, "L_inf": 20,"L_J": 5,"L_Y": 10, "Topt": 6.5, "width": 2, "kopt": 0.6,"xi":0.1,"r":0.3, "m_J": .04, "m_Y": .05, "m_A": .05, "delta_t": 0.1, "K":100}
    #final time
    #param_bound=np.array([[0,1],[0,1],[3,6],[1,4],[0,1],[0,0.5],[0,1],[0,1],[0,1],[100, 500]])
    T_FINAL = 30
    T_forcast=40
    #number of patches
    #initial abundance in each patch and for each stage
    N_J0=100
    N_Y0=100
    N_A0= 100
    no_patches=10
    temp_max=10
    #Number of iteration
    rSize   = len(PARAMS)-3
    NUMBER_SIMS = 200000
    N_Species  = 100
    import abc_example
    rows=T_forcast
    cols=no_patches
    temperatures = np.ndarray(shape=(rows, cols), dtype=float, order='F')
    temperatures[:, 0]=np.linspace(0, temp_max,T_forcast)
    for x in range(1, cols):
        temperatures[:, x]=temperatures[:, x-1]+PARAMS["delta_t"]
    temp1=temperatures[0:T_FINAL,:]
    D={}
    for q in range(1,no_patches+1):
       D['A_bias'+ str(q)] = np.empty((N_Species, len(temperatures)))
       D['Y_bias'+ str(q)] = np.empty((N_Species, len(temperatures)))
       D['J_bias'+ str(q)] = np.empty((N_Species, len(temperatures)))
    for i in range(N_Species ):
        param=copy.deepcopy(PARAMS)
        param_actual=actual_params(param)
    #############################################################################
    #simulating summary statistics and retaining a matrix with first row observed summar (OS) statistics and the remaining rows simulated summary (SS) statistics for NUMBER_SIMS iterations.It equally retain the simulated parameters. i'e retain all (theta_i, S_i) for i=0:NUMBER_SIMS.  theta_i and s_i from the joint distribution.
        param_save, Obs_Sim         = run_sim(param_actual)
    ######################################################################################################
#normalize the rows of Obs_sim to have NOS in row 1 and NSS in the remaining rows. Substract rows i=2:NUMBER_SIMS from row 1 of Obs_sim (whic contain OS).Compute the eucleadean distance (d) between NSS and NOS then use it along side tolerance (δ), to determine all parameters and NSS corresponding to d ≤ δ.Choose δ such that δ × 100% of the NUMBER_SIMS simulated parameters and NSS are selected. retain the parameters that made this threshold (library), the weights ot be used in local linear regression and the NSS that meets the threshold (stats)
        library, dists, stats,stats_SS,  NSS_cutoff, library_index   = sum_stats(Obs_Sim, param_save)
        result =do_rejection(library)
        #library_reg=do_logit_transformation(library, param_bound)
        #result_reg, HPDR_reg=do_kernel_ridge(stats, library_reg, param_bound)
        PARAMS2={}
        PARAMS2 = {"L_0":result[0] , "L_inf": result[1],"L_J": 5,"L_Y": 10, "Topt": result[2], "width": result[3], "kopt": result[4],"xi":result[5],"r":result[6], "m_J": result[7], "m_Y":result[8] , "m_A": result[9], "delta_t": param_actual["delta_t"], "K": result[10]}
        N_J1, N_Y1, N_A1 = abc_example.species3(param_actual,temperatures, T_forcast, no_patches,N_J0, N_Y0,N_A0)
        N_J2, N_Y2, N_A2 = abc_example.species3(PARAMS2,temperatures, T_forcast, no_patches,N_J0, N_Y0,N_A0)
        for q in range(1,no_patches+1):
            D['A_bias'+ str(q)][i,:] = N_A1[:,q-1]-N_A2[:,q-1]
            D['Y_bias'+ str(q)][i,:] = N_Y1[:,q-1]-N_Y2[:,q-1]
            D['J_bias'+ str(q)][i,:] = N_J1[:,q-1]-N_J2[:,q-1]
    time=np.array(range(0,40))
    import plot
    for q in range(1,no_patches+1):
        plot.do_forcast(D['A_bias'+ str(q)],temperatures, time, 'A_bias'+ str(q))
        plot.do_forcast(D['Y_bias'+ str(q)],temperatures, time, 'Y_bias'+ str(q))
        plot.do_forcast(D['J_bias'+ str(q)],temperatures, time, 'J_bias'+ str(q))



