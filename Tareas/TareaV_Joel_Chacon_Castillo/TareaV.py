import time
from scipy.stats import uniform
from scipy.stats import gamma
from scipy import stats
import numpy as np
import math
#from numpy.polynomial.polynomial import polyval
from matplotlib import pyplot as plt

import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import sys
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(3)
def cdf(x, plot=True, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)
## shape parameter --> k and scale parameter --> theta
def Log_Gamma_distribution(x, k, theta):
    return (k-1.0)*np.log(x) - (x/theta)
def Gamma_distribution(x, k, theta):
    return (np.power(x, k-1.0)*np.exp(-x/theta))

def Computing_Slopes(S, G):
    S1 = S[0:(len(S)-1)]
    S2 = S[1:len(S)]
    G1 = G[0:(len(S)-1)]
    G2 = G[1:len(S)]
    return np.abs(G1-G2)/np.abs(S1-S2)
def sampling_g(S, H, Slopes, k, theta):
    ##compute probabilites based in Casella's book pag. 71.. it can be efficient precomputing this table and only modifying the interval of the new point
    alpha = Slopes
    beta = -Slopes*(S[0:(len(S)-1)]) + H[0:(len(H)-1)]
    Xi1 = S[0:(len(S)-1)] 
    Xi2 = S[1:len(S)]
    IntervalProbabilities = (np.exp(beta)*(  np.exp(alpha*Xi2) - np.exp(alpha*Xi1)  ))/alpha
    betaN = np.sum(IntervalProbabilities)
    idx = np.random.choice( np.arange(len(IntervalProbabilities)), p=IntervalProbabilities/betaN) 
    u = uniform.rvs()  
    inside = np.exp(alpha[idx]*Xi1[idx]) + u*( alpha[idx]*Xi2[idx] - alpha[idx]*Xi1[idx])
    x = (1.0/alpha[idx]) * np.log(inside)
    gx = np.exp(alpha[idx]*x + beta[idx])/betaN
    return x, gx, idx, betaN

def update_S(S, H, x, gx, Slopes):
    S = np.append(S,x)
    H = np.append(S,H)
    Sidx = np.argsort(S)
    S = S[Sidx]
    H =H[Sidx]
    Slopes = Computing_Slopes(S, H)

def Gamma(k, theta, maxite):
    S = np.arange(0.0001,7,0.01)# np.array([0.00001,0.3,6.0,7.0])
    H = Log_Gamma_distribution(S, k, theta)
  #  plt.plot(S,H)
  #  plt.show()
  #  exit()
    Slopes = Computing_Slopes(S, H)
    X = np.array([])
    ## sampling exponential
    ##iteratively...
    cont = 0
    while cont < maxite:
        ##get back x and its interval..
       [x, gx, idx, beta] = sampling_g(S, H, Slopes, k, theta) 
       fx = Gamma_distribution(x, k, theta)
       u = uniform.rvs()  
       f_lower = np.exp(( x - S[idx])*Slopes[idx]+ H[idx]) ### y = ((y_2-y_1)/(x_1-x_2))*( x - x_1) - y_1
       if u <= f_lower/(beta*gx):
           ##accept..
           X = np.append(X, x)
           cont+=1;
       else:
          if u <= (fx/gx):
             ##accept xg and update S
             X = np.append(X, x)
             cont+=1;
       update_S(S, H, x, fx, Slopes) 
    return X

def RandomGeneration(N):
    DX = np.array([1, 2, 3, 4, 5])
    idx = np.array([2,3,4,5])
    Period = 2**(31) 
    for i in range(0, N):
       xi = (107374182*DX[-1] + 104420*DX[-5])%(Period- 1)
       DX = np.append(DX, xi)
    return DX[0:len(DX)-1]/float(Period-1)
def Exercise_5():
    k = 2
    theta = 1 
    N = 10000
    points = Gamma(k,theta, N)
    points = np.sort(points)
    Y = Gamma_distribution(points, k, theta)
    #Imprimir el pdf de la libreria y el implementado..
    plt.plot(points, Y )
    plt.plot(points, gamma.pdf(points, a=k, scale=theta))
    
    x=points
    #imprimir el cdf de la libreria y el implementado..
    y = np.arange(len(x))/float(len(x))
    plt.plot(x, y)
    
    #x = gamma.cdf(k,theta, 1000)
    x =  np.random.gamma(k,  theta, N)
    x = np.sort(x)
    y = np.arange(len(x))/float(len(x))
    plt.plot(x, y)
    plt.legend(['pdf implementado', 'pdf stats.gamma', 'cdf implementado', 'cdf random.gamma'])
    plt.show()
    #plt.hist(points,  cumulative=True, label='CDF',histtype='step', alpha=0.8, color='k' )
    plt.show()
def Exercise_2():
    for i in range(0,3):
      points = RandomGeneration(10**i)
      test=stats.kstest(points, stats.uniform(loc=0.0, scale=1.0).cdf)
      print(str(10**i)+ " "+str(test))

    points = RandomGeneration(10000)
    #points = np.random.uniform(0,1,10000) # RandomGeneration(10000)
    plt.plot(range(0,len(points)), points, '.')

#    plt.hist(points,  cumulative=True, label='CDF',histtype='step', alpha=0.8, color='k' )
    plt.show()
#    print(points)

Exercise_2()
##Exercise_5()
