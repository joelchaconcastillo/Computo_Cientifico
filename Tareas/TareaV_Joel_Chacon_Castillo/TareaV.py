import time
from scipy.stats import uniform
from scipy.stats import gamma
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
    #return np.log(gamma.pdf(x, a=k, scale=theta))
    return (k-1.0)*np.log(x) - (x/theta)
def Gamma_distribution(x, k, theta):
    #return gamma.pdf(x, a=k, scale=theta)
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
    #S = np.array([0.00001,0.3,6.0,7.0])
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
       #fx = Log_Gamma_distribution(x, k, theta)
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
    #return [X, Gamma_distribution(X, k, theta)]
k = 2
theta = 1 
points = Gamma(k,theta, 1000)
points = np.sort(points)
Y = Gamma_distribution(points, k, theta)

plt.plot(points, Y )
plt.plot(points, gamma.pdf(points, a=k, scale=theta))


x=points
y = np.arange(len(x))/float(len(x))
plt.plot(x, y)
#plt.show()
#plt.hist(points,  cumulative=True, label='CDF',histtype='step', alpha=0.8, color='k' )
#plt.hist(points,  normed=True, cumulative=True, label='CDF',histtype='step', alpha=0.8, color='k' )
plt.show()
