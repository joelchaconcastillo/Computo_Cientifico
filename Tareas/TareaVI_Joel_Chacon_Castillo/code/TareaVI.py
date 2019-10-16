import time
from scipy.stats import uniform, gamma, beta, bernoulli, truncnorm
from scipy import stats
import numpy as np
import math
#from numpy.polynomial.polynomial import polyval
from matplotlib import pyplot as plt

import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import sys
np.set_printoptions(threshold=sys.maxsize)
#np.random.seed(3)


def My_Bernulli_Simulation(p):
    u = uniform.rvs(0,1.0)
    if u < p:
        return 1.0
    else:
        return 0.0

def f(p, n, r):
    fp =(p**r)*( (1.0-p)**(n-r))*np.cos(np.pi*p)
    if fp >=0 and fp <= 0.5:
      return fp
    else:
      return 0.0

def simulation_bernulli(n, p):
    r = 0
    for i in range(0,n):
       r += bernoulli.rvs(p)
       #r += My_Bernulli_Simulation(p)
    return r
def Metropolis_Hastings(n, p, maxite, r):
    X = np.array([])
    FX = np.array([])
    a_beta = r+1.0
    b_beta = n-r+1.0
    ##given X^(t)
    x_t =  uniform.rvs(0.0, 0.5)  
    X =np.append(X, x_t)
    FX =np.append(FX, f(x_t, n, r))
    cont = 0
    while cont < maxite:
       #generate 
       y_t = beta.rvs(a_beta, b_beta)
       fx = f(x_t, n, r)
       fy = f(y_t, n, r)
       if fx == 0:
           continue;
       cont +=1
       ratio_f = fy/fx
       ratio_q = beta.pdf(x_t, a_beta, b_beta)/beta.pdf(y_t, a_beta, b_beta)
       rho = min(ratio_f*ratio_q, 1.0)
       u_t =  uniform.rvs(0, 1.0)  
       if u_t <= rho:
           x_t = y_t
           X = np.append(X, y_t)
           FX =np.append(FX, fy)
       else:
           X = np.append(X, x_t)
           FX =np.append(FX, fx)
    plt.hist(X, label='CDF',histtype='step', alpha=0.8, color='k', density=True )
    plt.title("Beta instrumental distribution with r: " + str(r) + ", n:" + str(n)+", p:"+str(p))
    plt.show()
    return X, FX

def Metropolis_Hastings_Instrumental_Uniform(n, p, maxite, r):
    X = np.array([])
    FX = np.array([])
    a_beta = r+1.0
    b_beta = n-r+1.0
    ##given X^(t)
    x_t =  uniform.rvs(0.0, 0.5)  
    X =np.append(X, x_t)
    FX =np.append(FX, f(x_t, n, r))
    cont = 0
    while cont < maxite:
       #generate 
       y_t = beta.rvs(a_beta, b_beta)
       fx = f(x_t, n, r)
       fy = f(y_t, n, r)
       if fx == 0:
           continue;
       cont +=1
       ratio_f = fy/fx
       ratio_q = uniform.pdf(x_t)/uniform.pdf(y_t)
       rho = min(ratio_f*ratio_q, 1.0)
       u_t =  uniform.rvs(0, 1.0)  
       if u_t <= rho:
           x_t = y_t
           X = np.append(X, y_t)
           FX =np.append(FX, fy)
       else:
           X = np.append(X, x_t)
           FX =np.append(FX, fx)
    plt.hist(X, label='CDF',histtype='step', alpha=0.8, color='k', density=True, facecolor='green' )
    plt.title("Uniform instrumental distribution with r: " + str(r) + ", n:" + str(n)+", p:"+str(p))
    plt.show()
    return X, FX

def Metropolis_Hastings_Instrumental_Truncated_Normal(n, p, maxite, r):
    X = np.array([])
    FX = np.array([])
    a_beta = r+1.0
    b_beta = n-r+1.0
    ##given X^(t)
    x_t =  uniform.rvs(0.0, 0.5)  
    X =np.append(X, x_t)
    FX =np.append(FX, f(x_t, n, r))
    cont = 0
    while cont < maxite:
       #generate 
       y_t = beta.rvs(a_beta, b_beta)
       fx = f(x_t, n, r)
       fy = f(y_t, n, r)
       if fx == 0:
           continue;
       ratio_f = fy/fx
       qx = truncnorm.pdf(x_t, 0.0, 0.5)
       qy = truncnorm.pdf(y_t, 0,0.5)
       if qy == 0:
           continue;
       cont +=1
       ratio_q = qx/qy
       rho = min(ratio_f*ratio_q, 1.0)
       u_t =  uniform.rvs(0, 1.0)  
       if u_t <= rho:
           x_t = y_t
           X = np.append(X, y_t)
           FX =np.append(FX, fy)
       else:
           X = np.append(X, x_t)
           FX =np.append(FX, fx)
  #  x = np.sort(X)
   # y = np.arange(len(x))/float(len(x))
    plt.hist(X, label='CDF',histtype='step', alpha=0.8, color='k', density=True )
    plt.title("Truncated Norm instrumental distribution with r: " + str(r) + ", n:" + str(n)+", p:"+str(p))
    plt.show()
    return X, FX
p = 1.0/3.0
n = 5
maxite = 10000
r = simulation_bernulli(n, p)
print(r)
[X, FX] = Metropolis_Hastings(n, p, maxite, r)
[X, FX] = Metropolis_Hastings_Instrumental_Uniform(n, p, maxite, r)
[X, FX] = Metropolis_Hastings_Instrumental_Truncated_Normal(n, p, maxite, r)
plt.plot(X, FX, '.')
plt.show()
