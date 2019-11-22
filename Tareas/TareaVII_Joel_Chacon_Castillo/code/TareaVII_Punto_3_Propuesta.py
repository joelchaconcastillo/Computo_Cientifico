import time
from scipy.stats import uniform, gamma, beta, bernoulli, truncnorm, expon, multivariate_normal, lognorm, cauchy
from scipy.special import gamma as gg, factorial
import random
from numpy.random import choice

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

   
    
def Metropolis_Hastings(maxite, cov, mu, sigma1, sigma2, burnin, batch):
    delta = 1.0
    x_t = np.array([uniform.rvs(0.0, 5.0), uniform.rvs(0.0, 5.0)]) # en el soporte...
    #x_t =np.array([1000,1])# np.array([uniform.rvs(mu[0]-delta, mu[0]+delta), uniform.rvs(mu[1]-delta, mu[1]+delta)]) # en el soporte...
    X_walk =  np.copy(x_t)
    cov_prop1 = np.identity(2)*sigma1
    cov_prop2 = np.identity(2)*sigma2
    fx_t = multivariate_normal.logpdf(x_t, mu, cov)
    cont = 0.0
    w1 = 0
    w2 = 0
    pb1 = 0.5
    pb2 = 0.5
    total = 0.0
    PesosH = np.array([0.5, 0.5])
    while cont < maxite:
     total +=1
     idx = choice([0,1],p= [pb1, pb2])
     pb =  uniform.rvs(0.0, 1.0)
     if idx == 0:
       y_t = multivariate_normal.rvs(x_t, cov_prop1)
     if idx == 1:
       y_t = multivariate_normal.rvs(x_t, cov_prop2)
     if y_t[0] <= 0 or y_t[1] <=0:
        continue
     fy_t = multivariate_normal.logpdf(y_t, mu, cov)
     rho = fy_t-fx_t
     if np.log(uniform.rvs(0.0, 1.0)) <= rho:
       if idx ==0:
          w1 += 1
       if idx ==1:
          w2 += 1
       if cont > burnin:
        X_walk = np.vstack((X_walk, y_t))
       x_t = np.copy(y_t)
       fx_t = fy_t
       cont +=1
       if (cont % batch ) == 0:
         pb1 = w1/(w1+w2)
         pb2 = w2/(w1+w2)
         PesosH = np.vstack((PesosH, np.array([pb1, pb2])))
         w1 = 0.0
         w2 = 0.0
    print("Eficiencia " + str(cont/total))
    pesos = np.array([pb1, pb2])
    return X_walk[1:,:], cont/total, PesosH
maxite = 10000
cov = np.array([[1.0, 0.9],[0.9, 1.0]])
mu = np.array([3.0, 5.0])
sigma1 = 0.1
sigma2 = 5.0
burnin=100
batch = 100
X, eficiencia, PesosH = Metropolis_Hastings(maxite, cov, mu, sigma1, sigma2, burnin, batch)
FX = multivariate_normal.pdf(X, mu, cov)


plt.plot(np.log(FX))
plt.title("Evolucion de la cadena, Pesos finales = "+str(PesosH[-1,:])+"  \n Eficiencia="+str(eficiencia)+" \n " r" $x_0 = $"+str(X[0,:]) )
plt.xlabel("Iteracion")
plt.ylabel("log(f(x))")
plt.show()


dist = 5.0
xl, yl = np.mgrid[(mu[0]-dist):(mu[0]+dist):.1, (mu[1]-dist):(mu[1]+dist):.1]
zl = np.copy(xl)
for i in range(0, len(xl[0,:])):
  for j in range(0, len(xl[0,:])):	
     zl[i,j] = multivariate_normal.pdf(np.array([xl[i,j], yl[i,j]]), mu, cov)

plt.contourf(xl, yl, zl)
plt.plot(X[:,0], X[:,1], 'r.')
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")


plt.show()
plt.plot(PesosH[:,0])
plt.plot(PesosH[:,1])
plt.xlabel("Batch (cada 100 iteraciones)")
plt.ylabel("Valor del peso")
plt.legend([r"$w_1$ de $N_(0, "+str(sigma1)+" I)$", r"$w_2$ de $N_(0, "+str(sigma2)+" I)$"])

plt.show()

