import time
from scipy.stats import uniform, gamma, beta, bernoulli, truncnorm, expon, multivariate_normal, lognorm
from scipy.special import gamma as gg, factorial
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

   
    
def Metropolis_Hastings(maxite, cov, mu, sigma, burnin):
    delta = 1.0
    x_t = np.array([uniform.rvs(0.0, 5.0), uniform.rvs(0.0, 5.0)]) # en el soporte...
    #x_t =np.array([1000,1])# np.array([uniform.rvs(mu[0]-delta, mu[0]+delta), uniform.rvs(mu[1]-delta, mu[1]+delta)]) # en el soporte...
    X_walk =  np.copy(x_t)
    cov_prop = np.identity(2)*sigma
    #fx_t = multivariate_normal.pdf(x_t, mu, cov)
    fx_t = multivariate_normal.logpdf(x_t, mu, cov)
    cont = 0.0
    total = 0.0
    while cont < maxite:
     y_t = multivariate_normal.rvs(x_t, cov_prop)
     if y_t[0] <= 0 or y_t[1] <=0:
        continue
     #fy_t = multivariate_normal.pdf(y_t, mu, cov)
     fy_t = multivariate_normal.logpdf(y_t, mu, cov)
     #rho = min(1.0, fy_t/fx_t)
     rho = fy_t-fx_t# min(1.0, fy_t/fx_t)
     if np.log(uniform.rvs(0.0, 1.0)) <= rho:
       if cont > burnin:
        X_walk = np.vstack((X_walk, y_t))
       x_t = np.copy(y_t)
       fx_t = fy_t
       cont +=1
     total +=1
    print("Eficiencia " + str(cont/total))
    return X_walk[1:,:], cont/total
maxite = 1000
cov = np.array([[1.0, 0.9],[0.9, 1.0]])
mu = np.array([3.0, 5.0])
sigma = 0.1
burnin=100

X, eficiencia = Metropolis_Hastings(maxite, cov, mu, sigma, burnin)
FX = multivariate_normal.pdf(X, mu, cov)


plt.plot(np.log(FX))
plt.title("Evolucion de la cadena, \n Eficiencia="+str(eficiencia)+" \n sigma="+str(sigma)+ r" $x_0 = $"+str(X[0,:]) )
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

