import time
from scipy.stats import uniform, gamma, beta, bernoulli, truncnorm, norm, multivariate_normal, weibull_min, expon, loggamma
from scipy import stats, linalg
import numpy as np
import math
#from numpy.polynomial.polynomial import polyval
from matplotlib import pyplot as plt

import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import sys
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(3)

###############################Section point 1

def qi(X, Mu, v_sigma, rho, i): #X_i | X_j
      j = 1-i
      Mu_i_j = Mu[i] + rho*(v_sigma[i]/v_sigma[j])*(X[j]-Mu[j])
      Sigma_i_j = (v_sigma[i]**2)*(1.0-(rho**2))
      yt = np.copy(X)
      yt[i] = norm.rvs(Mu_i_j, Sigma_i_j)
      return yt
def energy(xt, yt, Mu, Sigma_Inv, v_sigma, rho, i):   #it is computed trough the logaritmic, X_j| x_i
    j = 1-i
    term1 = -0.5*np.dot(np.dot((yt- Mu), Sigma_Inv),yt-Mu)
    term2 = 0.5*np.dot(np.dot((xt- Mu), Sigma_Inv),xt-Mu)
    Mu_i_j_x = Mu[i] + rho*(v_sigma[i]/v_sigma[j])*(xt[j]-Mu[j])
    Mu_i_j_y = Mu[i] + rho*(v_sigma[i]/v_sigma[j])*(yt[j]-Mu[j])
    Sigma_i_j = (v_sigma[i]**2)*(1.0-rho**2)
    term3 = ((xt[i]-Mu_i_j_x)**2)/(2*(Sigma_i_j**2))
    term4 = ((xt[j]-Mu_i_j_y)**2)/(2*(Sigma_i_j**2))
    return term1+term2-term3+term4

def MH(maxite, Mu, Sigma, rho, v_sigma):
    Sigma_Inv = linalg.inv(Sigma) #precision matrix..
    U = (uniform.rvs(size=2)*2-3.0)
    xt = Mu+U# np.array([Mu[0]+uniform.rvs()*0.1, Mu[1]+uniform.rvs()*0.1])
    X = np.array([xt])
    cont = 0
    while cont < maxite:
      d = np.random.randint(2)
      yt = qi(xt, Mu, v_sigma, rho, d)
      ##check support ##it is not in the support
      if yt[d]==0: 
        continue	  
####In this case the accepting criteria is not required..
#      criteria = energy(xt, yt, Mu, Sigma_Inv, v_sigma, rho, d)
#      u = np.log(uniform.rvs())
#      if u < criteria: ##accepting..
      ##compute decision criteria
      X = np.vstack((X, yt))
      xt = np.copy(yt)
      cont +=1
    return X 
def Punto1():
   rho = 0.85
   maxite = 10000
   v_sigma = np.array([1,1]) # sigma vector
   Mu = np.array([1, 1])
   Sigma = np.array([[v_sigma[0]**2, rho*v_sigma[0]*v_sigma[1]], [rho*v_sigma[0]*v_sigma[1], v_sigma[1]**2]])
   x = MH(maxite, Mu, Sigma, rho, v_sigma)
   ##print contour ...
   xl, yl = np.mgrid[-3:5:.01, -3:5:.01]
   pos = np.empty(xl.shape + (2,))
   pos[:, :, 0] = xl
   pos[:, :, 1] = yl
   rv = multivariate_normal(Mu, Sigma)
   plt.contourf(xl, yl, rv.pdf(pos))
   plt.plot( x[:,0], x[:,1], '-r')
   plt.title("Simulacion de districion normal multivariada con \n " + r"$\rho$=" + str(rho) + r" $\mu$="+str(Mu)+" y max. ite.="+str(maxite) )
   plt.show()
   ####
   plt.plot(np.abs(x[1:maxite, 0]-x[0:(maxite-1), 0]))
   plt.xlabel("Tiempo")
   plt.ylabel("Tamanio de paso")
   plt.title("Desplazamientos en la variable " r"$X_1$ con " r"$\rho = $"+str(rho) )
   plt.show()
   plt.plot(np.abs(x[1:maxite, 1]-x[0:(maxite-1), 1]))
   plt.xlabel("Tiempo")
   plt.ylabel("Tamanio de paso")
   plt.title("Desplazamientos en la variable " r"$X_2$ con " r"$\rho = $"+str(rho) )
   plt.show()

Punto1()
