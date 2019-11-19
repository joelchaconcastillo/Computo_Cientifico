import time
from scipy.stats import uniform, gamma, beta, bernoulli, truncnorm, norm, multivariate_normal, weibull_min, expon, loggamma, pearsonr, hypergeom
from scipy import stats, linalg
from sklearn.utils import resample
import pandas as pd
import numpy as np
import math
#from numpy.polynomial.polynomial import polyval
from matplotlib import pyplot as plt

import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import sys
from numpy import genfromtxt

np.set_printoptions(threshold=sys.maxsize)
#np.random.seed(3)
def Metodo_Percentiles_Media_No_Parametrico(B,theta, data, conf):
  n = len(data)
  Bi = np.array([])
  for i in range(B):
#     Bi = np.append(Bi,np.mean(data[np.random.choice(n, n, replace=True)]))
     Bi =  np.append(Bi, np.mean(resample(data, replace=True, n_samples=n)))
  theta_hat = np.mean(Bi)
  icn = np.quantile(Bi, np.array([conf/2.0, 1.0-(conf/2.0)]))
  lim_inf = 2.0*theta_hat- icn[1]
  lim_sup = 2.0*theta_hat - icn[0]

  return Bi, theta_hat, lim_inf, lim_sup

def Metodo_Percentiles_Media_Gamma(B, data, alpha, beta, conf):
  n = len(data)
  ###Asumiendo que los datos vienen de una gamma
  mean = np.mean(data)
  std = np.std(data)
  shape = (mean/std)**2
  scale = (std**2)/mean
  Bi = np.array([])
  for i in range(B):
     Bi = np.append(Bi,np.mean(gamma.rvs(n, shape, 1.0/scale)))
  theta_hat = np.mean(Bi)
  icn = np.quantile(Bi, np.array([conf/2.0, 1.0-(conf/2.0)]))
  lim_inf = 2.0*theta_hat - icn[1]
  lim_sup = 2.0*theta_hat - icn[0]
  return Bi, theta_hat, lim_inf, lim_sup

def Intervalo_BCa_Media(B, data, conf):
  n = len(data)
  #calculo de la aceleracion..
  theta_i = np.array([])
  data2 = np.ma.array(data, mask=False)
  robs = np.mean(data)
  Bi = np.array([]) 
  for i in range(B):
     idx = np.random.choice(n, n, replace=True)
     Bi = np.append(Bi,  np.mean(data[idx])) 
 
  theta_barra = np.mean(Bi) 
  for i in range(n):
     data2.mask[i] = True
     theta_i = np.append(theta_i, np.mean(data2.compressed()) )
     data2.mask[i] = False
  theta_hat = np.mean(theta_i)
  diff = theta_hat - theta_i
  a = np.sum(np.power(diff,3)) / 6.0*(np.sum(np.power(diff,2)))**(3.0/2.0)
 
  z0 = norm.ppf(np.mean(Bi<robs))
  zalpha = norm.ppf(conf)
  zalpha_c = norm.ppf(1.0-conf)
  alpha1 = norm.cdf(z0 + (z0+zalpha)/(1.0 - a*(z0+zalpha)))
  alpha2 = norm.cdf(z0 + (z0+zalpha_c)/(1.0 - a*(z0+zalpha_c)))
  inter = np.quantile(Bi, np.array([alpha1, alpha2]))
  lim_inf = 2.0*theta_barra - inter[1]
  lim_sup = 2.0*theta_barra - inter[0]
  return Bi, theta_barra, lim_inf, lim_sup
     
def Metodo_Percentiles_Pearson_No_Parametrico(B, data, conf):
  n = len(data)
  Bi = np.array([])
  for i in range(B):
     idx = np.random.choice(n, n, replace=True)
     Bi = np.append(Bi,pearsonr(data[idx,0],data[idx,1])[0])
  theta_hat = np.mean(Bi)
  icn = np.quantile(Bi, np.array([conf/2.0, 1.0-(conf/2.0)]))
#  icn = np.quantile(Bi, np.array([0.05,0.95]))
  lim_inf = 2.0*theta_hat- icn[1]
  lim_sup = 2.0*theta_hat - icn[0]
  return Bi, theta_hat, lim_inf, lim_sup

def Intervalo_BCa_Pearson(B, data, conf):
  n = len(data)
  robs = pearsonr(data[:,0],data[:,1])[0]
  Bi = np.array([]) 
  for i in range(B):
     idx = np.random.choice(n, n, replace=True)
     Bi = np.append(Bi,pearsonr(data[idx,0],data[idx,1])[0])
  theta_barra = np.mean(Bi) 
  data2 = np.ma.array(data, mask=False)
  theta_i = np.array([])
  for i in range(n):
     data2.mask[i] = True
     a = data2[:,0].compressed()
     b = data2[:,1].compressed()
     theta_i = np.append(theta_i,pearsonr(a,b)[0] )
     data2.mask[i] = False
  theta_hat = np.mean(theta_i)
  diff = theta_hat - theta_i
  a = np.sum(np.power(diff,3)) / (6.0*(np.sum(np.power(diff,2)))**(1.5))
  z0 = norm.ppf(np.mean(Bi<robs))
  zalpha = norm.ppf(conf)
  zalpha_c = norm.ppf(1.0-conf)
  alpha1 = norm.cdf(z0 + (z0+zalpha)/(1.0 - a*(z0+zalpha)))
  alpha2 = norm.cdf(z0 + (z0+zalpha_c)/(1.0 - a*(z0+zalpha_c)))
  inter = np.quantile(Bi, np.array([alpha1, alpha2]))
  lim_inf = 2.0*theta_barra- inter[1]
  lim_sup = 2.0*theta_barra- inter[0]
  #print(lim_inf)
  #print(lim_sup)
  return Bi, theta_barra, lim_inf, lim_sup

def Metodo_Percentiles_Pearson_Normal(B, data, conf):
  n = len(data[:,0])
  ###Asumiendo que los datos vienen de una gamma
  Mu = np.mean(data, axis=0)
  Sigma = np.cov(data, rowvar=0)
  Bi = np.array([])
  for i in range(B):
     mues = multivariate_normal.rvs(Mu, Sigma, n)
     Bi = np.append(Bi,pearsonr(mues[:,0],mues[:,1])[0])
  theta_hat = np.mean(Bi)
  icn = np.quantile(Bi, np.array([conf/2.0, 1.0-(conf/2.0)]))
  lim_inf = 2.0*theta_hat - icn[1]
  lim_sup = 2.0*theta_hat - icn[0]
  return Bi, theta_hat, lim_inf, lim_sup

def Punto_1():
   data = np.array([14.18, 10.99, 3.38, 6.76, 5.56, 1.26, 4.05, 4.61, 1.78, 3.84, 4.69, 2.12, 2.39, 16.75, 4.19])
   n = len(data)
   alpha = 3.0
   beta = 2.0
   conf = 0.1
   B = 10000
   theta = alpha*beta
   sigma = alpha*(beta**2)
   Bi, theta_hat, lim_inf, lim_sup = Metodo_Percentiles_Media_No_Parametrico(B, theta, data, conf)

   p =  pd.DataFrame(Bi).hist(bins=58, range=(0,10), figsize=(9,9))
   plt.axvline(x=theta_hat, color='y')
   plt.axvline(x=theta, color='r')
   plt.axvline(x=lim_inf, color='m')
   plt.axvline(x=lim_sup, color='m')

   plt.title("Distribucion bootstrap no parametrico \n (línea roja: promedio real, línea amarilla: promedio empírico, líneas magenta: intervalos)")
   plt.show()

   print("==================Punto 1============================")
   print("========Bootstrap No parametrico - percentiles=========")
   print("Media teorica: "+str(theta))
   print("Media empirica: "+str(theta_hat))
   print("Intervalo de confianza al "+str(100.0*(1.0-conf)))
   print("["+str(lim_inf) + ","+str(lim_sup)+"]")

   Bi, theta_hat, lim_inf, lim_sup = Intervalo_BCa_Media(B, data, conf)

   p =  pd.DataFrame(Bi).hist(bins=58, range=(0,10), figsize=(9,9))
   plt.axvline(x=theta_hat, color='y')
   plt.axvline(x=theta, color='r')
   plt.axvline(x=lim_inf, color='m')
   plt.axvline(x=lim_sup, color='m')

   plt.title("Distribucion bootstrap no parametrico Bias-Corrected-Accelerated \n (línea roja: promedio real, línea amarilla: promedio empírico, líneas magenta: intervalos)")
   plt.show()


   print("========Bootstrap No parametrico -- Bias Corrected and Accelerated =========")
   print("Media teorica: "+str(theta))
   print("Media empirica: "+str(theta_hat))
   print("Intervalo de confianza al "+str(conf/2.0))
   print("["+str(lim_inf) + ","+str(lim_sup)+"]")



   Bi, theta_hat, lim_inf, lim_sup = Metodo_Percentiles_Media_Gamma(B, data, alpha, beta, conf)

   p =  pd.DataFrame(Bi).hist(bins=58, range=(0,10), figsize=(9,9))
   plt.axvline(x=theta_hat, color='y')
   plt.axvline(x=theta, color='r')
   plt.axvline(x=lim_inf, color='m')
   plt.axvline(x=lim_sup, color='m')

   plt.title("Distribucion bootstrap parametrico asumiendo una distribucion Gamma\n (línea roja: promedio real, línea amarilla: promedio empírico, líneas magenta: intervalos)")
   plt.show()

   print("========Bootstrap parametrico asumiendo distribucion Gamma - percentiles =========")
   print("Media teorica: "+str(theta))
   print("Media empirica: "+str(theta_hat))
   print("Intervalo de confianza al "+str(conf/2.0))
   print("["+str(lim_inf) + ","+str(lim_sup)+"]")

def Punto_2():
   data = genfromtxt('cd4.csv', delimiter=',', skip_header=1)[:,1:]
   conf = 0.1
   B = 10000
   Intervalo_BCa_Media(B, data, conf)
   Bi, theta_hat, lim_inf, lim_sup = Metodo_Percentiles_Pearson_No_Parametrico(B, data, conf)
   print("\n==================Punto 2============================")
   print("========Bootstrap No parametrico - percentiles=========")
   print("Media empirica: "+str(theta_hat))
   print("Intervalo de confianza al "+str(conf/2.0))
   print("["+str(lim_inf) + ","+str(lim_sup)+"]")
   p =  pd.DataFrame(Bi).hist(bins=58, range=(0,1), figsize=(9,9))
   plt.axvline(x=theta_hat, color='r')
   plt.axvline(x=lim_inf, color='m')
   plt.axvline(x=lim_sup, color='m')

   plt.title("Distribucion bootstrap no parametrico \n (línea roja: promedio empírico, líneas magenta: intervalos)")
   plt.show()



   Bi, theta_hat, lim_inf, lim_sup = Intervalo_BCa_Pearson(B, data, conf)
   print("========Bootstrap No parametrico -- Bias Corrected and Accelerated  =========")
   print("Media empirica: "+str(theta_hat))
   print("Intervalo de confianza al "+str(conf/2.0))
   print("["+str(lim_inf) + ","+str(lim_sup)+"]")
   p =  pd.DataFrame(Bi).hist(bins=58, range=(0,1), figsize=(9,9))
   plt.axvline(x=theta_hat, color='r')
   plt.axvline(x=lim_inf, color='m')
   plt.axvline(x=lim_sup, color='m')

   plt.title("Distribucion bootstrap no parametrico Bias-Corrected-Accelerated \n (línea roja: promedio empírico, líneas magenta: intervalos)")
   plt.show()

   Bi, theta_hat, lim_inf, lim_sup = Metodo_Percentiles_Pearson_Normal(B, data, conf)

   p =  pd.DataFrame(Bi).hist(bins=108, range=(0,1), figsize=(9,9))
   plt.axvline(x=theta_hat, color='r')
   plt.axvline(x=lim_inf, color='m')
   plt.axvline(x=lim_sup, color='m')

   plt.title("Distribucion bootstrap parametrico asumiendo una distribucion Normal\n (línea roja: promedio empírico, líneas magenta: intervalos)")
   plt.show()

   print("========Bootstrap parametrico asumiendo distribucion Gamma - percentiles =========")
   print("Media empirica: "+str(theta_hat))
   print("Intervalo de confianza al "+str(conf/2.0))
   print("["+str(lim_inf) + ","+str(lim_sup)+"]")



#Punto_1()
Punto_2()
