import time
from scipy.stats import uniform, gamma, beta, bernoulli, truncnorm, norm, multivariate_normal, weibull_min, expon, loggamma, pearsonr
from scipy import stats, linalg
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
     Bi = np.append(Bi,np.mean(data[np.random.choice(n, n, replace=True)]))
  theta_hat = np.mean(Bi)
  icn = np.quantile(Bi, np.array([conf/2.0, 1.0-(conf/2.0)]))
#  icn = np.quantile(Bi, np.array([0.05,0.95]))
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
   print("========Bootstrap No parametrico - percentiles=========")
   print("Media teorica: "+str(theta))
   print("Media empirica: "+str(theta_hat))
   print("Intervalo de confianza al "+str(conf/2.0))
   print("["+str(lim_inf) + ","+str(lim_sup)+"]")

   Bi, theta_hat, lim_inf, lim_sup = Intervalo_BCa_Media(B, data, conf)
   print("========Bootstrap No parametrico -- Bias Corrected and Accelerated =========")
   print("Media teorica: "+str(theta))
   print("Media empirica: "+str(theta_hat))
   print("Intervalo de confianza al "+str(conf/2.0))
   print("["+str(lim_inf) + ","+str(lim_sup)+"]")



   Bi, theta_hat, lim_inf, lim_sup = Metodo_Percentiles_Media_Gamma(B, data, alpha, beta, conf)
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
   print("========Bootstrap No parametrico - percentiles=========")
   print("Media empirica: "+str(theta_hat))
   print("Intervalo de confianza al "+str(conf/2.0))
   print("["+str(lim_inf) + ","+str(lim_sup)+"]")

   Bi, theta_hat, lim_inf, lim_sup = Intervalo_BCa_Pearson(B, data, conf)
   print("========Bootstrap No parametrico -- Bias Corrected and Accelerated  =========")
   print("Media empirica: "+str(theta_hat))
   print("Intervalo de confianza al "+str(conf/2.0))
   print("["+str(lim_inf) + ","+str(lim_sup)+"]")




Punto_1()
Punto_2()
