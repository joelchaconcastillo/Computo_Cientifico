import time
from scipy.stats import uniform
import numpy as np
import math
#from numpy.polynomial.polynomial import polyval
from matplotlib import pyplot as plt

import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import sys
np.set_printoptions(threshold=sys.maxsize)
#np.random.seed(0)
def Kappa(A):
  BEigs = np.linalg.eigvals(A)
  print(BEigs)
  return np.max(BEigs)/np.min(BEigs)

def Cholesky(A):
  R = np.copy(A)
  m = len(R[:,0])
  for k in range(0, m):
      for j in range(k+1, m):
          R[j,j:m] -= R[k, j:m]*(R[k,j]/R[k,k])
      R[k,k:m] = R[k,k:m]/np.sqrt(R[k,k])
  R  = np.triu(R, k=0)
  return R

def Backward(U, b):
   m = len(U[:,0]) ##get the number of rows..
   x = np.zeros(m)
   for j in range(m-1, -1, -1):
     x[j] = b[j]
     for k in range(j+1, m):
         x[j] -= x[k] * U[j,k]
     x[j] /=U[j,j]
   return x

#QR Factorization, Algorithm 23.1 taken of the Trefethen pag.175
def QR(A):
  V = np.copy(A)
  m,n = np.shape(A)
  Q = np.zeros([m,n], dtype=float)
  R = np.zeros([n,n], dtype=float)
  for i in range(0,n):
    R[i,i] = np.sqrt(np.dot(V[:,i], V[:,i]))
    Q[:,i] = V[:,i]/R[i,i]
    for j in range(i+1, n):
        R[i,j] = np.dot(np.transpose(Q[:,i]),V[:,j])
        V[:,j] -= R[i,j]*Q[:,i] 
  return Q,R

##Returns the vandermonde system and the coefficients coefficeints fitted
def Least_Squares(A, Y, scip = False):
 print("condition "+ str(np.linalg.cond(A) ))
 if scip == False:
  Q,R = QR(A)
 else:
  Q, R = scipy.linalg.qr(A,  mode='economic')
 return Backward(R, np.dot(np.transpose(Q), Y))

def Exercise1A():
   m=20
   n=50
#   A =  uniform.rvs( 0.0, 1.0, (m,n))
   A = np.random.uniform(0.0,1.0,(m,n))
   Q,R = QR(A)
   Q = np.copy(Q[0:m, 0:m])
   R = np.copy(R[0:m, 0:n])

   ###well-conditioned.....########################################
   #Q,R = scipy.linalg.qr(A, mode='economic')
#   #Lambda = np.arange(1.0, 21.0, 1.0)
   Lambda = np.linspace(1e2, 1.0, m)
   Lambda2 = Lambda + np.random.normal(0.0, 0.01, m)


   B = np.dot(np.dot(np.transpose(Q), np.diag(Lambda)), Q)
   Beps = np.dot(np.dot(np.transpose(Q), np.diag( np.abs(Lambda2))), Q)

   Start1 = time.time()
   R1 = Cholesky(B)
   End1 = time.time()
   
   Start2 = time.time()
   R2 = Cholesky(Beps)
   End2 = time.time()

   #print(str(np.linalg.cond(B,-2)) + " " + str(np.linalg.cond(Beps, -2) ))
   print(str( Kappa(B) ) + " " + str( Kappa(Beps)))
   print( str(End1-Start1) + " " +str(End2-Start2))
   exit()
########## ill-conditioned.....
   A =  uniform.rvs(0,10, (m,n))
#   A = np.random.uniform(0,1,(20, 50))
   Q,R = QR(A)
   Q = np.copy(Q[0:m, 0:m])
   R = np.copy(R[0:m, 0:n])
   #Q,R = scipy.linalg.qr(A, mode='economic')
   Lambda = np.linspace(30, 1.0, m)
   B = np.dot(np.dot(np.transpose(Q), np.diag(Lambda)), Q)
   Beps = np.dot(np.dot(np.transpose(Q), np.diag( np.abs(Lambda  + np.random.normal(0.0, 0.01, m) ))), Q)


   Start1 = time.time()
   R1 = Cholesky(B)
   End1 = time.time()
   
   Start2 = time.time()
   R2 = Cholesky(Beps)
   End2 = time.time()

#   print(scipy.linalg.cholesky(B))
   #Cholesky(Beps)
   print(str(np.linalg.cond(R1)) + " " + str(np.linalg.cond(R2) ))
   print( str(End1-Start1) + " " +str(End2-Start2))


def Exercise1B_C():
   m=200
   n=500
########## ill-conditioned.....
   A = np.random.uniform(0,1,(m, n))
   Q,R = QR(A)
   Q = np.copy(Q[0:m, 0:m])
   R = np.copy(R[0:m, 0:n])
   Lambda = np.linspace(2, 1e-5, m)
   B_ill = np.dot(np.dot(np.transpose(Q), np.diag(Lambda)), Q)

   Start1 = time.time()
   R1 = Cholesky(B_ill)
   End1 = time.time()
   
   Start2 = time.time()
   R2 = scipy.linalg.cholesky(B_ill)
   End2 = time.time()

   print(str(np.linalg.cond(B_ill) ))
   print( str(End1-Start1) + " " +str(End2-Start2))

def Exercise2A():
   n = 20
   d = 5
   beta = np.array([5,4,3,2,1])
#   Xv = np.random.uniform(0,1,(n))
   XM = np.random.uniform(0,1,(n, d)) #np.vander(Xv, d)
   ##simulating y....
   y = np.dot(XM, beta) + np.random.normal(0.0, 0.15, n)
   #######
   betahat = Least_Squares(XM, y)

   XMDelta = XM + np.random.normal(0.0, 0.01, (n, d))
   betap= Least_Squares(XMDelta, y)
   betac = np.dot(scipy.linalg.inv(np.dot(np.transpose(XMDelta),XMDelta)), np.transpose(XMDelta).dot(y) )
   print(beta)
   print(betahat)
   print(betap)
   print(betac)

def Exercise2B():
   n = 20
   d = 5
   beta = np.array([5,4,3,2,1])
#   Xv = np.random.uniform(0,1,(n))
   XM = np.random.uniform(0,1,(n, d)) 
   #XM[0,:] = np.random.uniform(0,1e-300,(1,d))+XM[1,:]
   XM[:,1] = XM[:,0]+ np.random.uniform(0,1e-6,(1,n))
   ##simulating y....
   y = np.dot(XM, beta) + np.random.normal(0.0, 0.15, n)
   #######
   betahat = Least_Squares(XM, y)

   XMDelta = XM + np.random.normal(0.0, 0.01, (n, d))
   XMDelta[:,1] = XMDelta[:,0]+ np.random.uniform(0,1e-6,(1,n))

   betap= Least_Squares(XMDelta, y)
   betac = np.dot(scipy.linalg.inv(np.dot(np.transpose(XMDelta),XMDelta)), np.transpose(XMDelta).dot(y) )
   print(beta)
   print(betahat)
   print(betap)
   print(betac)


#Exercise1A()
#Exercise1B_C()
#Exercise2A()
Exercise2B()
