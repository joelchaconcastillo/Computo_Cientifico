import time
import numpy as np
import math
#from numpy.polynomial.polynomial import polyval
from matplotlib import pyplot as plt

import scipy
import scipy.linalg   # SciPy Linear Algebra Library

np.random.seed(0)
def Backward(U, b):
   m = len(U[:,0]) ##get the number of rows..
   x = np.zeros(m)
   for j in range(m-1, -1, -1):
     x[j] = b[j]
     for k in range(j+1, m):
         x[j] -= x[k] * U[j,k]
     x[j] /=U[j,j]
   return x

#Cholesky Factorization, Algorithm 23.1 taken of the Trefethen pag.175
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

def Least_Squares(Y, p):
 #checking generic example...
 #A = np.array([[3.0, -6.0], [4.0, -8.0], [0.0, 1.0]])
 #b = np.array([-1.0, 7.0, 2.0])

 ##building Vandermonde system..
 A = np.zeros([len(Y), p+1])
 for i in range(p+1):
   A[:,i] = np.power(Y, i)
 #Q,R = QR(A)
 Q, R = scipy.linalg.qr(A,  mode='reduced')
 print(np.shape(A))
 print(np.shape(Q))
 print(np.shape(R))
 return A, Backward(R, np.dot(np.transpose(Q), Y))
 

def Exercise1():
 A =  np.array([[1.0, 0.0, 0.0, 0.0, 1.0],[-1.0, 1.0, 0.0, 0.0, 1.0],[-1.0, -1.0, 1.0, 0.0, 1.0],[-1.0, -1.0, -1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, 1.0]])
 Q,R = QR(A)
 print(A)
 print(Q)
 print(R)

def Exercise2():
# A =  np.array([[1.0, 0.0, 0.0, 0.0, 1.0],[-1.0, 1.0, 0.0, 0.0, 1.0],[-1.0, -1.0, 1.0, 0.0, 1.0],[-1.0, -1.0, -1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, 1.0]])
  Y = np.ones([3])
  Degree = 2
  [A, C] = Least_Squares(Y, Degree)
  print(A)
  print(A.dot(C))
  

def Exercise3():
    ###Artifitial data....
    nPoints = 500;
    sigma = 0.11
    X = np.arange(1.0, nPoints+1, 1.0)
    X = (4.0*np.pi*X)/nPoints
    Y = np.sin(X) +  np.random.normal(0.0, sigma, nPoints)

    Degree = 5
    ##Training....
    [A, C] = Least_Squares(Y, Degree) 
    print(A)


    ##Testing...
    xgrid = np.linspace(0, 15, 50)   
   
    YP = np.zeros(50)

    for i in range(Degree+1):
      YP += C[i]*np.power(xgrid, i)

    plt.plot(xgrid, YP)
    plt.plot(X, Y)
    plt.show()
    
    print(C)
    
def Exercise4():
    print("no yet")


#Exercise1()
#Exercise2()
Exercise3()
#Exercise4()
