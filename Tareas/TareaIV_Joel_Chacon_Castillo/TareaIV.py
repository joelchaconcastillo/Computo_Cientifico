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
np.random.seed(3)
def Kappa(A):
  BEigs = np.linalg.eigvals(A)
  return np.max(np.abs(BEigs))/np.min(np.abs(BEigs))

#QR Factorization, Algorithm 23.1 taken of the Trefethen pag.175
def QR(A):
  V = np.copy(A)
  m,n = np.shape(A)
  Q = np.zeros([m,n], dtype=float)
  R = np.zeros([n,n], dtype=float)
  for i in range(0,n):
    R[i,i] = np.sqrt(np.dot(V[:,i], V[:,i]))
#    print(R[i,i])
    Q[:,i] = V[:,i]/(R[i,i])
    for j in range(i+1, n):
        R[i,j] = np.dot(np.transpose(Q[:,i]),V[:,j])
        V[:,j] -= R[i,j]*Q[:,i] 
  return Q,R
def Practical_QR_iterative_without_shift(A):
    dim = (A[:,0]).size
    Qf = np.identity(dim)
    Ak = np.copy(A)#np.dot(np.dot( np.transpose(Q), np.copy(A)), Q)
    Tol = 1e-10
    for i in range(dim,1,-1):
     while np.abs(Ak[i-1, i-2]) > Tol:
        #Q,R = QR(Ak[0:i,0:i] - muI)
        Q, R = scipy.linalg.qr( Ak[0:i,0:i],  mode='economic')
        Ak[0:i, 0:i] = np.dot(R, Q)
        QAugmented = np.identity(dim)
        QAugmented[0:i, 0:i] = np.copy(Q)
        Qf = (Qf).dot(QAugmented)
    return Ak, (Qf)

def Practical_QR_iterative(A):
    dim = (A[:,0]).size
    Qf = np.identity(dim)
    Ak = np.copy(A)#np.dot(np.dot( np.transpose(Q), np.copy(A)), Q)
    Tol = 1e-10
    for i in range(dim,1,-1):
     while np.abs(Ak[i-1, i-2]) > Tol:
        muI = Ak[i-1, i-1]*np.identity(i)
        #Q,R = QR(Ak[0:i,0:i] - muI)
        Q, R = scipy.linalg.qr( Ak[0:i,0:i]-muI,  mode='economic')
        Ak[0:i, 0:i] = np.dot(R, Q) + muI
        QAugmented = np.identity(dim)
        QAugmented[0:i, 0:i] = np.copy(Q)
        Qf = (Qf).dot(QAugmented)
    return Ak, (Qf)

def Practical_QR_recursive(A):
    dim = (A[:,0]).size
    Ak = np.copy(A)#np.dot(np.dot( np.transpose(Q), np.copy(A)), Q)
    if dim == 1:
      return Ak[0]
    Tol = 1e-2
    while np.abs(Ak[dim-1, dim-2]) > Tol:
       muI = Ak[dim-1,dim-1]*np.identity(dim)
       Q,R = QR(Ak - muI)
       Ak = np.dot(R, Q) + muI
    Ak[0:(dim-1),0:(dim-1)], Qf[0:(dim-1),0:(dim-1)] = Practical_QR(Ak[0:(dim-1),0:(dim-1)])
    return Ak
def Exercise2():
  for N in range(1,6):
#    N=1
    epsilon = 10**N
    A = np.array([[8.0, 1.0, 0.0], [1.0, 4.0, epsilon], [0.0, epsilon, 1.0]])
    Sigma, Q = Practical_QR_iterative(A)
    print("=========================")
    print("N: "+str(N))
    print("Condition number: " + str(Kappa(A)))
    print("A:")
    print(A)
    print("Eigenvalues: ")
    print(Sigma)
    print("Eigenvectors: ")
    print(Q)
    #print(np.dot(np.dot(np.transpose(Q),Sigma), Q))
def Exercise5():
    #A = 1.0/3.0*np.array([[2.0, -2.0, 1.0], [1.0, 2.0, 2.0], [2.0, 1.0, -2.0]])
    ###Orthogonal matrix
    A = np.array([[0.6667, 0.7454, 0.0000], [-0.7454,0.6667, 0.0000], [0,0.0000,1.0000]])
    print(A) 
    Sigma, Q = Practical_QR_iterative_without_shift(A)
    print(Sigma)
    print(Q)
    print(np.dot(np.dot(np.transpose(Q),Sigma), Q))


Exercise2()
#Exercise5()
