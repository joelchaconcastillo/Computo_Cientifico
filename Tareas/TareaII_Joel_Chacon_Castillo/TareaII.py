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
def Least_Squares(X, Y, p, scip = False):
 #checking generic example...
 #A = np.array([[3.0, -6.0], [4.0, -8.0], [0.0, 1.0]])
 #b = np.array([-1.0, 7.0, 2.0])
 ##building Vandermonde system..
 A = np.zeros([len(Y), p+1])
 for i in range(p+1):
   A[:,i] = np.power(X, i)

 print("condition "+ str(np.linalg.cond(A) ))
 if scip == False:
  Q,R = QR(A)
 else:
  Q, R = scipy.linalg.qr(A,  mode='economic')
 return A, Backward(R, np.dot(np.transpose(Q), Y))
 

def Exercise1():
 A =  np.array([[1.0, 0.0, 0.0, 0.0, 1.0],[-1.0, 1.0, 0.0, 0.0, 1.0],[-1.0, -1.0, 1.0, 0.0, 1.0],[-1.0, -1.0, -1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, 1.0]])
 Q,R = QR(A)
 print(A)
 print(Q)
 print(R)

def Exercise2():
  ####### Artifitial data is emplyed
  nPoints = 5
  Degree = 3
  Y = np.arange(1.0, nPoints+1, 1.0)
  X = np.array([1.2, 5.6, 7.8, 5.6, 8.8])

  [A, C] = Least_Squares(X, Y, Degree-1) 
  Degree = 2
  print(A)
  print(A.dot(C))

def Ploting():
 ###Artifitial data....
    nPoints = 1000;
    sigma = 0.11
    X = np.arange(1.0, nPoints+1, 1.0)
    X = (4.0*np.pi*X)/nPoints
    Y = np.sin(X) +  np.random.normal(0.0, sigma, nPoints)
    plt.plot(X, Y,'r.', label= 'Generated Points' )
    for Degree in [3, 4, 6, 100]:
     #Degree = 5
     ##Training....
     [A, C] = Least_Squares(X, Y, Degree-1) 
#     print(A)

     Ntest = 1000
     ##Testing...
     xgrid = np.linspace(0.0, 12.0, Ntest)   
   
     YP = np.zeros(Ntest)

     for i in range(Degree):
       YP += C[i]*np.power(xgrid, i)
     plt.plot(xgrid, YP, label='P='+str(Degree))

    plt.legend(framealpha=1, frameon=True)
    plt.xlabel('X', fontsize=18)
    plt.ylabel('Y', fontsize=16)
    plt.title('Data with a training set of 1000 points')
#    plt.savefig('1000.eps')
    plt.show()

def Times1():
 ###Artifitial data....
  for nPoints in [100, 1000, 10000]:
    sigma = 0.11
    X = np.arange(1.0, nPoints+1, 1.0)
    X = (4.0*np.pi*X)/nPoints
    Y = np.sin(X) +  np.random.normal(0.0, sigma, nPoints)
    for Degree in [3, 4, 6, 100]:
     ##Training....
     start1 = time.time()
     [A, C] = Least_Squares(X, Y, Degree-1) 
     end1 = time.time()

     start2 = time.time()
     [A, C] = Least_Squares(X, Y, Degree-1, True)  ##Checking with scify..
     end2 = time.time()

     print(str(nPoints) + "/" + str(Degree) + " " +str(end1-start1) + " " + str(end2-start2))
    ##Only yhe fitting time is taken into account... cX
   ##  Ntest = 1000
   ##  ##Testing...
   ##  xgrid = np.linspace(0.0, 12.0, Ntest)   
   ##
   ##  YP = np.zeros(Ntest)

   ##  for i in range(Degree):
   ##    YP += C[i]*np.power(xgrid, i)


def Exercise3():
    Ploting()
    Times()
       
def Exercise4():
 ###Artifitial data....
  #for Degree in [3, 4, 6, 100, 200]:
  for Degree in np.arange(1, 1000, 100):
    nPoints = Degree*10
    sigma = 0.11
    X = np.arange(1.0, nPoints+1, 1.0)
    ###X = (4.0*np.pi*X)/100000#nPoints reformulation to avoid the condition issue ---Stability
    X = (4.0*np.pi*X)/nPoints
    Y = np.sin(X) 
    Y = np.random.normal(0.0, sigma, nPoints)
     ##Training....
    start1 = time.time()
    [A, C] = Least_Squares(X, Y, Degree-1) 
    end1 = time.time()

    start2 = time.time()
    [A, C] = Least_Squares(X, Y, Degree-1, True)  ##Checking with scify..
    end2 = time.time()
    print(str(nPoints) + "/" + str(Degree) + " " +str(end1-start1) + " " + str(end2-start2))



#Exercise1()
#Exercise2()
#Exercise3()
#Exercise4()
