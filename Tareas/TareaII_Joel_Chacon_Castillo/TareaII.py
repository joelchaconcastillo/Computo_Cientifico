import time
import numpy as np
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
  Q = np.zeros([m,m], dtype=float)
  R = np.zeros([m,n], dtype=float)
  for i in range(0,n):
    R[i,i] = np.sqrt(np.dot(V[:,i], V[:,i]))
    Q[:,i] = V[:,i]/R[i,i]
    for j in range(i+1, n):
        R[i,j] = np.dot(np.transpose(Q[:,i]),V[:,j])
        V[:,j] -= R[i,j]*Q[:,i] 
  return Q,R
def Least_Squares(A, b):
 Q,R = QR(A)
 return Backward(R, np.dot(np.transpose(Q), b))

 

def Exercise1():
 A =  np.array([[1.0, 0.0, 0.0, 0.0, 1.0],[-1.0, 1.0, 0.0, 0.0, 1.0],[-1.0, -1.0, 1.0, 0.0, 1.0],[-1.0, -1.0, -1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, 1.0]])
 Q,R = QR(A)
 print(A)
 print(Q)
 print(R)

def Exercise2():
 A =  np.array([[1.0, 0.0, 0.0, 0.0, 1.0],[-1.0, 1.0, 0.0, 0.0, 1.0],[-1.0, -1.0, 1.0, 0.0, 1.0],[-1.0, -1.0, -1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, 1.0]])
 b = np.ones([5,1])
 print(Least_Squares(A,b))


Exercise1()
Exercise2()
