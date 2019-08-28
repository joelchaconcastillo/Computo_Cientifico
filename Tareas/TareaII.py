import numpy as np

#Cholesky Factorization, Algorithm 23.1 taken of the Trefethen pag.175
def Gram_Schmidt(A):
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
def Forward(L,b):
   m = len(L[:,0]) ##get the number of rows..
   x = np.zeros(m)
   for j in range(0,m):
     x[j] = b[j]
     for k in range(0, j):
         x[j] -= x[k] * L[j,k]
     x[j] /=L[j,j]
   return x

#Gaussian Elimination with Partial Pivoting which is taken of Trefethen book pag. 160
def GEPP(A_Matrix):
   A = A_Matrix.copy()
   m = len(A)
   L = np.eye(m, m)
   U = A.copy()
   P = np.eye(m, m)
   for k in range(0, m-1):
      #Select i >= k to maximize | u_{ik} | 
      i = np.argmax( np.abs(U[k:m, k]))+k
      U[[k, i],k:m] = U[[i, k],k:m]
      L[[k, i],0:k] = L[[i, k],0:k]
      P[[k, i],:] = P[[i, k],:]
      #Interchange two rows
      #Interchange Permutation matrix
      for j in range(k+1, m):
         L[j, k] = np.copy(U[j,k]/U[k,k])
         U[j,k:m+1] = np.copy(U[j, k:m+1] - L[j,k]*U[k,k:m+1])
   return P, L, U

#define matrix..
#A = np.array([[2.0, 1.0, 1.0, 0.0], [4.0, 3.0, 3.0, 1.0],[8.0, 7.0, 9.0, 5.0], [6.0, 7.0 ,9.0, 8.0]]) ##Trefethen book example, Pag. 158 in book
####################################
#####################################
#############Exercise #3
#################################
#A =  np.array([[1.0, 0.0, 0.0, 0.0, 1.0],[-1.0, 1.0, 0.0, 0.0, 1.0],[-1.0, -1.0, 1.0, 0.0, 1.0],[-1.0, -1.0, -1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, 1.0]])
#P,L,U = GEPP(A)
#print(L)
#print(U)
#print(P)

##Entries random generation..
#A = np.random.uniform(0,1,(5,5))
#P,L,U = GEPP(A)
#print(L)
#print(U)
#print(P)


####################################
#####################################
#############Exercise #4
#################################


A =  np.array([[1.0, 0.0, 0.0, 0.0, 1.0],[-1.0, 1.0, 0.0, 0.0, 1.0],[-1.0, -1.0, 1.0, 0.0, 1.0],[-1.0, -1.0, -1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, 1.0]])
P,L,U = GEPP(A)
b = np.random.uniform(0,1,(5))
print(U)
y = Forward(L, np.dot(P,b))
x =  Backward(U, y)
print(x)

####################################
#####################################
#############Exercise #5
#################################

A = np.dot(A,np.transpose(A))
print(A)
R = Cholesky(A)
print(np.dot(np.transpose(R),R))



####################################
#####################################
#############Exercise #6
#################################
m=500
A = np.random.uniform(0,1,(m,m))
A = np.dot(A,np.transpose(A))
#print(A)
R = Cholesky(A)

P,L,U = GEPP(A)


#print(Backward(U, b))
#print(L)
#print(Forward(L, b))

#print(P)
#print(U)

#print(np.dot(L,U))
#print(np.dot(P,A))

