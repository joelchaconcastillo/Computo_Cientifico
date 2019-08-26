import time
import numpy as np
np.random.seed(0)

#Cholesky Factorization, Algorithm 23.1 taken of the Trefethen pag.175
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
   cont = 0
   for i in range(0,m):
     if np.abs(U[i,i]) > 1e-20:
        cont +=1 
   if cont < m:
      print("La matriz a factorizar no tiene rango completo") 
      exit(0)
   return P, L, U

def Exercise3():
 A =  np.array([[1.0, 0.0, 0.0, 0.0, 1.0],[-1.0, 1.0, 0.0, 0.0, 1.0],[-1.0, -1.0, 1.0, 0.0, 1.0],[-1.0, -1.0, -1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, 1.0]])
 P,L,U = GEPP(A)
 print("A Static")
 print(A)
 print("L")
 print(L)
 print("U")
 print(U)
 print("P")
 print(P)
 
 #Entries random generation..
 A = np.random.uniform(0,1,(5,5))
 P,L,U = GEPP(A)
 print("A Random")
 print(A)
 print("L")
 print(L)
 print("U")
 print(U)
 print("P")
 print(P)

def Exercise4():
   #Given a static matrix
   A1 =  np.array([[1.0, 0.0, 0.0, 0.0, 1.0],[-1.0, 1.0, 0.0, 0.0, 1.0],[-1.0, -1.0, 1.0, 0.0, 1.0],[-1.0, -1.0, -1.0, 1.0, 1.0],[-1.0, -1.0, -1.0, -1.0, 1.0]])
   #Random matrix
   A2 = np.random.uniform(0,1,(5,5))
   
   P1,L1,U1 = GEPP(A1)
   P2,L2,U2 = GEPP(A2)
   
   b1 = np.random.uniform(0,1,(5))
   b2 = np.random.uniform(0,1,(5))
   b3 = np.random.uniform(0,1,(5))
   
   print("Static Matrix")
   print(A1)
   print("P")
   print(P1)
   print("L")
   print(L1)
   print("U")
   print(U1)
   
   
   
   ##Static matrix solving
   
   b1
   y1 = Forward(L1, np.dot(P1,b1))
   x1 =  Backward(U1, y1)
   print("Solving with b:")
   print(b1)
   print("Answer:")
   print(x1)
   
   #b2
   y1 = Forward(L1, np.dot(P1,b2))
   x1 =  Backward(U1, y1)
   print("Solving with b:")
   print(b2)
   print("Answer:")
   print(x1)
   
   #b3
   y1 = Forward(L1, np.dot(P1,b3))
   x1 =  Backward(U1, y1)
   
   print("Solving with b:")
   print(b3)
   print("Answer:")
   print(x1)
   ###random matrix solving 
   print("Random Matrix=====================")
   print(A2)
   print("P")
   print(P2)
   print("L")
   print(L2)
   print("U")
   print(U2)
   
   #b1
   y2 = Forward(L2, np.dot(P2,b1))
   x2 =  Backward(U2, y2)
   print("Solving with b:")
   print(b1)
   print("Answer:")
   print(x2)
   
   #b2
   y2 = Forward(L2, np.dot(P2,b2))
   x2 =  Backward(U2, y2)
   print("Solving with b:")
   print(b2)
   print("Answer:")
   print(x2)
   
   
   #b3
   y2 = Forward(L2, np.dot(P2,b3))
   x2 =  Backward(U2, y2)
   print("Solving with b:")
   print(b3)
   print("Answer:")
   print(x2)


def Exercise5():
  A = np.random.uniform(0,1,(5,5))
  A = np.dot(A,np.transpose(A))
  print("Before factorization")
  print(A)
  R = Cholesky(A)
  print("After factorization")
  print(R)
  print("Checking R^*R")
  print(np.dot(np.transpose(R),R))
def Exercise6():
 m=500
 for i in range(10,10000, 200):
  A = np.random.uniform(0,1,(i,i))
  A = np.dot(A,np.transpose(A))
  
  start1 = time.time()
  R = Cholesky(A)
  end1 = time.time()

  start2 = time.time()
  P,L,U = GEPP(A)
  end2 = time.time()
  print(str(i)+ " " + str(end1-start1)+" "+str(end2-start2))



#define matrix..
#A = np.array([[2.0, 1.0, 1.0, 0.0], [4.0, 3.0, 3.0, 1.0],[8.0, 7.0, 9.0, 5.0], [6.0, 7.0 ,9.0, 8.0]]) ##Trefethen book example, Pag. 158 in book
####################################
#####################################
#############Exercise #3
#################################
#Exercise3()


##############################33####################################
#####################################
##==========================EXERCISE #4
#################################

#Exercise4()

####################################
#####################################
#############Exercise #5
#################################
#Exercise5()

####################################
#####################################
#############Exercise #6
#################################
#Exercise6()
