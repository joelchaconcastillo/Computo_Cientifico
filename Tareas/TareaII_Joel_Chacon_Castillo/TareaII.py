import time
import numpy as np
np.random.seed(0)

#Cholesky Factorization, Algorithm 23.1 taken of the Trefethen pag.175
def QR(A):
  V = np.copy(A)

  m,n = np.shape(A)
  Q = np.zeros(m,m)
  R = np.zeros(m,n)

  for i in range(0,m):
      R[i,i] = np.norm(V[:,i])
      Q[:,i] = V[:,i]/R[i,i]
    for j in range(i+1, n):
        R[i,j] = np.dot(np.transpose(Q[:i]),V[:,j])
        V[:,j] -= R[i,j]*Q[:,i] 
  return Q,R

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
