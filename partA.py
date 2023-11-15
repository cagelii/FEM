import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def stiffnessAssembler(x):
    N = len(x)-1
    A = np.zeros([N+1,N+1])
    for i in range(N):
        h = x[i+1]-x[i]
        A[i,i] += 1/h
        A[i,i+1] += -1/h
        A[i+1,i] += -1/h
        A[i+1,i+1] += 1/h
    A[0,0] += 10e6
    A[N,N] += 10e6 #dirichlet BC
    return A

def loadVectorAssembler(x,l,r):
    N = len(x)-1
    B = np.zeros(N+1)
    for i in range(N):
        h = x[i+1]-x[i]
        B[i] += rhs(x[i])*h/2
        B[i+1] += rhs(x[i+1])*h/2
    B[0] += l*10e6
    B[N] += r*10e6
    return B

def massAssembler(x):
    N = len(x)-1
    M = np.zeros([N+1,N+1])
    for i in range(N):
        h = x[i+1]-x[i]
        M[i,i] += h/3
        M[i,i+1] += h/6
        M[i+1,i] += h/6
        M[i+1,i+1] += h/3
    M[0,0] = 10e6
    M[N,N] = 10e6 #dirichlet BC
    return M

def discreteLaplacian(M,B):
    return -np.matmul(np.linalg.inv(M),B) #since Au=B

def rhs(x):
    res = 0
    if abs(0.5-abs(x)) <= 0.3:
        res = 10
    return res
    
def eta(x,alpha,laplace):
    N = len(x)-1
    res = np.zeros(N+1)
    for i in range(N):
        h = x[i+1]-x[i]
        a = alpha*laplace[i+1]+rhs(x[i+1]) #integrand of x+1
        b = alpha*laplace[i]+rhs(x[i]) #integrand of x
        c = (a**2+b**2)*h/2 #trapezoidal
        res[i] = np.sqrt(c)*h #error indicator
    return res

def main():
    alpha = 0.01
    n = 12
    l = -1
    r = 1
    x = np.linspace(l,r,num=n)
    errori = np.ones(1)
    while x.size<1e4 and np.sum(np.square(errori))>1e-3:
        B = loadVectorAssembler(x,0,0)
        M = massAssembler(x)
        laplace = discreteLaplacian(M,B)
        errori = eta(x,alpha,laplace)
        l = 0.9
        m = np.max(errori)
        for i,e in enumerate(errori):
            if e > l*m:
                x = np.append(x,(x[i+1]+x[i])/2)
        x.sort()

    A = alpha*stiffnessAssembler(x)
    B = loadVectorAssembler(x,0,0)
    M = massAssembler(x)
    u_h = np.linalg.solve(A,B)
    laplace = discreteLaplacian(M,B)
    errori = eta(x,alpha,laplace)
    res = [rhs(r)+alpha*laplace[i] for i,r in enumerate(x)]

    fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=1,ncols=4)
    fig.tight_layout()
    ax1.plot(x,u_h)
    ax1.set_xlabel('x')
    ax1.set_title('Solution')

    ax2.plot(x,res)
    ax2.set_xlabel('x')
    ax2.set_title('Residuals')

    ax3.plot(x,errori)
    ax3.set_xlabel('x')
    ax3.set_title('Error indicator')

    ax4.plot(x[1:],1/np.diff(x))
    ax4.set_xlabel('x')
    ax4.set_title('Mesh size distribution')
    plt.show()

if __name__ == "__main__":
    main()
