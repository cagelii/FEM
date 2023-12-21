from dolfin import *
from time import perf_counter as pc

mesh1=UnitSquareMesh(500,500)
V = FunctionSpace(mesh1,"Lagrange",1)
def boundary(x):
    return x[0]<DOLFIN_EPS or x[0]>1-DOLFIN_EPS or x[1]<DOLFIN_EPS or x[1]>1-DOLFIN_EPS

u0 = Constant(0.0)
bc = DirichletBC(V,u0,boundary)

# Define variational problem
u = TrialFunction( V )
v = TestFunction( V )
f = Expression("8*pow(M_PI,2)*sin(2*M_PI*x[0])*sin(2*M_PI*x[1])",degree = 2)
#g = Expression("sin(5*x[0])", degree = 2)
a = inner( grad ( u ) , grad ( v ) ) * dx
L = f * v * dx #+ g * v * ds

# Compute solution
u = Function(V)
start = pc()
solve( a == L , u , bc )
stopp = pc()
print(stopp-start)
# Save solution in VTK format
file = File("poisson.pvd")
file << u
# Plot solution
import matplotlib.pyplot as plt
plot(u)
plt.show()