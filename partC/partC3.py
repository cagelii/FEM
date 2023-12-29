from dolfin import *
set_log_active(False)
import numpy as np
import matplotlib.pyplot as plt

mesh=Mesh("3dmesh/sphere2.xml")
V = FunctionSpace(mesh,"CG",1)

#Data
T = 50
h = mesh.hmin()
dt = h
dts = np.linspace(0,T,int(np.round(T/dt,0))+1)
alpha = 0.01


#Boundary conditions
class DirichletBoundary(SubDomain):
	def inside(self,x,on_boundary):
		return on_boundary

g = Constant(0.0)
bc = DirichletBC(V,g,DirichletBoundary())

def func(data):

	#Initial condition
	indata = Expression("pow((R-sqrt(pow(x[0],2)+pow(x[1],2))),2)+pow(x[2],2) <= r ? rho :0", rho=data[0], R=data[1], r=data[2], degree=3)
	u0 = Function(V)
	u0 = interpolate(indata,V)

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)

	# Crank-Nicolson
	a = u/dt*v*dx+alpha*dot(grad(u)/2,grad(v))*dx
	L = u0/dt*v*dx-alpha*dot(grad(u0)/2,grad(v))*dx


	# Compute solution
	t = 0
	u = Function(V)

	# create file
	#file = File("partC3D/solution.pvd")

	#Set initial condition
	u.assign(u0)

	# Mass loss
	mass = np.zeros(3)

	# Copy initial data
	u_initial = Function(V)
	u_initial = interpolate(indata, V)

	# Define an integral functional
	M = (u_initial - u)*dx
	ts = [5,7,30,51]
	i = 0
	# Time loop
	while t<=T:
		u0.assign(u)

		#file << (u, t)
		solve(a == L, u, bc)
		if t > ts[i]:
			mass[i] = assemble(M)
			i += 1
		t += dt
	F = (mass[0]-10)**2+(mass[1]-15)**2+(mass[2]-30)**2
	print(F)
	return F

import scipy.optimize as optimize
from scipy.optimize import minimize

data = [32,0.45,0.13]
res = minimize(func, data, method = 'nelder-mead' , options = {'xtol':1e-3 , 'disp': True})
print(res)

with open('optimized.txt', 'w') as f:
    f.write(str(res))
