from dolfin import *
set_log_active(False)
import numpy as np
import matplotlib.pyplot as plt

mesh=Mesh("3dmesh/sphere2.xml")
V = FunctionSpace(mesh,"CG",1)

#Data
T = 20
h = mesh.hmin()
dt = h
dts = np.linspace(0,T,int(np.round(T/dt,0))+1)
alpha = 0.01
R = 0.445
r = 0.12677
rho = [10, 20, 40]
rho = 32.6

# Mass loss figure
fig, ax = plt.subplots(1,1)
labels = ['Rho = 10','Rho = 20','Rho = 40']

#Boundary conditions
class DirichletBoundary(SubDomain):
	def inside(self,x,on_boundary):
		return on_boundary

g = Constant(0.0)
bc = DirichletBC(V,g,DirichletBoundary())

for j in range(1): #range(len(rho)):

	#Initial condition
	indata = Expression("pow((R-sqrt(pow(x[0],2)+pow(x[1],2))),2)+pow(x[2],2) <= r ? rho :0", rho=rho, R=R, r=r, degree=3)
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
	mass = np.zeros(len(dts))

	# Copy initial data
	u_initial = Function(V)
	u_initial = interpolate(indata, V)
	# Define an integral functional
	M = (u_initial - u)*dx
	# compute the functional
	mass[0] = assemble(M)

	i = 0
	# Time loop
	while t<=T:
		i += 1
		u0.assign(u)

		#file << (u, t)
		solve(a == L, u, bc)
		mass[i] = assemble(M)
		t += dt

	ax.plot(dts,mass) #label=labels[j])
	ax.set_ylabel('Mass loss')
	ax.set_xlabel('Time (days)')

ax.legend()
plt.savefig("optimized.png")
