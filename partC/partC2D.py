from dolfin import *
set_log_active(False)

mesh=Mesh("2d_mesh/circle3.xml")
V = FunctionSpace(mesh,"CG",1)

#Data
T = 20
h = mesh.hmin()
dt = h
alpha = 0.01
R = 0.5
r = 0.2

#Boundary conditions
class DirichletBoundary(SubDomain):
	def inside(self,x,on_boundary):
		return on_boundary

g = Constant(0.0)
bc = DirichletBC(V,g,DirichletBoundary())

#Initial condition
indata = Expression("abs(R-sqrt(pow(x[0],2)+pow(x[1],2))) <= r ? 10: 0", R=R, r=r, degree=2)
u0 = Function(V)
u0 = interpolate(indata,V)


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

#Crank-Nicolson
a = u/dt*v*dx+alpha*dot(grad(u)/2,grad(v))*dx
L = u0/dt*v*dx-alpha*dot(grad(u0)/2,grad(v))*dx


# Compute solution
t = 0
u = Function(V)

#create file
file = File("partC2D/solution.pvd")

#Set initial condition
u.assign(u0)

#time loop
while t<=T:
	
	u0.assign(u)
	file << (u, t)
	solve(a == L, u, bc)
	

	t += dt