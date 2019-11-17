"""
This file will be used to plot and visualise the solutions to natural parameter continuation.
"""
import matplotlib.pyplot as plt
import numpy as np
from natural_parameter_continuation import npc
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

def cubic_equation(b):
     """Algebraic Cubic Equations"""
     return lambda x : x**3 - x + b


def hopf_bifurcation(beta):
    """Return a systems of equations relating to the hopf bifurcation"""
    return lambda t, X : [
            beta*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
            X[0]+beta*X[1]-X[1]*(X[0]**2+X[1]**2),
           ]

def hopf_bifurcation_modified(beta):
    """Return a systems of equations relating to the hopf bifurcation"""
    return lambda t, X : [
            beta*X[0]-X[1]+X[0]*(X[0]**2+X[1]**2)-X[0]*(X[0]**2+X[1]**2)**2,
            X[0]+beta*X[1]+X[1]*(X[0]**2+X[1]**2)-X[1]*(X[0]**2+X[1]**2)**2,
            ]

"""
Visualise solutions for the algebraic cubic equations
"""
X0=[1]
vary_par=dict(start=-2, stop=2, steps=100)
sol=npc(cubic_equation, X0, vary_par, method='solve', root_finder=fsolve)
fig1=plt.figure()
plt.plot(sol["params"], sol["results"])
plt.title("Cubic Solutions Plot")
#plt.show()
fig1.savefig("cubic_solutions_plot")

"""
Visualise solutions to the Hopf Bifurcation Normal form
"""
X0=np.array([0.33,0.33])
vary_par=dict(start=0, stop=2, steps=50)
b_vars=np.array([0.33,0.33])
t=(0,6.3)
sol=npc(hopf_bifurcation, X0, vary_par, t, method='shooting', boundary_vars=b_vars,  root_finder=fsolve,integrator=solve_ivp)
fig2=plt.figure()
plt.plot(sol["params"], sol["results"])
plt.title("Hopf Bifurcation Plot")
plt.show()
fig2.savefig("hopf_bif_plot")


"""
Visualise solutions to the Modified Hopf Bifurcation
"""
X0=np.array([1,1])
vary_par=dict(start=2, stop=-1, steps=50)
b_vars=np.array([1,1])
t=(0,6.35)
sol=npc(hopf_bifurcation_modified, X0, vary_par, t, method='shooting',                        boundary_vars=b_vars,  root_finder=fsolve,integrator=solve_ivp)
fig2=plt.figure()
plt.plot(sol["params"], sol["results"])
plt.title("Hopf Bifurcation Modified Plot")
plt.show()
fig2.savefig("hopf_bif_modified_plot")


