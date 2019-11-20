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


u0=np.array([1.00115261, 0.99997944])
p=(2,0)
b_vars=np.array([1,1])
t=(0,6.3)
n_steps=100
sol=npc(hopf_bifurcation, u0, p, t, b_vars, n_steps)

norm=list(map(lambda x : np.linalg.norm(x), sol["solutions"]))
fig1=plt.figure()
plt.plot(sol["params"], norm)
plt.xlabel("Param Value")
plt.ylabel("Solution U Norm")
plt.title("Continuous solution to Hopf Bifurcation")
plt.savefig('Continuous solution to Hopf Bifurcation')

#plt.show()

u0=np.array([1.00115261, 0.99997944])
p=(2, -1)
b_vars=np.array([1,1])
t=(0,6.3)
n_steps=100
sol=npc(hopf_bifurcation_modified, u0, p, t, b_vars, n_steps)

norm=list(map(lambda x : np.linalg.norm(x), sol["solutions"]))
fig2=plt.figure()
plt.plot(sol["params"], norm)
plt.xlabel("Param Value")
plt.ylabel("Solution U Norm")
plt.title("Continuous solution to Modified Hopf Bifurcation")
plt.savefig('Continuous solution to Modified Hopf Bifurcation')

