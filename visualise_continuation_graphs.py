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


u0=np.array([0.33,0.33])
p=(0,2)
b_vars=np.array([0.33,0.33])
t=(0,6.3)
n_steps=40
sol=npc(hopf_bifurcation, u0, p, t, b_vars, n_steps)

norm=list(map(lambda x : np.linalg.norm(x), sol["solutions"]))
plt.plot(sol["params"], norm)
plt.xlabel("Param Value")
plt.ylabel("Solution U Norm")
plt.title("Continuous solution to Hopf Bifurcation")


plt.show()
