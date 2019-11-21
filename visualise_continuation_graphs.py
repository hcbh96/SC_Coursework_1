import matplotlib.pyplot as plt
import numpy as np
from natural_parameter_continuation import npc
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


u0=np.array([1.00115261, 0.99997944, 0.63])
p=(0,2)
n_steps=10
sol=npc(hopf_bifurcation, u0, p, n_steps)

norm=list(map(lambda x : np.linalg.norm(x[0:-1]), sol["solutions"]))
fig1=plt.figure()
plt.plot(sol["params"], norm)
plt.xlabel("Param Value")
plt.ylabel("Solution U Norm")
plt.title("Continuous solution to Hopf Bifurcation")
#plt.savefig('Continuous solution to Hopf Bifurcation')

plt.show()

u0=np.array([1.00115261, 0.99997944, 6.3])
p=(2, -1)
t=(0,6.3)
n_steps=100
sol=npc(hopf_bifurcation_modified, u0, p, n_steps)

norm=list(map(lambda x : np.linalg.norm(x[0:-1]), sol["solutions"]))
fig2=plt.figure()
plt.plot(sol["params"], norm)
plt.xlabel("Param Value")
plt.ylabel("Solution U Norm")
plt.title("Continuous solution to Modified Hopf Bifurcation")
#plt.savefig('Continuous solution to Modified Hopf Bifurcation')
plt.show()
