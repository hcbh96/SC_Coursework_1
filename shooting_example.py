"""
ShootingMethod.py contains an implemtation of the shooting method to find the initial predator prey populations given a time period of 20.76s and a finishing condition of [pred, prey]=[0.38, 0.38]
"""

from scipy.optimize import newton
from scipy.integrate import odeint
import numpy as np

#define params
a=1
d=0.1
b=0.2

#expected starting params at boundary
boundary_vars=[0.38, 0.38]

#time range
t=np.linspace(0,20.76,10000)

# rate of change and pred and prey populations
def dXdt(X,t=0):
     """Return the change in pred and prey populations"""
     return np.array([X[0]*(1-X[0])-((a*X[0]*X[1])/(d+X[0])),
                  b*X[1]*(1-(X[1]/X[0]))])

#define function to pass to fsolve
def func_to_solve(X):
    """Return the difference between the shot and the initial_bv conditions"""
    sol=boundary_vars - odeint(dXdt,X,t)[-1]
    return sol

#define an initial guess for the starting conditions
X0=[0.51,0.5]

#calc the result using a secant method
res=newton(func_to_solve,X0,tol=1e-4,maxiter=50)
print(res)#returns [0.37231387 0.37548837]
