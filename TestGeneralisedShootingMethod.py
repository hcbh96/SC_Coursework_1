"""
ShootingMethod.py contains an implemtation of the shooting method to find the initial predator prey populations given a time period of 20.76s and a finishing condition of [pred, prey]=[0.38, 0.38]
"""

from scipy.optimize import newton
from scipy.integrate import odeint
import GeneralisedShootingMethod
import numpy as np

"""params such [a,b,d] below should be defined outside the function as it means one can write loops with varying params during numerical analysis"""
#define params
a=1; d=0.1; b=0.2

# rate of change and pred and prey populations
def dXdt_lotka_volterra(X,t=0):
     """Return the change in pred and prey populations"""
     return np.array([X[0]*(1-X[0])-((a*X[0]*X[1])/(d+X[0])),
                  b*X[1]*(1-(X[1]/X[0]))])

#define an initial guess for the starting conditions
X0=[0.51,0.5]

#expected starting params at boundary
boundary_vars=[0.38, 0.38]

#time range
t=np.linspace(0,20.76,10000)

#calc the result using a secant method
res=GeneralisedShootingMethod.solve(dXdt_lotka_volterra, X0, t, boundary_vars)
print(res)#returns [0.37231387 0.37548837]



"""
Testing out generalised shooting method on the Hopf Bifurcation
"""
sigma=-1;beta=1
def dXdt_hopf_bif(X, t=0):
    """Function to calculate the rate of change of the Hopf Bifurcation at position X"""
    return np.array([
            beta*X[0]-X[1]+sigma*X[0]*(X[0]**2+X[1]**2),
            X[0]+beta*X[1]+sigma*X[1]*(X[0]**2+X[1]**2)
            ])

# Initial conditions guess
X0=[1,0]
# Boundary Conditions
boundary_vars=[0,1]
# time range
t=np.linspace(0,360,100)
res=GeneralisedShootingMethod.solve(dXdt_hopf_bif, X0, t,          boundary_vars)
print(res)#returns [ 15484.77502465 -58041.44113902 ]
