import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

"""
We are using X=[u,v] to describe both poulations

- u is the prey pop
- v is the pred pop
"""

#define params
a=1
d=0.1
b=float(input("Enter the rate of predation b ∈ [0.1, 0.5] : "))

# rate of change and pred and prey populations
def dXdt(X, t=0):
     """Return the change in pred and prey populations"""
     return np.array([X[0]*(1-X[0])-((a*X[0]*X[1])/(d+X[0])),
                  b*X[1]*(1-(X[1]/X[0]))])

# initial pred and prey populations
X0=np.array([0.38,0.38])
t = np.linspace(0,80,10000) #set linspace to 0,24 as limit cycle happens at ~ 21s

def solve_int(info=False):
    # solve the ode using odeint as odeint suffices and is easy to use
    X, infodict = odeint(dXdt, X0, t, full_output=True)
    if (info):
        print(infodict['message'])
    return X

def plot_pred_pray(X):
    #plot popultions
    prey, pred = X.T
    f1 = plt.figure()
    plt.plot(t, prey, 'r-', label='Prey')
    plt.plot(t, pred  , 'b-', label='Pred')
    plt.grid(markevery=1)
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.xticks(np.arange(0,80,1))#plot ticks every second to help with approx time interval
    plt.ylabel('population')
    plt.title('Evolution of predator and prey populations')
    #f1.savefig('predator_and_prey_1.png')
    plt.show()
    return

X=solve_int()
plot_pred_pray(X)


""" I have Isolated a periodic orbit with period 20.94s starting conditions are a=1 d=0.1, b=0.2, pred_pop=0.38 prey_po = 0.38

An appropriate phase condition for the limit cycle is that the dy/dt and dx/dt must pass through 0 this works as both curves oscilate and so have gradient of 0 twice during an oscilation cycle at their peaks and troughs.

The time period for each phase is approx 20.76.

I will now attempt to construct a shooting method root finding problem"""

