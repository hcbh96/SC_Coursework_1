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
X0=np.array([0.6,0.3])
t = np.linspace(0,1000,100)

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
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.title('Evolution of predator and prey populations')
    f1.savefig('predator_and_prey_1.png')
    return

X=solve_int()
plot_pred_pray(X)

