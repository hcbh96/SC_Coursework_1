import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np

def lotka_volterra(b):
     """Return the change in pred and prey populations"""
      #define params
     a=1
     d=0.1
     return lambda t, X : [
             X[0]*(1-X[0])-((a*X[0]*X[1])/(d+X[0])),
             b*X[1]*(1-(X[1]/X[0]))
             ]

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

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)


def visualise(Y, t, fig_name):
    f1 = plt.figure()
    for y in Y:
        plt.plot(t, y, color=list(np.random.rand(3)))
        plt.grid()
        plt.xlabel('t')
        plt.ylabel('Y')
        plt.title('Equations Plot')
    f1.savefig(fig_name + '.png')
    #plt.show()
    return


""" I have Isolated a periodic orbit with period 20.94s starting conditions are a=1 d=0.1, b=0.2, pred_pop=0.38 prey_po = 0.38

An appropriate phase condition for the limit cycle is that the dy/dt and dx/dt must pass through 0 this     works as both curves oscilate and so have gradient of 0 twice during an oscilation cycle at their peaks     and troughs.

 The time period for each phase is approx 20.76.

"""

# initial pred and prey populations
X0=[0.38,0.38]
t = (0,80) #set linspace to 0,24 as limit cycle happens at ~ 21s
b=0.2
sol=solve_ivp(lotka_volterra(b), t, X0)
visualise(sol.y,sol.t, 'lotka_volterra')


"""
Plotting the Hopf Bifurcation normal form I will do this twice

- once for b=0
- once for b=2

This should provide reasonable data for testing

"""

# initial pred and prey populations
X0=[0.06,0.06]
t=(0,20) #set linspace to 0,24 as limit cycle happens at ~ 21s
b=0
sol=solve_ivp(hopf_bifurcation(b), t, X0)
visualise(sol.y,sol.t, 'hopf_bifurcation_start')

 # initial pred and prey populations
X0=[1,1]
t = (0,20) #set linspace to 0,24 as limit cycle happens at ~ 21s
b=2
sol=solve_ivp(hopf_bifurcation(b), t, X0)
visualise(sol.y,sol.t, 'hopf_bifurcation_end')


"""
Plotting modified Hopf Bifurcation normal form
 - once for b=-1
 - once for b=2

 This should provide reasonable data for testing
 """

# initial pred and prey populations
X0=[0.7,0.7]
t=(0,20) #set linspace to 0,24 as limit cycle happens at ~ 21s
b=2
sol=solve_ivp(hopf_bifurcation_modified(b), t, X0)
visualise(sol.y,sol.t, 'hopf_bifurcation_modified_start')

# initial pred and prey populations
X0=[0.5,0.5]
t = (0,20) #set linspace to 0,24 as limit cycle happens at ~ 21s
b=-1
sol=solve_ivp(hopf_bifurcation_modified(b), t, X0)
visualise(sol.y,sol.t, 'hopf_bifurcation_modified_end')


