from generalised_shooting_method import shooting
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from find_root import find_root
import warnings
import numpy as np

#TODO add handler for local minima
#TODO add collocation method
#TODO add tolerance and maxiter
#TODO make sure that if method = shooting t is set

def npc(func_wrapper, X0, vary_par, t=False, boundary_vars=[0], method='shooting', root_finder=fsolve, integrator=solve_ivp):
    """Function performs natural parameter continuation, i.e., it
simply increments the a parameter by a set amount and attempts
to find the solution for the new parameter value using the last
found solution as an initial guess.

        Parameters
        ____________________
        func_wrapper : a function in the form f(v) : g(x) where g(x) is the function to be solved which contains the parameter to be varies noted as v. For example def f(v): return lambda x, t=0 : x + v
            - If you are using solve_ivp (default) lambda must be of form lambda t, x : x + v
            - If you are using odeint lambda must be of form lambda x, t=0 : x + v

        X0 : guess of the solution to function g(x) with vary_param set to start

        vary_par : (start=1, end=2, steps=10) )the range over which to vary the vary_param

        boundary_vars=0 : optional, boundary conditions if a shooting method needs to be used

        step_size=0.1 : the step size to increment v by in each itteration

        max_steps=100 : optionsal, the max number of steps in the continuation method

        method='shooting' : optional the method to use to solve the equation can be shooting or just solve at the moment

        root_finder=fsolve : optional the root_finding method that you would like to use can be fsolve or root_finder atm

        integrator=solve_ivp : optional, the integrator to use to solve the boundary value problem if the shooting method is being used
        ______________________
        Returns : nd.array of solutions to each equation

        Warning the solution returned may be a minima, and the newton output is flatter than the fsolve output
"""

    # set i to vary params initial value
    v=vary_par['start']

    #steps
    steps=np.linspace(vary_par['start'], vary_par['stop'], vary_par['steps'])

    #define response array
    res=[]

    # create linspace
    # while parameter is below limit
    for v in steps:
        """By passing the function wrapper instead of the function I can update the function definition at run time allowing var_par to change with each iteration"""
        func=func_wrapper(v)
        try:# this is being used in case the root finder can not find a solution due to local minima
            if method == 'solve':
                sol=find_root(func, X0, root_finder) #TODO add tol and maxiter
            elif method == 'shooting':
                sol=shooting(func, X0, t, boundary_vars, integrator, root_finder, tol=1e-4, maxiter=50)
            # update initial guess
            X0=sol
            res.append(sol)
        except RuntimeError:
            warnings.warn("Probably stuck at a local stationary point vary_param: %s, X0 : %s" % (v, X0))

    return res

