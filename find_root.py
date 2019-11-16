from scipy.optimize import fsolve
from scipy.integrate import odeint
import numpy as np

def find_root(equation, X0, root_finder=fsolve, args=(),tol=1e-4, maxiter=50, full_output=True):
    """
    A function that returns the solution to an equation using either the fsolve or the newton method.

    Parameters:
    ___________
        equation : an ndarray containing the first order differtial equations to be solved
        root_finder=newton : the root_finder to use to find the solutions to the equation
        tol=1e-4 : float, optional the allowable error of the zero value
        maxiter=50 : int optional, maximum number of iterations
    ----------
    Returns : an ndarray containing the corrected initial values for the limit cycle.
"""
    #this will cause the function to throw if an unsupported root finder is supplied
    if root_finder.__name__ not in ['fsolve','newton']:
             raise AttributeError("This function only works with either scipy.newton or scipy.fsolve as a root_finder")
    # if the root finder is a newton
    if root_finder.__name__ == 'fsolve':
        res, infodict, ier, mesg = root_finder(equation, X0, args=args, xtol=tol, maxfev=maxiter, full_output=True)
        # handle non convergence
        if not ier == 1:
            raise RuntimeError("The following error was raised: '{0}', for equation: '{1}' with X0 values: '{2}'".format(mesg, equation.__name__, X0))
    # use the newton method to solve for the initial conditions
    elif root_finder.__name__ == 'newton':
        sol = root_finder(equation, X0, args=args, tol=tol, maxiter=maxiter, full_output=True)
        res = sol[0]
        print("sol: ".format(sol))
        try:
            converged=all(sol[1].converged)
        except:
            converged=sol[1].converged

        if not converged:
            print("Not converged")
            raise RuntimeError("The newton root finder did not find a root for equation: '{1}' with X0 values:  '{2}'".format(equation.__name__, X0))

    return res
