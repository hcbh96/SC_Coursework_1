from scipy.optimize import newton
from scipy.integrate import odeint
from ode_integrator import func_to_solve

def find_root(equation, X0, root_finder=newton, tol=1e-4, maxiter=50):
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
    # TODO **kwargs or *args here
    # if the root finder is a newton
    if root_finder.__name__ == 'fsolve':
        res = root_finder(equation,X0,xtol=tol, maxfev=maxiter)
    # use the newton method to solve for the initial conditions
    elif root_finder.__name__ == 'newton':
        res=root_finder(equation,X0,tol=tol, maxiter=maxiter)
    return res
