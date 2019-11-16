from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from ode_integrator import ode_integrator
from find_root import find_root
import numpy as np

def shooting(dXdt, X0, t, boundary_vars, integrator=solve_ivp, root_finder=fsolve, tol=1e-4, maxiter=50, _func_to_solve=ode_integrator, solve_derivative=False):
    """
    A function that returns an estimation of the starting condition of a BVP subject to the first order differential equations

    Parameters:
    ___________
        dXdt : an ndarray containing the first order differtial equations to be solved
        X0 : ndarray of the expected starting conditions of the equation
        t : ndarray containing a number of equally spaced samples in a closed interval relating to the period of oscillation
        boundary_vars : the intended target specified by the boundary conditions
        tol=1e-4 : float, optional the allowable error of the zero value
        maxiter=50 : int optional, maximum number of iterations
        integrator=newton : fsolve or newton (scipy.optimize) a root finder that can be used to find the approx answer
        root_finder=odeint : odeint or solve_ivp (scipy.integrate) an iterator to calculate the the result of the shot
    ----------
    Returns : an ndarray containing the corrected initial values for the limit cycle.
"""

    res = find_root(_func_to_solve, X0, root_finder, args=(dXdt, t, boundary_vars, integrator, solve_derivative))
    return res
