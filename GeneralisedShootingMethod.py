from scipy.optimize import newton
from scipy.integrate import odeint

#define function to pass to solve
def _func_to_solve(X, dXdt, t, boundary_vars):
    """Return the difference between the shot and the target            conditions"""
    sol=boundary_vars - odeint(dXdt,X,t)[-1]
    return sol

def solve(dXdt, X0, t, boundary_vars, tol=1e-4, maxiter=50):
    """
    A function that returns an estimation of the starting condition of a BVP subject to the first order differential equations

    Parameters:
    ___________
        dXdt : an ndarray containing the first order differtial equations to be solved
        X0 : ndarray of the expected starting conditions of the equation
        t : ndarray containing a number of equally spaced samples in a closed interval relating to the period of oscillation
        target : the intended target specified by the boundary conditions
        tol : float, optional the allowable error of the zero value
        maxiter : int optional, maximum number of iterations
    ----------
    Returns : an ndarray containing the corrected initial values for the limit cycle.
"""
    #TODO add in comment about what happens in the numerical root finder fails
    #TODO ask Dr Barton what sort of additional options would be worth including
    #TODO what exactly is meant by the phase condition why might dcxdt dt be better than an x value

    # use the newton method to solve for the initial conditions
    return newton(_func_to_solve,X0,args=(dXdt,t,boundary_vars),tol=tol, maxiter=maxiter)


