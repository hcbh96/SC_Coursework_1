from scipy.optimize import newton
from scipy.integrate import odeint
from ode_integrator import func_to_solve

def shooting(dXdt, X0, t, boundary_vars, integrator=odeint, root_finder=newton, tol=1e-4, maxiter=50, _func_to_solve=func_to_solve):
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
    #this will cause the function to throw if an unsupported root finder is supplied
    if root_finder.__name__ not in ['fsolve','newton']:
             raise AttributeError("This function only works with either scipy.newton or scipy.fsolve as a root_finder")


    # if the root finder is a newton
    if root_finder.__name__ == 'fsolve':
        res = root_finder(func_to_solve,X0,args=(dXdt,t,boundary_vars, integrator),xtol=tol, maxfev=maxiter)
    # use the newton method to solve for the initial conditions
    elif root_finder.__name__ == 'newton':
        res=root_finder(_func_to_solve,X0,args=(dXdt,t,boundary_vars, integrator),tol=tol, maxiter=maxiter)
    return res
