from scipy.optimize import newton
from scipy.integrate import odeint


#define function to pass to solve
def func_to_solve(X, dXdt, t, boundary_vars):
    """Return the difference between the shot and the target conditions"""
    sol=boundary_vars - odeint(dXdt,X,t)[-1]
    return sol

def solve(dXdt, X0, t, boundary_vars, tol=1e-4, maxiter=50):
    """Returns an estimation of the starting condition of a BVP subject to the first order differential equations

    Parameters:
        dXdt: an ndarray containing the first order differtial equations to be solved
        X0: ndarray of the expected starting conditions of the equation
        t: ndarray containing a number of equally spaced samples in a closed interval
        target: the intended target specified by the boundary conditions
        tol: float, optional the allowable error of the zero value
        maxiter: int optional, maximum number of iterations
"""
    return newton(func_to_solve,X0,args=(dXdt,t,boundary_vars),tol=tol, maxiter=maxiter)


