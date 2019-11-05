"""
GeneralisedShootingMethod.py contains a generalised version of the shooting method

One will be able to
- provide a function to solve
- import the method for use in a separate file
- set the tol, maxiter, boundary conditions
- initial guess
- fsolve or newton
- solve_ivp
- The function will return the """

from scipy.optimize import newton
from scipy.integrate import odeint


#define function to pass to fsolve
def func_to_solve(X, dXdt, t, boundary_vars):
    """Return the difference between the shot and the initial_bv conditions"""
    sol=boundary_vars - odeint(dXdt,X,t)[-1]
    return sol

def solve(dXdt, X0, t, boundary_vars, xtol=1e-4, maxfev=50):
    result=newton(func_to_solve,X0,args=(dXdt,t,boundary_vars))
    return result


