from generalised_shooting_method import shooting
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from find_root import find_root
import warnings
import numpy as np
import warnings
warnings.filterwarnings("error")#warning were still causing our guess to update      causing solutions to jump therefor I have started catching runtime warnings

#TODO add handler for local minima
#TODO add collocation method
#TODO add tolerance and maxiter
#TODO make sure that if method = shooting t is set

def npc(func_wrapper, u0, p, t, b_vars, n_steps=100):
    """Function performs natural parameter continuation, i.e., it simply increments the a parameter by a set amount and attempts to find the solution for the new parameter value using the last found solution as an initial guess.

        Parameters
        ____________________

        dudt : callable Right-hand side of the system. The calling signature is fun(t, y). Here t is a scalar, and there are two options for the ndarray y: It can either have shape (n,); then fun must return array_like with shape (n,). Alternatively it can have shape (n, k); then fun must return an array_like with shape (n, k), i.e. each column corresponds to a single column in y. The choice between the two options is determined by vectorized argument (see below). The vectorized implementation allows a faster approximation of the Jacobian by finite differences (required for stiff solvers).

        u0 : array-like
            guess of the solution to function at the starting param

        var_p : tuple
            Interval of parameter variation (p0, pf). The solver starts with p=p0 and re-calculates result until it reaches p=pf.

        b_vars :  array-like
            the boundary variable that specify the periodicity condition of the equation

        n_steps=100 : int optional,
                the number of equally spaced steps at which the iteration should be run

        ---------------------
        Returns : dict
            params - parameter value
            solutions - solution at corresponding parameter value
    """

    #steps #TODO change this to linspace
    steps=np.linspace(p[0], p[1], n_steps)

    #define response dict
    res= { "solutions":[],"params": [] }

    #Loop through param values running shooting to find solution
    for par in steps:
        """By passing the function wrapper instead of the function I can update the function definition at run time allowing var_par to change with each iteration"""
        # prep function
        dudt = func_wrapper(p)

        if not shooting:
            u, info, ier, msg = fsolve(dudt, u0, full_output=True)
            if ier == 1:
                print("Root finder found the solution u={} after {} function calls; the norm of the final residual is {}".format(u, info["nfev"], np.linalg.norm(info["fvec"])))
            else:
                print("Root finder failed with error message: {}".format(msg))
        else:
            u=shooting(u0, dudt, t, b_vars)

        #prep result to return
        if u is not None:
            u0=u # update initial guess
            res["params"].append(par)
            res["solutions"].append(u)

    return res


