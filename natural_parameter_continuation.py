from generalised_shooting_method import shooting
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np

def npc(func_wrapper, u0, p, t, b_vars, n_steps=100):
    """Function performs natural parameter continuation, i.e., it simply
    increments the a parameter by a set amount and attempts to find the
    solution for the new parameter value using the last found solution
    as an initial guess.

        USAGE: npc(func_wrapper, u0, p, t, b_vars, n_steps)


        INPUTS:

        func_wrapper: callable g(x) returns f(t, x).
            This Input should be a function wrapper which returns a
            function defining dudt

        u0 : array-like
            guess of the solution to function at the starting param

        p : tuple
            Interval of parameter variation (p0, pf). The solver starts with
            p=p0 and re-calculates result until it reaches p=pf.

        b_vars :  array-like
            the boundary variable that specify the periodicity condition of the
            equation

        n_steps=100 : int optional,
                the number of equally spaced steps at which the iteration
                should be run

        OUTPUT : dict
            params - parameter value for which solutions were calculated
            solutions - solution at corresponding parameter value

        NOTE: This function is currently failing its tests, this is most likely
        due to an issue with the shooting method
    """

    #steps
    steps=np.linspace(p[0], p[1], n_steps)

    #define response dict
    res= { "solutions":[],"params": [] }

    #Loop through param values running shooting to find solution
    for par in steps:
        """By passing the function wrapper instead of the function I can update
        the function definition at run time allowing var_par to change with
        each iteration
        """

        # prep function
        dudt = func_wrapper(par)
        if not shooting:
            u, info, ier, msg = fsolve(dudt, u0, full_output=True)
            if ier == 1:
                print("Root finder found the solution u={} after {} function
                        calls; the norm of the final residual is {}".format(u,
                            info["nfev"], np.linalg.norm(info["fvec"])))
            else:
                print("Root finder failed with error message: {}".format(msg))
        else:
            u=shooting(u0, dudt, t, b_vars)

        #prep result to return
        if u is not None:
            print("Updating U0: {} to u: {}".format(u0, u))
            u0=u # update initial guess
            b_vars=u
            res["params"].append(par)
            res["solutions"].append(u)

    return res


