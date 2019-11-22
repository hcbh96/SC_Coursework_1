from shooting import shooting
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np

def npc(func_wrapper, state_vec, p, n_steps=100, shoot=True):
    """Function performs natural parameter continuation, i.e., it simply
    increments the a parameter by a set amount and attempts to find the
    solution for the new parameter value using the last found solution
    as an initial guess.

        USAGE: npc(func_wrapper, state_vec, p, n_steps)


        INPUTS:

        func_wrapper: callable g(x) returns f(t, x).
            This Input should be a function wrapper which returns a
            function defining dudt

        state_vec : array-like
            guess of the solution to function in the form [u0,...uN,T], the
            T the final param is the expected period and is a necessary
            argument if shoot == True (default)

        p : tuple
            Interval of parameter variation (p0, pf). The solver starts with
            p=p0 and re-calculates result until it reaches p=pf.

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
        print("Running on parameter value {}".format(par))
        # prep function
        dudt = func_wrapper(par)
        if not shoot:
            u, info, ier, msg = fsolve(dudt, state_vec, full_output=True)
            if ier == 1:
                print("Root finder found the solution u={} after {} function calls, with paramater {}; the norm of the final residual is {}".format(u,info["nfev"], par, np.linalg.norm(info["fvec"])))
            else:
                u=None
                print("Root finder failed with error message: {}".format(msg))
        else:
            u=shooting(state_vec, dudt)

        #prep result to return
        if u is not None:
            state_vector=u
            res["params"].append(par)
            res["solutions"].append(u)

    return res


