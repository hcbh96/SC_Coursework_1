import numpy as np
from scipy.optimize import root
from scipy.optimize import fsolve
from shooting import shooting
def func_to_solve(v, func_wrapper, v_guess, delta_v, shoot):
    #TODO: Test params passed to this function
    #TODO: Decouple input functions and test input output
    # v=[u, p]

    # func to guess to dudt
    dudt=func_wrapper(v[-1])
    print("Fitting with param {}, using initial guess {}".format(v[-1], v[0:-1]))
    lam = lambda v : np.array([np.dot(delta_v, (v-v_guess))])
    if not shoot:
        s_r = fsolve(dudt, v[0:-1])
        print("s_r {}".format(s_r))
        print("lamda {}".format(lam(v)))
    else:
        s_r = shooting(v[0:-1], dudt)
        print("s_r {}".format(s_r))
        print("lamda {}".format(lam(v)))
    res = np.concatenate([s_r,lam(v)])
    print("Result {}".format(res))
    return res

#TODO: Ensure the function is properly documented
def pac(func_wrapper, v0, v1, p_range, step_size=0.1, shoot=True):
    """
    Function performs natural pseudo archlength continuation, The method is
    based on the observation that the "idealparametrisation of a curve is
    arch- length. Pseudo-arch-length is an approximation of the arch-length
    in the tangent space of the curve.

         USAGE: pac(func_wrapper, v0, v1, p_range, step_size)


         INPUTS:

         func_wrapper: callable g(x) returns f(t, x).
             This Input should be a function wrapper which returns a
             function defining dudt

         v0 : array-like
             guess of the solution to function in the form [u0,...uN,T,p], the
             T is the expected period and the p is the param value

         v1 : array-like
              guess of the solution to function in the form [u0,...uN,T,p], the
              T is the expected period and the p is the param value

         p_range : tuple
             Interval of parameter variation (p0, pf). The solver will break
             after it exceeds one of the p_range bounds

         step_size=0.1 : int optional,
                 the number of equally spaced steps at which the iteration
                 should be run

         OUTPUT : dict
             params - parameter value for which solutions were calculated
             solutions - solution at corresponding parameter value

         NOTE: This function is currently failing its tests, this is most likely
         due to an issue with the shooting method
    """
    #prep_solution
    solution = { "params": [], "solutions": [] }

    # set up loop and termination
    while v1[1] >= p_range[0] and v1[1] <= p_range[1] and v1[1] >= p_range[0] and v1[1] <= p_range[1]:

        # calc secant
        delta_v = v1 - v0 # only take the postional variables into acc

        # using v1 and delta_v calc v2 guess
        v_guess = v1 + delta_v*step_size

        # solve for root #TODO decouple functions and unit test
        print("v_guess length: {}".format(v_guess))
        v2=root(func_to_solve, v_guess,
                args=(func_wrapper, v_guess, delta_v, shoot), method='lm').x
        print("V2 : {}".format(v2))


        # reassign, re-run and prep solution
        v0=v1;v1=v2
        print("Solution for param {} is {}".format(v2[-2], v2[0:-2]))
        solution["params"].append(v1[-2])
        solution["solutions"].append(v1[0:-2])

    return solution

