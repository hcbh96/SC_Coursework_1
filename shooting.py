from scipy.optimize import root
from scipy.integrate import solve_ivp
import numpy as np

def phase_cond(u, dudt):
    res= np.array(dudt(0,u))
    return res

def periodicity_cond(u, dudt, T):
    # integrate the ode for time t from starting position U
    res = np.array(u - solve_ivp(dudt, (0, T), u).y[:,-1])
    return res

def g(state_vec, dudt):
    T = state_vec[-1]
    u=state_vec[0:-1]
    res = np.concatenate((
        periodicity_cond(u, dudt, T),
        phase_cond(u, dudt),
        ))
    return res


def shooting(state_vec, dudt):
    """
    A function that returns an estimation of the starting condition of a BVP
    subject to the first order differential equations

    USAGE: shooting(state_vec, dudt)

    INPUTS:

        state_vec : ndarray
            the state_vector to solve, [u0...uN,T] the final argument should be
            the expected period of the limit cycle or the period of the limit
            cycle.

        dudt : ndarray
            containing the first order differtial equations to be solved

    ----------
    OUTPUT : an ndarray containing the corrected initial values for the limit cycle.

    NOTE: This function is currently having issues when used with npc however it
    is also currently passing all of its tests
    """
    sol = root(g, state_vec, args=(dudt,), method="lm")
    if sol["success"] == True:
         print("Root finder found the solution u={} after {} function calls"
                 .format(sol["x"], sol["nfev"]))
         return sol["x"]
    else:
         print("Root finder failed with error message: {}".format(sol["message"]))
         return None
