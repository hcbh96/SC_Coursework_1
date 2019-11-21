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

    USAGE:

    INPUTS:

        u0 : ndarray
            the expected starting conditions of the equation

        dudt : ndarray
            containing the first order differtial equations to be solved

        t : 2-tuple of float.
            Interval of integration (t0, tf). The solver starts with t=t0 and
            integrates until it reaches t=tf.

        boundary_vars : The periodicity conditions that need to be satisfied

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
