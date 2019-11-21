from scipy.optimize import root
from scipy.integrate import solve_ivp
import numpy as np

def phase_cond(u, dudt, t):
    res= np.array(dudt(0,u))
    return res

def periodicity_cond(u, dudt, t, b_vars):
    # integrate the ode for time t from starting position U
    res = np.array(b_vars - solve_ivp(dudt, t, u).y[:,-1])
    return res

def g(u, dudt, t, b_vars):
    res = np.concatenate((
        periodicity_cond(u, dudt, t, b_vars),
        phase_cond(u, dudt, t),
        ))
    return res


def shooting(u0, dudt, t, b_vars):
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
    sol = root(g, u0, args=(dudt, t, b_vars), method="lm")
    if sol["success"] == True:
         print("Root finder found the solution u={} after {} function calls"
                 .format(sol["x"], sol["nfev"]))
         return sol["x"]
    else:
         print("Root finder failed with error message: {}".format(sol["message"]))
         return None
