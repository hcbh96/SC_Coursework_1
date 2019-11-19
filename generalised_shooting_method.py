from scipy.optimize import root
from scipy.integrate import solve_ivp
from ode_integrator import ode_integrator
from find_root import find_root
import numpy as np

def phase_cond(u, dudt, t):
    """
    Return the phase_cond of the equation (always 0)
    """
    res= np.array(dudt(0, solve_ivp(dudt, t, u).y[:,-1]))
    return res

def periodicity_cond(u, dudt, t, b_vars):
    """
    Returns the periodicity_cond of the equation
    """
    # integrate the ode for time t from starting position U
    res = np.array(b_vars - solve_ivp(dudt, t, u).y[:,-1])
    return res

def g(u, dudt, t, b_vars):
    res = np.concatenate((periodicity_cond(u, dudt, t, b_vars), phase_cond(u, dudt, t)))
    return res


def shooting(u0, dudt, t, b_vars):
    """
    A function that returns an estimation of the starting condition of a BVP subject to the first order differential equations

    Parameters:
    ___________

        dudt : an ndarray containing the first order differtial equations to be solved

        u0 : ndarray of the expected starting conditions of the equation

        p : any additional parameters for dudt

        t : 2-tuple of floatsi. Interval of integration (t0, tf). The solver starts with t=t0 and integrates until it reaches t=tf.

        boundary_vars : The periodicity conditions that need to be satisfied

    ----------
    Returns : an ndarray containing the corrected initial values for the limit cycle.
    """

    sol = root(g, u0, args=(dudt, t, b_vars), method="lm")

    if sol["success"] == True:
         print("Root finder found the solution u={} after {} function calls".format(sol["x"], sol["nfev"]))
         return sol["x"]
    else:
         print("Root finder failed with error message: {}".format(sol["message"]))
         return None
