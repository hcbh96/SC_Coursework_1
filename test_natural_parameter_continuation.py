"""This is a test file for the function natural_parameter_continuation, I have used this opportunity to practice TDD and use pytest as an alternative to unittest"""

import pytest
import math
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import newton
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from natural_parameter_continuation import npc

# TODO I need to write assertions here
def test_should_use_solve_method_via_fsolve():
    """This function seems to always find a root and then a local min"""
    # arrange
    def func_wrapper(v) :
        return lambda x: x**3 -x + v
    X0=[1]
    vary_par=dict(start=-2,stop=2,steps=100)


    #act
    sol=npc(func_wrapper, X0, vary_par, method='solve',root_finder=fsolve)


    #assert
    exp=1/math.sqrt(3)# this is the value of the local minima
    assert len(sol["params"]) == 60, "Should return 60 solutions actually
    returns    '{0}'".format(len(sol["params"]))
    assert len(sol["results"]) == 60, "Should return 60 solutions actually
    returns   '{0}'".format(len(sol["results"]))
    for i in range(len(sol["params"])):# checks validity of solution before bif
        calc_result = func_wrapper(sol["params"][i])(sol["solutions"][i])
        assert math.isclose(calc_result,0, abs_tol=1e-03), "The solution to the
        function given the parameter '{0}' should be 0 but instead equals {1}"
        .format(sol["params"][i], calc_result)


def test_run_npc_on_hopf_bifurcation_normal_form():
    """Returns two roots to the equation using the newton method"""
    # arrange
    def hopf_bifurcation(beta):
        """Return a systems of equations relating to the hopf bifurcation"""
        return lambda t, X : [
                beta*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+beta*X[1]-X[1]*(X[0]**2+X[1]**2),
               ]


    u0=np.array([0,0])
    p=(0,2)
    b_vars=np.array([0,0])
    t=(0,6.3)
    n_steps=40
    #act
    sol=npc(hopf_bifurcation, u0, p, t, b_vars, n_steps)


    #assert
    assert len(sol["results"]) == 40
    for i in range(len(sol["params"])):# checks validity of solution before bif
        calc_result = func_wrapper(sol["params"][i])(sol["results"][i])
        assert math.allclose(calc_result, 0, abs_tol=1e-03)


def test_return_solution_using_newton_and_solve_ivp():
    """Calculates the continuous solution to the Hopf bifurcation and asserts
    the first and last solution is correct i.e b=2 and b=-1 using a newton
    root_finder and solve_ivp integrator"""
    # arrange
    def func_wrapper(v):
        return lambda t, X : [
                v*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+v*X[1]-X[1]*(X[0]**2+X[1]**2),
                ]
    X0=[1, 1]
    vary_par=dict(start=0, stop=2, steps=10)
    b_vars=[1,1]
    t=(0,6.25)
    # act
    sol=npc(func_wrapper, X0, vary_par, t, method='shooting',
            boundary_vars=b_vars, integrator=solve_ivp, root_finder=newton)
    # assert
    for i in range(len(sol["params"])):
        calc_result = func_wrapper(sol["params"][i])(0, sol["solutions"][i])
        assert math.isclose(np.linalg.norm(calc_result),0, abs_tol=1e-03),
        "The solution to the function given    the parameter '{0}' should
        be 0 but instead equals {1}".format(sol["params"][i], calc_result)


def test_return_solution_using_newton_and_odeint():
    """Calculates the continuous solution to the Hopf bifurcation using the
    shooting method with odeint the solution returned is actually a local
    minima but the test takes account of that and you will se warning
    printed out"""
    # arrange
    def func_wrapper(v):
        # X t has to be t X when using solve_ivp
        return lambda X, t=0 : [
                v*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+v*X[1]-X[1]*(X[0]**2+X[1]**2),
                ]
    X0=np.array([0.33,0.33])
    vary_par=dict(start=0, stop=2, steps=10)
    b_vars=np.array([0.33,0.33])
    t=np.linspace(0,6.3)
    # act
    sol=npc(func_wrapper, X0, vary_par, t, method='shooting',
            boundary_vars=b_vars, root_finder=newton, integrator=odeint)
    # assert
    assert len(sol["params"]) == 8, "Should return 10 solutions actually
    returns    '{0}'".format(len(sol["params"]))
    for i in range(len(sol["params"])):# checks validity of all solutions
        calc_result = odeint(func_wrapper(sol["params"][i]),
                sol["solutions"][i], t)
        assert np.allclose(calc_result, b_vars, atol=1e-01), "Integrating
        the function given the param : '{0}', X0: '{1}' should return the
        b_vars: '{2}' but instead returns : '{3}'".format(sol["params"][i],
                sol["solutions"][i], b_vars, calc_result)

