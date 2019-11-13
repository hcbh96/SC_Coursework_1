"""This is a test file for the function natural_parameter_continuation, I have used this opportunity to practice TDD and use pytest as an alternative to unittest"""

import pytest
import math
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import newton
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from natural_parameter_continuation import npc

# NOTE this test shows an error as it is currently finding a local minima
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
    assert math.isclose(sol[0][0], 1.5214, rel_tol=1e-02)
    assert math.isclose(sol[-1][0], exp, rel_tol=1e-02) # exp is a local minima


def test_return_solution_when_newton_used():
    """Returns two roots to the equation using the newton method"""
    # arrange
    def func_wrapper(v) :
        return lambda x: x**3 -x + v
    X0=1
    vary_par=dict(start=-2,stop=-1,steps=100)


    #act
    sol=npc(func_wrapper, X0, vary_par, method='solve',root_finder=newton)


    #assert
    assert math.isclose(sol[0], 1.5214, rel_tol=1e-02)
    assert math.isclose(sol[-1], 1.3247, rel_tol=1e-02)


# NOTE when using the newton root finder in both cases the solution seems to blow up not sure why yet
def test_return_continuous_solution_to_hopf_bifurcation_using_newton_and_solve_ivp():
    """Calculates the continuous solution to the Hopf bifurcation and asserts the first and last solution is correct i.e b=2 and b=-1 using a newton root_finder and solve_ivp integrator"""
    # arrange
    def func_wrapper(v):
        return lambda t, X : [ #NOTE : X and t had to be switched here to run func properly
                v*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+v*X[1]-X[1]*(X[0]**2+X[1]**2),
                ]
    X0=[1, 1]
    vary_par=dict(start=0, stop=2, steps=10)
    b_vars=[1,1]
    t=(0,6.25)
    # act
    sol=npc(func_wrapper, X0, vary_par, t, method='shooting', boundary_vars=b_vars, integrator=solve_ivp, root_finder=newton)
    # assert
    print(sol)
    assert np.allclose(sol[0], [1, 1], atol=1e-02)
    assert np.allclose(sol[-1], [1,1], atol=1e-02)
    assert len(sol) == 10, "Should return 10 solutions actually returns %s " % len(sol)


# NOTE when using the newton root finder in both cases the solution seems to blow up not sure why yet
def test_return_continuous_solution_to_hopf_bifurcation_using_newton_and_odeint():
    """Calculates the continuous solution to the Hopf bifurcation using the shooting method with odeint the solution returned is actually a local minima but the test takes account of that and you will se warning printed out"""
    # arrange
    def func_wrapper(v):
        # X t has to be t X when using solve_ivp
        return lambda X, t=0 : [
                v*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+v*X[1]-X[1]*(X[0]**2+X[1]**2),
                ]
    X0=[1,1]
    vary_par=dict(start=0, stop=2, steps=10)
    b_vars=[1,1]
    t=(0,6.25)
    # act
    sol=npc(func_wrapper, X0, vary_par, t, method='shooting', boundary_vars=b_vars, root_finder=newton, integrator=odeint)
    # assert
    print(sol)
    assert np.allclose(sol[0], [1, 1], rtol=1e-02)
    assert np.allclose(sol[-1], [1,1], atol=1e-02)
    assert len(sol) == 10, "Should return 10 solutions actually returns %s " % len(sol)


def test_return_continuous_solution_to_hopf_bifurcation_using_fsolve_and_solve_ivp():
    """Calculates the continuous solution to the Hopf bifurcation this is having issues  as    if b         decreases instead of increase while array does not work"""
    # arrange
    def func_wrapper(v):
        # X t has to be t X when using solve_ivp
        return lambda t, X : [
                v*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+v*X[1]+X[1]*(X[0]**2+X[1]**2),
                ]
    X0=[1,1]
    vary_par=dict(start=0, stop=2, steps=10)
    b_vars=[1,1]
    t=(0,6.25)
    # act
    sol=npc(func_wrapper, X0, vary_par, t, method='shooting', boundary_vars=b_vars, root_finder=fsolve,integrator=solve_ivp)
    # assert
    assert np.allclose(sol[0], [1,1], atol=1e-01)
    assert np.allclose(sol[-1], [1, 1], atol=1e-01)
    assert len(sol) == 10, "Should return 10 solutions actually returns %s " % len(sol)


def test_return_continuous_solution_to_hopf_bifurcation_using_fsolve_and_odeint():
    """Calculates the continuous solution to the Hopf bifurcation this is having issues  as    if b decreases instead of increase while array does not work"""
    # arrange
    def func_wrapper(v):
        # X t has to be t X when using solve_ivp
        return lambda X, t=0 : [
                v*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+v*X[1]+X[1]*(X[0]**2+X[1]**2),
                ]
    X0=[1,1]
    vary_par=dict(start=0, stop=2, steps=10)
    b_vars=[1,1]
    t=(0,6.25)
    # act
    sol=npc(func_wrapper, X0, vary_par, t, method='shooting', boundary_vars=b_vars, root_finder=fsolve, integrator=odeint)
    # assert
    assert np.allclose(sol[0], [1, 1], atol=1e-01)
    assert np.allclose(sol[-1], [1,1], atol=1e-02)
    assert len(sol) == 10, "Should return 10 solutions actually returns %s " % len(sol)



def test_return_continuous_solution_to_modified_hopf_bifurcation_using_fsolve_and_solve_ivp():
    """Calculates the continuous solution to the Hopf bifurcation this is having issues  as    if           b         decreases instead of increase while array does not work"""
    # arrange
    def func_wrapper(v):
        # X t has to be t X when using solve_ivp
        return lambda t, X : [
                v*X[0]-X[1]+X[0]*(X[0]**2+X[1]**2)-X[0]*(X[0]**2+X[1]**2)**2,
                X[0]+v*X[1]+X[1]*(X[0]**2+X[1]**2)-X[1]*(X[0]**2+X[1]**2)**2,
                ]
    X0=[1,1]
    vary_par=dict(start=0, stop=2, steps=10)
    b_vars=[1,1]
    t=(0,6.25)
    # act
    sol=npc(func_wrapper, X0, vary_par, t, method='shooting', boundary_vars=b_vars, root_finder=fsolve,     integrator=solve_ivp)
    # assert
    assert np.allclose(sol[0], [1,1], atol=1e-01)
    assert np.allclose(sol[-1], [1, 1], atol=1e-01)
    assert len(sol) == 10, "Should return 10 solutions actually returns %s " % len(sol)


def test_return_continuous_solution_to_modified_hopf_bifurcation_using_fsolve_and_odeint():
    """Calculates the continuous solution to the Hopf bifurcation this is having issues  as    if b         decreases instead of increase while array does not work"""
    # arrange
    def func_wrapper(v):
        # X t has to be t X when using solve_ivp
        return lambda X, t=0 : [
                v*X[0]-X[1]+X[0]*(X[0]**2+X[1]**2)-X[0]*(X[0]**2+X[1]**2)**2,
                X[0]+v*X[1]+X[1]*(X[0]**2+X[1]**2)-X[1]*(X[0]**2+X[1]**2)**2,
            ]
    X0=[1,1]
    vary_par=dict(start=2, stop=-1, steps=10)
    b_vars=[1,1]
    t=(0,6.25)
    # act
    sol=npc(func_wrapper, X0, vary_par, t, method='shooting', boundary_vars=b_vars, root_finder=fsolve,     integrator=odeint)
    # assert
    assert np.allclose(sol[0], [1, 1], atol=1e-01)
    assert np.allclose(sol[-1], [1,1], atol=1e-01)
    assert len(sol) == 10, "Should return 10 solutions actually returns %s " % len(sol)
