"""This is a test file for the function natural_parameter_continuation, I have used this opportunity to practice TDD and use pytest as an alternative to unittest"""

import pytest
import math
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import newton
from natural_parameter_continuation import npc

# TODO this test shows an error as it is currently finding a local minima
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

def test_return_continuous_solution_to_hopf_bifurcation():
    """Calculates the continuous solution to the Hopf bifurcation this is having issues as if b decreases instead of increase while array does not work"""
    # arrange
    def func_wrapper(v):
        return lambda X, t=0 : [
                v*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2)**2,
                X[0]+v*X[1]+X[1]*(X[0]**2+X[1]**2)-X[1]*(X[0]**2+X[1]**2)**2,
                ]
    X0=[1,1]
    vary_par=dict(start=2, stop=-1, steps=100)
    b_vars=[1,1]

    # act
    sol=npc(func_wrapper, X0, vary_par, method='shooting', boundary_vars=b_vars)
    # assert
    print(sol[0])
    print(sol[-1])
    assert np.allclose(sol[0], [16.779, -4.695], atol=1e-02)
