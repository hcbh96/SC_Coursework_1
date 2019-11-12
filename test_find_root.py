import pytest
import math
from scipy.optimize import newton
from scipy.optimize import fsolve
from find_root import find_root

def test_should_find_the_root_using_newton():
    #arrange
    equation = lambda x : x + 2
    X0=6

    #act
    sol=find_root(equation, X0, root_finder=newton, tol=1e-4, maxiter=50)

    #assert
    assert math.isclose(sol, -2), "Should equal - 2"


def test_should_find_root_using_fsolve():
    #arrange
    equation = lambda x : x - 2
    X0=7

    #act
    sol=find_root(equation,X0,root_finder=fsolve, tol=1e-4, maxiter=50)

    #assert
    assert math.isclose(sol, + 2), "Should be equal to 2"

def test_should_throw_if_root_finder_is_not_recognised():
    #arrange
    eq = lambda x : x
    X0=7
    def fake_root_finder():
        return "Fake it to make it"
    throws=False
    #act
    try:
        find_root(eq,X0,root_finder=fake_root_finder)
    except AttributeError:
        throws = True

    assert throws == True, "The function should have a thrown an attribute error"
