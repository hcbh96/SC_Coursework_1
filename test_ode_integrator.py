from ode_integrator import func_to_solve
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

import numpy as np
import pytest

def test_throws_if_passed_random_integrator():
    # arrange
    X=1
    dXdt=lambda x : x
    t=(1,2)
    boundary_vars=0

    def random(X): return X #testing this fails with AttributeError if not passed odeint or solve_ivp

    # act
    throws=False
    try:
        print(boundary_vars)
        func_to_solve(X,dXdt,t,boundary_vars,integrator=random)
    except AttributeError:
        throws=True

    #assert
    assert throws==True, "Should throw as integrator is not odeint or solve_ivp"


# test odeint integrator
def test_odeint():
    #arrange
    X=np.array([6,6])
    dXdt=lambda x, t=0 : [1,1]
    t=np.linspace(0,10,1)
    boundary_vars=np.array([4,4])

    #act
    sol=func_to_solve(X, dXdt, t, boundary_vars, integrator=odeint)

    #assert
    assert np.all(sol) == np.all([2,2])

# test solve_ivp integrator
def test_sole_ivp():
    #arrange
    X=np.array([6,6])
    dXdt=lambda x, t=0 : [1,1]
    t=np.linspace(0,1)
    boundary_vars=np.array([4,4])

    #act
    sol=func_to_solve(X, dXdt, t, boundary_vars, integrator=solve_ivp)

    #assert
    assert np.all(sol) == np.all([2,2])

# test solve derivative
def test_solve_derivative():
    #arrange
    X=np.array([6,6])
    dXdt=lambda x, t=0 : [1,1]
    t=np.linspace(0,1)
    boundary_vars=np.array([4,4])

    #act
    sol=func_to_solve(X, dXdt, t, boundary_vars, integrator=solve_ivp, solve_derivative=True)

    #assert
    assert np.all(sol) == np.all([1,1])

