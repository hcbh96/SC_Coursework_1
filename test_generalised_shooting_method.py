"""
Test Generalised shooting method contains test aimed at ensuring the Generalised shooting method is up to scratch and working properly.
"""

from scipy.optimize import fsolve
from scipy.optimize import newton
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

from generalised_shooting_method import shooting
import numpy as np
import pytest

#define a test class
def test_on_constant_derivative_find_root_with_odeint():
    # rate of change set to conistant
    def dXdt(X, t=0):
        return [1]

    #define an initial guess
    X0=[1]

    #expected starting params at boundaries
    boundary_vars=[1]

    # define timespan
    t=np.linspace(0,1,10)

    #calc the result using a secant method
    res=shooting(dXdt, X0, t, boundary_vars, integrator=odeint)

    assert np.isclose(res, [0], atol=1e-03), "'{0}' be close to 0".format(res)


def test_with_varying_derivative_to_find_stationary_point_with_solve_ivp():
    """This function is designed to test the generalised shooting method to an accuracy of 6 decimal places on a very simple system of ODEs"""
    # rate of change set to constant
    def dXdt(t, X):
        return np.array([X[0],X[1]])

    #define an initial guess
    X0=np.array([1,1])

    #expected starting params at boundaries
    boundary_vars=np.array([1,1])

    # define timespan
    t=np.linspace(0,10)

    #calc the result using a secant method
    res=shooting(dXdt, X0, t, boundary_vars, integrator=solve_ivp, solve_derivative=True)

    assert np.allclose(res, [0.815,0.815], atol=1e-03), "'{0}' should be close to [1,1]".format(res)


def test_on_lotka_volterra():
     """This function is intended to test the generalised shooting method on the Lotka Volterra method ensuring that the absolute tolerance of the solution is within 2 decimal places

     params such [a,b,d] below should be defined outside the function as it means one can write loops with varying params during numerical analysis"""
     #define params
     a=1; d=0.1; b=0.2

     # rate of change and pred and prey populations
     def dXdt_lotka_volterra(t, X):
         """Return the change in pred and prey populations"""
         return np.array([
             X[0]*(1-X[0])-((a*X[0]*X[1])/(d+X[0])),
             b*X[1]*(1-(X[1]/X[0]))
             ])

     #define an initial guess for the starting conditions
     X0=[0.51,0.5]

     #expected starting params at boundary
     boundary_vars=[0.38, 0.38]

     #time range
     t=np.linspace(0,20.76)

     #calc the result using a secant method
     res=shooting(dXdt_lotka_volterra, X0, t, boundary_vars)
     assert np.allclose(res, 0.38, atol=1e-01), "'{0}' should be close to [0.38, 0.38]".format(res)


def test_on_hopf_bifurcation():
    """
    This function is intended to test the  generalised shooting method on the Hopf Bifurcation ensuring that the absolute tolerance is within 2 decimal places
    """
    sigma=1;beta=0
    def dXdt_hopf_bif(t, X):
        """Function to calculate the rate of change of the Hopf Bifurcation at position X"""
        return np.array([
                beta*X[0]-X[1]+sigma*X[0]*(X[0]**2+X[1]**2),
                X[0]+beta*X[1]+sigma*X[1]*(X[0]**2+X[1]**2)
                ])

    # make an initial condition guess
    X0=[0.6,0.6]

    # set the boundary conditions
    boundary_vars=[0,1.5]

    # define a time range
    t=np.linspace(0,6.25)

    #find the solution of the generalised shooting method
    res=shooting(dXdt_hopf_bif, X0, t, boundary_vars)
    expected = [ 0.15, 1.19 ]
    assert np.allclose(res, expected, atol=1e-02), "'{0}' should be close to '{1}'".format(res, expected)


def test_on_system_of_three():
     """
     This function is designed to ensure that the generalised shooting method works well when given a system of 3 first order ODEs

     When Testing out generalised shooting method for a system of equations containing three first order ODEs I decided it was in the interest of the reader to use the most basic example possible, to allow the system to be envisaged by the reader and give them undoubtable confidence that the test is testing for the correct output

     The specific point of this test is to ensure that my shooting method works on systems of first order linear ODEs of greater than 2 dimensions, this test focuses solely on doing that in the most simple way possible, i.e by having a system of three linearly independent first order ODE which all have solutions at 0.
"""
     sigma=-1;beta=1
     def dXdt(t, X):
         """Function to calculate the rate of change of the Hopf Bifurcation  at position X"""
         return np.array([
                 X[0],
                 X[1],
                 X[2]
                 ])

     # make an initial condition guess
     X0=[1,0,0.5]

     # set the boundary conditions
     boundary_vars=[1,1,1]

     # define a time range
     t=np.linspace(0,10)

     #find the solution of the generalised shooting method
     res=shooting(dXdt, X0, t, boundary_vars)
     expected = [0.82, 0.82, 0.82 ]
     assert np.allclose(res, expected, atol=1e-02), "'{0}' should be close to '{1}'".format(res, expected)


def test_unmatched_input_dimensions():
    #arrange
    def dXdt(t, X):
        return np.array([0,0])
    X0=[0,0,0] # this variable has too many inputs
    boundary_vars=[0,0]
    t=np.linspace(0,10)
    #act
    throws = False
    try:
        res=shooting(dXdt, X0, t, boundary_vars)
    except ValueError:
        throws = True
    #assert
    assert throws, "Function should have thrown"


def test_unmatched_input_dimensions_2():
    #arrange
    def dXdt(t, X):
        return np.array([0,0])
    X0=[0,0]
    boundary_vars=[0,0,0] # this variable has too many inputs
    t=np.linspace(0,10)
    throws = False
    #act
    try:
        res=shooting(dXdt, X0, t, boundary_vars)
    except ValueError:
        throws = True
    #assert
    assert throws, "Function should have throws"


def test_shooting_method_with_no_convergence():
    # rate of change and pred and prey populations
    def dXdt(X, t=0):
        """Return the change in pred and prey populations"""
        return 0#this has no roots

    #define an initial guess for the starting conditions
    X0=2

    #expected starting params at boundary
    boundary_vars=0

    #time range
    t=np.linspace(0,10)
    #calc the result using a secant method
    throws = False
    try:
        res=shooting(dXdt, X0, t, boundary_vars, maxiter=1, integrator=odeint, root_finder=newton)
    except RuntimeError:
        throws = True

    assert throws, "Function should have thrown"

def test_function_with_fsolve():
    sigma=-1;beta=1
    def dXdt(t, X):
      """Function to calculate the rate of change of the Hopf Bifurcation     at position X"""
      return np.array([X[0],X[1],X[2]])

    # make an initial condition guess
    X0=[1,0,0.5]

    # set the boundary conditions
    boundary_vars=[0,0,0]

    # define a time range
    t=np.linspace(0,10)

    #find the solution of the generalised shooting method
    res=shooting(dXdt, X0, t, boundary_vars, root_finder=fsolve)
    expected = [0 , 0, 0 ]
    assert np.allclose(res, expected, atol=1e-03), "'{0}' should be close to '{1}'".format(res, expected)


def test_should_solve_ivp():
     # rate of change set to conistant
     def dXdt(t, X):
         return [X[0],X[1]]

     #define an initial guess
     X0=[1,1]

     #expected starting params at boundaries
     boundary_vars=[1,1]

     # define timespan
     t=(1,10)

     #calc the result using a secant method
     res=shooting(dXdt, X0, t, boundary_vars, integrator=solve_ivp)

     assert np.allclose(res, [0,0], atol=1e-03), "'{0}' should be close to [0,0]".format(res)


def test_func_throws_with_non_defined_integrator():
    # rate of change set to conistant
    def dXdt(t, X):
      return [1]
    def random_integrator(X):
      return X[0]*2+1

    #define an initial guess
    X0=[1]

    #expected starting params at boundaries
    boundary_vars=[1]

    # define timespan
    t=np.linspace(0,1)

    throws=False
    #calc the result using a secant method
    try:
      res=shooting(dXdt, X0, t, boundary_vars,         integrator=random_integrator)
    except AttributeError:
      throws=True

    assert throws, "Fucntion should have thrown"


if __name__ == '__main__':
    unittest.main()
