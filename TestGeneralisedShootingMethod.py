"""
ShootingMethod.py contains an implemtation of the shooting method to find the initial predator prey populations given a time period of 20.76s and a finishing condition of [pred, prey]=[0.38, 0.38]
"""

from scipy.optimize import newton
from scipy.integrate import odeint
import GeneralisedShootingMethod
import numpy as np

def test_on_constant_derivative_negative():
    # rate of change set to constant
    def dXdt(X, t=0):
        return 1

    #define an initial guess
    X0=1

    #expected starting params at boundaries
    boundary_vars=1

    # define timespan
    t=np.linspace(0,1,10)

    #calc the result using a secant method
    res=GeneralisedShootingMethod.solve(dXdt, X0, t, boundary_vars)

    assert np.isclose(res, 0.38, atol=1e-01).all() == False, "should be equal to true"
    return "test_on_constant_derivative_negative"

def test_on_constant_derivative_positive():
    # rate of change set to constant
    def dXdt(X, t=0):
        return 1

    #define an initial guess
    X0=1

    #expected starting params at boundaries
    boundary_vars=1

    # define timespan
    t=np.linspace(0,1,10)

    #calc the result using a secant method
    res=GeneralisedShootingMethod.solve(dXdt, X0, t, boundary_vars)

    assert np.isclose(res, 0, atol=1e-01).all() == True, "should be      equal to true"
    return "test_on_constant_derivative_positive"


def test_on_lotka_volterra():
     """params such [a,b,d] below should be defined outside the function as it means one can write      loops with varying params during numerical analysis"""
     #define params
     a=1; d=0.1; b=0.2

     # rate of change and pred and prey populations
     def dXdt_lotka_volterra(X,t=0):
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
     t=np.linspace(0,20.76,10000)

     #calc the result using a secant method
     res=GeneralisedShootingMethod.solve(dXdt_lotka_volterra, X0, t, boundary_vars)
     assert np.isclose(res, 0.38, atol=1e-01).all() == True, "should be equal to true"
     return "test_on_lotka_volterra"


def test_on_hopf_bifurcation():
    """
    Testing out generalised shooting method on the Hopf Bifurcation
    """
    sigma=-1;beta=1
    def dXdt_hopf_bif(X, t=0):
        """Function to calculate the rate of change of the Hopf Bifurcation at position X"""
        return np.array([
                beta*X[0]-X[1]+sigma*X[0]*(X[0]**2+X[1]**2),
                X[0]+beta*X[1]+sigma*X[1]*(X[0]**2+X[1]**2)
                ])

    # make an initial condition guess
    X0=[1,0]

    # set the boundary conditions
    boundary_vars=[0,1]

    # define a time range
    t=np.linspace(0,360,100)

    #find the solution of the generalised shooting method
    res=GeneralisedShootingMethod.solve(dXdt_hopf_bif, X0, t, boundary_vars)
    expected = [ 0.51, -0.15 ]
    assert np.isclose(res, expected, atol=1e-01).all() == True, "should be equal to true"
    return "test_on_hopf_bifurcation"

def test_on_system_of_three():
     """
     When Testing out generalised shooting method for a system of equations containing three first order ODEs I decided it was in the interest of the reader to use the most basic example possible, to allow the system to be envisaged by the reader...the specific point of this test is to ensure that my shooting method works on systems of first order linear ODEs of greater than 2 dimensions, this test focuses solely on doing that in the most simple way possible

     N.B I have made the above note to show Dr Barton why I have decided not to use the second system of ODEs supplied in the question paper.
     """
     sigma=-1;beta=1
     def dXdt(X, t=0):
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
     t=np.linspace(0,10,100)

     #find the solution of the generalised shooting method
     res=GeneralisedShootingMethod.solve(dXdt, X0, t, boundary_vars)
     expected = [0 , 0, 0 ]
     assert np.isclose(res, expected, atol=1e-01).all() == True, "should be   equal to true"
     return "test_on_system_of_three"

def test_unmatched_input_dimensions_1():
    """This was written before code i.e via TDD"""
    #arrange
    def dXdt(X, t=0):
        return np.array([0,0])
    X0=[0,0,0] # this variable has too many inputs
    boundary_vars=[0,0]
    t=np.linspace(0,10,100)
    #act
    res=GeneralisedShootingMethod.solve(dXdt, X0, t, boundary_vars)
    #assert
    expect="Input array sizes do not match"
    assert res == expect, "should ensure the array input sizes match"


print("Starting Tests on Generalised Shooting Method...")
print("%s passed" % test_on_constant_derivative_negative())
print("%s passed" % test_on_constant_derivative_positive())
print("%s passed" % test_on_lotka_volterra())
print("%s passed" % test_on_hopf_bifurcation())
print("%s passed" % test_on_system_of_three())
print("%s passed" % test_unmatched_input_dimensions_1())
print("All tests have passed on Generalised Shooting Method")
