from scipy.optimize import fsolve
from scipy.optimize import newton
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import math
from context import methods
from methods.shooting import shooting
import numpy as np
import pytest

#define a test class
def test_on_constant_derivative_find_root_with_odeint():
    # rate of change set to conistant
    def dXdt(t, X):
        return [1]

    #define an initial guess
    X0=[0]

    #expected starting params at boundaries
    boundary_vars=[1]

    # define timespan
    t=(0,1)

    #calc the result using a secant method
    res=shooting(X0, dXdt, t, boundary_vars)
    exp = solve_ivp(dXdt, t, res).y[:,-1]
    assert np.isclose(boundary_vars, exp, atol=1e-03)


def test_on_lotka_volterra():
     """This function is intended to test the generalised shooting method
     on the Lotka Volterra method ensuring that the absolute tolerance of
     the solution is within 2 decimal places

     params such [a,b,d] below should be defined outside the function as
     it means one can write loops with varying params during numerical analysis
     """
     #define params
     a=1; d=0.1; b=0.2

     # rate of change and pred and prey populations
     def dXdt(t, X):
         """Return the change in pred and prey populations"""
         return np.array([
            X[0]*(1-X[0])-((a*X[0]*X[1])/(d+X[0])),
            b*X[1]*(1-(X[1]/X[0]))
            ])

     #define an initial guess for the starting conditions
     X0=[0.51,0.5]

     #expected starting params at boundary
     b_vars=[0.46543377, 0.3697647]

     #time range
     t=(0,20.76)

     #calc the result using a secant method
     res=shooting(X0, dXdt, t, b_vars)
     exp = solve_ivp(dXdt, t, res).y[:,-1]
     assert np.allclose(b_vars, exp, atol=1e-02)


def test_on_hopf_bif_nor_form_b_1():
    def dudt(t, X):
        """Return a systems of equations relating to the hopf bifurcation"""
        return np.array([
                1*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+1*X[1]-X[1]*(X[0]**2+X[1]**2),
               ])

    #define an initial guess for the starting conditions
    X0=[0.51,0.5]

    #expected starting params at boundary
    b_vars=[0.6978647 , 0.69902571]

    #time range
    t=(0,6.3)

    #calc the result using a secant method
    res=shooting(X0, dudt, t, b_vars)
    exp = solve_ivp(dudt, t, res).y[:,-1]
    assert np.allclose(b_vars, exp, atol=1e-02)

def test_on_hopf_bif_nor_form_b_0():
    def dudt(t, X):
        """Return a systems of equations relating to the hopf bifurcation"""
        return np.array([
                0*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+0*X[1]-X[1]*(X[0]**2+X[1]**2),
               ])
    #define an initial guess for the starting conditions
    X0=[0,0]
    #expected starting params at boundary
    b_vars=[0, 0]

    #time range
    t=(0,6.3)

    #calc the result using a secant method
    res=shooting(X0, dudt, t, b_vars)
    exp = solve_ivp(dudt, t, res).y[:,-1]
    assert np.allclose(b_vars, exp, atol=1e-02)

def test_on_hopf_bif_nor_form_b_2():
    def dudt(t, X):
        """Return a systems of equations relating to the hopf bifurcation"""
        return np.array([
                2*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+2*X[1]-X[1]*(X[0]**2+X[1]**2),
               ])
    #define an initial guess for the starting conditions
    X0=[1.4,1.4]
    #expected starting params at boundary
    b_vars=[1.00115261, 0.99997944]
    #time range
    t=(0,6.3)

    #calc the result using a secant method
    res=shooting(X0, dudt, t, b_vars)
    exp = solve_ivp(dudt, t, res).y[:,-1]
    assert np.allclose(b_vars, exp, atol=1e-02)

def test_on_hopf_bif_nor_form_b_2():
    beta=2
    def dudt(t, X):
        """Return a systems of equations relating to the hopf bifurcation"""
        return np.array([
             beta*X[0]-X[1]+X[0]*(X[0]**2+X[1]**2)-X[0]*(X[0]**2+X[1]**2)**2,
             X[0]+beta*X[1]+X[1]*(X[0]**2+X[1]**2)-X[1]*(X[0]**2+X[1]**2)**2,
             ])
    #define an initial guess for the starting conditions
    X0=[1,1]
    #expected starting params at boundary
    b_vars=[1.00099494, 0.99852453]
    #time range
    t=(0,6.3)

    #calc the result using a secant method
    res=shooting(X0, dudt, t, b_vars)
    exp = solve_ivp(dudt, t, res).y[:,-1]
    assert np.allclose(b_vars, exp, atol=1e-02)


if __name__ == '__main__':
    unittest.main()
