from scipy.optimize import fsolve
from scipy.optimize import newton
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import math
from context import methods
from methods.shooting import shooting
import numpy as np
import pytest

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
     X0=[0.51, 0.5, 20.76]

     #calc the result using a secant method
     res=shooting(X0, dXdt)
     u=res[0:-1]
     exp = solve_ivp(dXdt, (0,res[-1]), u).y[:,-1]
     assert np.allclose(exp, u, atol=1e-02)


def test_on_hopf_bif_nor_form_b_1():
    def dudt(t, X):
        """Return a systems of equations relating to the hopf bifurcation"""
        return np.array([
                1*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+1*X[1]-X[1]*(X[0]**2+X[1]**2),
               ])

    #define an initial guess for the starting conditions
    X0=[0.51,0.5,6.3]

    #calc the result using a secant method
    res=shooting(X0, dudt)
    u=res[0:-1]
    exp = solve_ivp(dudt, (0,res[-1]), u).y[:,-1]
    print("Norm: {}".format(np.linalg.norm(u)))
    assert np.allclose(exp, u, atol=1e-02)

def test_on_hopf_bif_nor_form_b_0():
    def dudt(t, X):
        """Return a systems of equations relating to the hopf bifurcation"""
        return np.array([
                0*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+0*X[1]-X[1]*(X[0]**2+X[1]**2),
               ])
    #define an initial guess for the starting conditions
    X0=[0,0,6.3]

    #calc the result using a secant method
    res=shooting(X0, dudt)
    u=res[0:-1]
    exp = solve_ivp(dudt, (0,res[-1]), u).y[:,-1]
    assert np.allclose(exp, u, atol=1e-02)

def test_on_hopf_bif_nor_form_b_2():
    def dudt(t, X):
        """Return a systems of equations relating to the hopf bifurcation"""
        return np.array([
                2*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+2*X[1]-X[1]*(X[0]**2+X[1]**2),
               ])
    #define an initial guess for the starting conditions
    X0=[1.4,1.4,6.3]

    #calc the result using a secant method
    res=shooting(X0, dudt)
    u=res[0:-1]
    exp = solve_ivp(dudt, (0,res[-1]), u).y[:,-1]
    assert np.allclose(exp, u, atol=1e-02)


def test_on_hopf_bif_nor_form_b_2():
    beta=2
    def dudt(t, X):
        """Return a systems of equations relating to the hopf bifurcation"""
        return np.array([
             beta*X[0]-X[1]+X[0]*(X[0]**2+X[1]**2)-X[0]*(X[0]**2+X[1]**2)**2,
             X[0]+beta*X[1]+X[1]*(X[0]**2+X[1]**2)-X[1]*(X[0]**2+X[1]**2)**2,
             ])
    #define an initial guess for the starting conditions
    X0=[1,1,6.3]

    #calc the result using a secant method
    res=shooting(X0, dudt)
    u=res[0:-1]
    exp = solve_ivp(dudt, (0,res[-1]), u).y[:,-1]
    assert np.allclose(exp, u, atol=1e-02)

if __name__ == '__main__':
    unittest.main()
