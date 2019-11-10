"""
Test Generalised shooting method contains test aimed at ensuring the Generalised shooting method is up to scratch and working properly.
"""

from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

from generalised_shooting_method import shooting
import numpy as np
import unittest

#define a test class
class TestGeneralisedShootingMethod(unittest.TestCase):
    def test_on_constant_derivative_negative(self):
        # rate of change set to conistant
        def dXdt(X, t=0):
            return 1

        #define an initial guess
        X0=1

        #expected starting params at boundaries
        boundary_vars=1

        # define timespan
        t=np.linspace(0,1,10)

        #calc the result using a secant method
        res=shooting(dXdt, X0, t, boundary_vars)

        self.assertFalse(np.isclose(res, 0.38, atol=1e-01).all())


    def test_on_constant_derivative_positive(self):
        """This function is designed to test the generalised shooting method to an accuracy of 6 decimal places on a very simple system of ODEs"""
        # rate of change set to constant
        def dXdt(X, t=0):
            return [1,1]

        #define an initial guess
        X0=[1,1]

        #expected starting params at boundaries
        boundary_vars=[1,1]

        # define timespan
        t=np.linspace(0,1,10)

        #calc the result using a secant method
        res=shooting(dXdt, X0, t, boundary_vars)

        self.assertTrue(np.isclose(res, [0,0], atol=1e-06).all())


    def test_on_lotka_volterra(self):
         """This function is intended to test the generalised shooting method on the Lotka Volterra method ensuring that the absolute tolerance of the solution is within 2 decimal places

         params such [a,b,d] below should be defined outside the function as it means one can write loops with varying params during numerical analysis"""
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
         res=shooting(dXdt_lotka_volterra, X0, t, boundary_vars)
         self.assertTrue(np.isclose(res, 0.38, atol=1e-02).all())


    def test_on_hopf_bifurcation(self):
        """
        This function is intended to test the  generalised shooting method on the Hopf Bifurcation ensuring that the absolute tolerance is within 2 decimal places
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
        res=shooting(dXdt_hopf_bif, X0, t, boundary_vars)
        expected = [ 0.51, -0.15 ]
        self.assertTrue(np.isclose(res, expected, atol=1e-02).all())


    def test_on_system_of_three(self):
         """
         This function is designed to ensure that the generalised shooting method works well when given a system of 3 first order ODEs

         When Testing out generalised shooting method for a system of equations containing three first order ODEs I decided it was in the interest of the reader to use the most basic example possible, to allow the system to be envisaged by the reader and give them undoubtable confidence that the test is testing for the correct output

         The specific point of this test is to ensure that my shooting method works on systems of first order linear ODEs of greater than 2 dimensions, this test focuses solely on doing that in the most simple way possible, i.e by having a system of three linearly independent first order ODE which all have solutions at 0.
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
         res=shooting(dXdt, X0, t, boundary_vars)
         expected = [0 , 0, 0 ]
         self.assertTrue(np.isclose(res, expected, atol=1e-01).all())


    def test_unmatched_input_dimensions_1(self):
        """This test should throw a RuntimeError due to unmatched dimension sizes between the boundary variables and the number of 1st order ODEs"""
        #arrange
        def dXdt(X, t=0):
            return np.array([0,0])
        X0=[0,0,0] # this variable has too many inputs
        boundary_vars=[0,0]
        t=np.linspace(0,10,100)
        #act
        throws = False
        try:
            res=shooting(dXdt, X0, t, boundary_vars)
        except RuntimeError:
            throws = True
        #assert
        self.assertTrue(throws)


    def test_unmatched_input_dimensions_2(self):
        """This test should throw a ValueError due to unmatched dimension sizes between the initial guess variables and the number of 1st order ODEs"""
        #arrange
        def dXdt(X, t=0):
            return np.array([0,0])
        X0=[0,0]
        boundary_vars=[0,0,0] # this variable has too many inputs
        t=np.linspace(0,10,100)
        throws = False
        #act
        try:
            res=shooting(dXdt, X0, t, boundary_vars)
        except ValueError:
            throws = True
        #assert
        self.assertTrue(throws)


    def test_shooting_method_with_no_convergence(self):
        """This function ensures that in the case that the shooting method does not converge an error is returned describing the issue"""
        #define params
        a=1; d=0.1; b=0.3

        # rate of change and pred and prey populations
        def dXdt(X,t=0):
            """Return the change in pred and prey populations"""
            return 0

        #define an initial guess for the starting conditions
        X0=2

        #expected starting params at boundary
        boundary_vars=0

        #time range
        t=np.linspace(0,10,100)
        #calc the result using a secant method
        throws = False
        try:
            res=shooting(dXdt, X0, t, boundary_vars, maxiter=1)
        except RuntimeError:
            throws = True

        self.assertTrue(throws)

    def test_function_with_fsolve(self):
        """
        This function is designed to ensure that the Generalised shooting method is able to work with other integrators such as fsolve
        """
        sigma=-1;beta=1
        def dXdt(X, t=0):
          """Function to calculate the rate of change of the Hopf Bifurcation     at position X"""
          return np.array([X[0],X[1],X[2]])

        # make an initial condition guess
        X0=[1,0,0.5]

        # set the boundary conditions
        boundary_vars=[1,1,1]

        # define a time range
        t=np.linspace(0,10,100)

        #find the solution of the generalised shooting method
        res=shooting(dXdt, X0, t, boundary_vars, root_finder=fsolve)
        expected = [0 , 0, 0 ]
        self.assertTrue(np.isclose(res, expected, atol=1e-01).all())


    """Other test I beleive are fairly important to perform for the sake of TDD are mapped out below I will complete them if I have time before the end of the course work"""
    #test to see if fsolve has been called with the correct params
    #test to see if newton has been called with the correct params
    #test to see if when called with newton the code still works
    #test to ensure params passed to newton are the correct params
    #test to ensure the function throws if neither newton nor fsolve are passed to the function
    def test_func_throws_with_non_defined_root_finder(self):
         # rate of change set to conistant
         def dXdt(X, t=0):
             return 1
         def random_root_finder():
             return ""

         #define an initial guess
         X0=1

         #expected starting params at boundaries
         boundary_vars=1

         # define timespan
         t=np.linspace(0,1,10)

         throws=False
         #calc the result using a secant method
         try:
             res=shooting(dXdt, X0, t, boundary_vars, root_finder=random_root_finder)
         except AttributeError:
             throws=True

         self.assertTrue(throws)

    #test to see if it works when odeint is passed to the function
    def test_should_solve_solve_ivp(self):
         # rate of change set to conistant
         def dXdt(X, t=0):
             return [1,1]

         #define an initial guess
         X0=[1,1]

         #expected starting params at boundaries
         boundary_vars=[1,1]

         # define timespan
         t=(1,10)

         #calc the result using a secant method
         res=shooting(dXdt, X0, t, boundary_vars, integrator=solve_ivp)

         self.assertFalse(np.isclose(res, [0.3], atol=1e-01).all())

    #test to see if when solve_ivp is passed to function it still works
    #test to see the params that are passed to solve_ivp
    #test to see the params that are passed to odeint
    #test to see if when neither odeint or solve_ivp it throws
    def test_func_throws_with_non_defined_integrator(self):
        # rate of change set to conistant
        def dXdt(X, t=0):
          return 1
        def random_integrator(X):
          return X*2+1

        #define an initial guess
        X0=1

        #expected starting params at boundaries
        boundary_vars=1

        # define timespan
        t=np.linspace(0,1,10)

        throws=False
        #calc the result using a secant method
        try:
          res=shooting(dXdt, X0, t, boundary_vars,         integrator=random_integrator)
        except AttributeError:
          throws=True

        self.assertTrue(throws)


if __name__ == '__main__':
    unittest.main()
