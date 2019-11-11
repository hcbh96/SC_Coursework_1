from generalised_shooting_method import shooting
from scipy.optimize import fsolve
from find_root import find_root

def npc(func_wrapper, X0, vary_par, boundary_vars=[0], step_size=0.1, max_steps=100, method='shooting', root_finder=fsolve):
    """Function performs natural parameter continuation, i.e., it
simply increments the a parameter by a set amount and attempts
to find the solution for the new parameter value using the last
found solution as an initial guess.

        Parameters
        ____________________
        func_wrapper : a function in the form f(v) : g(x) where g(x) is the function to be solved which contains the parameter to be varies noted as v. For example def f(v): return lambda x : x + v

        X0 : guess of the solution to function g(x) with vary_param set to start

        vary_par : the range over which to vary the vary_param

        boundary_vars=0 : optional, boundary conditions if a shooting method needs to be used

        step_size=0.1 : the step size to increment v by in each itteration

        max_steps=100 : optionsal, the max number of steps in the continuation method

        method='shooting' : optional the method to use to solve the equation can be shooting or just solve at the moment

        root_finder=fsolve : optional the root_finding method that you would like to use can be fsolve or root_finder atm

"""

    # set i to vary params initial value
    v=vary_par['start']

    #steps
    steps=0

    # while parameter is below limit
    while v <= vary_par['stop'] and steps <= max_steps:
        """By passing the function wrapper instead of the function I can update the function definition at run time allowing var_par to change with each iteration"""
        func=func_wrapper(v)
        # split method here TODO add a colloction option
        if method == 'solve':
            sol=find_root(func, X0, root_finder) #TODO add tol and maxiter
        elif method.__name__ == 'shooting':
           sol= method(func, X0, t, boundary_vars, integrator=odeint, root_finder=newton, tol=1e-4, maxiter=50)

        # update initial guess
        X0=sol

        print([v,sol])


        # incremement paramater by a set amount
        v=v+step_size


def func_wrapper(v) :
    return lambda x: x**3 -x + v

X0=1
vary_par=dict(start=-2,stop=2)
npc(func_wrapper, X0, vary_par, method='solve',root_finder=fsolve)
