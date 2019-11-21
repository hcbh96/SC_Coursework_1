import numpy as np
from scipy.optimize import root
from generalised_shooting_method import shooting

def func_to_solve(v, func_wrapper, t, b_vars, delta_v, v_guess):
    #TODO: Test params passed to this function
    #TODO: Decouple input functions and test input output
    # v=[u, p]

    # func to guess to dudt
    dudt=func_wrapper(v[1])
    return [
        shooting(v[0], dudt, t, b_vars),
        lambda v : np.dot(delta_v, (v-v_guess)),
        ]

#TODO: Ensure the function is properly documented
def pseudo_archlength_continuation(func_wrapper, v0, v1, t,
        step_size, p_range, b_vars):
    """
    step_size : +ive if steps increasing, -ive if steps decreasing
    """
    #prep_solution
    solution = { "params": [], "solutions": [] }

    # set up loop and termination
    while v1[1] >= p_range[0]
    and v1[1] <= p_range[1]
    and v1[1] >= p_range[0]
    and v1[1] <= p_range[1]:

        # calc secant
        delta_v = v1 - v0

        # using v1 and delta_v calc v2 guess
        v_guess = v1 + delta_v*step_size

        # solve for root #TODO decouple functions and unit test
        v2=root(func_to_solve, v_guess,
                args=(func_wrapper, t, b_vars, v_guess, delta_v)).x
        print("V2 : {}".format(v2))


        # reassign, re-run and prep solution
        v0=v1;v1=v2
        solution["params"].append(v1[1])
        solution["solutions"].append(v1[0])



def func_wrapper(v):
    # update this to take [u,p]
    return lambda t, X : np.array([
        v*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
        X[0]+v*X[1]-X[1]*(X[0]**2+X[1]**2),
        ])

u0=np.array([1,1])
p0=0
v0=np.array([u0, p0])
u1=np.array([1.5,1.5])
p1=0.1
v1=np.array([u1,p1])

p_range=[0,2]
b_vars=np.array([1,1])

step_size=0.1

t=(0,6.35)

sol=pseudo_archlength_continuation(func_wrapper, v0, v1,
        t, step_size, p_range, b_vars)
print("Solution: {}".format(sol))
