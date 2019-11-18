from scipy.optimize import fsolve
from generalised_shooting_method import shooting
"""
1. start with two solutions [[u0, t0, p0],[u1,t1,p1]]
2. write v=[u,t,p] extended state vector
3. predict next solution v3_hat=v2+delta_v1_v2
4. solve [u3-uT, du0, delta_v1_v2(*dot*)(v2-v2_hat)]=0
"""
def func_to_solve(v, dudt, t, b_vars, v_guess, delta_v):
    #TODO: Test params passed to this function
    #TODO: Decouple input functions and test input output
    # v=[u, p]
    return [
        shooting(dudt, v, t, b_vars, solve_derivative=False),
        shooting(dudt, v, t, b_vars, solve_derivative=True),
        lambda v : np.dot((v-v_guess), delta_v),
        ]

#TODO: Ensure the function is properly documented
def pseudo_archlength_continuation(dudt, v0, v1, t, step_size, p_range, b_vars):
    """

    step_size : +ive if steps increasing, -ive if steps decreasing
    """

    # set up loop and termination
    while p >= p_range[0] and p <= p_range[1]
        # calc secant
        delta_v = v1 - v0

        # using v1 and delta_v calc v2 guess
        v_guess = v1 + delta_v*step_size

        # solve for root #TODO decouple functions and unit test
        v2=fsolve(func_to_solve, v_guess, args=(dudt, t, b_vars, v_guess, delta_v))
        #[shoot(periodicity), shoot(phase), np.dot((v2-v_guess), delta_v)=0]

        # reassign and re-run
        v0=v1;v1=v2


#NOTE could have an error handler incase user supplies v0 or v1 with p outside p_range

def dudt(v):
    # update this to take [u,p]
    return t, v : np.array([
        v*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
        X[0]+v*X[1]-X[1]*(X[0]**2+X[1]**2),
        ])

u0=[1,1]
p0=2
v0=[u0, p0]
vary_par=dict(start=2, stop=-1, steps=50)
b_vars=[1,1]
t=(0,6.35)

pseudo_archlength_continuation()
