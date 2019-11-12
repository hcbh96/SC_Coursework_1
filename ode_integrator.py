from scipy.integrate import odeint

# TODO : allow any integrator to be used and the args for the integrator to be passed in a more dynamic fashion i.e "args=()"

#define function to pass to solve
def func_to_solve(X, dXdt, t, boundary_vars, integrator=odeint, solve_derivative=False):
    """Returns the integration of a defined derivative from a defined intial value over a defined period. Additional options include adding in boundary conditions to help with the shooting method and a choice of method when using the solve_ivp integrator.

    ____________________

    Parameters :

    X : nparray of scalars defining a tarting position for the integrator
    dXdt : np array defining a system of first order ODEs
    t : time period over which to run the integration - this argument must be passed in correctly:
        - for solve_ivp : t is a 2-tuple of floats. Interval of integration (t0, tf). The solver starts with t=t0 and integrates until it reaches t=tf.
        - for odeint : t is an array. A sequence of time points for which to solve for y. The initial value point should be the first element of this sequence. This sequence must be monotonically increasing or monotonically decreasing; repeated values are allowed.
    integrator=odeint : optional, can either be odeint or solve_ivp
    """

    if integrator.__name__ not in ['solve_ivp', 'odeint']:
        raise AttributeError("This function only works with either solve_ivp or odeint from scipy.integrate")
    if integrator.__name__ == 'odeint':
        sol=boundary_vars - integrator(dXdt,X,t)[-1]
    elif integrator.__name__ == 'solve_ivp':
        sol=boundary_vars-integrator(dXdt, t, X).y[-1,-1]
    if solve_derivative:
        sol=dXdt(sol)

    return sol