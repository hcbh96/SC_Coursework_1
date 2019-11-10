from generalised_shooting_method import shooting

def npc(ode, X0, par0, vary_par, step_size=0.1, mex_steps=100, discretisation=shooting):
    """Function performs natural parameter continuation, i.e., it
simply increments the a parameter by a set amount and attempts
to find the solution for the new parameter value using the last
found solution as an initial guess."""
    #par0 input the initial par0 ensure this is a scalar or array of scalars
    #ensure vary_par give n index from the parameters array can not be greater than par0 array length
    #ensure step_size is a scalar
    #ensure max steps is 100
    #ensure discretion is a function
    #ensure a solver has been psed to the func

    # set i to vary params initial value
    i=vary_par.min

    # while parameter is below limit
    while i <= vary_par.max:

        #if discretisation
            #run shooting

        #if not discretisation
            #run fsolve = TODO maybe put _func_to_solve into its own folder

        # run discretisation method
        discretisation(dXdt, X0, t, boundary_vars, integrator=odeint, root_finder=newton, tol=1e-4, maxiter=50)
        # print result of the shooting method

        # update initial guess

        # incremement paramater by a set amount

ode=False
X0=1
par0=dict(a=1)
vary_par=dict(key='c',start=0,stop=2)
discretisation=False
func_to_solve=lambda X, t=0 : X**3 - X + c


