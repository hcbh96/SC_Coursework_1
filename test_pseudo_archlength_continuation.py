import pytest
from arch_length_continuation import pac
import numpy as np
from scipy.integrate import solve_ivp

def test_solve_cubic():
    """This function seems to always find a root and then a local min"""
    # arrange
    def func_wrapper(v) :
        return lambda x: x**3 -x + v
    V0=np.array([1,1])
    V1=np.array([0.98,1.1])
    p_range=(-2,2)
    step_size=0.1

    #act
    sol=pac(func_wrapper, V0, V1, p_range, step_size, shoot=False)


    #assert
    assert len(sol["params"]) == 5


def test_run_puc_on_hopf_bifurcation_normal_form():
    """Returns two roots to the equation using the newton method"""
    # arrange
    def hopf_bifurcation_norm(beta):
        """Return a systems of equations relating to the hopf bifurcation"""
        return lambda t, X : [
                beta*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+beta*X[1]-X[1]*(X[0]**2+X[1]**2),
               ]


    V0=np.array([1.4, 1.4, 6.2822916, 2])
    V1=np.array([1.3, 1.3, 6.2822916, 1.9])
    p_range=(0,2)
    step_size=-1
    #act
    sol=pac(hopf_bifurcation_norm, V0, V1, p_range, step_size)
    #assert
    assert len(sol["params"]) > 0
    assert len(sol["solutions"]) > 0
    for i in range(len(sol["params"])):
         u=sol["solutions"][i][0:-1]
         T=sol["solutions"][i][-1]
         exp = solve_ivp(hopf_bifurcation_norm(sol["params"][i]), (0,T),u).y[:,-1]
         assert np.allclose(exp,u, atol=1e-03)

