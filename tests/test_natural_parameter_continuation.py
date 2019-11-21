import pytest
import numpy as np
from scipy.integrate import solve_ivp
from context import methods
from methods.pcontinuation import npc


def test_should_use_solve_method_via_fsolve():
    """This function seems to always find a root and then a local min"""
    # arrange
    def func_wrapper(v) :
        return lambda x: x**3 -x + v
    X0=[1, 10]
    p=(-2,2)


    #act
    sol=npc(func_wrapper, X0, p)


    #assert
    assert len(sol["params"]) > 0
    assert len(sol["solutions"]) > 0
    for i in range(len(sol["params"])):
        u=sol["solutions"][i][0:-1]
        T=sol["solutions"][i][-1]
        exp = solve_ivp(func_wrapper(sol["params"][i]), (0,T),u).y[:,-1]
        assert np.allclose(exp,u, atol=1e-03)


def test_run_npc_on_hopf_bifurcation_normal_form():
    """Returns two roots to the equation using the newton method"""
    # arrange
    def hopf_bifurcation_norm(beta):
        """Return a systems of equations relating to the hopf bifurcation"""
        return lambda t, X : [
                beta*X[0]-X[1]-X[0]*(X[0]**2+X[1]**2),
                X[0]+beta*X[1]-X[1]*(X[0]**2+X[1]**2),
               ]


    state_vec=np.array([0.28843231, 0.31117759, 6.2822916 ])
    p=(0,2)
    n_steps=10
    #act
    sol=npc(hopf_bifurcation_norm, state_vec, p, n_steps)


    #assert
    assert len(sol["params"]) > 0
    assert len(sol["solutions"]) > 0
    for i in range(len(sol["params"])):
         u=sol["solutions"][i][0:-1]
         T=sol["solutions"][i][-1]
         exp = solve_ivp(hopf_bifurcation_norm(sol["params"][i]), (0,T),u).y[:,-1]
         assert np.allclose(exp,u, atol=1e-03)

