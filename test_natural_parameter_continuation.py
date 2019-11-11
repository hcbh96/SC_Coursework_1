"""This is a test file for the function natural_parameter_continuation, I have used this opportunity to practice TDD and use pytest as an alternative to unittest"""

import pytest
from natural_parameter_continuation import npc

def test_should_use_lambda_as_():
    # arrange
    ode=lambda X,t-0 : [1,b]# system of ODEs
    X0=[2,3]# initial guess
    par0=dict(a=1)
    vary_par=dict(b=2,start=0,stop=2)

    npc(ode, X0, par0, vary_par)
    # act

    # asses
