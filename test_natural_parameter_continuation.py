"""This is a test file for the function natural_parameter_continuation, I have used this opportunity to practice TDD and use pytest as an alternative to unittest"""

import pytest
import natural_parameter_continuation as npc

@pytest.mark.natural_parameter_continuation_set1

def test_no_ODE_passed_to_eq():
    """This function should ensure that the npc.solve fails with a TypeError if the ODE pased to npc.solve is not a function"""
    #arrange
    ODE=[] # the ODE to use
    X0=1, # the initial state
    par0=1, # the initial parameters
    #act
    throws=False
    try:
        npc.solve(ODE, X0, par0)
    except TypeError:
        throws=True

    assert throws, "This should be equal to True as the function should have thrown"
