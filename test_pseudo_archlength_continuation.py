import pytest
from pseudo_archlength_continuation import pseudo_archlength_continuation as pac

def test_ensure_fsolve_is_called_each_loop():
    #arrange

    #act
    pac(dudt, [])
    #assert

