"""
Unit tests for models.
"""

import unittest
import numpy as np

from model_reduction.ode_model import OdeModel
from model_reduction.models.lotka_volterra import LotkaVolterra

class TwoDimTest(OdeModel):

    """Docstring for TwoDimTest. """

    def __init__(self):
        """TODO: to be defined1. """

        A = [
            [3, -4],
            [1, -1]
        ]
        B = [
            [0],
            [0]
        ]
        super().__init__(A, B)

class TestOdeModel(unittest.TestCase):
    """Tests for OdeModel class. """

    def test_init(self):
        """
        Test that a new object is created successfully and that errors are
        thrown if matrices are not defined.
        """
        test_model = TwoDimTest()
        self.assertTrue(test_model.state_mat is not None)

    def test_simulation(self):
        """
        Test that a new object is created successfully and that errors are
        thrown if matrices are not defined.
        """
        two_dim_model = TwoDimTest()
        init = [0, 0]
        t_start = 0
        t_stop = 5
        dt = 0.1
        two_dim_model.simulate(
            init, t_start, t_stop, dt
        )

    def test_nonlinear_system(self):
        """TODO: Docstring for test_nonlinear_system.
        :returns: TODO

        """

        lotka_volterra = LotkaVolterra()
        init = [10, 10]
        t_start = 0
        t_stop = 100
        dt = 0.1
        lotka_volterra.simulate(
            init, t_start, t_stop, dt
        )

if __name__ == "__main__":
    unittest.main()
