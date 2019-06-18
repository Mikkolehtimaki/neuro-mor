"""
Unit tests for models.
"""

import unittest
import numpy as np

from model_reduction.reduction_method import ReductionMethod
from model_reduction.ode_model import OdeModel
from model_reduction.models.lotka_volterra import LotkaVolterra
from model_reduction.pod import POD
from model_reduction.deim import DEIM
from model_reduction.deim import Algorithm as deimalg

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

class TestReductionMethods(unittest.TestCase):
    """Tests for OdeModel class. """

    def test_snapshots(self):
        """
        Test that snapshot generation gives the correct shape matrices.
        """

        # Create random data with 5 variables and 100 timesteps
        y_dim = 5
        x_dim = 100
        test_data = np.random.rand(y_dim, x_dim)
        redu_method = ReductionMethod(TwoDimTest())

        snapshots = redu_method.create_snapshots(test_data)
        self.assertTrue(snapshots.shape[0] == y_dim and snapshots.shape[1] <
                        x_dim)

    def test_basis_dims(self):
        """
        Test that reduced basis calculation gives correct matrices

        """

        rows = 100
        cols = 500
        test_mat = np.random.rand(rows, cols)

        rank = 25
        left_basis, singular_vals = ReductionMethod.get_basis_vectors(
            test_mat)
        left_basis = left_basis[:, :rank]
        singular_vals = singular_vals[:rank]

        # Check that calculated matrices have correct dimensions
        self.assertTrue(singular_vals.shape[0] == rank)
        self.assertTrue(left_basis.shape[0] == rows)
        self.assertTrue(left_basis.shape[1] == rank)

    def test_pod(self):
        """
        Test POD reduction method
        """

        # Workflow is to create an ode model, simulate it, then pass it to the
        # reduction method (if the method uses empirical data, like POD)
        lotka_volterra = LotkaVolterra()
        init = [10, 10]
        t_start = 0
        t_stop = 100
        dt = 0.1
        lotka_volterra.simulate(
            init, t_start, t_stop, dt
        )

        pod = POD(lotka_volterra)

        # Reduce to rank two (actually full rank)
        rank = 2
        pod.reduce(rank)

        # Check that calculated matrices have correct dimensions
        self.assertTrue(pod.reduced_state_mat.shape[0] == rank)
        self.assertTrue(pod.reduced_state_mat.shape[1] == rank)
        self.assertTrue(pod.reduced_input_mat.shape[0] == rank)
        self.assertTrue(pod.reduced_input_mat.shape[1] ==
                        pod.ode_model.input_mat.shape[1])

        solutions = pod.simulate(init, t_start, t_stop, dt)

        # Project back
        solutions = np.array(solutions).T
        solutions = np.matmul(pod.pod_basis, solutions)

        orig_sol = np.array(lotka_volterra.solutions).T

        # import matplotlib.pyplot as plt
        # plt.title("POD vs Original")
        # plt.plot(orig_sol[0, :])
        # plt.plot(solutions[0, :], c='r')
        # plt.show()

        # Results should be identical with plain POD if no dimensions were
        # discarded
        for idx, element in enumerate(orig_sol[0, :]):
            self.assertAlmostEqual(element, solutions[0, idx])

    def test_deim_algorithm(self):
        """
        Test DEIM interpolation index / matrix algorithm

        """

        lotka_volterra = LotkaVolterra()
        init = [10, 10]
        t_start = 0
        t_stop = 100
        dt = 0.1
        rank = 1

        lotka_volterra.simulate(init, t_start, t_stop, dt)
        deim = DEIM(lotka_volterra)
        nonlin_snaps = deim.create_snapshots(
            np.array(lotka_volterra.nonlinear_values).T
        )
        deim_basis, _ = DEIM.get_basis_vectors(nonlin_snaps)
        deim_basis = deim_basis[:, :rank]

        P, U, inds = DEIM.deim(deim_basis)

        # Test that returned values have correct shapes
        self.assertTrue(P.shape[1] == rank)
        self.assertTrue(U.shape[1] == rank)
        self.assertTrue(len(inds) == rank)
        for ind, i in enumerate(inds):
            self.assertTrue(P[i, ind] == 1)

    def test_deim(self):
        """
        Test DEIM simulation
        """

        lotka_volterra = LotkaVolterra()
        init = [10, 10]
        t_start = 0
        t_stop = 100
        dt = 0.1
        rank = 2

        lotka_volterra.simulate(init, t_start, t_stop, dt)

        deim = DEIM(lotka_volterra)

        # Test different rank combinations, none should error
        # deim.reduce(1, 2)
        # deim.reduce(2, 1)
        # Final simulation with full ranks
        deim.reduce(2, rank)

        solutions = deim.simulate(init, t_start, t_stop, dt)

        # Project back
        solutions = np.array(solutions).T
        solutions = np.matmul(deim.pod_basis, solutions)

        orig_sol = np.array(lotka_volterra.solutions).T

        # import matplotlib.pyplot as plt
        # plt.title("DEIM vs Original")
        # plt.plot(orig_sol[0, :])
        # plt.plot(solutions[0, :], c='r')
        # plt.show()

        # Results should be identical with plain POD if no dimensions were discarded
        for idx, element in enumerate(orig_sol[0, :]):
            self.assertAlmostEqual(element, solutions[0, idx])

if __name__ == "__main__":
    unittest.main()
