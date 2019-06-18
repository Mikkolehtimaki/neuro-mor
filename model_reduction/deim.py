"""
Model reduction with Proper Orthogonal decomposition and Discrete empirical
interpolation method
"""

from enum import Enum

import numpy as np

from model_reduction.pod import POD
from model_reduction.ode_model import OdeModel

class Algorithm(Enum):
    """
    Available algorithms for calculating a DEIM basis and interpolation points.
    """

    DEIM = 1

class DEIM(POD):

    """Docstring for DEIM. """

    def __init__(self, ode_model: OdeModel):

        """
        Initialize and object for POD+DEIM reduction
        """

        super().__init__(ode_model)
        self.deim_inds = None
        self.deim_ind_mat = None
        self.deim_singular_values = None
        self.deim_basis = None
        self.precompute_deim = None
        self.funcs = None
        self.deim_rank = None
        self.full_state = None

    @staticmethod
    def deim(basis: "2D array of left singular vectors"):
        """
        DEIM algorithm for fiding interpolation indices and DEIM projection matrix
        :returns: Tuple of sampling point matrix, projection basis and sampling indices
        """

        # Start by selecting the first interpolation index from the largest element
        # in the first basis column vector
        ind = np.argmax(abs(basis[:, 0]))

        # The U matrix is used to calculate the best interpolation indices. It is
        # assembled from column vectors. It is initialized with the first left
        # singular vector.
        U = np.matrix(basis[:, 0]).T

        # P will hold the final interpolation indices in the form of natural basis
        # column vectors. Each column will be unique. Must be a matrix.
        # Initialize with natural basis of index of largest value of first basis
        # vector
        P = np.matrix(np.zeros(np.shape(basis[:, 0]))).T
        P[ind, 0] = 1
        # Also save the found indices for later
        indices = [ind]

        # Rest of the indices are found with the following algorithm
        # Iterate from second index over all columns of input basis
        for vec in range(1, np.shape(basis)[1]):

            u_l = np.matrix(basis[:, vec]).T
            # Solve (P.T*U)c = P.T*u_l for c, must have matrices as input
            c = np.linalg.solve(
                np.matmul(P.T, U),
                np.matmul(P.T, u_l)
            )

            # Compute a residual that determines the index of the next DEIM
            # interpolation point
            residual = u_l - np.matmul(U, c)
            ind = np.argmax(abs(residual))

            # Grow the matrices used for next iteration
            U = np.concatenate((U, u_l), axis=1)
            P = np.concatenate((P, np.zeros(u_l.shape)), axis=1)
            P[ind, vec] = 1
            indices.append(ind)

        return P, U, indices

    def reduce(self, pod_rank: int, deim_rank: int, snapshot_inds: list = None,
               algorithm: Algorithm=Algorithm.DEIM):
        """
        Reduce a model with given POD and DEIM ranks.
        :pod_rank: POD rank of the reduced model
        :deim_rank: DEIM rank of the reduced model
        :snapshot_inds: Indices for snapshot gathering. If given, snapshots will not be acquired
        with static or adaptive methods
        :algorithm: Algorithm for calculating DEIM interpolation point indices and basis
        """

        # Do POD stuff first
        super().reduce(pod_rank)
        self.deim_rank = deim_rank

        # First step is to collect both total and nonlinear snapshots
        if snapshot_inds is not None:
            nonlinear_snapshots = np.array(self.ode_model.nonlinear_values).T
            nonlinear_snapshots = nonlinear_snapshots[:, snapshot_inds]
        else:
            nonlinear_snapshots = self.create_snapshots(
                np.array(self.ode_model.nonlinear_values).T
            )

        # Get the singular values and left singular vectors for DEIM
        full_deim_basis, self.deim_singular_values = self.get_basis_vectors(
            nonlinear_snapshots
        )
        self.deim_basis = full_deim_basis[:, :deim_rank].copy()

        # Note that we could also get the basis for full rank, and save that
        # Then to get reduced models, could just drop columns from those..

        # Next we need interpolation indices, those are given by the DEIM
        # algorithm
        if algorithm == Algorithm.DEIM:

            self.deim_ind_mat, _, self.deim_inds = self.deim(self.deim_basis)

            self.precompute_deim = np.array(np.matmul(
                self.pod_basis.T,
                np.matmul(
                    self.deim_basis,
                    np.linalg.inv(np.matmul(self.deim_ind_mat.T, self.deim_basis))
                )
            ))

        else:

            raise ValueError('Unknown DEIM algorithm requested')

        self.funcs = [self.ode_model.nonlinear_vector[idx] for idx in self.deim_inds]

    def reduced_model_call(self, current_time, state):
        """
        Evaluates the reduced model.

        :returns: new state, given current time and state

        """

        # Evaluate nonlinear vector at the indices chosen by DEIM
        full_state = np.matmul(self.pod_basis, state)
        full_state = self.ode_model.set_state(full_state)
        nonlin_eval = [f(current_time) for f in self.funcs]

        # Project the nonlinear values to reduced space in place with using the
        # precomputed DEIM basis
        # return np.dot(self.reduced_state_mat, state) + \
        return np.dot(self.reduced_state_mat, np.dot(self.pod_basis.T, full_state)) + \
            np.dot(self.precompute_deim, nonlin_eval) + \
            np.dot(self.reduced_input_mat,
                   self.ode_model.input_vector(current_time))
