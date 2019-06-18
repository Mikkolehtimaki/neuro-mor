"""
Model reduction with Proper Orthogonal Decomposition (POD).

POD can be applied to linear and nonlinear models, but will not efficiently
reduce nonlinear models.
"""

import numpy as np

from model_reduction.reduction_method import ReductionMethod
from model_reduction.ode_model import OdeModel

class POD(ReductionMethod):

    """
    Docstring for POD.

    Should the workflow be this:
    1. Initialize with model to be reduced with
        a. Model to be reduced (OdeModel object)
        b. Rank of reduced model
        c. Snapshot collection method
    2. Call reduce, which does work that can be run offline, like actual
    snapshot collection and matrix calculations
    3. Simulate
    4. Gather results
    """

    def __init__(self, ode_model: OdeModel):
        """
        Initialize an object for POD reduction
        """

        super().__init__(ode_model)
        self.pod_basis = None
        self.full_pod_basis = None
        self.singular_values = None
        self.reduced_state_mat = None
        self.reduced_input_mat = None
        self.pod_rank = None

    def reduce(self, rank: int, snapshot_inds: list = None, centering: bool=False):
        """
        Reduce a model to the given rank.
        :rank: Rank of the reduced model
        :snapshot_inds: Indices for snapshot gathering. If given, snapshots will not be acquired
        with static or adaptive methods
        """

        self.pod_rank = rank

        # First step is to get snapshots.
        # They are stored in a list of lists so we must fetch them and use an
        # array with dimensions that reduction method handles
        if snapshot_inds is not None:
            snapshots = np.array(self.ode_model.solutions).T
            snapshots = snapshots[:, snapshot_inds]
        else:
            snapshots = self.create_snapshots(
                np.array(self.ode_model.solutions).T, centering=centering)

        # TODO: Cache the snapshots, if method is called again it would save
        # time

        # Get the singular values and left singular vectors
        self.full_pod_basis, self.singular_values = self.get_basis_vectors(snapshots)
        # It is extremely important to use copy() of the VIEW to create a NEW
        # array, instead of saving just the view to the original FULL
        # dimensional basis. The effect to run time is HUGE.
        self.pod_basis = self.full_pod_basis[:, :rank].copy()
        # self.pod_basis = np.eye(self.pod_basis.shape[0])[:, :rank]

        if len(self.singular_values[:rank]) != rank:
            raise RuntimeWarning("POD REDUCTION COULD NOT ACHIEVE DESIRED RANK! "
                                 "(possibly not enough empirical data provided)")

        # Calculate POD reduced matrices
        self.reduced_state_mat = np.matmul(
            np.matmul(self.pod_basis.T, self.ode_model.state_mat),
            self.pod_basis)
        self.reduced_input_mat = np.matmul(self.pod_basis.T, self.ode_model.input_mat)

    def reduced_model_call(self, current_time, state):
        """
        Evaluates the reduced model.

        :returns: new state, given current time and state

        """

        # TODO: To optimize POD, would make sense to have the ode_model declare
        # itself as linear or nonlinear, and use a corresponding evaluation
        # method. For linear models, this implementation has a lot of
        # unnecessary overhead

        # Evaluate nonlinear values with the full state vector
        full_state = np.dot(self.pod_basis, state)
        full_state = self.ode_model.set_state(full_state)
        nonlin_eval = [f(current_time) for f in
                       self.ode_model.nonlinear_vector]

        # Project the nonlinear values to reduced space in place
        # return np.dot(self.reduced_state_mat, state) + \
        return np.dot(self.reduced_state_mat, np.dot(self.pod_basis.T, full_state)) + \
            np.dot(self.pod_basis.T, nonlin_eval) + \
            np.dot(self.reduced_input_mat,
                   self.ode_model.input_vector(current_time))

    def integrate_step(self, current_time, timestep):
        """
        Integrate forward for one timestep.

        :timestep: step size in seconds.
        """

        # Use the same solver as the underlying model had specified
        self.state = self.solver(self.reduced_model_call, self.state, current_time,
                                 timestep)

        # Project back after integration to check for any callbacks
        # This must be done in the original space.
        # After callbacks, go back to reduced space.

        # full_state = np.matmul(self.pod_basis, self.state)
        # self.ode_model.set_state(full_state)
        # full_state = self.ode_model.callback(current_time)
        # self.state = np.matmul(self.pod_basis.T, full_state)
        return self.state

    def to_reduced_basis(self, array):
        """
        Convert a given 1D or 2D array to reduced basis
        """
        reduced = np.dot(self.pod_basis.T, array)
        if len(reduced) != self.reduced_state_mat.shape[0]:
            raise ValueError("Length of initial value vector does not "
                             "correspond to state matrix")
        return reduced
