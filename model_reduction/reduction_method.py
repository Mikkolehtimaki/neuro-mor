"""
Base class for model reduction methods.
"""

from enum import Enum

import numpy as np

from model_reduction.ode_model import OdeModel

class SnapshotMethod(Enum):
    """
    Available algorithms for snapshots selection
    """

    fixed = 1
    adaptive = 2

class ReductionMethod():
    """
    ReductionMethod objects are used together with OdeModel objects to create
    reduced models.

    A reduction method must be able to
    - create snapshots from model simulations
    - access the models matrices (A, B, nonlin, u)
    - store the reduced matrices as internal variables
    - implement a method for simulating the reduced model
    """

    # It could be possible to have options, like snapshots generation related
    # settings, as instance variables.. would it be more OOP?

    def __init__(self, ode_model: OdeModel):
        """TODO: to be defined1. """

        self.ode_model = ode_model
        self.solver = ode_model.solver

        # Settings to be changed before calling reduce() or other methods on
        # reduction objects
        self.snapshot_method = SnapshotMethod.fixed
        self.snapshot_threshold = 1
        self.snapshot_interval = 5
        self.reduced_state = None
        self.state = None
        self.full_state = None

    def create_snapshots(self, solutions: list, centering: bool=False):
        """
        Get snapshots from the given data.

        :solutions: should be a matrix (np array) with variables on y and
        timesteps on x.
        :method: one of
            - fixed: constant step sampling of trajectories
            - adaptive: choose entries with euclidian distance higher than
              adaptive_threshold to existing snapshots
        :interval: Step size to use when collecting snapshots with a fixed
        interval
        :returns: TODO

        """

        # Do type checking to make sure indexing works properly
        if type(solutions) != np.ndarray:
            raise ValueError("Solutions matrix should be a numpy array")

        if self.snapshot_method == SnapshotMethod.fixed:

            snapshots = solutions[:, ::self.snapshot_interval]

        elif self.snapshot_method == SnapshotMethod.adaptive:

            print("Using adaptive snapshot collection method with threshold %s"
                  % self.snapshot_threshold)

            # Initialize with first element
            snapshots = [solutions[:, 0]]
            variables = np.shape(solutions)[0]

            # Then iterate over solutions to check which ones should be added.
            # List comp for numpy matrices will iterate rows, we want to
            # iterate columns now
            num_solutions = np.shape(solutions)[1]
            print('Found {} solutions'.format(num_solutions))
            for ind in range(1, num_solutions):
                diff = np.linalg.norm(solutions[:, ind]-snapshots[-1])/variables
                if  diff > self.snapshot_threshold:
                    snapshots.append(solutions[:, ind])

            # for sol in solutions.T:
                # Create a distance matrix first
                # If every distance is greater than threshold, add it to snaps
                # dist_mat = [np.linalg.norm(x-sol) for x in snapshots]
                # if all([x > self.snapshot_threshold for x in dist_mat]):
                #     snapshots = np.vstack((snapshots, sol))

            snapshots = np.array(snapshots).T
            print('Collected {} snapshots'.format(snapshots.shape[1]))

        else:

            raise ValueError("Requested snapshot method is not supported")

        # Centering could be done here, or could be its own function if there
        # are many ways to do it or customize it..
        if centering:
            snapshots = np.subtract(snapshots.T, np.mean(snapshots, axis=1)).T

        return snapshots

    @staticmethod
    def get_basis_vectors(data: list):
        """
        Returns the projection basis vectors.

        :data: Time trajectory data with variables in y and time in x
        :rank: dimensionality of the returned projection space
        :returns: first _rank_ ordered left singular vectors and singular values
        """

        # Use np svd to get the basis.
        left_basis_vectors, singular_values, _ = np.linalg.svd(
            data, full_matrices=False)
        return left_basis_vectors, singular_values

    def to_reduced_basis(self, array):
        """
        Convert a given 1D or 2D array to reduced basis
        """
        raise NotImplementedError("Method must be implemented in a subclass")

    def integrate_step(self, current_time, state):
        """
        Calculate the value of the state vector at the next integration step
        """
        raise NotImplementedError("Method must be implemented in a subclass")

    def reduced_model_call(self, current_time, state):
        """
        Evaluates the reduced model.

        :returns: new state, given current time and state

        """
        raise NotImplementedError("Method must be implemented in a subclass")

    def simulate(self, initial_value, t_start, t_stop, timestep):
        """
        Simulate the model from start to stop current_time.

        Currently the only supported integration method is RK4.
        """

        solutions = []
        self.full_state = initial_value
        self.ode_model.set_state(initial_value)

        # Project the initial value to reduced space
        reduced_init = self.to_reduced_basis(initial_value)
        self.state = reduced_init
        solutions.append(reduced_init)

        # Create steps for simulation. Not including end point so that the
        # values stay more pretty.
        steps = self.ode_model.generate_timesteps(t_start, t_stop, timestep)

        state = reduced_init
        for step in steps[1:]:

            # state = self.solver(self.integrate_step, state, step, timestep)
            state = self.integrate_step(step, timestep)
            solutions.append(state)

        return solutions
