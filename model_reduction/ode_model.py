"""
Base class for a model.

A model is a group of ordinary differential equations
and it is described with matrices.
"""

import numpy as np

from model_reduction.integration_methods import rk23

class OdeModel():

    """Docstring for OdeModel. """

    def __init__(self, state_mat_A, input_mat_B, object_index: int = 0):
        """
        Initializes an ODE model.

        The matrices A and B are required to create an object.
        After initialization, the deriving class should also set properties for
        self.nonlinear_vector and self.input_vector.
        """

        # Initialize the internal matrices. These are required for most ODE
        # systems, hence B being required
        self.state_mat = np.array(state_mat_A)
        self.input_mat = np.array(input_mat_B)

        # Values for every variable in the current moment
        self.object_index = object_index

        # Initialize nonlinear part, which is a vector valued function.
        # Defaults to all zero vector. Should be set to a new value when
        # creating systems with nonlinearities
        self.nonlinear_vector = self.default_nonlinear()

        # Initialize a vector for inputs to the system. Should be updated if
        # the system has inputs.
        self.input_vector = self.default_u_t

        # Initialize a callback function that is called in every loop of
        # simulate() function
        self.callback = self.default_callback

        # By default use rk23 solver
        self.solver = rk23

        # Initialize lists for saving simulation trajectories and nonlinear
        # intermediates for model reduction
        self.nonlinear_values = []
        self.solutions = []
        self.timesteps = 0.05
        self.state = None

    def default_nonlinear(self):
        """TODO: Docstring for nonlinear_vector.
        :returns: TODO

        Evaluate nonlinear vector of ODE system.

        The default implementation assumes nonlinear part is zero. The user
        should assign a new fuction to self.nonlinear_vector
        in the inheriting classes.

        This should return a vector of functions with length of the state vector.
        """

        funcs = [lambda t: 0]*self.state_mat.shape[0]
        return funcs

    def default_u_t(self, current_time):
        """
        current_time dependent inputs to the system.

        The default implementation assumes it is zero. The user should override
        this method in inheriting classes.

        Should return a vector with size B.shape[1]
        """

        return np.zeros((self.input_mat.shape[1]))

    def default_callback(self, current_time):
        """
        Placeholder for callback during simulation.

        Default implementation will not modify the state
        """

        return self.state

    def model_call(self, current_time, state):
        """
        Evaluates the model equations, given the current current_time and state.

        :returns: Next state, given current time and state
        """

        # Update the state so that the nonlinear functions use the current state even in the
        # middle of runge kutta steps
        state = self.set_state(state)

        # Only need to feed in time because nonlinear evaluation uses the model's internal state
        self.nonlin_eval = [f(current_time) for f in self.nonlinear_vector]
        return np.dot(self.state_mat, state) + \
            self.nonlin_eval + \
            np.dot(self.input_mat, self.input_vector(current_time))

    def generate_timesteps(self, t_start: float, t_stop: float, timestep: float):
        """
        Generates evenly spaced timesteps for simulation.

        :t_start: Start of simulation
        :t_stop: Stop of simulation
        :timestep: dt of simulation
        :returns: array of timesteps

        """
        steps = np.linspace(t_start, t_stop, num=int(t_stop/timestep)+1,
                            endpoint=True)
        self.timesteps = steps
        return steps

    def integrate_step(self, current_time, timestep):
        """
        Integrate forward for one timestep.

        :current_time: time of the system
        :timestep: step size in seconds.
        """

        # Set the state of the system to the return value from ode solver
        # so that callbacks can be checked and an updated state eventually returned
        new_state = self.solver(self.model_call, self.state, current_time, timestep)
        # self.set_state(self.model_call(current_time, state))

        # Keep nonlinear values for reduction purposes
        self.nonlinear_values.append(self.nonlin_eval)

        # run callbacks
        self.set_state(new_state)
        self.state = self.callback(current_time)

        return self.state

    def set_state(self, new_state):
        """
        Update the internal state to given state
        """
        self.state = new_state
        return self.state

    def simulate(self, initial_value, t_start, t_stop, timestep):
        """
        Simulate the model from start to stop time with given timestep.

        Simulation trajectories / solutions are saved internally, so that a
        model can be simulated numerous time with different parameters or
        initial values before reduction.

        :initial_value: Array of initial values
        :t_start: Simulation timestep, corresponding to initial values
        :t_stop: last included simulation timestep
        :returns: Solutions for the current parameters and initial value
        """

        if len(initial_value) != self.state_mat.shape[0]:
            raise ValueError("Length of initial value vector does not "
                             "correspond to state matrix")

        self.timestep = timestep
        solutions = []
        self.nonlinear_values = []
        solutions.append(initial_value)

        # Create steps for simulation. Not including end point so that the
        # values stay more pretty.
        steps = self.generate_timesteps(t_start, t_stop, timestep)

        initial_value = np.array(initial_value)
        state = self.set_state(initial_value)

        for current_step in steps[1:]:

            # state = self.solver(self.integrate_step, state, current_step, timestep)
            state = self.integrate_step(current_step, timestep)
            solutions.append(state)

        self.solutions = solutions
        return solutions
