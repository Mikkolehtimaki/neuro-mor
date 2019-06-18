"""
Lotka Volterra implementation
"""

import numpy as np

from model_reduction.ode_model import OdeModel

class LotkaVolterra(OdeModel):

    """Docstring for LotkaVolterra. """

    alpha = 0.1
    beta = 0.02
    gamma = 0.4
    delta = 0.02

    def __init__(self):
        """TODO: to be defined1. """
        A = [
            [self.alpha, 0],
            [0, -self.gamma]
        ]
        B = [[0], [0]]

        super().__init__(A, B)

        # Set the property to point to the updated nonlinear vector
        self.nonlinear_vector = self.nonlinear()

    # This is the way to override the default nonlinear function
    # Just keep signature the same as in base class
    def nonlinear(self):
        """
        Lotka-Volterra nonlinear part.
        """

        return [
            lambda t: -self.beta*self.state[0]*self.state[1],
            lambda t: self.delta*self.state[0]*self.state[1]
        ]

if __name__ == "__main__":

    lv = LotkaVolterra()
    init = [10, 10]
    t_start = 0
    t_stop = 100
    dt = 0.1
    lv.simulate(
        init, t_start, t_stop, dt
    )

    # Plot for visual verification
    import matplotlib.pyplot as plt
    sol = lv.solutions
    plt.plot(sol)
    plt.show()
