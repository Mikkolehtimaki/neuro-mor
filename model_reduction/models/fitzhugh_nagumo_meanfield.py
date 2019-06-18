"""
Model definition for a discretized FN model
"""

import math

import numpy as np

from model_reduction.ode_model import OdeModel
from model_reduction.integration_methods import rk23

class FitzHughNagumoMeanfield(OdeModel):

    """Docstring for HodgkinHuxleyCompartment. """

    def __init__(self, n_v: int = 15, n_w: int = 15, n_y: int = 15):
        """
        Initialize a FitzHugh-Nagumo model
        :n_v: discretization points in V direction
        :n_w: discretization points in W direction
        :n_y: discretization points in Y direction
        """

        self.n_v = n_v
        self.n_w = n_w
        self.n_y = n_y

        # Initial condition
        self.mean_v0 = 0.0
        self.mean_w0 = 0.5
        self.mean_y0 = 0.3
        self.sigma_v0 = 0.4
        self.sigma_w0 = 0.4
        self.sigma_y0 = 0.05

        # Grid parameters
        self.v_min = -3.0
        self.v_max = 3.0
        self.w_min = -2.0
        self.w_max = 2.0
        self.y_min = 0.01
        self.y_max = 1.0

        # Discretization step size
        self.h_v = (self.v_max - self.v_min)/self.n_v
        self.h_w = (self.w_max - self.w_min)/self.n_w
        self.h_y = (self.y_max - self.y_min)/self.n_y

        # Discretized arrays
        self.discrete_v = [self.v_min + i*self.h_v for i in range(self.n_v)]
        self.discrete_w = [self.w_min + i*self.h_w for i in range(self.n_w)]
        self.discrete_y = [self.y_min + i*self.h_y for i in range(self.n_y)]

        # Model parameters
        self.Sigma_ext = 0.0
        self.Sigma_J = 0.2
        self.V_rev = 1.0    # Synaptic reverse potential
        self.Alpha = 1.0    # Rate at which the synapses turn ON ...
        self.Beta = 1.0     # ... and OFF
        self.T_max = 1.0    # Maximum amplitude, ...
        self.Lambda = 0.2   # ... slope parameter ...
        self.V_T = 2.0      # ... and firing threshold of the sigmoidal function in the synaptic SDE (3rd equation of the model)
        self.A = 0.1        # Parameters of the function \chi (see Eq. (3) in the paper)
        self.B = 0.5
        self.a = 0.7        # These are...
        self.b = 0.8        # ...the parameters...
        self.c = 0.08       # ...of the FitzHugh-Nagumo model
        self.I_ext = 0.4        # Amplitude and ...
        self.Jbar = 1.0      # Mean value and ...

        self.update()
        self.input_vector = np.array([0])

        super().__init__(self.state_mat, self.input_mat, 0)

        self.nonlinear_vector = self.generate_nonlinear_vector()
        self.integral_points = self.generate_integral_points()
        self.solver = rk23

    def generate_state_mat(self):
        """
        Return the state matrix, given the current instance variables (model
        parameters).

        :returns: State matrix A

        """

        rows = []
        for i in range(0, self.n_v):
            for j in range(0, self.n_w):
                for k in range(0, self.n_y):

                    # How to handle the boundary points? If i, j, k too close to boundary, drop
                    # those terms?
                    # row = np.zeros(int(self.n_y*self.n_w*self.n_v))
                    # if k < 2 or j < 2 or i < 2:
                    #     rows.append(row)
                    #     continue

                    # if k > self.n_y - 3 or j > self.n_w - 3 or i > self.n_v - 3:
                    #     rows.append(row)
                    #     continue

                    mat = np.zeros((self.n_v, self.n_w, self.n_y))

                    mat[i, j, k] = -30*self.Sigma_ext**2/(24*self.h_v**2) - 30*self.sigma_Y_squared(
                        self.discrete_v[i], self.discrete_y[k])/(24*self.h_y**2)

                    if i > 1:
                        mat[i-2, j, k] = -self.Sigma_ext**2/(24*self.h_v**2) - 1/(12*self.h_v)*(
                            self.discrete_v[i-2] - self.discrete_v[i-2]**3/3 - self.discrete_w[j]
                            + self.I_ext)

                    if i > 0:
                        mat[i-1, j, k] = 16*self.Sigma_ext**2/(24*self.h_v**2) + 8/(12*self.h_v)*(
                            self.discrete_v[i-1] - self.discrete_v[i-1]**3/3 - self.discrete_w[j]
                            + self.I_ext)

                    if i < self.n_v - 1:
                        mat[i+1, j, k] = 16*self.Sigma_ext**2/(24*self.h_v**2) - 8/(12*self.h_v)*(
                            self.discrete_v[i+1] - self.discrete_v[i+1]**3/3 - self.discrete_w[j]
                            + self.I_ext)

                    if i < self.n_v - 2:
                        mat[i+2, j, k] = -self.Sigma_ext**2/(24*self.h_v**2) + 1/(12*self.h_v)*(
                            self.discrete_v[i+2] - self.discrete_v[i+2]**3/3 - self.discrete_w[j]
                            + self.I_ext)

                    if j > 1:
                        mat[i, j-2, k] = -1/(12*self.h_w)*(self.c*(
                            self.discrete_v[i] + self.a - self.b*self.discrete_w[j-2]))

                    if j > 0:
                        mat[i, j-1, k] = 8/(12*self.h_w)*(self.c*(
                            self.discrete_v[i] + self.a - self.b*self.discrete_w[j-1]))

                    if j < self.n_w - 1:
                        mat[i, j+1, k] = -8/(12*self.h_w)*(self.c*(
                            self.discrete_v[i] + self.a - self.b*self.discrete_w[j+1]))

                    if j < self.n_w - 2:
                        mat[i, j+2, k] = 1/(12*self.h_w)*(self.c*(
                            self.discrete_v[i] + self.a - self.b*self.discrete_w[j+2]))

                    if k > 1:
                        mat[i, j, k-2] = -self.sigma_Y_squared(
                            self.discrete_v[i], self.discrete_y[k-2])/(24*self.h_y**2) - (
                                1/(12*self.h_y))*(self.Alpha*self.synaptic_S(self.discrete_v[i])*(
                                    1-self.discrete_y[k-2]) - self.Beta*self.discrete_y[k-2])

                    if k > 0:
                        mat[i, j, k-1] = 16*self.sigma_Y_squared(
                            self.discrete_v[i], self.discrete_y[k-1])/(24*self.h_y**2) + 8*(
                                1/(12*self.h_y))*(self.Alpha*self.synaptic_S(self.discrete_v[i])*(
                                    1-self.discrete_y[k-1]) - self.Beta*self.discrete_y[k-1])

                    if k < self.n_y - 1:
                        mat[i, j, k+1] = 16*self.sigma_Y_squared(
                            self.discrete_v[i], self.discrete_y[k+1])/(24*self.h_y**2) - 8*(
                                1/(12*self.h_y))*(self.Alpha*self.synaptic_S(self.discrete_v[i])*(
                                    1-self.discrete_y[k+1]) - self.Beta*self.discrete_y[k+1])

                    if k < self.n_y - 2:
                        mat[i, j, k+2] = -self.sigma_Y_squared(
                            self.discrete_v[i], self.discrete_y[k+2])/(24*self.h_y**2) + (
                                1/(12*self.h_y))*(self.Alpha*self.synaptic_S(self.discrete_v[i])*(
                                    1-self.discrete_y[k+2]) - self.Beta*self.discrete_y[k+2])

                    rows.append(mat.ravel(order='C'))
        return np.vstack(rows)


    def update(self):
        """
        Updates the internal state and input matrices.

        This should be called _manually_ after parameters are changed.
        """
        self.state_mat = self.generate_state_mat()
        self.input_mat = self.generate_input_mat()
        self.nonlinear_vector = self.generate_nonlinear_vector()
        self.integral_points = self.generate_integral_points()

    def generate_input_mat(self):
        """
        Return a list of functions describing the nonlinear term.
        :returns: List of functions, same length as indices

        """

        return np.zeros((self.n_y*self.n_w*self.n_v, 1))

    def generate_nonlinear_vector(self):
        """
        Return a list of functions describing the nonlinear term.
        :returns: List of functions, same length as indices

        """
        # Every function in this list must have identical signature
        funcs = []

        for i in range(0, self.n_v):
            for j in range(0, self.n_w):
                for k in range(0, self.n_y):

                    fun = self.make_nonlinear(k, j, i)
                    funcs.append(fun)

        return funcs

    def make_nonlinear(self, k: 'y', j: 'w', i: 'v'):
        """
        Return the nonlinear part of the equation in coordinates y, w, v
        """

        def f(current_time):

            # if k < 2 or j < 2 or i < 2:
            #     return 0

            # if k > self.n_y - 3 or j > self.n_w - 3 or i > self.n_v - 3:
            #     return 0

            state = self.state

            integral = 125.0*self.h_v*self.h_w*self.h_y*np.sum(np.multiply(
                self.integral_points, state))/23887872.0

            p_v_w_y = state[k + self.n_y*j + self.n_y*self.n_w*i]
            value = -30*math.pow(self.Sigma_J*integral*(self.discrete_v[i]-self.V_rev),
                                 2)*p_v_w_y/(24*self.h_v**2)
            if i >= 2: # v_i_minus_2
                p_v_minus_2 = state[k + self.n_y*j + self.n_y*self.n_w*(i-2)]
                value += (-math.pow(self.Sigma_J*integral*(self.discrete_v[i-2] - self.V_rev),
                                    2)/(24*self.h_v**2) + self.Jbar*integral*(
                                        self.discrete_v[i-2] - self.V_rev)/(12*self.h_v))*p_v_minus_2
            if i >= 1: # v_i_minus_1
                p_v_minus_1 = state[k + self.n_y*j + self.n_y*self.n_w*(i-1)]
                value += (16*math.pow(self.Sigma_J*integral*(self.discrete_v[i-1] - self.V_rev),
                                      2)/(24*self.h_v**2) - 8*self.Jbar*integral*(
                                          self.discrete_v[i-1] - self.V_rev)/(12*self.h_v))*p_v_minus_1
            if i < self.n_v-1: # v_i_plus_1
                p_v_plus_1 = state[k + self.n_y*j + self.n_y*self.n_w*(i+1)]
                value += (16*math.pow(self.Sigma_J*integral*(self.discrete_v[i+1] - self.V_rev),
                                      2)/(24*self.h_v**2) + 8*self.Jbar*integral*(
                                          self.discrete_v[i+1] - self.V_rev)/(12*self.h_v))*p_v_plus_1
            if i < self.n_v-2: # v_i_plus_2
                p_v_plus_2 = state[k + self.n_y*j + self.n_y*self.n_w*(i+2)]
                value += (-math.pow(self.Sigma_J*integral*(self.discrete_v[i+2] - self.V_rev),
                                    2)/(24*self.h_v**2) - self.Jbar*integral*(
                                        self.discrete_v[i+2] - self.V_rev)/(12*self.h_v))*p_v_plus_2
            return value
        return f

    def generate_integral_points(self):
        """
        indices for triple integral
        """

        # Generate indices for integration points
        int_points_V = [0]*(self.n_v)
        int_points_W = np.zeros((self.n_v, self.n_w))
        int_points_Y = np.zeros((self.n_v, self.n_w, self.n_y))
        # Variables for integrating over the grid of p(V)
        x_1 = 0
        dx = 1

        # Newton-cotes points
        coeffs = [19, 75, 50, 50, 75, 19]
        for i in range(1, int(self.n_v/5)):
            interp_points = [x_1+(5*i-m)*dx for m in range(5, -1, -1)]
            # At every point of V, add up the coefficients
            for ind, point in enumerate(interp_points):
                int_points_V[point] += coeffs[ind]

        # For W, every point corresponds to an array
        for i in range(1, int(self.n_w/5)):
            interp_points = [x_1+(5*i-m)*dx for m in range(5, -1, -1)]
            for ind, point in enumerate(interp_points):
                int_points_W[:, point] = np.add(int_points_W[:, point], np.multiply(coeffs[ind], int_points_V))

        # For integration over Y, we need to coefficients in a cube
        for i in range(1, int(self.n_y/5)):
            interp_points = [x_1+(5*i-m)*dx for m in range(5, -1, -1)]
            interp_points_y = [self.y_min+(5*i-m)*self.h_y for m in range(5, -1, -1)]
            for ind, point in enumerate(interp_points):
                int_points_Y[:, :, point] = np.add(
                    int_points_Y[:, :, point], np.multiply(coeffs[ind]*interp_points_y[ind], int_points_W))

        return np.array(int_points_Y).astype(np.float64).ravel(order='C')

    def synaptic_S(self, V: float):
        """
        Concentration of transmitter released into the synaptic cleft. See equation 8 in propagation
        of chaos paper.

        :voltage: variable V
        :returns: S(V)
        """

        return self.T_max/(1+math.exp(self.Lambda*(self.V_T-V)))

    def chi(self, Y: float):
        """
        Function chi(Y), Eq 3 in original paper. We can only implement the case 0 < x < 1,
        because the boundaries of Y are [0, 1]? At least the original author's implementation
        only considers this case

        """

        return self.A*math.exp(-self.B/(1-(math.pow(2*Y-1, 2))))

    def sigma_Y_squared(self, V: float, Y: float):
        """
        Equation 9 from the paper. It is squared in the model (Eq 32) so here it is written squared
        (gets rid of one nasty sqrt).
        """

        return (self.Alpha*self.synaptic_S(V)*(1-Y) + self.Beta*Y)*math.pow(self.chi(Y), 2)

    def get_initial_values(self):
        """
        Return initial values.
        """
        initial_values = np.zeros(int(self.n_v*self.n_w*self.n_y), dtype=float)
        for i in range(2, self.n_v-2):
            for j in range(2, self.n_w-2):
                for k in range(2, self.n_y-2):

                    initial_values[k + self.n_y*j + self.n_y*self.n_w*i] = (
                        1./(pow(2.*np.pi, 3/2.)*self.sigma_v0*self.sigma_w0*self.sigma_y0))*np.exp(
                            -0.5*(pow((self.v_min + i*self.h_v-self.mean_v0)/self.sigma_v0, 2) +
                                  pow((self.w_min + j*self.h_w-self.mean_w0)/self.sigma_w0, 2) +
                                  pow((self.y_min + k*self.h_y-self.mean_y0)/self.sigma_y0, 2)))

        return initial_values

    def compute_VW_density(self, state):
        """
        Computes the marginal probability density over Y, at the given state.
        Also returns the volume of the probability mass under the curve.

        :returns: density, hyper_volume
        """
        state = state.copy().reshape((self.n_v, self.n_w, self.n_y))
        marginal_density_VW = np.zeros((self.n_v, self.n_w), dtype=float)
        for i in range(0, self.n_v):
            for j in range(0, self.n_w):
                for k in range(1, int(self.n_y/5.)):
                    marginal_density_VW[i, j] += 19*state[i, j, 5*k-5] \
                    + 75*state[i, j, 5*k-4] + 50*state[i, j, 5*k-3] + 50*state[i, j, 5*k-2] \
                        + 75*state[i, j, 5*k-1] + 19*state[i, j, 5*k]

                marginal_density_VW[i, j] = 5*self.h_y*marginal_density_VW[i, j]/288.

        marginal_density_V = np.zeros(self.n_v, dtype=float)

        for i in range(0, self.n_v):
            for j in range(1, int(self.n_w/5.)):
                marginal_density_V[i] += 19*marginal_density_VW[i, 5*j-5] \
                    + 75*marginal_density_VW[i, 5*j-4] + 50*marginal_density_VW[i, 5*j-3] \
                    + 50*marginal_density_VW[i, 5*j-2] + 75*marginal_density_VW[i, 5*j-1] \
                    + 19*marginal_density_VW[i, 5*j]

            marginal_density_V[i] = 5*self.h_w*marginal_density_V[i]/288.

        hyper_volume = 0.
        for i in range(1, int(self.n_v/5.)):
            hyper_volume += 19*marginal_density_V[5*i-5] + 75*marginal_density_V[5*i-4] \
                + 50*marginal_density_V[5*i-3] + 50*marginal_density_V[5*i-2] \
                + 75*marginal_density_V[5*i-1] + 19*marginal_density_V[5*i]
        hyper_volume = 5*self.h_v*hyper_volume/288.
        return marginal_density_VW, hyper_volume


if __name__ == "__main__":

    import time
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
    points_v = 30
    points_w = 25
    points_y = 20
    model = FitzHughNagumoMeanfield(points_v, points_w, points_y)
    inits = model.get_initial_values()

    print('S: %s' % model.synaptic_S(0.5))
    print('chi: %s' % model.chi(0.5))
    print('Sigma_Y: %s' % model.sigma_Y_squared(0.5, 0.5))

    t_i = 0.0      # Initial and ...
    h_t = 0.01
    t_f = 2.2 - h_t     # ... final time instant for the numerical integration

    start = time.time()
    sol = model.simulate(inits, t_i, t_f, h_t)
    stop = time.time() - start
    print('Time elapsed: {}s'.format(stop))

    final_sol = sol[-1]

    density_VW, hyper_vol = model.compute_VW_density(final_sol)

    MAX = density_VW.max()

    print('hyper vol: %s' % hyper_vol)
    print('MAX: %s' % MAX)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    W_grid, V_grid = np.meshgrid(model.discrete_w, model.discrete_v)

    surf = ax.plot_surface(V_grid, W_grid, density_VW, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.set_zlim3d(0.0, MAX)

    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

    ax.set_title('u(t_fixed = %.2f, V, W), Volume = %.2f' % (t_f, hyper_vol))
    ax.set_xlabel('V')
    ax.set_ylabel('W')
    ax.set_zlabel('u(t_fixed = ' + str(t_f) + ',V,W)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('fokker_planck_vectorized.png')
    plt.show()
