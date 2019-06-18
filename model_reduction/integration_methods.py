"""
Integration methods
"""

def euler(f, x, t, dt: float = 0.5):
    """
    Forward euler integration
    """
    return x + dt*f(t, x)

def rk23(f, x, t, dt):
    """
    Integrate one step of dt with Runge-Kutta order 2 with alpha = 2/3.

    :f: function with signature f(t, x)
    :x: state variables
    :t: time
    :dt: timestep
    """

    k_1 = f(t, x)
    k_2 = f(t + 2*dt/3, x + (2*dt/3)*k_1)
    return x + dt*(0.25*k_1 + 0.75*k_2)
