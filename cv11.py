import numpy as np
from scipy.integrate import odeint
from visualization import plot_double_pendulum

G = 9.81
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0

THETA1_0 = 2 * np.pi / 6
THETA2_0 = 5 * np.pi / 8
OMEGA1_0 = 0.0
OMEGA2_0 = 0.0

T_MAX = 20.0
DT = 0.05


def get_derivative(state, t, l1, l2, m1, m2):
    theta1, omega1, theta2, omega2 = state
    delta = theta1 - theta2
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)
    denom = m1 + m2 * sin_d ** 2

    alpha1 = (
        m2 * G * np.sin(theta2) * cos_d
        - m2 * sin_d * (l1 * omega1 ** 2 * cos_d + l2 * omega2 ** 2)
        - (m1 + m2) * G * np.sin(theta1)
    ) / (l1 * denom)

    alpha2 = (
        (m1 + m2) * (l1 * omega1 ** 2 * sin_d - G * np.sin(theta2) + G * np.sin(theta1) * cos_d)
        + m2 * l2 * omega2 ** 2 * sin_d * cos_d
    ) / (l2 * denom)

    return omega1, alpha1, omega2, alpha2


if __name__ == "__main__":
    t = np.arange(0, T_MAX, DT)
    state0 = [THETA1_0, OMEGA1_0, THETA2_0, OMEGA2_0]
    od = odeint(get_derivative, state0, t, args=(L1, L2, M1, M2))

    theta1 = od[:, 0]
    theta2 = od[:, 2]

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    plot_double_pendulum(x1, y1, x2, y2)
