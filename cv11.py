import numpy as np
from scipy.integrate import odeint
from visualization import plot_double_pendulum

# Gravitační zrychlení [m/s²]
G = 9.81

# Délky tyčí [m] a hmotnosti závaží [kg]
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0

# Počáteční úhly [rad] — θ₁ = 60°, θ₂ = 112.5°
THETA1_0 = 2 * np.pi / 6
THETA2_0 = 5 * np.pi / 8

# Počáteční úhlové rychlosti — kyvadlo začíná v klidu
OMEGA1_0 = 0.0
OMEGA2_0 = 0.0

# Délka simulace [s] a časový krok [s]
T_MAX = 60.0
DT = 0.05


def get_derivative(state, t, l1, l2, m1, m2):
    """Vrátí derivace stavu (θ̇₁, θ̈₁, θ̇₂, θ̈₂) pro odeint.

    Rovnice jsou odvozeny z Euler-Lagrangeova formalismu pro dvojité kyvadlo.
    Stav: [θ₁, ω₁, θ₂, ω₂]
    """
    theta1, omega1, theta2, omega2 = state

    delta = theta1 - theta2
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)

    # Společný jmenovatel obou rovnic: l * (m1 + m2·sin²(θ1−θ2))
    denom = m1 + m2 * sin_d ** 2

    # Úhlové zrychlení prvního kyvadla θ̈₁
    alpha1 = (
        m2 * G * np.sin(theta2) * cos_d
        - m2 * sin_d * (l1 * omega1 ** 2 * cos_d + l2 * omega2 ** 2)
        - (m1 + m2) * G * np.sin(theta1)
    ) / (l1 * denom)

    # Úhlové zrychlení druhého kyvadla θ̈₂
    alpha2 = (
        (m1 + m2) * (l1 * omega1 ** 2 * sin_d - G * np.sin(theta2) + G * np.sin(theta1) * cos_d)
        + m2 * l2 * omega2 ** 2 * sin_d * cos_d
    ) / (l2 * denom)

    return omega1, alpha1, omega2, alpha2


if __name__ == "__main__":
    # Pole časových kroků a počáteční stav [θ₁, ω₁, θ₂, ω₂]
    t = np.arange(0, T_MAX, DT)
    state0 = [THETA1_0, OMEGA1_0, THETA2_0, OMEGA2_0]

    # Numerická integrace soustavy diferenciálních rovnic
    # od[:, 0] = θ₁,  od[:, 2] = θ₂  (sloupce 1 a 3 jsou úhlové rychlosti)
    od = odeint(get_derivative, state0, t, args=(L1, L2, M1, M2))

    theta1 = od[:, 0]
    theta2 = od[:, 2]

    # Kartézské souřadnice závaží z úhlů
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    plot_double_pendulum(x1, y1, x2, y2)
